# ==============================================================================
# AeroSweep — Modul Training
# File: src/aerosweep/neuro/train_cv.py
#
# Tanggung jawab:
#   1. Membaca konfigurasi training dari configs/cv.yaml
#   2. Memvalidasi dataset hasil preprocessing sudah siap
#   3. Fine-tuning model YOLO11n-seg (instance segmentation)
#   4. Menyimpan bobot terbaik ke models/nn_weights/aerosweep_trained/
#   5. Mengekspor ringkasan hasil training
#
# Dipanggil oleh: scripts/02_train_cv.py
# ==============================================================================

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# Setup Logger
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AeroSweep.train_cv")


# ==============================================================================
# KELAS UTAMA: YOLOTrainer
# ==============================================================================

class YOLOTrainer:
    """
    Kelas untuk fine-tuning model YOLO instance segmentation
    pada dataset DroneWaste yang sudah dipreprocess.

    Alur kerja:
        trainer = YOLOTrainer(config_path="configs/cv.yaml", root_dir=ROOT_DIR)
        trainer.validate_dataset()
        trainer.train()
        trainer.export_summary()

    Contoh penggunaan langsung:
        trainer = YOLOTrainer("configs/cv.yaml", Path("."))
        trainer.validate_dataset()
        trainer.train()
    """

    def __init__(
        self,
        config_path: str,
        root_dir: Path,
        override_epochs:  Optional[int]   = None,
        override_batch:   Optional[int]   = None,
        override_device:  Optional[str]   = None,
        override_workers: Optional[int]   = None,
    ):
        """
        Parameters
        ----------
        config_path      : Path ke configs/cv.yaml
        root_dir         : Path root project (untuk resolve path relatif)
        override_epochs  : Override jumlah epoch dari cv.yaml (opsional)
        override_batch   : Override batch size dari cv.yaml (opsional)
        override_device  : Override device dari cv.yaml, misal "cpu" / "0" / "0,1"
        override_workers : Override jumlah DataLoader workers
        """
        self.config_path = Path(config_path)
        self.root_dir    = Path(root_dir)

        # Override dari CLI (lebih prioritas dari cv.yaml)
        self.override_epochs  = override_epochs
        self.override_batch   = override_batch
        self.override_device  = override_device
        self.override_workers = override_workers

        # Load konfigurasi
        self.config = self._load_config()

        # Ekstrak parameter training dari config
        self.train_cfg   = self.config.get("training", {})
        self.dataset_cfg = self.config

        # Path-path penting
        self.dataset_path = self.root_dir / self.config.get("path", "data/vision")
        self.model_name   = self.train_cfg.get("model", "yolo11n.pt")
        self.model_path   = self.root_dir / self.model_name
        self.output_dir   = self.root_dir / self.train_cfg.get(
            "project", "models/nn_weights"
        )
        self.run_name     = self.train_cfg.get("name", "aerosweep_trained")

        # Path model terbaik hasil training
        self.best_model_path = (
            self.output_dir / self.run_name / "weights" / "best.pt"
        )

        # Resolusi parameter final (override > config)
        self.epochs  = self.override_epochs  or self.train_cfg.get("epochs",  100)
        self.batch   = self.override_batch   or self.train_cfg.get("batch",    16)
        self.device  = self.override_device  or self.train_cfg.get("device",    0)
        self.workers = self.override_workers or self.train_cfg.get("workers",   8)
        self.imgsz   = self.train_cfg.get("imgsz",         512)
        self.lr0     = self.train_cfg.get("lr0",          0.01)
        self.lrf     = self.train_cfg.get("lrf",          0.01)
        self.momentum     = self.train_cfg.get("momentum",    0.937)
        self.weight_decay = self.train_cfg.get("weight_decay", 0.0005)
        self.patience     = self.train_cfg.get("patience",       20)
        self.augment      = self.train_cfg.get("augment",       True)
        self.seed         = self.train_cfg.get("seed",           42)
        self.verbose      = self.train_cfg.get("verbose",       True)

        logger.info(
            f"YOLOTrainer diinisialisasi: model={self.model_name}, "
            f"epochs={self.epochs}, batch={self.batch}, device={self.device}"
        )

    # --------------------------------------------------------------------------
    # PRIVATE: Load konfigurasi
    # --------------------------------------------------------------------------

    def _load_config(self) -> dict:
        """Baca cv.yaml dan kembalikan sebagai dictionary."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"File konfigurasi tidak ditemukan: {self.config_path}"
            )
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Konfigurasi dimuat dari: {self.config_path}")
        return config

    # --------------------------------------------------------------------------
    # PUBLIC: Validasi dataset sebelum training
    # --------------------------------------------------------------------------

    def validate_dataset(self) -> bool:
        """
        Validasi dataset hasil preprocessing sudah siap untuk training.

        Pengecekan:
        1. Folder train/images dan train/labels ada dan tidak kosong
        2. Folder val/images dan val/labels ada dan tidak kosong
        3. cv.yaml sudah memiliki nama kelas (bukan placeholder)
        4. Model pretrained tersedia di root project
        5. Jumlah file gambar dan label seimbang (tidak ada yang hilang)

        Returns
        -------
        True jika semua validasi lulus.

        Raises
        ------
        SystemExit jika ada validasi yang gagal.
        """
        logger.info("Memvalidasi dataset sebelum training...")
        errors   = []
        warnings = []

        processed_dir = self.dataset_path / "processed"

        # --- Cek folder train ---
        train_img_dir = processed_dir / "train" / "images"
        train_lbl_dir = processed_dir / "train" / "labels"

        if not train_img_dir.exists():
            errors.append(
                f"Folder train/images tidak ditemukan: {train_img_dir}\n"
                f"  → Jalankan scripts/01_preprocessing.py terlebih dahulu."
            )
        else:
            train_imgs = list(train_img_dir.glob("*.jpg"))
            if not train_imgs:
                errors.append(
                    f"Folder train/images kosong: {train_img_dir}\n"
                    f"  → Jalankan scripts/01_preprocessing.py terlebih dahulu."
                )
            else:
                logger.info(f"  Train images  : {len(train_imgs)} file ✓")

        if not train_lbl_dir.exists():
            errors.append(f"Folder train/labels tidak ditemukan: {train_lbl_dir}")
        else:
            train_lbls = list(train_lbl_dir.glob("*.txt"))
            logger.info(f"  Train labels  : {len(train_lbls)} file ✓")

        # --- Cek folder val ---
        val_img_dir = processed_dir / "val" / "images"
        val_lbl_dir = processed_dir / "val" / "labels"

        if not val_img_dir.exists():
            errors.append(
                f"Folder val/images tidak ditemukan: {val_img_dir}\n"
                f"  → Jalankan scripts/01_preprocessing.py terlebih dahulu."
            )
        else:
            val_imgs = list(val_img_dir.glob("*.jpg"))
            if not val_imgs:
                errors.append(f"Folder val/images kosong: {val_img_dir}")
            else:
                logger.info(f"  Val images    : {len(val_imgs)} file ✓")

        if not val_lbl_dir.exists():
            errors.append(f"Folder val/labels tidak ditemukan: {val_lbl_dir}")
        else:
            val_lbls = list(val_lbl_dir.glob("*.txt"))
            logger.info(f"  Val labels    : {len(val_lbls)} file ✓")

        # --- Cek cv.yaml sudah diupdate (bukan placeholder) ---
        nc    = self.config.get("nc", 0)
        names = self.config.get("names", {})

        if nc == 0 or not names:
            errors.append(
                "cv.yaml belum diupdate dengan nama kelas.\n"
                "  → Jalankan scripts/01_preprocessing.py terlebih dahulu\n"
                "    agar bagian 'nc' dan 'names' terisi dari JSON."
            )
        else:
            logger.info(f"  Jumlah kelas  : {nc} ✓")
            logger.info(f"  Nama kelas    : {list(names.values())[:3]}... ✓")

        # --- Cek model pretrained tersedia ---
        if not self.model_path.exists():
            errors.append(
                f"Model pretrained tidak ditemukan: {self.model_path}\n"
                f"  → Pastikan {self.model_name} ada di root project.\n"
                f"    Ultralytics akan otomatis download jika ada koneksi internet."
            )
        else:
            logger.info(f"  Model pretrained: {self.model_path} ✓")

        # --- Cek keseimbangan file gambar dan label ---
        if train_img_dir.exists() and train_lbl_dir.exists():
            train_imgs_stem = {f.stem for f in train_img_dir.glob("*.jpg")}
            train_lbls_stem = {f.stem for f in train_lbl_dir.glob("*.txt")}
            missing_labels  = train_imgs_stem - train_lbls_stem

            if missing_labels:
                warnings.append(
                    f"{len(missing_labels)} file gambar train tidak memiliki "
                    f"file label yang sesuai. Contoh: "
                    f"{list(missing_labels)[:3]}"
                )

        # --- Tampilkan hasil validasi ---
        if warnings:
            for w in warnings:
                logger.warning(f"  ⚠ {w}")

        if errors:
            logger.error("Validasi dataset GAGAL:")
            for e in errors:
                logger.error(f"  ✗ {e}")
            sys.exit(1)

        logger.info("Validasi dataset: PASSED ✓")
        return True

    # --------------------------------------------------------------------------
    # PUBLIC: Training utama
    # --------------------------------------------------------------------------

    def train(self) -> dict:
        """
        Jalankan fine-tuning YOLO instance segmentation.

        Menggunakan Ultralytics YOLO API yang sudah diuji dan
        terdokumentasi di https://docs.ultralytics.com/modes/train/

        Proses:
        1. Load model pretrained yolo11n.pt
        2. Fine-tune pada dataset DroneWaste
        3. Simpan best.pt ke models/nn_weights/aerosweep_trained/weights/
        4. Update path model di cv.yaml untuk digunakan inferensi

        Returns
        -------
        Dictionary berisi ringkasan hasil training (metrics).
        """
        logger.info("=" * 60)
        logger.info("Memulai fine-tuning YOLO Instance Segmentation")
        logger.info("=" * 60)
        self._print_training_plan()

        # Buat direktori output
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Load model pretrained
        # ------------------------------------------------------------------
        logger.info(f"Memuat model: {self.model_path}")
        try:
            model = YOLO(str(self.model_path))
        except Exception as e:
            logger.error(f"Gagal memuat model: {e}")
            raise
        if getattr(model, "task", None) != "segment":
            raise ValueError(
                "Model bukan instance segmentation (task != 'segment'). "
                "Gunakan bobot *-seg.pt atau model segmentation."
            )

        # ------------------------------------------------------------------
        # Mulai training
        # Semua parameter didokumentasikan di:
        # https://docs.ultralytics.com/modes/train/#train-settings
        # ------------------------------------------------------------------
        start_time = time.time()

        try:
            results = model.train(
                # --- Dataset ---
                data        = str(self.config_path),
                imgsz       = self.imgsz,

                # --- Training schedule ---
                epochs      = self.epochs,
                batch       = self.batch,
                patience    = self.patience,

                # --- Optimizer ---
                lr0          = self.lr0,
                lrf          = self.lrf,
                momentum     = self.momentum,
                weight_decay = self.weight_decay,

                # --- Augmentasi ---
                augment     = self.augment,

                # --- Hardware ---
                device      = self.device,
                workers     = self.workers,

                # --- Output ---
                project     = str(self.output_dir),
                name        = self.run_name,
                exist_ok    = True,    # Overwrite run sebelumnya

                # --- Reproduktibilitas ---
                seed        = self.seed,

                # --- Logging ---
                verbose     = self.verbose,

                # --- Simpan model ---
                # save=True      : simpan best.pt dan last.pt
                # save_period=-1 : tidak simpan checkpoint per epoch
                save        = True,
                save_period = self.train_cfg.get("save_period", -1),

                # --- Pretrained ---
                # True = gunakan bobot pretrained (fine-tuning)
                # False = training dari scratch
                pretrained  = True,
            )

        except KeyboardInterrupt:
            logger.warning("Training dihentikan oleh pengguna (Ctrl+C).")
            logger.warning(
                "Model terakhir tersimpan di: "
                f"{self.output_dir / self.run_name / 'weights' / 'last.pt'}"
            )
            raise

        except Exception as e:
            logger.error(f"Error selama training: {e}")
            raise

        elapsed = time.time() - start_time
        logger.info(f"Training selesai dalam {elapsed / 60:.1f} menit ✓")

        # ------------------------------------------------------------------
        # Salin best.pt ke lokasi yang dikonfigurasi di cv.yaml
        # ------------------------------------------------------------------
        self._sync_best_model()

        # ------------------------------------------------------------------
        # Update path model di cv.yaml untuk inferensi
        # ------------------------------------------------------------------
        self._update_inference_model_path()

        # ------------------------------------------------------------------
        # Ringkasan metrics
        # ------------------------------------------------------------------
        summary = self._extract_metrics(results, elapsed)
        return summary

    # --------------------------------------------------------------------------
    # PRIVATE: Sync best model ke path inferensi
    # --------------------------------------------------------------------------

    def _sync_best_model(self):
        """
        Pastikan best.pt tersimpan di lokasi yang benar sesuai cv.yaml.
        Ultralytics menyimpan di: output_dir/run_name/weights/best.pt
        """
        best_src = self.output_dir / self.run_name / "weights" / "best.pt"

        if best_src.exists():
            logger.info(f"Model terbaik tersimpan di: {best_src} ✓")
            # Juga simpan salinan dengan nama yang lebih deskriptif
            best_copy = self.output_dir / "aerosweep_best.pt"
            shutil.copy2(best_src, best_copy)
            logger.info(f"Salinan model disimpan di: {best_copy} ✓")
        else:
            logger.warning(
                f"best.pt tidak ditemukan di: {best_src}\n"
                f"  Cek folder {self.output_dir / self.run_name / 'weights'}"
            )

    def _update_inference_model_path(self):
        """
        Update bagian inference.model_path di cv.yaml
        agar scripts/03_inference_extraction.py otomatis
        menggunakan model terbaik hasil training ini.
        """
        best_model = self.output_dir / self.run_name / "weights" / "best.pt"

        if not best_model.exists():
            logger.warning("Tidak dapat update cv.yaml: best.pt tidak ditemukan.")
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Update path model inferensi
        # Simpan sebagai path relatif dari root project
        relative_path = str(best_model.relative_to(self.root_dir))
        config.setdefault("inference", {})["model_path"] = relative_path

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(
            f"cv.yaml diperbarui: inference.model_path = {relative_path} ✓"
        )

    # --------------------------------------------------------------------------
    # PRIVATE: Ekstrak dan tampilkan metrics
    # --------------------------------------------------------------------------

    def _extract_metrics(self, results, elapsed: float) -> dict:
        """
        Ekstrak metrics utama dari hasil training Ultralytics.

        Ultralytics results object memiliki atribut:
        - results.results_dict : dictionary semua metrics
        - results.maps         : mAP per kelas

        Parameters
        ----------
        results : Objek hasil model.train() dari Ultralytics
        elapsed : Waktu training dalam detik

        Returns
        -------
        Dictionary ringkasan metrics.
        """
        summary = {
            "elapsed_seconds" : elapsed,
            "elapsed_minutes" : round(elapsed / 60, 1),
            "model"           : self.model_name,
            "epochs"          : self.epochs,
            "imgsz"           : self.imgsz,
            "best_model_path" : str(
                self.output_dir / self.run_name / "weights" / "best.pt"
            ),
        }

        # Ekstrak metrics dari results jika tersedia
        try:
            if hasattr(results, "results_dict") and results.results_dict:
                rd = results.results_dict
                summary.update({
                    # Metrics segmentasi instance
                    "metrics/mAP50(M)"    : rd.get("metrics/mAP50(M)",    "N/A"),
                    "metrics/mAP50-95(M)" : rd.get("metrics/mAP50-95(M)", "N/A"),
                    "metrics/precision(M)": rd.get("metrics/precision(M)","N/A"),
                    "metrics/recall(M)"   : rd.get("metrics/recall(M)",   "N/A"),
                    # Loss akhir
                    "train/box_loss"      : rd.get("train/box_loss",      "N/A"),
                    "train/seg_loss"      : rd.get("train/seg_loss",      "N/A"),
                    "train/cls_loss"      : rd.get("train/cls_loss",      "N/A"),
                })
        except Exception as e:
            logger.warning(f"Tidak dapat mengekstrak metrics detail: {e}")

        return summary

    def _print_training_plan(self):
        """Tampilkan parameter training sebelum dimulai."""
        logger.info("Parameter training:")
        logger.info(f"  Model pretrained : {self.model_path}")
        logger.info(f"  Dataset config   : {self.config_path}")
        logger.info(f"  Epochs           : {self.epochs}")
        logger.info(f"  Batch size       : {self.batch}")
        logger.info(f"  Image size       : {self.imgsz}x{self.imgsz}")
        logger.info(f"  Learning rate    : {self.lr0} → {self.lr0 * self.lrf:.6f}")
        logger.info(f"  Early stopping   : patience={self.patience} epoch")
        logger.info(f"  Device           : {self.device}")
        logger.info(f"  Workers          : {self.workers}")
        logger.info(f"  Output dir       : {self.output_dir / self.run_name}")
        logger.info(f"  Augmentasi       : {self.augment}")
        logger.info(f"  Seed             : {self.seed}")

    # --------------------------------------------------------------------------
    # PUBLIC: Export ringkasan training ke file teks
    # --------------------------------------------------------------------------

    def export_summary(self, summary: dict):
        """
        Simpan ringkasan hasil training ke outputs/training_summary.txt
        untuk dokumentasi dan referensi tim.

        Parameters
        ----------
        summary : Dictionary hasil dari _extract_metrics()
        """
        summary_path = self.root_dir / "outputs" / "training_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("AeroSweep — Ringkasan Hasil Training\n")
            f.write("=" * 60 + "\n\n")

            for key, value in summary.items():
                f.write(f"  {key:<30}: {value}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("Langkah berikutnya:\n")
            f.write("  python scripts/03_inference_extraction.py\n")
            f.write("=" * 60 + "\n")

        logger.info(f"Ringkasan training disimpan di: {summary_path} ✓")
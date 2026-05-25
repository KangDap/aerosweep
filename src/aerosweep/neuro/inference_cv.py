# ==============================================================================
# AeroSweep — Modul Inferensi & Ekstraksi Fitur
# File: src/aerosweep/neuro/inference_cv.py
#
# Tanggung jawab:
#   1. Memuat model YOLO terlatih (best.pt)
#   2. Menjalankan inferensi pada grid citra UAV 512x512
#   3. Mengekstrak 6 metrik matematis dari hasil segmentasi:
#       - ID Grid
#       - Koordinat Centroid (X, Y)
#       - Kepadatan Area (%)
#       - Jumlah Sampah (Instance)
#       - Kategori Dominan Material
#       - Confidence Score (rata-rata)
#   4. Menangani edge case: grid tanpa deteksi (0 objek)
#   5. Mengembalikan hasil sebagai list of dict → diekspor ke CSV
#
# Dipanggil oleh: scripts/03_inference_extraction.py
#
# OUTPUT AKHIR: outputs/output_fitur_ann_ke_fuzzy.csv
# Kolom CSV: grid_id, centroid_x, centroid_y, area_density_pct,
#            jumlah_instance, kategori_dominan, confidence_score
# ==============================================================================

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
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
logger = logging.getLogger("AeroSweep.inference_cv")


# ==============================================================================
# KONSTANTA
# ==============================================================================

# Ukuran grid (piksel) — harus sama dengan imgsz training
GRID_SIZE = 512

# Total piksel dalam satu grid (512 x 512)
TOTAL_GRID_PIXELS = GRID_SIZE * GRID_SIZE  # = 262.144 piksel

# Nilai default saat grid tidak memiliki deteksi apapun
NO_DETECTION_DEFAULTS = {
    "centroid_x"       : -1,      # -1 = tidak ada objek
    "centroid_y"       : -1,      # -1 = tidak ada objek
    "area_density_pct" : 0.0,     # 0% = tidak ada sampah
    "jumlah_instance"  : 0,       # 0 objek terdeteksi
    "kategori_dominan" : "none",  # Tidak ada kategori
    "confidence_score" : 0.0,     # 0 = tidak ada deteksi
}


# ==============================================================================
# KELAS UTAMA: FeatureExtractor
# ==============================================================================

class FeatureExtractor:
    """
    Kelas utama untuk inferensi YOLO dan ekstraksi fitur matematis
    dari hasil instance segmentation pada grid citra UAV.

    Alur kerja per grid:
        1. Load citra grid (512x512)
        2. Jalankan YOLO inference → dapatkan masks + boxes + classes + confs
        3. Gabungkan semua mask → binary combined mask
        4. Hitung centroid dari combined mask menggunakan cv2.moments
        5. Hitung area density dari total piksel mask
        6. Hitung jumlah instance dari len(boxes)
        7. Tentukan kategori dominan dari kelas dengan piksel terbanyak
        8. Hitung rata-rata confidence score
        9. Kembalikan sebagai dictionary

    Contoh penggunaan:
        extractor = FeatureExtractor(
            model_path  = "models/nn_weights/aerosweep_trained/weights/best.pt",
            config_path = "configs/cv.yaml",
        )
        results = extractor.run_batch(grid_dir="data/vision/test_grids")
        # results → list of dict, siap diekspor ke CSV
    """

    def __init__(
        self,
        model_path:   str,
        config_path:  str,
        conf_threshold: float = 0.25,
        iou_threshold:  float = 0.45,
        device:         str   = "0",
        save_annotated: bool  = False,
        annotated_dir:  Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_path      : Path ke best.pt hasil training
        config_path     : Path ke configs/cv.yaml
        conf_threshold  : Confidence threshold (deteksi di bawah ini diabaikan)
        iou_threshold   : IoU threshold untuk NMS
        device          : Device inferensi ("0"=GPU, "cpu"=CPU)
        save_annotated  : Simpan citra dengan overlay mask hasil inferensi
        annotated_dir   : Direktori penyimpanan citra ter-annotasi
        """
        self.model_path      = Path(model_path)
        self.config_path     = Path(config_path)
        self.conf_threshold  = conf_threshold
        self.iou_threshold   = iou_threshold
        self.device          = device
        self.save_annotated  = save_annotated
        self.annotated_dir   = Path(annotated_dir) if annotated_dir else None

        # Validasi path
        self._validate_paths()

        # Load konfigurasi → nama kelas
        self.config      = self._load_config()
        self.class_names = self._load_class_names()

        # Load model YOLO
        self.model = self._load_model()

        # Siapkan direktori output annotasi jika diperlukan
        if self.save_annotated and self.annotated_dir:
            self.annotated_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"FeatureExtractor siap: "
            f"model={self.model_path.name}, "
            f"kelas={len(self.class_names)}, "
            f"conf={self.conf_threshold}, "
            f"device={self.device}"
        )

    # --------------------------------------------------------------------------
    # PRIVATE: Inisialisasi
    # --------------------------------------------------------------------------

    def _validate_paths(self):
        """Validasi path model dan config ada sebelum inferensi."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {self.model_path}\n"
                f"Jalankan scripts/02_train_cv.py terlebih dahulu."
            )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config tidak ditemukan: {self.config_path}"
            )
        logger.info("Validasi path model & config: OK ✓")

    def _load_config(self) -> dict:
        """Baca cv.yaml."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def _load_class_names(self) -> dict:
        """
        Baca nama kelas dari cv.yaml.
        Kembalikan sebagai dict {class_id: nama_kelas}.

        Fallback ke dict kosong jika belum diupdate oleh preprocessing.
        """
        names = self.config.get("names", {})
        if not names:
            logger.warning(
                "Nama kelas tidak ditemukan di cv.yaml. "
                "Kelas akan ditampilkan sebagai ID numerik."
            )
            return {}

        # Pastikan key bertipe int (YAML kadang membaca sebagai string)
        class_names = {int(k): str(v) for k, v in names.items()}
        logger.info(f"Nama kelas dimuat: {len(class_names)} kelas")
        return class_names

    def _load_model(self) -> YOLO:
        """
        Load model YOLO dari best.pt.

        Ultralytics YOLO secara otomatis mendeteksi tipe model
        (detection vs segmentation) dari arsitektur yang tersimpan
        di dalam file .pt.
        """
        logger.info(f"Memuat model dari: {self.model_path}")
        try:
            model = YOLO(str(self.model_path))
            if getattr(model, "task", None) != "segment":
                raise RuntimeError(
                    "Model bukan instance segmentation (task != 'segment'). "
                    "Gunakan bobot *-seg.pt atau model segmentation."
                )
            logger.info(f"Model berhasil dimuat ✓")
            return model
        except Exception as e:
            raise RuntimeError(f"Gagal memuat model YOLO: {e}")

    # --------------------------------------------------------------------------
    # PUBLIC: Inferensi batch pada folder grid
    # --------------------------------------------------------------------------

    def run_batch(self, grid_dir: str) -> list[dict]:
        """
        Jalankan inferensi pada seluruh grid di dalam folder
        dan ekstrak 6 metrik untuk setiap grid.

        Parameters
        ----------
        grid_dir : Path ke folder berisi file grid .jpg / .png

        Returns
        -------
        List of dict, setiap dict = 1 baris data untuk CSV.
        Urutan kolom:
            grid_id, centroid_x, centroid_y, area_density_pct,
            jumlah_instance, kategori_dominan, confidence_score

        Edge case:
            Grid tanpa deteksi tetap masuk ke list dengan nilai
            default dari NO_DETECTION_DEFAULTS.
        """
        grid_path = Path(grid_dir)

        if not grid_path.exists():
            raise FileNotFoundError(
                f"Direktori grid tidak ditemukan: {grid_path}"
            )

        # Kumpulkan semua file gambar (jpg dan png)
        grid_files = sorted(
            list(grid_path.glob("*.jpg")) +
            list(grid_path.glob("*.png"))
        )

        if not grid_files:
            raise ValueError(
                f"Tidak ada file gambar (.jpg/.png) di: {grid_path}"
            )

        logger.info("=" * 60)
        logger.info(f"Memulai inferensi batch")
        logger.info(f"  Direktori grid : {grid_path}")
        logger.info(f"  Jumlah grid    : {len(grid_files)}")
        logger.info(f"  Conf threshold : {self.conf_threshold}")
        logger.info(f"  IoU threshold  : {self.iou_threshold}")
        logger.info("=" * 60)

        all_results   = []
        total_detected = 0
        empty_grids    = 0

        for idx, grid_file in enumerate(grid_files, start=1):

            # Progress log setiap 50 grid
            if idx % 50 == 0 or idx == 1 or idx == len(grid_files):
                logger.info(
                    f"  Memproses grid {idx}/{len(grid_files)}: "
                    f"{grid_file.name}"
                )

            # Jalankan inferensi + ekstraksi fitur untuk satu grid
            row = self.process_single_grid(grid_file)
            all_results.append(row)

            # Statistik
            if row["jumlah_instance"] == 0:
                empty_grids += 1
            else:
                total_detected += row["jumlah_instance"]

        logger.info("=" * 60)
        logger.info("Inferensi batch selesai!")
        logger.info(f"  Total grid diproses    : {len(grid_files)}")
        logger.info(f"  Grid dengan deteksi    : {len(grid_files) - empty_grids}")
        logger.info(f"  Grid tanpa deteksi     : {empty_grids}")
        logger.info(f"  Total instance deteksi : {total_detected}")
        logger.info("=" * 60)

        return all_results

    # --------------------------------------------------------------------------
    # PUBLIC: Inferensi satu grid
    # --------------------------------------------------------------------------

    def process_single_grid(self, grid_path: Path) -> dict:
        """
        Proses satu file grid: inferensi + ekstraksi 6 metrik.

        Parameters
        ----------
        grid_path : Path ke file gambar grid (512x512)

        Returns
        -------
        Dictionary dengan 7 key:
            grid_id, centroid_x, centroid_y, area_density_pct,
            jumlah_instance, kategori_dominan, confidence_score
        """
        grid_id = grid_path.stem  # Nama file tanpa ekstensi

        # ------------------------------------------------------------------
        # 1. Baca citra grid
        # ------------------------------------------------------------------
        image = cv2.imread(str(grid_path))
        if image is None:
            logger.warning(f"Gagal membaca citra: {grid_path}. Grid dilewati.")
            return self._empty_row(grid_id)

        img_h, img_w = image.shape[:2]

        # ------------------------------------------------------------------
        # 2. Jalankan YOLO inference
        # Dokumentasi: https://docs.ultralytics.com/modes/predict/
        # ------------------------------------------------------------------
        try:
            predictions = self.model.predict(
                source  = str(grid_path),
                conf    = self.conf_threshold,
                iou     = self.iou_threshold,
                device  = self.device,
                verbose = False,   # Matikan output per-gambar agar log bersih
                retina_masks = True,  # Mask resolusi penuh (bukan downsampled)
            )
        except Exception as e:
            logger.error(f"Error inferensi pada {grid_id}: {e}")
            return self._empty_row(grid_id)

        # Ultralytics predict() selalu kembalikan list, ambil elemen pertama
        result = predictions[0]

        # ------------------------------------------------------------------
        # 3. Cek apakah ada deteksi
        # ------------------------------------------------------------------
        has_masks = (
            result.masks is not None and
            len(result.masks) > 0
        )
        has_boxes = (
            result.boxes is not None and
            len(result.boxes) > 0
        )

        # Edge case: tidak ada deteksi sama sekali
        if not has_masks or not has_boxes:
            logger.debug(f"Tidak ada deteksi pada grid: {grid_id}")
            return self._empty_row(grid_id)

        # ------------------------------------------------------------------
        # 4. Ekstrak data mentah dari hasil prediksi
        # ------------------------------------------------------------------

        # Masks: array boolean (N, H, W) — satu mask per instance
        # .data memberikan tensor, .numpy() konversi ke numpy array
        masks_tensor = result.masks.data.cpu().numpy()  # shape: (N, H, W)

        # Bounding boxes: tensor (N, 4) — [x1, y1, x2, y2]
        boxes = result.boxes.xyxy.cpu().numpy()         # shape: (N, 4)

        # Class IDs: tensor (N,) — integer
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # shape: (N,)

        # Confidence scores: tensor (N,) — float [0.0, 1.0]
        confidences = result.boxes.conf.cpu().numpy()   # shape: (N,)

        # Jumlah instance terdeteksi
        n_instances = len(boxes)

        # ------------------------------------------------------------------
        # 5. Resize masks ke ukuran asli grid jika berbeda
        # Ultralytics kadang menghasilkan mask dengan ukuran berbeda
        # dari ukuran input karena internal downsampling
        # ------------------------------------------------------------------
        masks_resized = self._resize_masks(masks_tensor, img_h, img_w)

        # ------------------------------------------------------------------
        # 6. Ekstrak 6 metrik
        # ------------------------------------------------------------------

        # --- Metrik 1 & 2: Centroid (X, Y) menggunakan cv2.moments ---
        centroid_x, centroid_y = self._compute_centroid(masks_resized)

        # --- Metrik 3: Kepadatan Area (%) ---
        area_density_pct = self._compute_area_density(masks_resized, img_h, img_w)

        # --- Metrik 4: Jumlah Instance ---
        jumlah_instance = n_instances

        # --- Metrik 5: Kategori Dominan ---
        kategori_dominan = self._compute_dominant_category(
            masks_resized, class_ids
        )

        # --- Metrik 6: Rata-rata Confidence Score ---
        confidence_score = float(np.mean(confidences))

        # ------------------------------------------------------------------
        # 7. Simpan citra ter-annotasi jika diminta
        # ------------------------------------------------------------------
        if self.save_annotated and self.annotated_dir:
            self._save_annotated_image(
                image            = image,
                result           = result,
                grid_id          = grid_id,
                centroid_x       = centroid_x,
                centroid_y       = centroid_y,
                area_density_pct = area_density_pct,
            )

        # ------------------------------------------------------------------
        # 8. Susun baris hasil
        # ------------------------------------------------------------------
        row = {
            "grid_id"          : grid_id,
            "centroid_x"       : round(centroid_x, 2),
            "centroid_y"       : round(centroid_y, 2),
            "area_density_pct" : round(area_density_pct, 4),
            "jumlah_instance"  : jumlah_instance,
            "kategori_dominan" : kategori_dominan,
            "confidence_score" : round(confidence_score, 4),
        }

        logger.debug(
            f"Grid {grid_id}: "
            f"inst={jumlah_instance}, "
            f"density={area_density_pct:.2f}%, "
            f"dominan={kategori_dominan}, "
            f"conf={confidence_score:.3f}"
        )

        return row

    # --------------------------------------------------------------------------
    # PRIVATE: Ekstraksi Metrik — detail implementasi
    # --------------------------------------------------------------------------

    def _resize_masks(
        self,
        masks: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """
        Resize array masks ke dimensi target jika ukurannya berbeda.

        Parameters
        ----------
        masks    : Array (N, H_mask, W_mask) — boolean atau float
        target_h : Tinggi target (ukuran grid asli)
        target_w : Lebar target (ukuran grid asli)

        Returns
        -------
        Array (N, target_h, target_w) — boolean
        """
        n         = masks.shape[0]
        mask_h    = masks.shape[1]
        mask_w    = masks.shape[2]

        # Tidak perlu resize jika sudah sama
        if mask_h == target_h and mask_w == target_w:
            return masks.astype(bool)

        # Resize setiap mask satu per satu
        resized = np.zeros((n, target_h, target_w), dtype=bool)
        for i in range(n):
            # cv2.resize butuh uint8 (0 atau 255), bukan boolean
            mask_uint8 = (masks[i] * 255).astype(np.uint8)
            resized_mask = cv2.resize(
                mask_uint8,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST  # Pertahankan tepi mask
            )
            resized[i] = resized_mask > 127

        return resized

    def _compute_centroid(
        self,
        masks: np.ndarray,
    ) -> tuple[float, float]:
        """
        Hitung koordinat centroid (X, Y) dari GABUNGAN seluruh mask
        sampah di dalam grid menggunakan cv2.moments.

        Metode:
        1. Gabungkan semua mask instance menjadi satu binary mask
           (logical OR — piksel sampah = 1, background = 0)
        2. Hitung momen spasial dari combined mask
        3. Centroid X = M10 / M00, Centroid Y = M01 / M00

        Rumus:
            M00 = total area (jumlah piksel = 1)
            M10 = Σ(x * piksel(x,y))
            M01 = Σ(y * piksel(x,y))
            Cx  = M10 / M00
            Cy  = M01 / M00

        Parameters
        ----------
        masks : Array (N, H, W) boolean — mask per instance

        Returns
        -------
        (centroid_x, centroid_y) dalam koordinat piksel.
        Mengembalikan (-1.0, -1.0) jika M00 = 0 (mask kosong).
        """
        # Gabungkan semua mask → combined binary mask (H, W)
        # np.any(axis=0): piksel bernilai True jika ADA mask yang menutupinya
        combined_mask = np.any(masks, axis=0).astype(np.uint8)

        # Hitung momen menggunakan OpenCV
        # cv2.moments menerima array uint8 (0 atau 1)
        moments = cv2.moments(combined_mask)

        # M00 = zeroth moment = total area piksel yang bernilai 1
        M00 = moments["m00"]

        # Edge case: M00 = 0 berarti mask benar-benar kosong
        if M00 == 0:
            logger.debug("M00 = 0, centroid tidak dapat dihitung.")
            return -1.0, -1.0

        # Centroid: titik pusat massa piksel sampah
        centroid_x = moments["m10"] / M00  # Komponen X (horizontal)
        centroid_y = moments["m01"] / M00  # Komponen Y (vertikal)

        return float(centroid_x), float(centroid_y)

    def _compute_area_density(
        self,
        masks:   np.ndarray,
        img_h:   int,
        img_w:   int,
    ) -> float:
        """
        Hitung kepadatan area sampah dalam grid (dalam persen).

        Rumus:
            Kepadatan (%) = (total_piksel_sampah / total_piksel_grid) × 100

        Di mana:
            total_piksel_sampah = jumlah piksel yang tertutup MINIMAL
                                  satu mask instance (setelah union)
            total_piksel_grid   = img_h × img_w (= 512 × 512 = 262.144)

        Catatan: Menggunakan union mask (bukan sum) agar piksel yang
        ditutup lebih dari satu mask tidak dihitung dua kali.

        Parameters
        ----------
        masks  : Array (N, H, W) boolean
        img_h  : Tinggi grid dalam piksel
        img_w  : Lebar grid dalam piksel

        Returns
        -------
        Float persentase area [0.0, 100.0]
        """
        # Union semua mask → setiap piksel dihitung maksimal sekali
        combined_mask = np.any(masks, axis=0)  # shape: (H, W), dtype: bool

        # Hitung jumlah piksel yang bernilai True (tertutup sampah)
        total_waste_pixels = int(np.sum(combined_mask))

        # Total piksel grid
        total_grid_pixels = img_h * img_w

        # Kepadatan area dalam persen
        density_pct = (total_waste_pixels / total_grid_pixels) * 100.0

        return float(density_pct)

    def _compute_dominant_category(
        self,
        masks:     np.ndarray,
        class_ids: np.ndarray,
    ) -> str:
        """
        Tentukan kategori material yang paling dominan di dalam grid.

        Metode:
            Untuk setiap kelas unik yang terdeteksi, jumlahkan piksel
            dari SEMUA mask yang termasuk kelas tersebut.
            Kelas dengan total piksel terbanyak = kategori dominan.

        Contoh:
            Instance 0: class=Plastic, 1200 piksel
            Instance 1: class=Plastic, 800 piksel
            Instance 2: class=Metal,   2500 piksel
            → Plastic total: 2000 piksel
            → Metal total  : 2500 piksel
            → Dominan: Metal

        Catatan: Piksel overlap antar mask dari KELAS BERBEDA tetap
        dihitung per kelas (tidak di-union), karena kita ingin
        mengukur "representasi" tiap kelas, bukan area eksklusifnya.

        Parameters
        ----------
        masks     : Array (N, H, W) boolean
        class_ids : Array (N,) integer — class ID per instance

        Returns
        -------
        Nama kelas dominan sebagai string.
        Mengembalikan "unknown" jika tidak bisa ditentukan.
        """
        if len(class_ids) == 0:
            return "none"

        # Akumulasi total piksel per kelas
        # key: class_id, value: total piksel
        pixel_counts: dict[int, int] = {}

        for i, class_id in enumerate(class_ids):
            # Jumlah piksel dari mask instance ke-i
            pixel_count = int(np.sum(masks[i]))

            if class_id not in pixel_counts:
                pixel_counts[class_id] = 0
            pixel_counts[class_id] += pixel_count

        # Temukan kelas dengan piksel terbanyak
        if not pixel_counts:
            return "unknown"

        dominant_class_id = max(pixel_counts, key=pixel_counts.get)

        # Resolve ke nama kelas dari mapping
        dominant_name = self.class_names.get(
            dominant_class_id,
            f"class_{dominant_class_id}"  # Fallback: tampilkan ID numerik
        )

        return str(dominant_name)

    # --------------------------------------------------------------------------
    # PRIVATE: Edge case handler
    # --------------------------------------------------------------------------

    def _empty_row(self, grid_id: str) -> dict:
        """
        Kembalikan baris DataFrame dengan nilai default untuk grid
        yang tidak memiliki deteksi apapun (0 objek).

        Nilai default:
            centroid_x/y   : -1    → sinyal "tidak ada objek"
            area_density   : 0.0   → 0% area sampah
            jumlah_instance: 0     → nol deteksi
            kategori_dominan: none → tidak ada kategori
            confidence_score: 0.0  → tidak ada confidence

        Nilai -1 untuk centroid dipilih secara sengaja (bukan NaN)
        agar mudah difilter oleh sistem logika Fuzzy downstream.

        Parameters
        ----------
        grid_id : Nama unik grid

        Returns
        -------
        Dictionary baris kosong yang siap dimasukkan ke DataFrame.
        """
        row = {"grid_id": grid_id}
        row.update(NO_DETECTION_DEFAULTS)
        return row

    # --------------------------------------------------------------------------
    # PRIVATE: Simpan citra ter-annotasi
    # --------------------------------------------------------------------------

    def _save_annotated_image(
        self,
        image:            np.ndarray,
        result,
        grid_id:          str,
        centroid_x:       float,
        centroid_y:       float,
        area_density_pct: float,
    ):
        """
        Simpan citra grid dengan overlay:
        - Mask berwarna semi-transparan per instance
        - Bounding box dan label kelas + confidence
        - Titik centroid gabungan (lingkaran merah)
        - Teks ringkasan metrik di sudut atas

        Parameters
        ----------
        image            : Array citra OpenCV asli (H, W, 3)
        result           : Objek hasil prediksi Ultralytics
        grid_id          : Nama grid (untuk nama file output)
        centroid_x/y     : Koordinat centroid gabungan
        area_density_pct : Persentase kepadatan area sampah
        """
        # Buat salinan citra agar tidak memodifikasi original
        annotated = image.copy()

        # Gunakan Ultralytics built-in plot() untuk overlay mask + box
        # plot() mengembalikan array BGR dengan anotasi sudah tergambar
        annotated = result.plot(
            conf   = True,   # Tampilkan confidence score
            labels = True,   # Tampilkan nama kelas
            masks  = True,   # Tampilkan mask segmentasi
            boxes  = True,   # Tampilkan bounding box
            line_width = 1,
        )

        # Gambar titik centroid gabungan (lingkaran merah tebal)
        if centroid_x >= 0 and centroid_y >= 0:
            cx, cy = int(centroid_x), int(centroid_y)
            # Lingkaran luar putih (border)
            cv2.circle(annotated, (cx, cy), 8, (255, 255, 255), -1)
            # Lingkaran dalam merah
            cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
            # Crosshair horizontal
            cv2.line(annotated, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1)
            # Crosshair vertikal
            cv2.line(annotated, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)

        # Teks ringkasan metrik di sudut kiri atas
        n_inst = len(result.boxes) if result.boxes is not None else 0
        lines  = [
            f"ID: {grid_id}",
            f"Inst: {n_inst}",
            f"Density: {area_density_pct:.2f}%",
            f"Centroid: ({centroid_x:.1f}, {centroid_y:.1f})",
        ]

        # Background semi-transparan untuk teks
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (230, 20 + len(lines) * 18), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

        for i, line in enumerate(lines):
            cv2.putText(
                annotated,
                line,
                (5, 18 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Simpan ke direktori annotated
        out_path = self.annotated_dir / f"{grid_id}_annotated.jpg"
        cv2.imwrite(str(out_path), annotated)
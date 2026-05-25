# ==============================================================================
# AeroSweep — Entry Point Training
# File: scripts/02_train_cv.py
#
# Tanggung jawab:
#   1. Membaca konfigurasi dari configs/cv.yaml
#   2. Memanggil YOLOTrainer dari src/aerosweep/neuro/train_cv.py
#   3. Memvalidasi dataset sebelum training dimulai
#   4. Menjalankan fine-tuning YOLO11n-seg
#   5. Mengekspor ringkasan hasil training
#
# Cara menjalankan (dari ROOT project):
#   python scripts/02_train_cv.py
#   python scripts/02_train_cv.py --epochs 50 --batch 8
#   python scripts/02_train_cv.py --device cpu
#   python scripts/02_train_cv.py --dry-run  ← validasi saja
#
# Urutan eksekusi pipeline:
#   01_preprocessing.py → [02_train_cv.py] → 03_inference_extraction.py
# ==============================================================================

import argparse
import logging
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------------------
# Pastikan ROOT project ada di sys.path
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.aerosweep.neuro.train_cv import YOLOTrainer

# ------------------------------------------------------------------------------
# Setup Logger
# ------------------------------------------------------------------------------
log_dir = ROOT_DIR / "outputs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "training.log"),
    ],
)
logger = logging.getLogger("AeroSweep.02_train_cv")


# ==============================================================================
# ARGUMEN CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse argumen command line.

    Semua argumen bersifat opsional — jika tidak diberikan,
    nilai diambil dari configs/cv.yaml.
    Override CLI selalu lebih prioritas dari nilai di YAML.
    """
    parser = argparse.ArgumentParser(
        description=(
            "AeroSweep — Fine-tuning YOLO Instance Segmentation\n"
            "pada dataset DroneWaste untuk deteksi sampah dari UAV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh penggunaan:\n"
            "  # Training dengan konfigurasi default dari cv.yaml\n"
            "  python scripts/02_train_cv.py\n\n"
            "  # Override epoch dan batch\n"
            "  python scripts/02_train_cv.py --epochs 50 --batch 8\n\n"
            "  # Training di CPU (untuk testing)\n"
            "  python scripts/02_train_cv.py --device cpu --epochs 2\n\n"
            "  # Validasi dataset saja tanpa training\n"
            "  python scripts/02_train_cv.py --dry-run\n\n"
            "  # Multi-GPU (GPU 0 dan 1)\n"
            "  python scripts/02_train_cv.py --device 0,1\n"
        ),
    )

    # --- Konfigurasi ---
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cv.yaml",
        help="Path ke file konfigurasi cv.yaml.",
    )

    # --- Override parameter training ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override jumlah epoch training dari cv.yaml. "
            "Berguna untuk eksperimen cepat."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help=(
            "Override batch size dari cv.yaml. "
            "Kurangi jika terjadi CUDA out of memory."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Override device dari cv.yaml. "
            "Contoh: '0' (GPU pertama), '0,1' (multi-GPU), 'cpu'."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Override jumlah DataLoader workers dari cv.yaml. "
            "Kurangi jika RAM terbatas atau OS Windows."
        ),
    )

    # --- Mode operasi ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validasi dataset dan konfigurasi saja tanpa menjalankan training. "
            "Gunakan ini sebelum training sesungguhnya."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help=(
            "Lewati validasi dataset (tidak direkomendasikan). "
            "Hanya gunakan jika sudah yakin dataset valid."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Lanjutkan training dari checkpoint terakhir (last.pt). "
            "Berguna jika training sebelumnya terputus."
        ),
    )

    return parser.parse_args()


# ==============================================================================
# FUNGSI UTILITAS
# ==============================================================================

def print_system_info():
    """
    Tampilkan informasi sistem dan library sebelum training.
    Berguna untuk debugging dan dokumentasi eksperimen.
    """
    import platform
    import torch

    logger.info("=" * 60)
    logger.info("INFORMASI SISTEM")
    logger.info("=" * 60)
    logger.info(f"  OS              : {platform.system()} {platform.release()}")
    logger.info(f"  Python          : {sys.version.split()[0]}")
    logger.info(f"  PyTorch         : {torch.__version__}")

    # Info GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"  CUDA tersedia   : Ya (GPU: {gpu_count})")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem  = torch.cuda.get_device_properties(i).total_memory
            gpu_mem_gb = gpu_mem / (1024 ** 3)
            logger.info(f"  GPU [{i}]         : {gpu_name} ({gpu_mem_gb:.1f} GB)")
    else:
        logger.info("  CUDA tersedia   : Tidak (akan menggunakan CPU)")
        logger.warning(
            "Training dengan CPU akan sangat lambat. "
            "Pertimbangkan menggunakan Google Colab jika tidak ada GPU."
        )

    try:
        import ultralytics
        logger.info(f"  Ultralytics     : {ultralytics.__version__}")
    except Exception:
        pass

    logger.info("=" * 60)


def check_preprocessing_done(root_dir: Path, config_path: Path) -> bool:
    """
    Cek apakah script 01_preprocessing.py sudah pernah dijalankan
    dengan melihat keberadaan folder dan file hasil preprocessing.

    Parameters
    ----------
    root_dir    : Path root project
    config_path : Path ke configs/cv.yaml

    Returns
    -------
    True jika preprocessing sudah selesai, False jika belum.
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_path  = root_dir / config.get("path", "data/vision")
    processed_dir = dataset_path / "processed"

    # Cek keberadaan folder hasil preprocessing
    checks = [
        processed_dir / "train" / "images",
        processed_dir / "train" / "labels",
        processed_dir / "val"   / "images",
        processed_dir / "val"   / "labels",
    ]

    missing = [str(p) for p in checks if not p.exists()]

    if missing:
        logger.error(
            "Preprocessing belum dijalankan! Folder berikut tidak ditemukan:"
        )
        for m in missing:
            logger.error(f"  ✗ {m}")
        logger.error(
            "\nJalankan preprocessing terlebih dahulu:\n"
            "  python scripts/01_preprocessing.py"
        )
        return False

    # Cek cv.yaml sudah diupdate (nc bukan 0)
    nc    = config.get("nc", 0)
    names = config.get("names", {})

    if nc == 0 or not names:
        logger.error(
            "cv.yaml belum diupdate dengan nama kelas.\n"
            "Jalankan: python scripts/01_preprocessing.py"
        )
        return False

    return True


def handle_resume(root_dir: Path, config_path: Path) -> str:
    """
    Tangani mode resume training dari checkpoint last.pt.

    Parameters
    ----------
    root_dir    : Path root project
    config_path : Path ke configs/cv.yaml

    Returns
    -------
    Path ke last.pt sebagai string, atau None jika tidak ada.
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    output_dir = root_dir / train_cfg.get("project", "models/nn_weights")
    run_name   = train_cfg.get("name", "aerosweep_trained")
    last_pt    = output_dir / run_name / "weights" / "last.pt"

    if last_pt.exists():
        logger.info(f"Checkpoint ditemukan: {last_pt}")
        return str(last_pt)
    else:
        logger.warning(
            f"Checkpoint last.pt tidak ditemukan di: {last_pt}\n"
            "Training akan dimulai dari awal."
        )
        return None


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Entry point utama script training."""

    # --------------------------------------------------------------------------
    # 1. Parse argumen
    # --------------------------------------------------------------------------
    args = parse_args()

    logger.info("=" * 60)
    logger.info("AeroSweep — Training Pipeline")
    logger.info("=" * 60)

    # --------------------------------------------------------------------------
    # 2. Tampilkan informasi sistem
    # --------------------------------------------------------------------------
    print_system_info()

    # --------------------------------------------------------------------------
    # 3. Resolve path konfigurasi
    # --------------------------------------------------------------------------
    config_path = ROOT_DIR / args.config

    if not config_path.exists():
        logger.error(f"File konfigurasi tidak ditemukan: {config_path}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # 4. Cek preprocessing sudah selesai
    # --------------------------------------------------------------------------
    if not check_preprocessing_done(ROOT_DIR, config_path):
        sys.exit(1)

    logger.info("Pengecekan awal preprocessing: PASSED ✓")

    # --------------------------------------------------------------------------
    # 5. Tangani mode resume
    # --------------------------------------------------------------------------
    resume_path = None
    if args.resume:
        resume_path = handle_resume(ROOT_DIR, config_path)
        if resume_path:
            logger.info(f"Mode RESUME aktif dari: {resume_path}")

    # --------------------------------------------------------------------------
    # 6. Inisialisasi YOLOTrainer
    # --------------------------------------------------------------------------
    trainer = YOLOTrainer(
        config_path      = str(config_path),
        root_dir         = ROOT_DIR,
        override_epochs  = args.epochs,
        override_batch   = args.batch,
        override_device  = args.device,
        override_workers = args.workers,
    )

    # --------------------------------------------------------------------------
    # 7. Mode: dry-run — validasi saja
    # --------------------------------------------------------------------------
    if args.dry_run:
        logger.info("Mode DRY-RUN aktif...")
        trainer.validate_dataset()
        logger.info(
            "Semua validasi PASSED ✓\n"
            "Siap untuk training. Jalankan tanpa --dry-run."
        )
        return

    # --------------------------------------------------------------------------
    # 8. Validasi dataset (kecuali di-skip)
    # --------------------------------------------------------------------------
    if not args.skip_validation:
        trainer.validate_dataset()
    else:
        logger.warning(
            "--skip-validation aktif. Validasi dataset dilewati. "
            "Pastikan dataset sudah valid."
        )

    # --------------------------------------------------------------------------
    # 9. Konfirmasi sebelum training dimulai
    # --------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Semua pengecekan selesai. Training akan dimulai...")
    logger.info(
        "Tekan Ctrl+C untuk membatalkan. "
        "Training otomatis menyimpan last.pt jika dihentikan."
    )
    logger.info("=" * 60)

    # Jeda singkat agar pengguna bisa membaca log sebelum training dimulai
    time.sleep(2)

    # --------------------------------------------------------------------------
    # 10. Jalankan training
    # --------------------------------------------------------------------------
    start_time = time.time()

    try:
        summary = trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training dibatalkan oleh pengguna.")
        logger.warning(
            "Untuk melanjutkan dari checkpoint terakhir:\n"
            "  python scripts/02_train_cv.py --resume"
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training gagal dengan error: {e}")
        logger.error(
            "Tips debugging:\n"
            "  1. Kurangi batch size: --batch 4\n"
            "  2. Gunakan CPU untuk test: --device cpu --epochs 2\n"
            "  3. Cek log lengkap di outputs/training.log"
        )
        raise

    total_elapsed = time.time() - start_time

    # --------------------------------------------------------------------------
    # 11. Export ringkasan hasil training
    # --------------------------------------------------------------------------
    trainer.export_summary(summary)

    # --------------------------------------------------------------------------
    # 12. Ringkasan akhir
    # --------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING SELESAI")
    logger.info("=" * 60)
    logger.info(
        f"  Waktu total        : {total_elapsed / 60:.1f} menit"
    )
    logger.info(
        f"  Model terbaik      : {summary.get('best_model_path', 'N/A')}"
    )
    logger.info(
        f"  mAP50 (Mask)       : {summary.get('metrics/mAP50(M)', 'N/A')}"
    )
    logger.info(
        f"  mAP50-95 (Mask)    : {summary.get('metrics/mAP50-95(M)', 'N/A')}"
    )
    logger.info(
        f"  Ringkasan disimpan : outputs/training_summary.txt"
    )
    logger.info(
        f"  Log training       : outputs/training.log"
    )
    logger.info("=" * 60)
    logger.info("Langkah berikutnya: jalankan scripts/03_inference_extraction.py")
    logger.info("=" * 60)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
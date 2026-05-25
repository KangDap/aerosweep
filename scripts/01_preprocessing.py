# ==============================================================================
# AeroSweep — Entry Point Preprocessing
# File: scripts/01_preprocessing.py
#
# Tanggung jawab:
#   1. Membaca konfigurasi dari configs/cv.yaml
#   2. Memanggil DroneWasteDataset dari src/aerosweep/neuro/dataset.py
#   3. Auto-update cv.yaml dengan nama kelas dari JSON
#   4. Menjalankan sliding window + konversi COCO → YOLO
#   5. Split dataset train/val
#   6. Mencetak ringkasan hasil preprocessing
#
# Cara menjalankan (dari ROOT project):
#   python scripts/01_preprocessing.py
#   python scripts/01_preprocessing.py --config configs/cv.yaml
#   python scripts/01_preprocessing.py --dry-run  ← hanya validasi, tidak proses
#
# ==============================================================================

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# ------------------------------------------------------------------------------
# Pastikan ROOT project ada di sys.path agar import src/ bisa berjalan
# Ini diperlukan karena script dijalankan dari folder scripts/
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.aerosweep.neuro.dataset import DroneWasteDataset

# ------------------------------------------------------------------------------
# Setup Logger
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # Log ke terminal
        logging.StreamHandler(sys.stdout),
        # Log ke file untuk dokumentasi
        logging.FileHandler(ROOT_DIR / "outputs" / "preprocessing.log"),
    ],
)
logger = logging.getLogger("AeroSweep.preprocessing")


# ==============================================================================
# FUNGSI UTAMA
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse argumen command line."""
    parser = argparse.ArgumentParser(
        description="AeroSweep — Preprocessing pipeline untuk dataset DroneWaste",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cv.yaml",
        help="Path ke file konfigurasi cv.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Jalankan validasi saja tanpa memproses data. "
            "Berguna untuk mengecek path dan konfigurasi."
        ),
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        default=False,
        help="Lewati tahap split train/val (hanya lakukan sliding window).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        default=False,
        help="Hanya tampilkan ringkasan dataset tanpa memproses apapun.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """
    Baca file cv.yaml dan kembalikan sebagai dictionary.

    Parameters
    ----------
    config_path : Path ke configs/cv.yaml

    Returns
    -------
    Dictionary konfigurasi
    """
    if not config_path.exists():
        logger.error(f"File konfigurasi tidak ditemukan: {config_path}")
        logger.error("Pastikan configs/cv.yaml sudah dibuat sebelum menjalankan script ini.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Konfigurasi dimuat dari: {config_path}")
    return config


def validate_config(config: dict, root_dir: Path):
    """
    Validasi semua konfigurasi wajib ada dan bernilai logis
    sebelum proses berat dimulai.

    Parameters
    ----------
    config   : Dictionary konfigurasi dari cv.yaml
    root_dir : Path root project

    Raises
    ------
    SystemExit jika ada konfigurasi yang tidak valid.
    """
    logger.info("Memvalidasi konfigurasi...")
    errors = []

    # --- Validasi path dataset ---
    dataset_path = root_dir / config.get("path", "")
    if not dataset_path.exists():
        errors.append(f"Path dataset tidak ditemukan: {dataset_path}")

    # Validasi direktori citra raw
    images_dir = dataset_path / "raw" / "images"
    if not images_dir.exists():
        errors.append(f"Direktori citra tidak ditemukan: {images_dir}")
    else:
        # Cek apakah ada file gambar di dalamnya
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not image_files:
            errors.append(
                f"Tidak ada file gambar (.jpg/.png) di: {images_dir}"
            )
        else:
            logger.info(f"  Ditemukan {len(image_files)} file gambar di {images_dir}")

    # Validasi file COCO JSON
    coco_json = dataset_path / "raw" / "dronewaste_v1.0.json"
    if not coco_json.exists():
        errors.append(f"File COCO JSON tidak ditemukan: {coco_json}")
    else:
        logger.info(f"  File COCO JSON ditemukan: {coco_json}")

    # --- Validasi parameter preprocessing ---
    preproc = config.get("preprocessing", {})

    patch_size = preproc.get("patch_size", 512)
    if not isinstance(patch_size, int) or patch_size <= 0:
        errors.append(f"patch_size harus integer positif, dapat: {patch_size}")

    overlap = preproc.get("overlap", 0.2)
    if not (0.0 <= overlap < 1.0):
        errors.append(f"overlap harus antara 0.0 dan 1.0, dapat: {overlap}")

    train_val_split = preproc.get("train_val_split", 0.8)
    if not (0.0 < train_val_split < 1.0):
        errors.append(
            f"train_val_split harus antara 0.0 dan 1.0, dapat: {train_val_split}"
        )

    min_visibility = preproc.get("min_visibility", 0.1)
    if not (0.0 <= min_visibility <= 1.0):
        errors.append(
            f"min_visibility harus antara 0.0 dan 1.0, dapat: {min_visibility}"
        )

    # --- Tampilkan semua error sekaligus ---
    if errors:
        logger.error("Ditemukan error konfigurasi:")
        for err in errors:
            logger.error(f"  ✗ {err}")
        logger.error("Perbaiki konfigurasi di configs/cv.yaml sebelum melanjutkan.")
        sys.exit(1)

    logger.info("Validasi konfigurasi: PASSED ✓")


def print_preprocessing_plan(config: dict, root_dir: Path):
    """
    Tampilkan rencana preprocessing sebelum dieksekusi.
    Berguna untuk konfirmasi parameter sebelum proses berat dimulai.
    """
    preproc      = config.get("preprocessing", {})
    dataset_path = root_dir / config.get("path", "")
    patch_size   = preproc.get("patch_size", 512)
    overlap      = preproc.get("overlap", 0.2)
    stride       = int(patch_size * (1.0 - overlap))

    logger.info("=" * 60)
    logger.info("RENCANA PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"  Source citra    : {dataset_path / 'raw' / 'images'}")
    logger.info(f"  Source JSON     : {dataset_path / 'raw' / 'dronewaste_v1.0.json'}")
    logger.info(f"  Output dir      : {dataset_path / 'processed'}")
    logger.info(f"  Patch size      : {patch_size} x {patch_size} piksel")
    logger.info(f"  Overlap         : {overlap * 100:.0f}%")
    logger.info(f"  Stride          : {stride} piksel")
    logger.info(f"  Train/Val split : {preproc.get('train_val_split', 0.8) * 100:.0f}% / "
                f"{(1 - preproc.get('train_val_split', 0.8)) * 100:.0f}%")
    logger.info(f"  Min visibility  : {preproc.get('min_visibility', 0.1) * 100:.0f}%")
    logger.info(f"  Random seed     : {preproc.get('random_seed', 42)}")
    logger.info("=" * 60)


def main():
    """Entry point utama script preprocessing."""

    # --------------------------------------------------------------------------
    # 1. Parse argumen
    # --------------------------------------------------------------------------
    args = parse_args()

    # Buat folder outputs jika belum ada (untuk file log)
    (ROOT_DIR / "outputs").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AeroSweep — Preprocessing Pipeline")
    logger.info("=" * 60)

    # --------------------------------------------------------------------------
    # 2. Load dan validasi konfigurasi
    # --------------------------------------------------------------------------
    config_path = ROOT_DIR / args.config
    config      = load_config(config_path)
    validate_config(config, ROOT_DIR)

    # Ekstrak parameter dari config
    preproc      = config.get("preprocessing", {})
    dataset_path = ROOT_DIR / config["path"]

    coco_json_path = dataset_path / "raw" / "dronewaste_v1.0.json"
    images_dir     = dataset_path / "raw" / "images"
    output_dir     = dataset_path / "processed"

    patch_size      = preproc.get("patch_size",      512)
    overlap         = preproc.get("overlap",          0.2)
    train_val_split = preproc.get("train_val_split",  0.8)
    min_visibility  = preproc.get("min_visibility",   0.1)
    random_seed     = preproc.get("random_seed",       42)

    # --------------------------------------------------------------------------
    # 3. Inisialisasi DroneWasteDataset
    # --------------------------------------------------------------------------
    dataset = DroneWasteDataset(
        coco_json_path  = str(coco_json_path),
        images_dir      = str(images_dir),
        output_dir      = str(output_dir),
        config_path     = str(config_path),
        patch_size      = patch_size,
        overlap         = overlap,
        train_val_split = train_val_split,
        min_visibility  = min_visibility,
        random_seed     = random_seed,
    )

    # --------------------------------------------------------------------------
    # 4. Mode: summary-only — hanya tampilkan statistik
    # --------------------------------------------------------------------------
    if args.summary_only:
        dataset.print_dataset_summary()
        logger.info("Mode summary-only selesai. Tidak ada data yang diproses.")
        return

    # Tampilkan ringkasan dataset
    dataset.print_dataset_summary()

    # Tampilkan rencana preprocessing
    print_preprocessing_plan(config, ROOT_DIR)

    # --------------------------------------------------------------------------
    # 5. Mode: dry-run — validasi saja, tidak proses
    # --------------------------------------------------------------------------
    if args.dry_run:
        logger.info("Mode DRY-RUN aktif — tidak ada data yang diproses.")
        logger.info("Semua validasi PASSED. Siap untuk dijalankan tanpa --dry-run.")
        return

    # --------------------------------------------------------------------------
    # 6. Update cv.yaml dengan nama kelas dari JSON (Opsi B)
    # --------------------------------------------------------------------------
    logger.info("Langkah 1/3: Memperbarui cv.yaml dengan nama kelas dari JSON...")
    dataset.update_yaml()
    logger.info("cv.yaml berhasil diperbarui ✓")

    # --------------------------------------------------------------------------
    # 7. Jalankan sliding window + konversi anotasi COCO → YOLO
    # --------------------------------------------------------------------------
    logger.info("Langkah 2/3: Menjalankan sliding window & konversi anotasi...")
    start_time = time.time()
    dataset.run()
    elapsed = time.time() - start_time
    logger.info(f"Sliding window selesai dalam {elapsed:.1f} detik ✓")

    # --------------------------------------------------------------------------
    # 8. Split dataset train/val
    # --------------------------------------------------------------------------
    if not args.skip_split:
        logger.info("Langkah 3/3: Membagi dataset ke train/val...")
        dataset.split_dataset()
        logger.info("Split dataset selesai ✓")
    else:
        logger.info("Langkah 3/3: Dilewati (--skip-split aktif)")

    # --------------------------------------------------------------------------
    # 9. Ringkasan akhir
    # --------------------------------------------------------------------------
    total_elapsed = time.time() - start_time

    # Hitung jumlah patch hasil
    train_patches = len(list((output_dir / "train" / "images").glob("*.jpg")))
    val_patches   = len(list((output_dir / "val"   / "images").glob("*.jpg")))

    logger.info("=" * 60)
    logger.info("PREPROCESSING SELESAI")
    logger.info("=" * 60)
    logger.info(f"  Total patch train  : {train_patches}")
    logger.info(f"  Total patch val    : {val_patches}")
    logger.info(f"  Total patch        : {train_patches + val_patches}")
    logger.info(f"  Waktu total        : {total_elapsed:.1f} detik")
    logger.info(f"  Output tersimpan di: {output_dir}")
    logger.info(f"  Log tersimpan di   : {ROOT_DIR / 'outputs' / 'preprocessing.log'}")
    logger.info("=" * 60)
    logger.info("Langkah berikutnya: jalankan scripts/02_train_cv.py")
    logger.info("=" * 60)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
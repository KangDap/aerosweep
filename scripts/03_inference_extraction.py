# ==============================================================================
# AeroSweep — Entry Point Inferensi & Ekstraksi Fitur
# File: scripts/03_inference_extraction.py
#
# Tanggung jawab:
#   1. Membaca konfigurasi dari configs/cv.yaml
#   2. Memanggil FeatureExtractor dari src/aerosweep/neuro/inference_cv.py
#   3. Menjalankan inferensi batch pada seluruh grid citra UAV
#   4. Mengumpulkan 6 metrik per grid ke dalam Pandas DataFrame
#   5. Mengekspor DataFrame ke outputs/output_fitur_ann_ke_fuzzy.csv
#   6. Mencetak ringkasan statistik hasil ekstraksi
#
# Cara menjalankan (dari ROOT project):
#   python scripts/03_inference_extraction.py
#   python scripts/03_inference_extraction.py --grid-dir data/vision/test_grids
#   python scripts/03_inference_extraction.py --conf 0.3 --device cpu
#   python scripts/03_inference_extraction.py --dry-run
#   python scripts/03_inference_extraction.py --save-annotated
#
# Urutan eksekusi pipeline:
#   01_preprocessing.py → 02_train_cv.py → [03_inference_extraction.py]
#
# OUTPUT: outputs/output_fitur_ann_ke_fuzzy.csv
# ==============================================================================

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# ------------------------------------------------------------------------------
# Pastikan ROOT project ada di sys.path
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.aerosweep.neuro.inference_cv import FeatureExtractor

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
        logging.FileHandler(log_dir / "inference.log"),
    ],
)
logger = logging.getLogger("AeroSweep.03_inference_extraction")


# ==============================================================================
# SKEMA KOLOM CSV
# Urutan ini WAJIB konsisten — tim Fuzzy bergantung pada urutan kolom ini
# ==============================================================================
CSV_COLUMNS = [
    "grid_id",           # Nama unik grid: img_stem_grid_row_col
    "centroid_x",        # Koordinat X centroid gabungan mask (piksel), -1 jika kosong
    "centroid_y",        # Koordinat Y centroid gabungan mask (piksel), -1 jika kosong
    "area_density_pct",  # Kepadatan area sampah (%), 0.0 jika kosong
    "jumlah_instance",   # Jumlah objek terdeteksi (integer), 0 jika kosong
    "kategori_dominan",  # Nama kelas material terdominasi, "none" jika kosong
    "confidence_score",  # Rata-rata confidence score [0.0-1.0], 0.0 jika kosong
]


# ==============================================================================
# ARGUMEN CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse argumen command line."""
    parser = argparse.ArgumentParser(
        description=(
            "AeroSweep — Inferensi & Ekstraksi Fitur\n"
            "Menjalankan YOLO instance segmentation pada grid UAV\n"
            "dan mengekspor 6 metrik ke CSV untuk tim GA-Fuzzy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh penggunaan:\n"
            "  # Inferensi dengan konfigurasi default\n"
            "  python scripts/03_inference_extraction.py\n\n"
            "  # Tentukan folder grid dan threshold kustom\n"
            "  python scripts/03_inference_extraction.py \\\n"
            "      --grid-dir data/vision/test_grids --conf 0.3\n\n"
            "  # Simpan citra hasil anotasi + jalankan di CPU\n"
            "  python scripts/03_inference_extraction.py \\\n"
            "      --save-annotated --device cpu\n\n"
            "  # Validasi konfigurasi saja\n"
            "  python scripts/03_inference_extraction.py --dry-run\n"
        ),
    )

    # --- Konfigurasi ---
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cv.yaml",
        help="Path ke file konfigurasi cv.yaml.",
    )

    # --- Override parameter inferensi ---
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override path model dari cv.yaml. "
            "Contoh: models/nn_weights/aerosweep_trained/weights/best.pt"
        ),
    )
    parser.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help=(
            "Override direktori grid dari cv.yaml. "
            "Default: data/vision/test_grids"
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help=(
            "Override confidence threshold dari cv.yaml. "
            "Range: 0.0–1.0. Default: 0.25"
        ),
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help=(
            "Override IoU threshold dari cv.yaml. "
            "Range: 0.0–1.0. Default: 0.45"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Override device dari cv.yaml. "
            "Contoh: '0' (GPU), 'cpu'."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Override nama file CSV output. "
            "Default: output_fitur_ann_ke_fuzzy.csv"
        ),
    )

    # --- Mode operasi ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validasi konfigurasi dan path saja tanpa inferensi.",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        default=False,
        help=(
            "Simpan citra grid dengan overlay mask, bounding box, "
            "dan titik centroid ke outputs/annotated_grids/."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help=(
            "Append hasil ke CSV yang sudah ada (jika ada). "
            "Default: timpa (overwrite) file CSV lama."
        ),
    )

    return parser.parse_args()


# ==============================================================================
# FUNGSI UTILITAS
# ==============================================================================

def load_config(config_path: Path) -> dict:
    """Baca cv.yaml dan kembalikan sebagai dictionary."""
    if not config_path.exists():
        logger.error(f"File konfigurasi tidak ditemukan: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Konfigurasi dimuat dari: {config_path}")
    return config


def resolve_inference_params(
    config:   dict,
    args:     argparse.Namespace,
    root_dir: Path,
) -> dict:
    """
    Gabungkan parameter inferensi dari cv.yaml dan argumen CLI.
    CLI selalu lebih prioritas dari cv.yaml.

    Parameters
    ----------
    config   : Dictionary dari cv.yaml
    args     : Argumen dari CLI
    root_dir : Path root project

    Returns
    -------
    Dictionary parameter inferensi yang sudah diresolved.
    """
    infer_cfg = config.get("inference", {})
    dataset_path = root_dir / config.get("path", "data/vision")

    # Resolve setiap parameter: CLI > cv.yaml > default hardcoded
    params = {
        "model_path": str(
            root_dir / (
                args.model or
                infer_cfg.get(
                    "model_path",
                    "models/nn_weights/aerosweep_trained/weights/best.pt"
                )
            )
        ),
        "grid_dir": str(
            root_dir / (
                args.grid_dir or
                infer_cfg.get("source", "data/vision/test_grids")
            )
        ),
        "conf_threshold": (
            args.conf or
            infer_cfg.get("conf_threshold", 0.25)
        ),
        "iou_threshold": (
            args.iou or
            infer_cfg.get("iou_threshold", 0.45)
        ),
        "device": str(
            args.device or
            infer_cfg.get("device", "0")
        ),
        "output_dir": str(
            root_dir / infer_cfg.get("output_dir", "outputs")
        ),
        "output_csv": (
            args.output_csv or
            infer_cfg.get("output_csv", "output_fitur_ann_ke_fuzzy.csv")
        ),
        "save_annotated": (
            args.save_annotated or
            infer_cfg.get("save_annotated", False)
        ),
        "annotated_dir": str(
            root_dir / infer_cfg.get("annotated_dir", "outputs/annotated_grids")
        ),
    }

    return params


def validate_inference_params(params: dict) -> bool:
    """
    Validasi parameter inferensi sebelum proses dimulai.

    Parameters
    ----------
    params : Dictionary parameter hasil resolve_inference_params()

    Returns
    -------
    True jika semua validasi lulus, sys.exit(1) jika gagal.
    """
    logger.info("Memvalidasi parameter inferensi...")
    errors = []

    # Cek model tersedia
    model_path = Path(params["model_path"])
    if not model_path.exists():
        errors.append(
            f"Model tidak ditemukan: {model_path}\n"
            f"  → Jalankan scripts/02_train_cv.py terlebih dahulu.\n"
            f"  → Atau periksa path di configs/cv.yaml → inference.model_path"
        )
    else:
        import os
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"  Model          : {model_path} ({model_size_mb:.1f} MB) ✓")

    # Cek folder grid tersedia dan tidak kosong
    grid_dir = Path(params["grid_dir"])
    if not grid_dir.exists():
        errors.append(
            f"Direktori grid tidak ditemukan: {grid_dir}\n"
            f"  → Pastikan data/vision/test_grids/ berisi file gambar.\n"
            f"  → Atau gunakan: --grid-dir <path>"
        )
    else:
        grid_files = (
            list(grid_dir.glob("*.jpg")) +
            list(grid_dir.glob("*.png"))
        )
        if not grid_files:
            errors.append(
                f"Tidak ada file gambar di: {grid_dir}\n"
                f"  → Masukkan file .jpg atau .png ke folder tersebut."
            )
        else:
            logger.info(f"  Grid directory : {grid_dir} ({len(grid_files)} file) ✓")

    # Cek range threshold
    conf = params["conf_threshold"]
    iou  = params["iou_threshold"]
    if not (0.0 < conf <= 1.0):
        errors.append(f"conf_threshold harus antara 0.0 dan 1.0, dapat: {conf}")
    if not (0.0 < iou <= 1.0):
        errors.append(f"iou_threshold harus antara 0.0 dan 1.0, dapat: {iou}")

    # Tampilkan semua error
    if errors:
        logger.error("Validasi parameter GAGAL:")
        for err in errors:
            logger.error(f"  ✗ {err}")
        sys.exit(1)

    logger.info("Validasi parameter inferensi: PASSED ✓")
    return True


def print_inference_plan(params: dict):
    """Tampilkan rencana inferensi sebelum dieksekusi."""
    grid_files = (
        list(Path(params["grid_dir"]).glob("*.jpg")) +
        list(Path(params["grid_dir"]).glob("*.png"))
    )

    logger.info("=" * 60)
    logger.info("RENCANA INFERENSI")
    logger.info("=" * 60)
    logger.info(f"  Model           : {params['model_path']}")
    logger.info(f"  Grid directory  : {params['grid_dir']}")
    logger.info(f"  Jumlah grid     : {len(grid_files)}")
    logger.info(f"  Conf threshold  : {params['conf_threshold']}")
    logger.info(f"  IoU threshold   : {params['iou_threshold']}")
    logger.info(f"  Device          : {params['device']}")
    logger.info(f"  Output CSV      : {params['output_dir']}/{params['output_csv']}")
    logger.info(f"  Save annotated  : {params['save_annotated']}")
    logger.info("=" * 60)


# ==============================================================================
# FUNGSI EKSPOR CSV
# ==============================================================================

def build_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Konversi list of dict hasil inferensi ke Pandas DataFrame.

    Memastikan:
    - Urutan kolom konsisten sesuai CSV_COLUMNS
    - Tipe data setiap kolom benar
    - Tidak ada baris duplikat berdasarkan grid_id

    Parameters
    ----------
    results : List of dict dari FeatureExtractor.run_batch()

    Returns
    -------
    DataFrame dengan kolom sesuai CSV_COLUMNS dan tipe data yang benar.
    """
    if not results:
        logger.warning("List hasil inferensi kosong. DataFrame akan kosong.")
        return pd.DataFrame(columns=CSV_COLUMNS)

    # Buat DataFrame dari list of dict
    df = pd.DataFrame(results)

    # Pastikan semua kolom ada (tambahkan dengan NaN jika ada yang hilang)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            logger.warning(f"Kolom '{col}' tidak ditemukan, diisi dengan nilai default.")
            df[col] = None

    # Atur urutan kolom sesuai skema
    df = df[CSV_COLUMNS]

    # --- Paksakan tipe data yang benar ---
    df["grid_id"]           = df["grid_id"].astype(str)
    df["centroid_x"]        = pd.to_numeric(df["centroid_x"],        errors="coerce")
    df["centroid_y"]        = pd.to_numeric(df["centroid_y"],        errors="coerce")
    df["area_density_pct"]  = pd.to_numeric(df["area_density_pct"],  errors="coerce")
    df["jumlah_instance"]   = pd.to_numeric(df["jumlah_instance"],   errors="coerce").fillna(0).astype(int)
    df["kategori_dominan"]  = df["kategori_dominan"].astype(str)
    df["confidence_score"]  = pd.to_numeric(df["confidence_score"],  errors="coerce")

    # Hapus duplikat berdasarkan grid_id (jaga-jaga mode append)
    n_before = len(df)
    df = df.drop_duplicates(subset=["grid_id"], keep="last")
    n_after  = len(df)
    if n_before != n_after:
        logger.warning(f"Ditemukan {n_before - n_after} baris duplikat, dihapus.")

    logger.info(f"DataFrame dibangun: {len(df)} baris, {len(df.columns)} kolom")
    return df


def export_to_csv(
    df:          pd.DataFrame,
    output_path: Path,
    append_mode: bool = False,
) -> Path:
    """
    Ekspor DataFrame ke file CSV.

    Mode overwrite (default):
        Timpa file CSV lama dengan hasil baru.

    Mode append (--append):
        Baca CSV lama, gabungkan dengan hasil baru,
        deduplikasi berdasarkan grid_id, simpan kembali.
        Berguna untuk memproses dataset besar secara bertahap.

    Parameters
    ----------
    df          : DataFrame hasil inferensi
    output_path : Path file CSV output
    append_mode : True = append ke file lama, False = overwrite

    Returns
    -------
    Path file CSV yang disimpan.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append_mode and output_path.exists():
        # Baca data lama
        logger.info(f"Mode APPEND: membaca file lama dari {output_path}")
        df_old = pd.read_csv(output_path)
        logger.info(f"  Data lama      : {len(df_old)} baris")

        # Gabungkan: data lama + data baru
        df_combined = pd.concat([df_old, df], ignore_index=True)

        # Deduplikasi: kalau grid_id sama, pakai data terbaru (keep="last")
        n_before = len(df_combined)
        df_combined = df_combined.drop_duplicates(
            subset=["grid_id"], keep="last"
        )
        n_after = len(df_combined)

        if n_before != n_after:
            logger.info(
                f"  Duplikat dihapus: {n_before - n_after} baris "
                f"(grid yang sama diperbarui dengan hasil terbaru)"
            )

        df_final = df_combined[CSV_COLUMNS]
        logger.info(f"  Data final     : {len(df_final)} baris (lama + baru)")

    else:
        # Overwrite mode
        if output_path.exists():
            logger.info(f"File CSV lama akan ditimpa: {output_path}")
        df_final = df

    # Simpan ke disk
    df_final.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"CSV berhasil disimpan: {output_path} ✓")

    return output_path


def print_dataframe_summary(df: pd.DataFrame):
    """
    Tampilkan ringkasan statistik DataFrame sebelum diekspor.
    Membantu tim Fuzzy memahami distribusi data yang akan diterima.
    """
    if df.empty:
        logger.warning("DataFrame kosong, tidak ada statistik yang ditampilkan.")
        return

    n_total     = len(df)
    n_detected  = len(df[df["jumlah_instance"] > 0])
    n_empty     = len(df[df["jumlah_instance"] == 0])
    detect_rate = (n_detected / n_total * 100) if n_total > 0 else 0.0

    logger.info("=" * 60)
    logger.info("RINGKASAN HASIL EKSTRAKSI FITUR")
    logger.info("=" * 60)
    logger.info(f"  Total grid diproses    : {n_total}")
    logger.info(f"  Grid dengan deteksi    : {n_detected} ({detect_rate:.1f}%)")
    logger.info(f"  Grid tanpa deteksi     : {n_empty}")
    logger.info("")

    # Statistik numerik (hanya grid dengan deteksi)
    df_detected = df[df["jumlah_instance"] > 0]

    if not df_detected.empty:
        logger.info("  Statistik (grid dengan deteksi):")
        logger.info(
            f"  Area Density (%)"
            f"   min={df_detected['area_density_pct'].min():.2f}"
            f"   max={df_detected['area_density_pct'].max():.2f}"
            f"   mean={df_detected['area_density_pct'].mean():.2f}"
        )
        logger.info(
            f"  Jumlah Instance "
            f"   min={df_detected['jumlah_instance'].min()}"
            f"   max={df_detected['jumlah_instance'].max()}"
            f"   mean={df_detected['jumlah_instance'].mean():.1f}"
        )
        logger.info(
            f"  Confidence Score"
            f"   min={df_detected['confidence_score'].min():.3f}"
            f"   max={df_detected['confidence_score'].max():.3f}"
            f"   mean={df_detected['confidence_score'].mean():.3f}"
        )

        # Distribusi kategori dominan
        logger.info("")
        logger.info("  Distribusi kategori dominan:")
        cat_counts = df_detected["kategori_dominan"].value_counts()
        for cat, count in cat_counts.items():
            pct = count / n_detected * 100
            bar = "█" * min(int(pct / 2), 25)
            logger.info(f"    {cat:<25} {count:4d} ({pct:5.1f}%) {bar}")

    logger.info("=" * 60)


def add_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan kolom metadata tambahan ke DataFrame sebelum diekspor.
    Kolom ini membantu tim Fuzzy melakukan traceability dan audit.

    Kolom tambahan:
        processed_at : Timestamp saat inferensi dijalankan
        has_detection: Boolean flag (1 = ada deteksi, 0 = tidak ada)

    Parameters
    ----------
    df : DataFrame utama

    Returns
    -------
    DataFrame dengan kolom metadata tambahan.
    """
    # Timestamp inferensi
    df["processed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Flag boolean: 1 jika ada deteksi, 0 jika tidak
    df["has_detection"] = (df["jumlah_instance"] > 0).astype(int)

    return df


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Entry point utama script inferensi dan ekstraksi fitur."""

    # --------------------------------------------------------------------------
    # 1. Parse argumen
    # --------------------------------------------------------------------------
    args = parse_args()

    logger.info("=" * 60)
    logger.info("AeroSweep — Inference & Feature Extraction Pipeline")
    logger.info("=" * 60)

    # --------------------------------------------------------------------------
    # 2. Load konfigurasi
    # --------------------------------------------------------------------------
    config_path = ROOT_DIR / args.config
    config      = load_config(config_path)

    # --------------------------------------------------------------------------
    # 3. Resolve parameter inferensi (CLI > cv.yaml > default)
    # --------------------------------------------------------------------------
    params = resolve_inference_params(config, args, ROOT_DIR)

    # --------------------------------------------------------------------------
    # 4. Validasi parameter
    # --------------------------------------------------------------------------
    validate_inference_params(params)

    # --------------------------------------------------------------------------
    # 5. Tampilkan rencana inferensi
    # --------------------------------------------------------------------------
    print_inference_plan(params)

    # --------------------------------------------------------------------------
    # 6. Mode dry-run: validasi saja
    # --------------------------------------------------------------------------
    if args.dry_run:
        logger.info("Mode DRY-RUN aktif — tidak ada inferensi yang dijalankan.")
        logger.info("Semua validasi PASSED ✓")
        logger.info("Siap untuk inferensi. Jalankan tanpa --dry-run.")
        return

    # --------------------------------------------------------------------------
    # 7. Inisialisasi FeatureExtractor
    # --------------------------------------------------------------------------
    extractor = FeatureExtractor(
        model_path      = params["model_path"],
        config_path     = str(config_path),
        conf_threshold  = params["conf_threshold"],
        iou_threshold   = params["iou_threshold"],
        device          = params["device"],
        save_annotated  = params["save_annotated"],
        annotated_dir   = params["annotated_dir"],
    )

    # --------------------------------------------------------------------------
    # 8. Jalankan inferensi batch
    # --------------------------------------------------------------------------
    logger.info("Memulai inferensi batch pada grid UAV...")
    start_time = time.time()

    try:
        results = extractor.run_batch(grid_dir=params["grid_dir"])
    except KeyboardInterrupt:
        logger.warning("Inferensi dihentikan oleh pengguna.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Inferensi gagal: {e}")
        raise

    elapsed = time.time() - start_time
    logger.info(f"Inferensi selesai dalam {elapsed:.1f} detik ✓")

    # --------------------------------------------------------------------------
    # 9. Bangun DataFrame dari hasil inferensi
    # --------------------------------------------------------------------------
    logger.info("Membangun DataFrame...")
    df = build_dataframe(results)

    # --------------------------------------------------------------------------
    # 10. Tambahkan kolom metadata
    # --------------------------------------------------------------------------
    df = add_metadata_columns(df)

    # --------------------------------------------------------------------------
    # 11. Tampilkan ringkasan statistik
    # --------------------------------------------------------------------------
    print_dataframe_summary(df)

    # --------------------------------------------------------------------------
    # 12. Preview 5 baris pertama DataFrame di log
    # --------------------------------------------------------------------------
    logger.info("Preview 5 baris pertama DataFrame:")
    logger.info("\n" + df.head().to_string(index=False))

    # --------------------------------------------------------------------------
    # 13. Ekspor ke CSV
    # --------------------------------------------------------------------------
    output_csv_path = Path(params["output_dir"]) / params["output_csv"]

    logger.info(f"Mengekspor CSV ke: {output_csv_path}")
    export_to_csv(
        df          = df,
        output_path = output_csv_path,
        append_mode = args.append,
    )

    # --------------------------------------------------------------------------
    # 14. Ringkasan akhir
    # --------------------------------------------------------------------------
    total_elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("EKSTRAKSI FITUR SELESAI")
    logger.info("=" * 60)
    logger.info(f"  Total grid diproses : {len(results)}")
    logger.info(f"  Total baris CSV     : {len(df)}")
    logger.info(f"  Waktu total         : {total_elapsed:.1f} detik")
    logger.info(f"  Output CSV          : {output_csv_path}")
    logger.info(f"  Log inferensi       : {log_dir / 'inference.log'}")
    if params["save_annotated"]:
        logger.info(f"  Citra anotasi       : {params['annotated_dir']}")
    logger.info("=" * 60)
    logger.info("File CSV siap dikirim ke tim GA-Fuzzy.")
    logger.info("=" * 60)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
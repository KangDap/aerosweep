"""
src/aerosweep/neuro/train_cv.py
================================
Pipeline training YOLOv8x untuk AeroSweep dengan K-Fold Cross Validation.

Cara pakai:
  # Training fold 0 (default):
  python src/aerosweep/neuro/train_cv.py

  # Training fold tertentu:
  python src/aerosweep/neuro/train_cv.py --fold 2

  # Training semua fold sekaligus:
  python src/aerosweep/neuro/train_cv.py --all_folds

  # Pakai config custom:
  python src/aerosweep/neuro/train_cv.py --config configs/cv.yaml --fold 0

Output:
  models/nn_weights/runs/AeroSweep/fold_0/weights/best.pt
  models/nn_weights/runs/AeroSweep/fold_0/weights/last.pt
  models/nn_weights/runs/AeroSweep/fold_0/results.csv   ← loss & mAP per epoch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Pastikan root project ada di sys.path agar bisa import modul aerosweep
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # src/aerosweep/neuro/ → root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aerosweep.neuro.dataset import DroneWasteDataset  # noqa: E402


# ---------------------------------------------------------------------------
# UTILITAS
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    """Membaca cv.yaml dan mengembalikan dict konfigurasi."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def prepare_fold(cfg: dict, fold_idx: int) -> Path:
    """
    Menjalankan DroneWasteDataset.prepare() untuk fold tertentu.
    Mengembalikan path ke dataset.yaml Ultralytics.
    """
    coco_json    = ROOT / cfg["data"]["coco_json"]
    images_dir   = ROOT / cfg["data"]["images_dir"]
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    grid_size    = tuple(cfg["data"]["grid_size"])
    n_folds      = cfg["cross_validation"]["n_folds"]
    seed         = cfg["cross_validation"]["seed"]

    ds = DroneWasteDataset(
        coco_json=coco_json,
        images_dir=images_dir,
        output_dir=processed_dir,
        seed=seed,
    )

    paths = ds.prepare(
        grid_size=grid_size,
        n_folds=n_folds,
        fold_idx=fold_idx,
        copy_images=False,      # symlink lebih cepat; fallback ke copy di Windows
    )

    return paths["yaml_path"]


def train_fold(cfg: dict, fold_idx: int, dataset_yaml: Path) -> Path:
    """
    Melatih YOLOv8x untuk satu fold.
    Mengembalikan path ke best.pt.
    """
    # Import Ultralytics di sini agar error muncul lebih jelas jika belum terinstall
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics belum terinstall. Jalankan:\n"
            "  pip install ultralytics"
        )

    t_cfg  = cfg["train"]
    m_cfg  = cfg["model"]
    o_cfg  = cfg["output"]

    # Nama eksperimen: fold_0, fold_1, dst.
    experiment = o_cfg["experiment_name"].replace("{fold_idx}", str(fold_idx))

    runs_dir = ROOT / o_cfg["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Mulai training — Fold {fold_idx}/{cfg['cross_validation']['n_folds'] - 1}")
    print(f"  Model      : {m_cfg['pretrained_weights']}")
    print(f"  Dataset    : {dataset_yaml}")
    print(f"  Epochs     : {t_cfg['epochs']}  |  Batch : {t_cfg['batch']}")
    print(f"  Output     : {runs_dir / o_cfg['project'] / experiment}")
    print(f"{'='*60}\n")

    model = YOLO(m_cfg["pretrained_weights"])

    results = model.train(
        data        = str(dataset_yaml),
        epochs      = t_cfg["epochs"],
        imgsz       = cfg["data"]["imgsz"],
        batch       = t_cfg["batch"],
        workers     = t_cfg["workers"],
        device      = t_cfg["device"],

        # Optimizer & LR
        optimizer   = t_cfg["optimizer"],
        lr0         = t_cfg["lr0"],
        lrf         = t_cfg["lrf"],
        momentum    = t_cfg["momentum"],
        weight_decay= t_cfg["weight_decay"],

        # Warmup
        warmup_epochs   = t_cfg["warmup_epochs"],
        warmup_momentum = t_cfg["warmup_momentum"],
        warmup_bias_lr  = t_cfg["warmup_bias_lr"],

        # Early stopping
        patience    = t_cfg["patience"],

        # Augmentasi
        hsv_h       = t_cfg["hsv_h"],
        hsv_s       = t_cfg["hsv_s"],
        hsv_v       = t_cfg["hsv_v"],
        degrees     = t_cfg["degrees"],
        translate   = t_cfg["translate"],
        scale       = t_cfg["scale"],
        shear       = t_cfg["shear"],
        perspective = t_cfg["perspective"],
        flipud      = t_cfg["flipud"],
        fliplr      = t_cfg["fliplr"],
        mosaic      = t_cfg["mosaic"],
        mixup       = t_cfg["mixup"],
        copy_paste  = t_cfg["copy_paste"],
        close_mosaic= t_cfg["close_mosaic"],

        # Output & logging
        project     = str(runs_dir / o_cfg["project"]),
        name        = experiment,
        exist_ok    = t_cfg["exist_ok"],
        plots       = t_cfg["plots"],
        cache       = t_cfg["cache"],
    )

    # Path ke best weights
    best_pt = Path(results.save_dir) / "weights" / "best.pt"

    # Salin juga ke folder nn_weights/ untuk akses mudah
    weights_dir = ROOT / o_cfg["weights_dir"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / f"fold_{fold_idx}_best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, dest)
        print(f"\n[train] Best weights disalin ke: {dest}")

    return best_pt


def run_training(cfg: dict, fold_idx: int) -> None:
    """Prepare dataset + train untuk satu fold."""
    start = time.time()

    print(f"\n[train] Menyiapkan dataset untuk fold {fold_idx} ...")
    dataset_yaml = prepare_fold(cfg, fold_idx)

    best_pt = train_fold(cfg, fold_idx, dataset_yaml)

    elapsed = time.time() - start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)

    print(f"\n{'='*60}")
    print(f"  Training fold {fold_idx} selesai dalam {h}j {m}m {s}d")
    print(f"  Best weights : {best_pt}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AeroSweep — Training YOLOv8x dengan K-Fold CV"
    )
    parser.add_argument(
        "--config",
        default="configs/cv.yaml",
        help="Path ke file konfigurasi (default: configs/cv.yaml)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Index fold yang akan dilatih (0-based). "
             "Jika tidak diisi, pakai fold_idx dari config.",
    )
    parser.add_argument(
        "--all_folds",
        action="store_true",
        help="Latih semua fold secara berurutan.",
    )
    args = parser.parse_args()

    # Load config
    config_path = ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config tidak ditemukan: {config_path}")
    cfg = load_config(config_path)

    n_folds = cfg["cross_validation"]["n_folds"]

    if args.all_folds:
        print(f"[train] Mode: semua {n_folds} fold")
        for fold_idx in range(n_folds):
            run_training(cfg, fold_idx)
    else:
        fold_idx = args.fold if args.fold is not None else cfg["cross_validation"]["fold_idx"]
        if not (0 <= fold_idx < n_folds):
            raise ValueError(f"fold_idx harus antara 0 dan {n_folds - 1}, dapat: {fold_idx}")
        run_training(cfg, fold_idx)


if __name__ == "__main__":
    main()

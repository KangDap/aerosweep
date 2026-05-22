"""
src/aerosweep/neuro/dataset.py
================================
Modul dataset untuk AeroSweep — menangani:
  1. Parsing anotasi COCO (DroneWaste)
  2. Class merging: 20 kelas → 5 superkategori
  3. Konversi format COCO → YOLO (.txt per gambar)
  4. K-Fold split untuk cross-validation
  5. Grid mapping: agregasi deteksi per sel → output tabular untuk Fuzzy

Struktur output:
  data/vision/processed/
    ├── images/train/   ← symlink atau copy gambar
    ├── images/val/
    ├── labels/train/   ← file .txt format YOLO
    ├── labels/val/
    └── dataset.yaml    ← config Ultralytics

  data/tabular/
    └── grid_results.csv  ← [image_id, grid_row, grid_col,
                               jumlah_sampah, kepadatan_pct, superclass_counts]

Penggunaan:
  from src.aerosweep.neuro.dataset import DroneWasteDataset

  ds = DroneWasteDataset(
      coco_json="data/vision/raw/annotations.json",
      images_dir="data/vision/raw/images",
      output_dir="data/vision/processed",
  )
  ds.prepare(grid_size=(8, 8), n_folds=5, fold_idx=0)
"""

from __future__ import annotations

import json
import math
import shutil
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1. SUPERCLASS MAPPING  (20 DroneWaste kelas → 5 superkategori)
# ---------------------------------------------------------------------------
# Kunci  : nama kategori asli di COCO JSON DroneWaste (lowercase, strip spasi)
# Nilai  : (superclass_id, superclass_name)
#
# Catatan: nama kategori exak ada di annotations.json.  Kalau ada yang tidak
#          cocok, method _build_category_map() akan mencetak peringatan
#          sehingga mudah diperbaiki tanpa crash.

SUPERCLASS_MAP: Dict[str, Tuple[int, str]] = {
    # ── 0 : Plastik & Kemasan ─────────────────────────────────────────
    # Semua material berbasis plastik dan kertas/kemasan ringan
    "plastic packaging":                    (0, "plastic_packaging"),
    "plastic":                              (0, "plastic_packaging"),   # id:13 — tumpukan plastik
    "paper":                                (0, "plastic_packaging"),   # id:16 — kertas/karton

    # ── 1 : Logam & Kaca ──────────────────────────────────────────────
    # Limbah logam, besi tua, peralatan elektrikal berat
    "metal barrels":                        (1, "metal_glass"),         # id:8
    "scrap":                                (1, "metal_glass"),         # id:12 — besi tua
    "appliances":                           (1, "metal_glass"),         # id:5 — peralatan rumah tangga besar
    "foundry":                              (1, "metal_glass"),         # id:17 — abu/slag industri

    # ── 2 : Konstruksi & Puing ────────────────────────────────────────
    # Puing bangunan, tanah galian, material jalan
    "construction and demolition materials":(2, "construction"),        # id:2
    "asphalt milling":                      (2, "construction"),        # id:3
    "excavation materials":                 (2, "construction"),        # id:4
    "pallets":                              (2, "construction"),        # id:11 — palet kayu industri
    "rubble":                               (2, "construction"),        # id:1 — tanah/bebatuan

    # ── 3 : Berbahaya ─────────────────────────────────────────────────
    # Limbah B3: kendaraan, ban, elektronik, asbes
    "electronic equipment":                 (3, "hazardous"),           # id:6
    "vehicles":                             (3, "hazardous"),           # id:14
    "tyres":                                (3, "hazardous"),           # id:15
    "asbestos":                             (3, "hazardous"),           # id:18 — sangat berbahaya

    # ── 4 : Organik & Campuran ────────────────────────────────────────
    # Furnitur, kayu non-industri, tekstil, campuran tak terklasifikasi
    "furniture":                            (4, "organic_mixed"),       # id:7
    "mixed items":                          (4, "organic_mixed"),       # id:20
    "wood":                                 (4, "organic_mixed"),       # id:10 — limbah kayu
    "textile":                              (4, "organic_mixed"),       # id:19 — limbah tekstil
}

NUM_SUPERCLASSES = 5
SUPERCLASS_NAMES = [
    "plastic_packaging",   # 0
    "metal_glass",         # 1
    "construction",        # 2
    "hazardous",           # 3
    "organic_mixed",       # 4
]


# ---------------------------------------------------------------------------
# 2. KELAS UTAMA
# ---------------------------------------------------------------------------

class DroneWasteDataset:
    """
    Mengelola dataset DroneWaste dari raw COCO JSON hingga siap training YOLO
    dan menghasilkan tabel grid untuk modul Fuzzy.

    Parameters
    ----------
    coco_json   : Path ke file annotations.json (format COCO)
    images_dir  : Direktori berisi gambar-gambar (.jpg)
    output_dir  : Root output → data/vision/processed/
    seed        : Random seed untuk reproducibility
    """

    def __init__(
        self,
        coco_json: str | Path,
        images_dir: str | Path,
        output_dir: str | Path,
        seed: int = 42,
    ) -> None:
        self.coco_json   = Path(coco_json)
        self.images_dir  = Path(images_dir)
        self.output_dir  = Path(output_dir)
        self.seed        = seed

        self._coco: dict            = {}
        # cat_id (COCO) → (superclass_id, superclass_name)
        self._cat_map: Dict[int, Tuple[int, str]] = {}
        # image_id → list of annotations
        self._ann_by_img: Dict[int, List[dict]] = defaultdict(list)
        # image_id → image info dict
        self._img_info: Dict[int, dict] = {}

        self._loaded = False

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def prepare(
        self,
        grid_size: Tuple[int, int] = (8, 8),
        n_folds: int = 5,
        fold_idx: int = 0,
        copy_images: bool = False,
    ) -> Dict[str, Path]:
        """
        Pipeline lengkap satu panggilan:
          load → build_map → split → write_yolo → write_grid_csv

        Parameters
        ----------
        grid_size   : (n_rows, n_cols) pembagian sel per gambar
        n_folds     : jumlah fold untuk k-fold CV
        fold_idx    : fold yang dipakai sebagai validasi (0-based)
        copy_images : True = salin gambar; False = buat symlink (lebih cepat)

        Returns
        -------
        dict berisi path-path output penting
        """
        self._load_coco()
        self._build_category_map()
        train_ids, val_ids = self._kfold_split(n_folds, fold_idx)

        yolo_dir = self._write_yolo_labels(train_ids, val_ids, copy_images)
        yaml_path = self._write_dataset_yaml(yolo_dir)
        grid_csv  = self._write_grid_csv(grid_size)

        print(f"\n{'='*55}")
        print(f"  DroneWaste dataset siap")
        print(f"  Train : {len(train_ids)} gambar")
        print(f"  Val   : {len(val_ids)} gambar  (fold {fold_idx}/{n_folds})")
        print(f"  YAML  : {yaml_path}")
        print(f"  Grid  : {grid_csv}")
        print(f"{'='*55}\n")

        return {
            "yolo_dir":  yolo_dir,
            "yaml_path": yaml_path,
            "grid_csv":  grid_csv,
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Menghitung jumlah instance per superkategori dari seluruh dataset.
        Berguna untuk analisis class imbalance sebelum training.
        """
        if not self._loaded:
            self._load_coco()
            self._build_category_map()

        counts: Dict[str, int] = defaultdict(int)
        for anns in self._ann_by_img.values():
            for ann in anns:
                cid = ann["category_id"]
                if cid in self._cat_map:
                    _, sc_name = self._cat_map[cid]
                    counts[sc_name] += 1
        return dict(counts)

    def compute_grid_features(
        self,
        image_id: int,
        detections: Optional[List[dict]] = None,
        grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Menghitung fitur per sel grid dari satu gambar.
        Dapat dipanggil saat training (dari anotasi GT) maupun saat inference
        (dari deteksi YOLO).

        Parameters
        ----------
        image_id   : ID gambar di COCO
        detections : list of dict dengan key:
                       {bbox: [x,y,w,h], superclass_id: int, confidence: float}
                     Kalau None, maka digunakan anotasi ground truth.
        grid_size  : (n_rows, n_cols)

        Returns
        -------
        np.ndarray shape (n_rows, n_cols, 2+NUM_SUPERCLASSES)
          axis-2: [jumlah_sampah, kepadatan_pct, count_sc0, ..., count_sc4]
        """
        if not self._loaded:
            self._load_coco()
            self._build_category_map()

        img_info = self._img_info[image_id]
        W = img_info["width"]
        H = img_info["height"]
        n_rows, n_cols = grid_size

        cell_w = W / n_cols
        cell_h = H / n_rows

        # shape: (n_rows, n_cols, 2 + NUM_SUPERCLASSES)
        # channels: [count, density_pct, sc0_cnt, sc1_cnt, ..., sc4_cnt]
        features = np.zeros((n_rows, n_cols, 2 + NUM_SUPERCLASSES), dtype=np.float32)

        if detections is None:
            # Gunakan ground truth dari COCO
            raw_anns = self._ann_by_img.get(image_id, [])
            boxes = []
            for ann in raw_anns:
                cid = ann["category_id"]
                if cid not in self._cat_map:
                    continue
                sc_id, _ = self._cat_map[cid]
                x, y, w, h = ann["bbox"]
                boxes.append({"bbox": [x, y, w, h], "superclass_id": sc_id})
        else:
            boxes = detections

        for box in boxes:
            x, y, w, h = box["bbox"]
            sc_id       = box["superclass_id"]
            bbox_area   = w * h  # piksel²

            # Centroid objek → tentukan sel mana
            cx = x + w / 2
            cy = y + h / 2
            col = min(int(cx / cell_w), n_cols - 1)
            row = min(int(cy / cell_h), n_rows - 1)

            cell_area = cell_w * cell_h
            density_contribution = (bbox_area / cell_area) * 100.0  # %

            features[row, col, 0] += 1                     # jumlah_sampah
            features[row, col, 1] += density_contribution  # kepadatan_%
            features[row, col, 2 + sc_id] += 1             # count per superclass

        # Clip kepadatan di 100% (overlap bbox bisa melebihi 100)
        features[:, :, 1] = np.clip(features[:, :, 1], 0.0, 100.0)

        return features

    # ------------------------------------------------------------------
    # PRIVATE — LOADING & PARSING
    # ------------------------------------------------------------------

    def _load_coco(self) -> None:
        """Membaca annotations.json dan membangun index cepat."""
        print(f"[dataset] Membaca {self.coco_json} ...")
        with open(self.coco_json, "r") as f:
            self._coco = json.load(f)

        # Index: image_id → info
        for img in self._coco["images"]:
            self._img_info[img["id"]] = img

        # Index: image_id → [annotations]
        for ann in self._coco["annotations"]:
            self._ann_by_img[ann["image_id"]].append(ann)

        self._loaded = True
        print(f"[dataset] {len(self._img_info)} gambar, "
              f"{len(self._coco['annotations'])} anotasi dimuat.")

    def _build_category_map(self) -> None:
        """
        Memetakan category_id COCO → (superclass_id, superclass_name).
        Menampilkan warning untuk nama kategori yang tidak ada di SUPERCLASS_MAP.
        """
        unmapped = []
        for cat in self._coco["categories"]:
            name_key = cat["name"].lower().strip()
            if name_key in SUPERCLASS_MAP:
                self._cat_map[cat["id"]] = SUPERCLASS_MAP[name_key]
            else:
                # Fallback: masuk ke organic_mixed (id=4)
                self._cat_map[cat["id"]] = (4, "organic_mixed")
                unmapped.append(cat["name"])

        if unmapped:
            print(f"[dataset] WARNING: {len(unmapped)} kategori tidak ada di "
                  f"SUPERCLASS_MAP, di-fallback ke 'organic_mixed':")
            for n in unmapped:
                print(f"          → '{n}'")
            print("          Periksa SUPERCLASS_MAP di dataset.py jika ini salah.")

        mapped_ok = len(self._coco["categories"]) - len(unmapped)
        print(f"[dataset] Category map: {mapped_ok} terpetakan, "
              f"{len(unmapped)} fallback.")

    # ------------------------------------------------------------------
    # PRIVATE — K-FOLD SPLIT
    # ------------------------------------------------------------------

    def _kfold_split(
        self, n_folds: int, fold_idx: int
    ) -> Tuple[List[int], List[int]]:
        """
        Membagi image_id menjadi train/val berdasarkan k-fold.
        Stratified berdasarkan apakah gambar memiliki anotasi atau tidak
        (memastikan background frame terdistribusi merata).
        """
        all_ids = sorted(self._img_info.keys())

        # Pisahkan gambar dengan dan tanpa anotasi
        has_ann  = [i for i in all_ids if len(self._ann_by_img.get(i, [])) > 0]
        no_ann   = [i for i in all_ids if len(self._ann_by_img.get(i, [])) == 0]

        rng = random.Random(self.seed)
        rng.shuffle(has_ann)
        rng.shuffle(no_ann)

        def fold_split(ids: List[int]) -> Tuple[List[int], List[int]]:
            fold_size = math.ceil(len(ids) / n_folds)
            start     = fold_idx * fold_size
            end       = min(start + fold_size, len(ids))
            val   = ids[start:end]
            train = ids[:start] + ids[end:]
            return train, val

        train_ann,  val_ann  = fold_split(has_ann)
        train_noann, val_noann = fold_split(no_ann)

        train_ids = train_ann  + train_noann
        val_ids   = val_ann    + val_noann

        rng.shuffle(train_ids)
        rng.shuffle(val_ids)

        print(f"[dataset] K-Fold {fold_idx+1}/{n_folds}: "
              f"train={len(train_ids)}, val={len(val_ids)}")
        return train_ids, val_ids

    # ------------------------------------------------------------------
    # PRIVATE — YOLO LABEL WRITER
    # ------------------------------------------------------------------

    def _write_yolo_labels(
        self,
        train_ids: List[int],
        val_ids: List[int],
        copy_images: bool,
    ) -> Path:
        """
        Menulis file label YOLO (.txt) dan menyiapkan struktur folder
        yang kompatibel dengan Ultralytics:

          processed/
            images/train/  images/val/
            labels/train/  labels/val/
        """
        base = self.output_dir
        for split in ("train", "val"):
            (base / "images" / split).mkdir(parents=True, exist_ok=True)
            (base / "labels" / split).mkdir(parents=True, exist_ok=True)

        for split_name, ids in [("train", train_ids), ("val", val_ids)]:
            img_out_dir = base / "images" / split_name
            lbl_out_dir = base / "labels" / split_name

            for image_id in ids:
                info     = self._img_info[image_id]
                src_img  = self.images_dir / info["file_name"]
                dst_img  = img_out_dir / info["file_name"]

                # -- salin / symlink gambar --
                if not dst_img.exists():
                    if copy_images:
                        shutil.copy2(src_img, dst_img)
                    else:
                        try:
                            dst_img.symlink_to(src_img.resolve())
                        except (OSError, NotImplementedError):
                            # Windows mungkin perlu hak admin untuk symlink
                            shutil.copy2(src_img, dst_img)

                # -- tulis label YOLO --
                W = info["width"]
                H = info["height"]
                stem = Path(info["file_name"]).stem
                lbl_path = lbl_out_dir / f"{stem}.txt"

                lines = []
                for ann in self._ann_by_img.get(image_id, []):
                    cid = ann["category_id"]
                    if cid not in self._cat_map:
                        continue
                    sc_id, _ = self._cat_map[cid]

                    # COCO bbox: [x_top_left, y_top_left, width, height]
                    # YOLO bbox: [cx, cy, w, h]  — semua dinormalisasi 0-1
                    bx, by, bw, bh = ann["bbox"]
                    cx = (bx + bw / 2) / W
                    cy = (by + bh / 2) / H
                    nw = bw / W
                    nh = bh / H

                    # Validasi agar tidak ada nilai di luar [0,1]
                    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
                    nw, nh = max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))

                    if nw > 0 and nh > 0:
                        lines.append(f"{sc_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                with open(lbl_path, "w") as f:
                    f.write("\n".join(lines))

        print(f"[dataset] Label YOLO ditulis ke {base}")
        return base

    def _write_dataset_yaml(self, yolo_dir: Path) -> Path:
        """
        Menulis dataset.yaml yang dibutuhkan oleh Ultralytics YOLO.
        """
        yaml_path = yolo_dir / "dataset.yaml"
        names_str = "\n".join(
            f"  {i}: {name}" for i, name in enumerate(SUPERCLASS_NAMES)
        )
        content = (
            f"# AeroSweep — DroneWaste dataset config\n"
            f"# Auto-generated oleh dataset.py\n\n"
            f"path: {yolo_dir.resolve()}\n"
            f"train: images/train\n"
            f"val:   images/val\n\n"
            f"nc: {NUM_SUPERCLASSES}\n"
            f"names:\n{names_str}\n"
        )
        with open(yaml_path, "w") as f:
            f.write(content)
        print(f"[dataset] dataset.yaml ditulis ke {yaml_path}")
        return yaml_path

    # ------------------------------------------------------------------
    # PRIVATE — GRID CSV WRITER
    # ------------------------------------------------------------------

    def _write_grid_csv(self, grid_size: Tuple[int, int]) -> Path:
        """
        Menghasilkan tabel grid untuk seluruh dataset dan menyimpannya
        sebagai CSV di data/tabular/grid_results.csv.

        Kolom output:
          image_id, file_name, grid_row, grid_col,
          jumlah_sampah, kepadatan_pct,
          count_plastic_packaging, count_metal_glass,
          count_construction, count_hazardous, count_organic_mixed
        """
        tabular_dir = Path("data/tabular")
        tabular_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tabular_dir / "grid_results.csv"

        n_rows, n_cols = grid_size

        header = (
            "image_id,file_name,grid_row,grid_col,"
            "jumlah_sampah,kepadatan_pct,"
            "count_plastic_packaging,count_metal_glass,"
            "count_construction,count_hazardous,count_organic_mixed\n"
        )

        with open(csv_path, "w") as f:
            f.write(header)

            for image_id, info in self._img_info.items():
                features = self.compute_grid_features(
                    image_id, detections=None, grid_size=grid_size
                )
                # features shape: (n_rows, n_cols, 2+5)
                for r in range(n_rows):
                    for c in range(n_cols):
                        feat = features[r, c]
                        count    = int(feat[0])
                        density  = round(float(feat[1]), 4)
                        sc_cnts  = [int(feat[2 + s]) for s in range(NUM_SUPERCLASSES)]

                        # Hanya tulis sel yang memiliki setidaknya 1 deteksi
                        # (sel kosong bisa di-include dengan mengubah kondisi ini)
                        if count > 0:
                            row_str = (
                                f"{image_id},{info['file_name']},"
                                f"{r},{c},"
                                f"{count},{density},"
                                + ",".join(str(s) for s in sc_cnts)
                                + "\n"
                            )
                            f.write(row_str)

        print(f"[dataset] Grid CSV ditulis ke {csv_path}")
        return csv_path


# ---------------------------------------------------------------------------
# 3. FUNGSI UTILITAS STANDALONE
# ---------------------------------------------------------------------------

def verify_coco_json(coco_json: str | Path) -> None:
    """
    Verifikasi cepat struktur COCO JSON DroneWaste.
    Cetak statistik dasar dan daftar nama kategori asli —
    berguna untuk menyesuaikan SUPERCLASS_MAP.
    """
    with open(coco_json) as f:
        data = json.load(f)

    print("=" * 50)
    print("  COCO JSON Verification")
    print("=" * 50)
    print(f"  Gambar     : {len(data.get('images', []))}")
    print(f"  Anotasi    : {len(data.get('annotations', []))}")
    print(f"  Kategori   : {len(data.get('categories', []))}")
    print()
    print("  Nama kategori asli:")
    for cat in sorted(data.get("categories", []), key=lambda x: x["id"]):
        print(f"    [{cat['id']:3d}] {cat['name']}")
    print("=" * 50)


def coco_to_yolo_single(
    ann: dict,
    img_width: int,
    img_height: int,
    superclass_id: int,
) -> Optional[str]:
    """
    Konversi satu anotasi COCO ke satu baris YOLO.
    Mengembalikan None jika bbox tidak valid.
    """
    bx, by, bw, bh = ann["bbox"]
    if bw <= 0 or bh <= 0:
        return None
    cx = (bx + bw / 2) / img_width
    cy = (by + bh / 2) / img_height
    nw = bw / img_width
    nh = bh / img_height
    return f"{superclass_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


# ---------------------------------------------------------------------------
# 4. CONTOH PENGGUNAAN (jalankan sebagai script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AeroSweep — Dataset Preparation")
    parser.add_argument(
        "--coco_json",
        default="data/vision/raw/dronewaste_v1.0.json",
        help="Path ke file annotations.json DroneWaste",
    )
    parser.add_argument(
        "--images_dir",
        default="data/vision/raw/images",
        help="Direktori gambar raw",
    )
    parser.add_argument(
        "--output_dir",
        default="data/vision/processed",
        help="Direktori output processed",
    )
    parser.add_argument(
        "--grid_rows", type=int, default=8,
        help="Jumlah baris grid per gambar",
    )
    parser.add_argument(
        "--grid_cols", type=int, default=8,
        help="Jumlah kolom grid per gambar",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Jumlah fold untuk k-fold CV",
    )
    parser.add_argument(
        "--fold_idx", type=int, default=0,
        help="Fold yang dipakai sebagai validasi (0-based)",
    )
    parser.add_argument(
        "--verify_only", action="store_true",
        help="Hanya verifikasi COCO JSON tanpa memproses",
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_coco_json(args.coco_json)
    else:
        ds = DroneWasteDataset(
            coco_json   = args.coco_json,
            images_dir  = args.images_dir,
            output_dir  = args.output_dir,
        )

        # Cek distribusi kelas sebelum prepare
        print("\n[dataset] Distribusi superkategori (ground truth):")
        dist = ds.get_class_distribution()
        for name, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {name:<25} : {cnt:>5} instance")

        paths = ds.prepare(
            grid_size = (args.grid_rows, args.grid_cols),
            n_folds   = args.n_folds,
            fold_idx  = args.fold_idx,
        )

        print("\nOutput paths:")
        for k, v in paths.items():
            print(f"  {k:<12} : {v}")
# ==============================================================================
# AeroSweep — Modul Dataset & Preprocessing
# File: src/aerosweep/neuro/dataset.py
# Tanggung jawab:
#   1. Membaca citra UAV resolusi tinggi
#   2. Membaca anotasi COCO JSON (dronewaste_v1.0.json)
#   3. Overlapping sliding window → patch 512x512
#   4. Konversi anotasi poligon COCO → format segmentasi YOLO (.txt)
#   5. Auto-update configs/cv.yaml dengan nama kelas dari JSON
# Dipanggil oleh: scripts/01_preprocessing.py
# ==============================================================================

import json
import os
import shutil
import random
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from pycocotools import mask as coco_mask_utils
from pycocotools.coco import COCO

# ------------------------------------------------------------------------------
# Setup Logger
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AeroSweep.dataset")


# ==============================================================================
# KELAS UTAMA: DroneWasteDataset
# ==============================================================================

class DroneWasteDataset:
    """
    Kelas utama untuk memproses dataset DroneWaste.

    Alur kerja:
        1. Inisialisasi → load COCO JSON + validasi
        2. update_yaml()  → tulis nama kelas ke configs/cv.yaml
        3. run()          → sliding window + konversi anotasi → simpan ke disk
        4. split_dataset() → bagi hasil ke train/val

    Contoh penggunaan:
        ds = DroneWasteDataset(
            coco_json_path="data/vision/raw/dronewaste_v1.0.json",
            images_dir="data/vision/raw/images",
            output_dir="data/vision/processed",
            config_path="configs/cv.yaml",
        )
        ds.update_yaml()
        ds.run()
        ds.split_dataset()
    """

    def __init__(
        self,
        coco_json_path: str,
        images_dir: str,
        output_dir: str,
        config_path: str,
        patch_size: int = 512,
        overlap: float = 0.2,
        train_val_split: float = 0.8,
        min_visibility: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        coco_json_path   : Path ke file dronewaste_v1.0.json
        images_dir       : Path ke folder citra UAV asli
        output_dir       : Path ke folder output patch + label
        config_path      : Path ke configs/cv.yaml
        patch_size       : Ukuran patch sliding window (piksel), default 512
        overlap          : Fraksi overlap antar patch (0.0–1.0), default 0.2
        train_val_split  : Rasio data training (0.0–1.0), default 0.8
        min_visibility   : Minimum fraksi anotasi yang harus terlihat di patch
        random_seed      : Seed untuk reproduktibilitas
        """
        self.coco_json_path  = Path(coco_json_path)
        self.images_dir      = Path(images_dir)
        self.output_dir      = Path(output_dir)
        self.config_path     = Path(config_path)
        self.patch_size      = patch_size
        self.overlap         = overlap
        self.train_val_split = train_val_split
        self.min_visibility  = min_visibility
        self.random_seed     = random_seed

        # Seed global untuk reproduktibilitas
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Validasi path input
        self._validate_paths()

        # Load COCO JSON menggunakan pycocotools
        logger.info(f"Memuat COCO JSON dari: {self.coco_json_path}")
        self.coco = COCO(str(self.coco_json_path))

        # Bangun mapping: coco_category_id → yolo_class_id (0-indexed)
        # COCO ID tidak selalu mulai dari 0, YOLO wajib 0-indexed
        self.coco_id_to_yolo_id, self.class_names = self._build_class_mapping()

        logger.info(
            f"Dataset dimuat: {len(self.coco.imgs)} citra, "
            f"{len(self.coco.anns)} anotasi, "
            f"{len(self.class_names)} kelas"
        )

        # Siapkan direktori output
        self._prepare_output_dirs()

        # Hitung stride sliding window dari patch_size dan overlap
        # stride = jarak antar titik awal setiap patch
        self.stride = int(self.patch_size * (1.0 - self.overlap))

    # --------------------------------------------------------------------------
    # PRIVATE: Validasi & Inisialisasi
    # --------------------------------------------------------------------------

    def _validate_paths(self):
        """Validasi semua path wajib ada sebelum proses dimulai."""
        if not self.coco_json_path.exists():
            raise FileNotFoundError(
                f"File COCO JSON tidak ditemukan: {self.coco_json_path}\n"
                f"Pastikan dronewaste_v1.0.json ada di data/vision/raw/"
            )
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Direktori citra tidak ditemukan: {self.images_dir}\n"
                f"Pastikan citra UAV ada di data/vision/raw/images/"
            )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"File konfigurasi tidak ditemukan: {self.config_path}\n"
                f"Pastikan configs/cv.yaml sudah dibuat."
            )
        logger.info("Validasi path: OK")

    def _build_class_mapping(self) -> tuple[dict, dict]:
        """
        Bangun mapping dari COCO category_id ke YOLO class_id (0-indexed).

        COCO tidak menjamin ID kelas mulai dari 0 atau berurutan.
        YOLO mengharuskan class_id dimulai dari 0 dan berurutan.

        Returns
        -------
        coco_id_to_yolo_id : dict {coco_id: yolo_id}
        class_names        : dict {yolo_id: nama_kelas}
        """
        # Urutkan kategori berdasarkan COCO ID untuk konsistensi
        categories = sorted(
            self.coco.loadCats(self.coco.getCatIds()),
            key=lambda x: x["id"]
        )

        coco_id_to_yolo_id = {}
        class_names = {}

        for yolo_id, cat in enumerate(categories):
            coco_id_to_yolo_id[cat["id"]] = yolo_id
            class_names[yolo_id] = cat["name"]

        logger.info(f"Mapping kelas dibangun: {len(class_names)} kelas")
        for yolo_id, name in class_names.items():
            logger.debug(f"  YOLO ID {yolo_id}: {name}")

        return coco_id_to_yolo_id, class_names

    def _prepare_output_dirs(self):
        """Buat direktori output jika belum ada."""
        dirs = [
            self.output_dir / "train" / "images",
            self.output_dir / "train" / "labels",
            self.output_dir / "val"   / "images",
            self.output_dir / "val"   / "labels",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Direktori output disiapkan di: {self.output_dir}")

    # --------------------------------------------------------------------------
    # PUBLIC: Update cv.yaml dengan nama kelas dari JSON
    # --------------------------------------------------------------------------

    def update_yaml(self):
        """
        Baca configs/cv.yaml, update bagian 'nc' dan 'names' dengan
        data kelas yang dibaca langsung dari dronewaste_v1.0.json,
        lalu tulis kembali ke disk.

        Dipanggil SEBELUM run() agar cv.yaml siap saat training.
        """
        logger.info(f"Memperbarui cv.yaml di: {self.config_path}")

        # Baca YAML yang ada
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Update nc dan names
        config["nc"]    = len(self.class_names)
        config["names"] = self.class_names

        # Tulis kembali ke disk dengan komentar header tetap terjaga
        # Menggunakan allow_unicode=True agar nama kelas non-ASCII aman
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(
            f"cv.yaml diperbarui: nc={len(self.class_names)}, "
            f"names={list(self.class_names.values())}"
        )

    # --------------------------------------------------------------------------
    # PUBLIC: Entry point utama
    # --------------------------------------------------------------------------

    def run(self):
        """
        Jalankan seluruh pipeline preprocessing:
            - Iterasi setiap citra di COCO JSON
            - Terapkan sliding window → hasilkan patch 512x512
            - Konversi anotasi COCO → YOLO per patch
            - Simpan patch (.jpg) dan label (.txt) ke output_dir/all/

        Output sementara disimpan ke output_dir/all/ sebelum di-split.
        """
        logger.info("=" * 60)
        logger.info("Memulai preprocessing dataset DroneWaste")
        logger.info(f"  Patch size : {self.patch_size}x{self.patch_size}")
        logger.info(f"  Overlap    : {self.overlap * 100:.0f}%")
        logger.info(f"  Stride     : {self.stride} piksel")
        logger.info("=" * 60)

        # Direktori sementara untuk semua patch sebelum split
        all_images_dir = self.output_dir / "all" / "images"
        all_labels_dir = self.output_dir / "all" / "labels"
        all_images_dir.mkdir(parents=True, exist_ok=True)
        all_labels_dir.mkdir(parents=True, exist_ok=True)

        # Statistik proses
        total_patches    = 0
        total_with_annot = 0
        skipped_images   = 0

        # Iterasi semua citra di dataset
        img_ids = list(self.coco.imgs.keys())

        for img_id in tqdm(img_ids, desc="Memproses citra", unit="img"):
            img_info = self.coco.imgs[img_id]
            img_path = self.images_dir / img_info["file_name"]

            # Skip jika file citra tidak ditemukan di disk
            if not img_path.exists():
                logger.warning(f"Citra tidak ditemukan, dilewati: {img_path}")
                skipped_images += 1
                continue

            # Baca citra dengan OpenCV
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Gagal membaca citra: {img_path}")
                skipped_images += 1
                continue

            img_h, img_w = image.shape[:2]
            img_stem     = Path(img_info["file_name"]).stem

            # Muat semua anotasi untuk citra ini
            ann_ids  = self.coco.getAnnIds(imgIds=img_id)
            ann_list = self.coco.loadAnns(ann_ids)

            # Jalankan sliding window pada citra ini
            patches_count, annot_count = self._process_image_with_sliding_window(
                image        = image,
                img_w        = img_w,
                img_h        = img_h,
                img_stem     = img_stem,
                ann_list     = ann_list,
                out_img_dir  = all_images_dir,
                out_lbl_dir  = all_labels_dir,
            )

            total_patches    += patches_count
            total_with_annot += annot_count

        logger.info("=" * 60)
        logger.info(f"Preprocessing selesai!")
        logger.info(f"  Total patch dihasilkan : {total_patches}")
        logger.info(f"  Patch dengan anotasi   : {total_with_annot}")
        logger.info(f"  Patch kosong (no obj)  : {total_patches - total_with_annot}")
        logger.info(f"  Citra dilewati         : {skipped_images}")
        logger.info("=" * 60)

    # --------------------------------------------------------------------------
    # PRIVATE: Sliding Window pada satu citra
    # --------------------------------------------------------------------------

    def _process_image_with_sliding_window(
        self,
        image:       np.ndarray,
        img_w:       int,
        img_h:       int,
        img_stem:    str,
        ann_list:    list,
        out_img_dir: Path,
        out_lbl_dir: Path,
    ) -> tuple[int, int]:
        """
        Terapkan overlapping sliding window pada satu citra.

        Setiap patch diberi nama unik: {img_stem}_grid_{row}_{col}
        Contoh: UAV_site01_grid_0_0, UAV_site01_grid_0_1, dst.

        Parameters
        ----------
        image       : Array citra OpenCV (H x W x C)
        img_w/img_h : Dimensi citra asli
        img_stem    : Nama file citra tanpa ekstensi
        ann_list    : List anotasi COCO untuk citra ini
        out_img_dir : Direktori output patch gambar
        out_lbl_dir : Direktori output label YOLO

        Returns
        -------
        (jumlah_patch_total, jumlah_patch_berisi_anotasi)
        """
        patches_count    = 0
        annot_count      = 0
        row_idx          = 0

        # Geser jendela dari atas ke bawah
        y_start = 0
        while y_start < img_h:
            # Clamp koordinat agar tidak melebihi batas citra
            y_end = min(y_start + self.patch_size, img_h)
            # Geser ke kiri kalau patch di ujung agar ukurannya tetap 512
            if y_end - y_start < self.patch_size and y_start > 0:
                y_start = max(0, img_h - self.patch_size)
                y_end   = img_h

            col_idx = 0
            x_start = 0

            # Geser jendela dari kiri ke kanan
            while x_start < img_w:
                x_end = min(x_start + self.patch_size, img_w)
                if x_end - x_start < self.patch_size and x_start > 0:
                    x_start = max(0, img_w - self.patch_size)
                    x_end   = img_w

                # Potong patch dari citra
                patch = image[y_start:y_end, x_start:x_end]

                # Pad patch ke patch_size jika citra lebih kecil dari patch_size
                patch = self._pad_patch(patch)

                # Nama unik patch: stem_grid_row_col
                grid_name = f"{img_stem}_grid_{row_idx}_{col_idx}"

                # Konversi anotasi COCO ke YOLO untuk patch ini
                yolo_labels = self._convert_annotations_to_yolo(
                    ann_list = ann_list,
                    patch_x  = x_start,
                    patch_y  = y_start,
                    patch_w  = x_end - x_start,
                    patch_h  = y_end - y_start,
                    img_w    = img_w,
                    img_h    = img_h,
                )

                # Simpan patch gambar (.jpg)
                img_out_path = out_img_dir / f"{grid_name}.jpg"
                cv2.imwrite(str(img_out_path), patch)

                # Simpan label YOLO (.txt) — kosong jika tidak ada anotasi
                lbl_out_path = out_lbl_dir / f"{grid_name}.txt"
                with open(lbl_out_path, "w") as f:
                    f.write("\n".join(yolo_labels))

                patches_count += 1
                if yolo_labels:
                    annot_count += 1

                # Pindah ke patch berikutnya (horizontal)
                if x_end >= img_w:
                    break
                x_start += self.stride
                col_idx += 1

            # Pindah ke patch berikutnya (vertikal)
            if y_end >= img_h:
                break
            y_start += self.stride
            row_idx += 1

        return patches_count, annot_count

    def _pad_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Pad patch dengan warna hitam ke ukuran patch_size x patch_size.
        Diperlukan untuk patch di tepi citra yang lebih kecil dari patch_size.

        Parameters
        ----------
        patch : Array patch yang mungkin lebih kecil dari patch_size

        Returns
        -------
        Patch berukuran tepat patch_size x patch_size
        """
        h, w = patch.shape[:2]
        if h == self.patch_size and w == self.patch_size:
            return patch  # Tidak perlu padding

        padded = np.zeros(
            (self.patch_size, self.patch_size, 3),
            dtype=np.uint8
        )
        padded[:h, :w] = patch
        return padded

    # --------------------------------------------------------------------------
    # PRIVATE: Konversi Anotasi COCO → YOLO per Patch
    # --------------------------------------------------------------------------

    def _convert_annotations_to_yolo(
        self,
        ann_list: list,
        patch_x:  int,
        patch_y:  int,
        patch_w:  int,
        patch_h:  int,
        img_w:    int,
        img_h:    int,
    ) -> list[str]:
        """
        Konversi anotasi COCO (poligon) ke format segmentasi YOLO untuk
        satu patch spesifik.

        Format YOLO Segmentation per baris:
            class_id x1 y1 x2 y2 ... xn yn
        Semua koordinat dinormalisasi ke [0.0, 1.0] relatif terhadap patch.

        Parameters
        ----------
        ann_list : Semua anotasi COCO untuk citra ini
        patch_x  : Koordinat x kiri-atas patch di citra asli
        patch_y  : Koordinat y kiri-atas patch di citra asli
        patch_w  : Lebar patch di citra asli (sebelum padding)
        patch_h  : Tinggi patch di citra asli (sebelum padding)

        Returns
        -------
        List string YOLO label, satu string per objek.
        List kosong jika tidak ada objek yang terlihat di patch.
        """
        yolo_labels = []

        for ann in ann_list:
            # Ambil YOLO class ID dari mapping
            yolo_class_id = self.coco_id_to_yolo_id.get(ann["category_id"])
            if yolo_class_id is None:
                continue  # Kelas tidak dikenal, lewati

            # Ambil bounding box COCO: [x_min, y_min, width, height]
            bbox = ann.get("bbox", [])
            if not bbox:
                continue

            bx, by, bw, bh = bbox

            # Cek apakah bounding box beririsan dengan patch ini
            # Irisan antara bbox dan patch
            inter_x1 = max(bx, patch_x)
            inter_y1 = max(by, patch_y)
            inter_x2 = min(bx + bw, patch_x + patch_w)
            inter_y2 = min(by + bh, patch_y + patch_h)

            # Tidak ada irisan → lewati anotasi ini
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue

            # Hitung rasio visibilitas bbox dalam patch
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            bbox_area  = bw * bh
            if bbox_area <= 0:
                continue

            visibility = inter_area / bbox_area
            if visibility < self.min_visibility:
                continue  # Objek terlalu sedikit terlihat → lewati

            # Proses segmentasi poligon
            segmentation = ann.get("segmentation", [])

            if not segmentation:
                continue

            # Tangani format RLE (Run-Length Encoding) dari COCO
            if isinstance(segmentation, dict):
                # Format RLE → decode ke binary mask → ambil kontur
                binary_mask = coco_mask_utils.decode(segmentation)
                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue
                # Ambil kontur terbesar
                largest = max(contours, key=cv2.contourArea)
                poly    = largest.flatten().tolist()
                polygons = [poly]

            else:
                # Format poligon biasa: list of list koordinat
                polygons = segmentation

            # Proses setiap poligon dalam anotasi
            for poly in polygons:
                if len(poly) < 6:
                    # Poligon valid minimal 3 titik (6 koordinat)
                    continue

                # Konversi koordinat poligon ke ruang patch + normalisasi
                clipped_poly = self._clip_polygon_to_patch(
                    poly    = poly,
                    patch_x = patch_x,
                    patch_y = patch_y,
                    patch_w = self.patch_size,  # Gunakan patch_size untuk normalisasi
                    patch_h = self.patch_size,
                )

                if clipped_poly is None:
                    continue

                # Format: "class_id x1 y1 x2 y2 ... xn yn"
                coords_str = " ".join(
                    f"{v:.6f}" for v in clipped_poly
                )
                yolo_labels.append(f"{yolo_class_id} {coords_str}")

        return yolo_labels

    def _clip_polygon_to_patch(
        self,
        poly:    list,
        patch_x: int,
        patch_y: int,
        patch_w: int,
        patch_h: int,
    ) -> Optional[list]:
        """
        Clip dan normalisasi koordinat poligon ke ruang koordinat patch.

        Langkah:
        1. Geser koordinat relatif terhadap sudut kiri-atas patch
        2. Clip ke dalam batas [0, patch_size]
        3. Normalisasi ke [0.0, 1.0]

        Parameters
        ----------
        poly    : List koordinat poligon [x1, y1, x2, y2, ...]
        patch_x : Offset x patch di citra asli
        patch_y : Offset y patch di citra asli
        patch_w : Lebar patch (untuk normalisasi)
        patch_h : Tinggi patch (untuk normalisasi)

        Returns
        -------
        List koordinat ternormalisasi, atau None jika poligon tidak valid.
        """
        # Pisahkan x dan y
        xs = poly[0::2]
        ys = poly[1::2]

        # Geser ke ruang patch dan clip ke [0, patch_size]
        xs_local = [
            max(0.0, min(float(x) - patch_x, patch_w)) for x in xs
        ]
        ys_local = [
            max(0.0, min(float(y) - patch_y, patch_h)) for y in ys
        ]

        # Normalisasi ke [0.0, 1.0]
        xs_norm = [x / patch_w for x in xs_local]
        ys_norm = [y / patch_h for y in ys_local]

        # Susun kembali: [x1, y1, x2, y2, ...]
        coords_norm = []
        for x, y in zip(xs_norm, ys_norm):
            coords_norm.extend([x, y])

        # Validasi: minimal 3 titik unik (6 nilai)
        if len(coords_norm) < 6:
            return None

        # Validasi: poligon tidak boleh degenerasi (semua titik sama)
        unique_x = set(xs_norm)
        unique_y = set(ys_norm)
        if len(unique_x) < 2 or len(unique_y) < 2:
            return None

        return coords_norm

    # --------------------------------------------------------------------------
    # PUBLIC: Split dataset → train / val
    # --------------------------------------------------------------------------

    def split_dataset(self):
        """
        Bagi semua patch dari output_dir/all/ ke:
            - output_dir/train/images/ dan output_dir/train/labels/
            - output_dir/val/images/   dan output_dir/val/labels/

        Rasio split diambil dari self.train_val_split (default 0.8).
        Split dilakukan berdasarkan nama file gambar (bukan per citra asli)
        secara acak dengan seed yang sudah diset.

        Setelah split selesai, folder output_dir/all/ dihapus untuk
        menghemat storage.
        """
        logger.info("Memulai split dataset train/val...")

        all_images_dir = self.output_dir / "all" / "images"
        all_labels_dir = self.output_dir / "all" / "labels"

        # Kumpulkan semua file gambar patch
        all_image_files = sorted(all_images_dir.glob("*.jpg"))

        if not all_image_files:
            logger.error(
                "Tidak ada patch ditemukan di output_dir/all/images/. "
                "Jalankan run() terlebih dahulu."
            )
            return

        # Acak urutan file dengan seed
        random.shuffle(all_image_files)

        # Hitung jumlah data train
        n_total = len(all_image_files)
        n_train = int(n_total * self.train_val_split)

        train_files = all_image_files[:n_train]
        val_files   = all_image_files[n_train:]

        logger.info(
            f"Split: {n_total} patch total → "
            f"{len(train_files)} train / {len(val_files)} val"
        )

        # Salin file ke direktori train/val
        for split_name, file_list in [("train", train_files), ("val", val_files)]:
            img_dst = self.output_dir / split_name / "images"
            lbl_dst = self.output_dir / split_name / "labels"

            for img_path in tqdm(
                file_list,
                desc=f"Menyalin ke {split_name}",
                unit="file"
            ):
                # Salin gambar
                shutil.copy2(img_path, img_dst / img_path.name)

                # Salin label (nama sama, ekstensi .txt)
                lbl_path = all_labels_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
                else:
                    # Buat file label kosong jika tidak ada
                    # (patch tanpa objek tetap diperlukan untuk training)
                    (lbl_dst / (img_path.stem + ".txt")).touch()

        # Hapus folder sementara 'all' setelah split selesai
        logger.info("Menghapus folder sementara 'all'...")
        shutil.rmtree(self.output_dir / "all")

        logger.info("Split dataset selesai!")
        logger.info(
            f"  Train → {self.output_dir / 'train'}"
        )
        logger.info(
            f"  Val   → {self.output_dir / 'val'}"
        )

    # --------------------------------------------------------------------------
    # PUBLIC: Utilitas — ringkasan statistik dataset
    # --------------------------------------------------------------------------

    def print_dataset_summary(self):
        """
        Tampilkan ringkasan statistik dataset:
        - Jumlah citra, anotasi, kelas
        - Distribusi anotasi per kelas
        """
        logger.info("=" * 60)
        logger.info("RINGKASAN DATASET DroneWaste")
        logger.info("=" * 60)
        logger.info(f"File JSON : {self.coco_json_path}")
        logger.info(f"Total citra    : {len(self.coco.imgs)}")
        logger.info(f"Total anotasi  : {len(self.coco.anns)}")
        logger.info(f"Total kelas    : {len(self.class_names)}")
        logger.info("")
        logger.info("Distribusi anotasi per kelas:")

        # Hitung jumlah anotasi per kelas
        class_counts = {yolo_id: 0 for yolo_id in self.class_names}
        for ann in self.coco.anns.values():
            yolo_id = self.coco_id_to_yolo_id.get(ann["category_id"])
            if yolo_id is not None:
                class_counts[yolo_id] += 1

        for yolo_id, name in self.class_names.items():
            count = class_counts[yolo_id]
            bar   = "█" * min(count, 50)
            logger.info(f"  [{yolo_id:2d}] {name:<25} {count:4d} {bar}")

        logger.info("=" * 60)
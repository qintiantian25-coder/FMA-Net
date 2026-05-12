"""
add_new_dataset.py

Usage:
  python add_new_dataset.py --src ROOT_DIR [--dataset-root DATA_ROOT] [--dry-run] [--apply]

This script inspects subfolders under ROOT_DIR (each containing sequential frames),
sorts them by image count, selects the top-4 for training, the next 1 for validation,
and the remaining folders for testing. It copies the images into the project's `data`
structure (renaming them to 1.png, 2.png, ...) and runs `fangzhen_adaptive.run_consistent_simulation`
to synthesize blurred images and masks (adaptively scaling blind pixel parameters).

Default behavior is a dry-run that prints the planned actions. Use --apply to perform
the file copy and simulation steps.
"""
import os
import sys
import argparse
import csv
import shutil
import tempfile
import re

import cv2
import numpy as np


def natural_sort_key(s):
    """自然排序key函数，用于数字排序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def list_image_count(folder):
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    try:
        files = [f for f in os.listdir(folder) if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]
        return len(files)
    except Exception:
        return 0


def find_next_indexed_subdir(base_dir):
    # Find next available 3-digit numeric subdir name (e.g. '004')
    existing = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    nums = []
    for name in existing:
        try:
            nums.append(int(name))
        except Exception:
            continue
    n = 1
    if nums:
        n = max(nums) + 1
    return f"{n:03d}"


def pad_image_center(img, target_w, target_h, fill_value=0):
    h, w = img.shape[:2]
    if w > target_w or h > target_h:
        raise ValueError(f"Image {w}x{h} is larger than target canvas {target_w}x{target_h}")
    left = (target_w - w) // 2
    top = (target_h - h) // 2
    if img.ndim == 2:
        canvas = np.full((target_h, target_w), fill_value, dtype=img.dtype)
        canvas[top:top + h, left:left + w] = img
    else:
        canvas = np.full((target_h, target_w, img.shape[2]), fill_value, dtype=img.dtype)
        canvas[top:top + h, left:left + w, :] = img
    return canvas, left, top


def crop_image(img, left, top, crop_w, crop_h):
    return img[top:top + crop_h, left:left + crop_w]


def read_image_size(folder):
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(exts):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img.shape[1], img.shape[0]
    return None, None


def pad_folder_to_canvas(src_folder, dst_folder, target_w, target_h):
    os.makedirs(dst_folder, exist_ok=True)
    meta = None
    # 获取所有图像文件并自然排序
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort(key=natural_sort_key)
    for idx, fname in enumerate(image_files, start=1):
        src_path = os.path.join(src_folder, fname)
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        padded, left, top = pad_image_center(img, target_w, target_h, fill_value=0)
        cv2.imwrite(os.path.join(dst_folder, f"{idx}.png"), padded)
        if meta is None:
            meta = {'orig_w': img.shape[1], 'orig_h': img.shape[0], 'left': left, 'top': top}
    return meta


def crop_simulation_outputs_to_original(blur_dir, mask_dir, left, top, orig_w, orig_h):
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    for fname in sorted(os.listdir(blur_dir)):
        if not fname.lower().endswith(exts):
            continue
        p = os.path.join(blur_dir, fname)
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        cv2.imwrite(p, crop_image(img, left, top, orig_w, orig_h))

    mask_png = os.path.join(mask_dir, 'blind_pixel_mask.png')
    if os.path.exists(mask_png):
        img = cv2.imread(mask_png, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            cv2.imwrite(mask_png, crop_image(img, left, top, orig_w, orig_h))

    blind_csv = os.path.join(mask_dir, 'blind_pixel_coords.csv')
    if os.path.exists(blind_csv):
        rows = []
        with open(blind_csv, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or ['x', 'y', 'original_gray', 'simulated_gray']
            for row in reader:
                try:
                    x = int(float(row['x'])) - left
                    y = int(float(row['y'])) - top
                except Exception:
                    continue
                if 0 <= x < orig_w and 0 <= y < orig_h:
                    row['x'] = str(x)
                    row['y'] = str(y)
                    rows.append(row)
        with open(blind_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    flash_csv = os.path.join(mask_dir, 'flash_pixel_coords.csv')
    if os.path.exists(flash_csv):
        rows = []
        with open(flash_csv, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or ['frame_name', 'x', 'y', 'original_gray', 'simulated_gray', 'mode']
            for row in reader:
                try:
                    x = int(float(row['x'])) - left
                    y = int(float(row['y'])) - top
                except Exception:
                    continue
                if 0 <= x < orig_w and 0 <= y < orig_h:
                    row['x'] = str(x)
                    row['y'] = str(y)
                    rows.append(row)
        with open(flash_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def prepare_dataset_assignments(src_root, top_train=4, val_num=1):
    # scan folders
    items = []
    for name in sorted(os.listdir(src_root)):
        p = os.path.join(src_root, name)
        if os.path.isdir(p):
            cnt = list_image_count(p)
            items.append((name, p, cnt))
    # sort by count desc
    items.sort(key=lambda x: x[2], reverse=True)
    if len(items) < 9:
        print(f"Warning: found only {len(items)} subfolders under {src_root}; expected 9 or more.")
    train = items[:top_train]
    val = items[top_train:top_train + val_num]
    test = items[top_train + val_num:]      # 剩余所有文件夹作为测试集
    remainder = []                          # 没有未分配
    return train, val, test, remainder, items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Root folder that contains the folders of frames')
    parser.add_argument('--dataset-root', default=os.path.join(os.path.dirname(__file__), 'data'), help='Project data root (default: ./data)')
    parser.add_argument('--top-train', type=int, default=4, help='Number of folders with most images to assign to training')
    parser.add_argument('--val-num', type=int, default=1, help='Number of folders after training to assign to validation')
    parser.add_argument('--pad-to-w', type=int, default=0, help='If >0, pad smaller source frames to this width before simulation (not needed with adaptive fangzhen)')
    parser.add_argument('--pad-to-h', type=int, default=0, help='If >0, pad smaller source frames to this height before simulation')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Do not copy or run simulation, just print plan (default)')
    parser.add_argument('--apply', action='store_true', help='Perform copy and run simulation. Overrides --dry-run')
    args = parser.parse_args()

    dry_run = not args.apply

    src_root = os.path.abspath(args.src)
    data_root = os.path.abspath(args.dataset_root)

    if not os.path.isdir(src_root):
        print(f"src root not found: {src_root}")
        sys.exit(1)

    train, val, test, remainder, all_items = prepare_dataset_assignments(src_root, top_train=args.top_train, val_num=args.val_num)

    print("Found subfolders (sorted by image count desc):")
    for name, p, cnt in all_items:
        print(f"  {name}: {cnt} images -> {p}")

    print("\nPlanned assignment:")
    print("  TRAIN:")
    for name, p, cnt in train:
        print(f"    {name} ({cnt})")
    print("  VAL:")
    for name, p, cnt in val:
        print(f"    {name} ({cnt})")
    print("  TEST:")
    for name, p, cnt in test:
        print(f"    {name} ({cnt})")
    if remainder:
        print("  REMAINING (not assigned):")
        for name, p, cnt in remainder:
            print(f"    {name} ({cnt})")

    if dry_run:
        print('\nDry run mode - no files will be copied or modified. Use --apply to execute.')
        return

    # 导入自适应仿真模块
    from fangzhen_adaptive import run_consistent_simulation

    # map roles to dataset subroots
    role_map = [
        ('train', 'train_sharp', 'train_blur', 'train_mask', train),
        ('val', 'val_sharp', 'val_blur', 'val_mask', val),
        ('test', 'test_sharp', 'test_blur', 'test_mask', test),
    ]

    for role, sharp_dirname, blur_dirname, mask_dirname, group in role_map:
        for name, p, cnt in group:
            # determine next available subdir (3-digit) under sharp_dir
            sharp_root = os.path.join(data_root, sharp_dirname)
            blur_root = os.path.join(data_root, blur_dirname)
            mask_root = os.path.join(data_root, mask_dirname)
            os.makedirs(sharp_root, exist_ok=True)
            os.makedirs(blur_root, exist_ok=True)
            os.makedirs(mask_root, exist_ok=True)

            new_subdir = find_next_indexed_subdir(sharp_root)
            dest_sharp = os.path.join(sharp_root, new_subdir)
            dest_blur = os.path.join(blur_root, new_subdir)
            dest_mask = os.path.join(mask_root, new_subdir)

            os.makedirs(dest_sharp, exist_ok=True)

            # 自动检测源图像尺寸并在必要时 pad 到 640x512 后仿真
            TARGET_W, TARGET_H = 640, 512

            orig_w, orig_h = read_image_size(p)
            if orig_w is None:
                print(f"[WARN] Could not read image size from {p}. Copying as-is and running simulation on copied images.")
                # 直接复制原始文件到 dest_sharp
                print(f"Copying {p} -> {dest_sharp}")
                image_files = [f for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                image_files.sort(key=natural_sort_key)
                for idx, fname in enumerate(image_files, start=1):
                    new_name = f"{idx}.png"
                    shutil.copy2(os.path.join(p, fname), os.path.join(dest_sharp, new_name))
                print(f"Running simulation for {dest_sharp} -> blur:{dest_blur} mask:{dest_mask}")
                run_consistent_simulation(dest_sharp, dest_blur, dest_mask)
            elif orig_w == TARGET_W and orig_h == TARGET_H:
                # 已经是目标尺寸，直接复制并仿真
                print(f"Source images already {TARGET_W}x{TARGET_H} - copying {p} -> {dest_sharp}")
                image_files = [f for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                image_files.sort(key=natural_sort_key)
                for idx, fname in enumerate(image_files, start=1):
                    new_name = f"{idx}.png"
                    shutil.copy2(os.path.join(p, fname), os.path.join(dest_sharp, new_name))
                print(f"Running simulation for {dest_sharp} -> blur:{dest_blur} mask:{dest_mask}")
                run_consistent_simulation(dest_sharp, dest_blur, dest_mask)
            else:
                # 需要 pad 到目标尺寸
                print(f"Source images {orig_w}x{orig_h} -> need pad to {TARGET_W}x{TARGET_H}. Padding and copying.")
                temp_root = tempfile.mkdtemp(prefix='pad_src_')
                try:
                    temp_src = os.path.join(temp_root, 'src')
                    os.makedirs(temp_src, exist_ok=True)
                    meta = pad_folder_to_canvas(p, temp_src, TARGET_W, TARGET_H)
                    if meta is None:
                        print(f"[WARN] No valid images found in {p}; skipping this folder.")
                        continue
                    # copy padded sharp images into dest_sharp
                    print(f"Copying padded images -> {dest_sharp}")
                    padded_files = [f for f in os.listdir(temp_src) if f.lower().endswith('.png')]
                    padded_files.sort(key=natural_sort_key)
                    for fname in padded_files:
                        shutil.copy2(os.path.join(temp_src, fname), os.path.join(dest_sharp, fname))

                    # run adaptive simulation on the padded dest_sharp so parameters scale appropriately
                    print(f"Running simulation for padded images {dest_sharp} -> blur:{dest_blur} mask:{dest_mask}")
                    run_consistent_simulation(dest_sharp, dest_blur, dest_mask)
                finally:
                    shutil.rmtree(temp_root, ignore_errors=True)

    print('Dataset augmentation completed.')


if __name__ == '__main__':
    main()
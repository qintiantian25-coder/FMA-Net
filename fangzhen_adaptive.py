import cv2
import numpy as np
import random
import os
import glob
import csv
import re


# ------------------------- 辅助函数 -------------------------
def get_random_dark_color():
    """通用暗色采样 (0-100灰度)"""
    return random.randint(0, 10) if random.random() < 0.3 else random.randint(10, 100)


def get_mostly_black_color():
    """专为暗主导块采样的深度黑色 (0-15灰度，确保视觉上极明显)"""
    return random.randint(0, 5) if random.random() < 0.9 else random.randint(5, 15)


def natural_sort_key(s):
    """自然排序key函数"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# ------------------------- 盲元生成函数（参数化）-------------------------
def gen_mostly_black_tight_blob(w, h, target_pts, dominant_type, forbidden, margin_block):
    """
    生成高度粘合的黑主导或白主导块
    margin_block: 选址边界（避免太靠近边缘）
    """
    pts_dict = {}
    cx, cy = 0, 0
    # 选址避让
    for _ in range(50):
        tx, ty = random.randint(margin_block, w - margin_block), random.randint(margin_block, h - margin_block)
        if not any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            cx, cy = tx, ty
            break
    else:
        cx, cy = random.randint(margin_block, w - margin_block), random.randint(margin_block, h - margin_block)

    current_pts = [(cx, cy)]

    if dominant_type == 'white':
        pts_dict[(cx, cy)] = 255
    else:
        pts_dict[(cx, cy)] = get_mostly_black_color()

    while len(pts_dict) < target_pts:
        base_x, base_y = random.choice(current_pts)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        dx, dy = random.choice(dirs)
        nx, ny = base_x + dx, base_y + dy
        if 5 <= nx < w - 5 and 5 <= ny < h - 5 and (nx, ny) not in pts_dict:
            if dominant_type == 'white':
                c = 255 if random.random() < 0.8 else get_random_dark_color()
            else:
                if random.random() < 0.90:
                    c = get_mostly_black_color()
                else:
                    c = 255
            pts_dict[(nx, ny)] = c
            current_pts.append((nx, ny))

    pts = [(x, y, c) for (x, y), c in pts_dict.items()]
    return pts, (cx, cy)


def gen_extra_long_lines(w, h, line_start_margin, line_len_min, line_len_max):
    """超长破碎线，参数：起点边界，长度范围"""
    pts = []
    for _ in range(random.randint(2, 3)):
        cx, cy = random.randint(line_start_margin, w // 2), random.randint(line_start_margin, h // 2)
        dx, dy = random.choice([(1, 0), (0, 1), (1, 1), (1, -1), (2, 1)])
        length = random.randint(line_len_min, line_len_max)
        for _ in range(length):
            if 5 <= cx < w - 5 and 5 <= cy < h - 5:
                c = get_random_dark_color() if random.random() < 0.4 else 255
                pts.append((cx, cy, c))
            if random.random() < 0.05:
                sx, sy = cx + random.randint(-1, 1), cy + random.randint(-1, 1)
                if 5 <= sx < w - 5 and 5 <= sy < h - 5:
                    pts.append((sx, sy, get_random_dark_color()))
            cx, cy = cx + dx, cy + dy
            if not (0 <= cx < w and 0 <= cy < h):
                break
    return pts


def grow_compact_blob(w, h, target_w, target_d, forbidden, margin_block):
    """大型/中型块：白色核心+暗色污染边缘"""
    pts = []
    cx, cy = 0, 0
    for _ in range(50):
        tx, ty = random.randint(margin_block, w - margin_block), random.randint(margin_block, h - margin_block)
        if not any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            cx, cy = tx, ty
            break
    else:
        cx, cy = random.randint(margin_block, w - margin_block), random.randint(margin_block, h - margin_block)

    w_pts = set([(cx, cy)])
    bnd = [(cx, cy)]
    while len(w_pts) < target_w and bnd:
        px, py = random.choice(bnd)
        dirs = [(1, 0)] * 4 + [(-1, 0)] * 4 + [(0, 1)] * 4 + [(0, -1)] * 4 + [(1, 1)]
        dx, dy = random.choice(dirs)
        nx, ny = px + dx, py + dy
        if 5 <= nx < w - 5 and 5 <= ny < h - 5 and (nx, ny) not in w_pts:
            w_pts.add((nx, ny))
            bnd.append((nx, ny))
        if len(bnd) > target_w // 2:
            bnd.pop(0)

    d_pts = set()
    bnd_d = list(w_pts)
    while len(d_pts) < target_d and bnd_d:
        px, py = random.choice(bnd_d)
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)])
        nx, ny = px + dx, py + dy
        if 5 <= nx < w - 5 and 5 <= ny < h - 5 and (nx, ny) not in w_pts and (nx, ny) not in d_pts:
            d_pts.add((nx, ny))
            bnd_d.append((nx, ny))

    for x, y in d_pts:
        pts.append((x, y, get_random_dark_color()))
    for x, y in w_pts:
        c = get_random_dark_color() if random.random() < 0.05 else 255
        pts.append((x, y, c))
    return pts, (cx, cy)


def gen_cross_invalid_pixels(w, h, num_crosses, forbidden, occupied, margin_cross, retry=80):
    """
    十字形盲元，margin_cross 为坐标边界（通常较小）
    """
    pts = []
    centers = []
    arms = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

    for _ in range(num_crosses):
        cx, cy = None, None
        for _ in range(retry):
            tx = random.randint(margin_cross, w - 1 - margin_cross)
            ty = random.randint(margin_cross, h - 1 - margin_cross)
            if not (1 <= tx < w - 1 and 1 <= ty < h - 1):
                continue
            if any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
                continue
            cross_coords = [(tx + dx, ty + dy) for dx, dy in arms]
            if any((x, y) in occupied for x, y in cross_coords):
                continue
            cx, cy = tx, ty
            break
        if cx is None:
            continue
        pts.append((cx, cy, random.randint(0, 5)))
        occupied.add((cx, cy))
        for dx, dy in arms[1:]:
            x, y = cx + dx, cy + dy
            pts.append((x, y, random.randint(100, 150)))
            occupied.add((x, y))
        centers.append((cx, cy))
    return pts, centers, occupied


def gen_flash_anchor_positions(w, h, num_anchors, forbidden, occupied, margin_flash, min_dist, block_size=1, retry=120):
    """生成闪元候选位置（单像素）"""
    anchors = []
    tries = 0
    while len(anchors) < num_anchors and tries < retry * max(1, num_anchors):
        tries += 1
        tx = random.randint(margin_flash, w - 1 - margin_flash)
        ty = random.randint(margin_flash, h - 1 - margin_flash)
        block_coords = [(tx + dx, ty + dy) for dx in range(block_size) for dy in range(block_size)]
        if any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            continue
        if any((x, y) in occupied for x, y in block_coords):
            continue
        if any(abs(tx - ax) < min_dist and abs(ty - ay) < min_dist for ax, ay in anchors):
            continue
        anchors.append((tx, ty))
        for xy in block_coords:
            occupied.add(xy)
    return anchors, occupied


def sample_flash_pixels_for_frame(img, flash_anchors, min_active, max_active, block_size=1):
    """每帧随机激活闪元"""
    if not flash_anchors:
        return [], []
    active_num = min(len(flash_anchors), random.randint(min_active, max_active))
    active_positions = random.sample(flash_anchors, active_num)
    pts = []
    records = []
    for x, y in active_positions:
        if not (0 <= y < img.shape[0] and 0 <= x < img.shape[1]):
            continue
        orig_gray = int(img[y, x][0])
        if random.random() < 0.5:
            delta = random.randint(40, 120)
            c = random.randint(0, 30) if random.random() < 0.7 else max(0, orig_gray - delta)
            mode = "darken"
        else:
            delta = random.randint(40, 120)
            c = random.randint(225, 255) if random.random() < 0.7 else min(255, orig_gray + delta)
            mode = "brighten"
        for dx in range(block_size):
            for dy in range(block_size):
                px, py = x + dx, y + dy
                if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                    pts.append((px, py, c))
        records.append([x, y, orig_gray, c, mode])
    return pts, records


# ------------------------- 主仿真函数（自适应参数）-------------------------
def run_consistent_simulation(src_dir, dst_dir, mask_dir):
    """
    根据输入图像的实际尺寸自动调整盲元密度和大小。
    基准参考尺寸：640x512（原数据集）
    """
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 支持多种图片格式
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP')
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(src_dir, ext)))
    img_paths = sorted(set(img_paths), key=natural_sort_key)
    if not img_paths:
        print(f"警告：{src_dir} 中没有支持的图片文件")
        return

    first_img = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
    if first_img is None:
        print(f"无法读取第一张图像：{img_paths[0]}")
        return

    h, w = first_img.shape[:2]

    # ---------- 自适应参数计算 ----------
    # 基准尺寸（原设计对应的尺寸）
    ref_w, ref_h = 640, 512
    ref_short = min(ref_w, ref_h)           # 512
    ref_area = ref_w * ref_h                 # 327680

    short = min(w, h)
    area = w * h

    size_ratio = short / ref_short           # 例如 256/512 = 0.5
    area_ratio = area / ref_area             # 例如 (320*256)/(640*512)=0.25

    # 散布点数量
    base_cnt1 = max(150, int(800 * area_ratio))
    base_cnt2 = max(400, int(2000 * area_ratio))

    # 块状盲元尺寸
    tight_target = max(8, int(32 * area_ratio))
    large_w = max(15, int(130 * area_ratio))
    large_d = max(15, int(150 * area_ratio))
    medium_w = max(10, int(45 * area_ratio))
    medium_d = max(10, int(60 * area_ratio))

    # 选址边界与避让半径（基于短边比例）
    margin_block = max(20, int(120 * size_ratio))       # 块状盲元选址边界
    line_start_margin = max(10, int(30 * size_ratio))   # 超长线起点边界
    line_len_min = max(40, int(120 * size_ratio))
    line_len_max = max(60, int(180 * size_ratio))

    # 十字形盲元数量及边界
    cross_num = max(8, int(40 * area_ratio))
    margin_cross = max(4, int(8 * size_ratio))          # 十字形边界

    # 闪元数量
    flash_num = max(3, int(10 * area_ratio))
    margin_flash = max(4, int(8 * size_ratio))

    # 禁忌区域半径缩放
    radius_scale = size_ratio
    init_forbidden_radius = int(180 * radius_scale)
    blob_forbidden_radius = int(80 * radius_scale)
    large_forbidden_radius = int(100 * radius_scale)
    cross_forbidden_radius = int(20 * radius_scale)
    flash_forbidden_radius = int(12 * radius_scale)

    # ---------- 生成静态盲元 ----------
    all_static_blind_params = []
    forbidden = [(w // 4, h // 4, init_forbidden_radius)]

    # 1. 基础散布点
    for (wl, hl, cnt) in [(w, h, base_cnt1), (w // 2, h // 2, base_cnt2)]:
        for _ in range(cnt):
            tx, ty = random.randint(0, wl - 2), random.randint(0, hl - 2)
            all_static_blind_params.append((tx, ty, get_random_dark_color()))
            all_static_blind_params.append((tx + random.choice([0, 1]), ty + random.choice([0, 1]), 255))

    # 2. 超长破碎线
    all_static_blind_params += gen_extra_long_lines(w, h, line_start_margin, line_len_min, line_len_max)

    # 3. 粘合不规则块（两白两黑）
    for _ in range(2):
        p, center = gen_mostly_black_tight_blob(w, h, tight_target, 'white', forbidden, margin_block)
        all_static_blind_params += p
        forbidden.append((center[0], center[1], blob_forbidden_radius))
    for _ in range(2):
        p, center = gen_mostly_black_tight_blob(w, h, tight_target, 'dark', forbidden, margin_block)
        all_static_blind_params += p
        forbidden.append((center[0], center[1], blob_forbidden_radius))

    # 4. 大型/中型块
    configs = [(large_w, large_d, 2), (medium_w, medium_d, 2)]
    for wt, dt, n in configs:
        for _ in range(n):
            p, center = grow_compact_blob(w, h, wt, dt, forbidden, margin_block)
            all_static_blind_params += p
            forbidden.append((center[0], center[1], large_forbidden_radius))

    # 5. 十字形盲元
    occupied = {(x, y) for x, y, _ in all_static_blind_params}
    cross_pts, cross_centers, occupied = gen_cross_invalid_pixels(
        w, h, cross_num, forbidden, occupied, margin_cross, retry=80
    )
    all_static_blind_params += cross_pts
    for cx, cy in cross_centers:
        forbidden.append((cx, cy, cross_forbidden_radius))

    # 6. 闪元候选点
    flash_anchors, occupied = gen_flash_anchor_positions(
        w, h, flash_num, forbidden, occupied,
        margin_flash, min_dist=24, block_size=1, retry=120
    )
    for ax, ay in flash_anchors:
        forbidden.append((ax, ay, flash_forbidden_radius))

    # ---------- 渲染保存 ----------
    mask_img = np.zeros((h, w), dtype=np.uint8)
    csv_records = []
    flash_records = []

    for idx, p in enumerate(img_paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法读取图像：{p}，跳过")
            continue
        out_img = img.copy()

        # 静态盲元
        for x, y, c in all_static_blind_params:
            if 0 <= y < h and 0 <= x < w:
                out_img[y, x] = [c, c, c]
                if idx == 0:
                    mask_img[y, x] = 255
                    orig_gray = img[y, x][0]
                    csv_records.append([x, y, orig_gray, c])

        # 闪元
        min_active = max(1, flash_num // 2)
        max_active = max(min_active + 1, flash_num)
        flash_pts, frame_flash_records = sample_flash_pixels_for_frame(
            img, flash_anchors, min_active=min_active, max_active=max_active, block_size=1
        )
        for x, y, c in flash_pts:
            if 0 <= y < h and 0 <= x < w:
                out_img[y, x] = [c, c, c]
        if frame_flash_records:
            frame_name = os.path.basename(p)
            for rec in frame_flash_records:
                flash_records.append([frame_name] + rec)

        out_filename = os.path.splitext(os.path.basename(p))[0] + '.png'
        cv2.imwrite(os.path.join(dst_dir, out_filename), out_img)

    # 保存掩码和静态盲元CSV
    cv2.imwrite(os.path.join(mask_dir, "blind_pixel_mask.png"), mask_img)
    with open(os.path.join(mask_dir, "blind_pixel_coords.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'original_gray', 'simulated_gray'])
        writer.writerows(csv_records)

    # 保存闪元记录
    if flash_records:
        with open(os.path.join(mask_dir, "flash_pixel_coords.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_name', 'x', 'y', 'original_gray', 'simulated_gray', 'mode'])
            writer.writerows(flash_records)

    print(f"仿真完成：{len(img_paths)} 张图像已处理，盲元掩码位于 {mask_dir}；闪元候选点数：{len(flash_anchors)}")


if __name__ == "__main__":
    # 示例（可根据需要修改）
    DATA_BASE = r"D:\project\FGAF-Net\data"
    run_consistent_simulation(
        src_dir=os.path.join(DATA_BASE, "val_sharp", "001"),
        dst_dir=os.path.join(DATA_BASE, "val_blur", "001"),
        mask_dir=os.path.join(DATA_BASE, "val_mask", "001")
    )
    print("所有仿真任务完成。")
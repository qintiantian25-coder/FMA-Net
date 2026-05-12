import cv2
import numpy as np
import random
import os
import glob
import csv
import re  # 新增用于自然排序


# --- 基础工具 ---
def get_random_dark_color():
    """通用暗色采样 (0-100灰度)"""
    return random.randint(0, 10) if random.random() < 0.3 else random.randint(10, 100)


def get_mostly_black_color():
    """专为暗主导块采样的深度黑色 (0-15灰度，确保视觉上极明显)"""
    # 绝大部分为纯黑(0-5)，极少部分为深黑(5-15)
    return random.randint(0, 5) if random.random() < 0.9 else random.randint(5, 15)


def natural_sort_key(s):
    """自然排序的key函数，用于数字排序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# ==========================================
# 仿真模块：深度强化型混合聚合块 (无缝粘合，无外圈)
# ==========================================

def gen_mostly_black_tight_blob(w, h, target_pts, dominant_type, forbidden):
    """
    生成高度粘合且视觉上极明显的黑主导块或白主导块
    """
    pts_dict = {}
    cx, cy = 0, 0
    # 1. 选址避让
    for _ in range(50):
        tx, ty = random.randint(120, w - 120), random.randint(120, h - 120)
        if not any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            cx, cy = tx, ty
            break
    else:
        cx, cy = random.randint(120, w - 120), random.randint(120, h - 120)

    # 2. 紧凑生长：核心思想是新点必须在已有点的 1 像素邻域内
    current_pts = [(cx, cy)]

    # 颜色分配逻辑：在这里应用 dominant_type 的极性变化
    if dominant_type == 'white':
        pts_dict[(cx, cy)] = 255
    else:
        # 暗主导初始点直接使用深度黑色
        pts_dict[(cx, cy)] = get_mostly_black_color()

    while len(pts_dict) < target_pts:
        # 从现有像素中选基点，强制 8 邻域紧密生长
        base_x, base_y = random.choice(current_pts)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        dx, dy = random.choice(dirs)
        nx, ny = base_x + dx, base_y + dy

        if 5 <= nx < w - 5 and 5 <= ny < h - 5 and (nx, ny) not in pts_dict:
            if dominant_type == 'white':
                # 80% 白色，20% 通用随机灰
                c = 255 if random.random() < 0.8 else get_random_dark_color()
            else:
                # 90% 深度黑，10% 白色穿插
                if random.random() < 0.90:
                    c = get_mostly_black_color()
                else:
                    c = 255

            pts_dict[(nx, ny)] = c
            current_pts.append((nx, ny))

    # 转换格式
    pts = [(x, y, c) for (x, y), c in pts_dict.items()]
    return pts, (cx, cy)


def gen_extra_long_lines(w, h):
    """超长破碎线：单像素主干，极少毛刺"""
    pts = []
    for _ in range(random.randint(2, 3)):
        cx, cy = random.randint(30, w // 2), random.randint(30, h // 2)
        dx, dy = random.choice([(1, 0), (0, 1), (1, 1), (1, -1), (2, 1)])
        length = random.randint(120, 180)
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


def grow_compact_blob(w, h, target_w, target_d, forbidden):
    """大型/中型块：带污染边缘 (使用通用暗色采样)"""
    pts = []
    cx, cy = 0, 0
    for _ in range(50):
        tx, ty = random.randint(120, w - 120), random.randint(120, h - 120)
        if not any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            cx, cy = tx, ty
            break
    else:
        cx, cy = random.randint(120, w - 120), random.randint(120, h - 120)

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


def gen_cross_invalid_pixels(w, h, num_crosses, forbidden, occupied=None, margin=8, retry=80):
    """
    生成十字形无效像元（每个 5 个像素）：
    - 中心：深黑色 [0, 5]
    - 四个臂（上下左右）：中灰色 [60, 100]
    避免与禁忌区域和已占用坐标重叠。
    """
    if occupied is None:
        occupied = set()

    pts = []
    centers = []
    arms = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

    for _ in range(num_crosses):
        cx, cy = None, None

        for _ in range(retry):
            tx = random.randint(margin, w - 1 - margin)
            ty = random.randint(margin, h - 1 - margin)

            # 确保 4 邻域内的像素在边界内
            if not (1 <= tx < w - 1 and 1 <= ty < h - 1):
                continue

            # 区域级避让
            if any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
                continue

            cross_coords = [(tx + dx, ty + dy) for dx, dy in arms]
            # 像素级避让：与现有盲像素不重叠
            if any((x, y) in occupied for x, y in cross_coords):
                continue

            cx, cy = tx, ty
            break

        if cx is None:
            continue

        # 中心像素：深黑色
        pts.append((cx, cy, random.randint(0, 5)))
        occupied.add((cx, cy))

        # 臂部像素：中灰色
        for dx, dy in arms[1:]:
            x, y = cx + dx, cy + dy
            pts.append((x, y, random.randint(100, 150)))
            occupied.add((x, y))

        centers.append((cx, cy))

    return pts, centers, occupied


def gen_flash_anchor_positions(w, h, num_anchors, forbidden, occupied=None, margin=8, retry=120, min_dist=32,
                                block_size=1):
    """
    生成闪元候选位置（固定的 10 个方块锚点）：
    - 每个位置对应一个 block_size x block_size 的方块左上角
    - 避让已有盲元区域与已占用坐标
    - 位置之间保持一定最小间距，避免过于聚集
    """
    if occupied is None:
        occupied = set()

    anchors = []
    tries = 0
    while len(anchors) < num_anchors and tries < retry * max(1, num_anchors):
        tries += 1
        # Ensure chosen top-left (anchor) is at least `margin` from borders.
        # Use w-1 and h-1 so block_size==1 places anchors up to the last pixel inside margin.
        tx = random.randint(margin, w - 1 - margin)
        ty = random.randint(margin, h - 1 - margin)

        block_coords = [(tx + dx, ty + dy) for dx in range(block_size) for dy in range(block_size)]

        # 边界与避让检查：对整个方块生效
        if any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            continue
        if any((x, y) in occupied for x, y in block_coords):
            continue
        if any(abs(tx - ax) < min_dist and abs(ty - ay) < min_dist for ax, ay in anchors):
            continue

        anchors.append((tx, ty))
        # mark all pixels in the block as occupied (for block_size>1) — for single-pixel flashes this is just that pixel
        for xy in block_coords:
            occupied.add(xy)

    return anchors, occupied


def sample_flash_pixels_for_frame(img, flash_anchors, min_active=5, max_active=6, block_size=1):
    """
    从固定闪元候选位置中，随机挑选 5~6 个在当前帧激活。
    - 每个闪元为 block_size x block_size 方块
    - 可随机变亮或变暗
    - 返回：当前帧的闪元参数列表 [(x, y, c), ...] 和 CSV 记录
    """
    if not flash_anchors:
        return [], []

    active_num = min(len(flash_anchors), random.randint(min_active, max_active))
    active_positions = random.sample(flash_anchors, active_num)

    pts = []
    records = []
    for x, y in active_positions:
        # anchor (x,y) denotes the top-left of a block_size x block_size region.
        # For true single-pixel flashes block_size==1 and only that pixel is changed.
        # Compute the original gray from the anchor location for record-keeping.
        if not (0 <= y < img.shape[0] and 0 <= x < img.shape[1]):
            continue
        orig_gray = int(img[y, x][0])
        if random.random() < 0.5:
            # darken
            delta = random.randint(40, 120)
            c = random.randint(0, 30) if random.random() < 0.7 else max(0, orig_gray - delta)
            mode = "darken"
        else:
            # brighten
            delta = random.randint(40, 120)
            c = random.randint(225, 255) if random.random() < 0.7 else min(255, orig_gray + delta)
            mode = "brighten"

        # Append all pixels within the block (for block_size>1). When block_size==1 this is a single pixel.
        for dx in range(block_size):
            for dy in range(block_size):
                px, py = x + dx, y + dy
                if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                    pts.append((px, py, c))

        # keep one record per anchor (top-left) for CSV/logging
        records.append([x, y, orig_gray, c, mode])

    return pts, records


# ==========================================
# 批处理执行引擎（彩色图像适配）
# ==========================================

def run_consistent_simulation(src_dir, dst_dir, mask_dir):
    """
    对 src_dir 中的所有图像（支持 png, jpg, jpeg, bmp 等），施加相同的静态盲元模式，
    输出带盲元的彩色图像到 dst_dir，同时生成统一的掩码和坐标 CSV。
    输出图像格式统一为 PNG。
    """
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 支持多种图片扩展名
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP')
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(src_dir, ext)))
    # 去重并自然排序
    img_paths = sorted(set(img_paths), key=natural_sort_key)
    if not img_paths:
        print(f"警告：{src_dir} 中没有支持的图片文件（png/jpg/jpeg/bmp）")
        return

    first_img = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
    if first_img is None:
        print(f"无法读取第一张图像：{img_paths[0]}")
        return

    h, w = first_img.shape[:2]   # 动态读取真实图像尺寸，例如 512x640
    all_static_blind_params = []
    forbidden = [(w // 4, h // 4, 180)]

    # 1. 基础散布 (0-100 通用暗色)
    for wl, hl, cnt in [(w, h, 800), (w // 2, h // 2, 2000)]:
        for _ in range(cnt):
            tx, ty = random.randint(0, wl - 2), random.randint(0, hl - 2)
            all_static_blind_params.append((tx, ty, get_random_dark_color()))
            all_static_blind_params.append((tx + random.choice([0, 1]), ty + random.choice([0, 1]), 255))

    # 2. 超长破碎线
    all_static_blind_params += gen_extra_long_lines(w, h)

    # 3. 四个高度粘合的 30 像素不规则块（两白两黑）
    for _ in range(2):   # 白主导
        p, center = gen_mostly_black_tight_blob(w, h, 32, 'white', forbidden)
        all_static_blind_params += p
        forbidden.append((center[0], center[1], 80))
    for _ in range(2):   # 黑主导
        p, center = gen_mostly_black_tight_blob(w, h, 32, 'dark', forbidden)
        all_static_blind_params += p
        forbidden.append((center[0], center[1], 80))

    # 4. 大型/中型块
    configs = [(130, 150, 2), (45, 60, 2)]
    for wt, dt, n in configs:
        for _ in range(n):
            p, center = grow_compact_blob(w, h, wt, dt, forbidden)
            all_static_blind_params += p
            forbidden.append((center[0], center[1], 100))

    # 4.5 十字形无效像元（中心黑 + 四臂中灰）
    occupied = {(x, y) for x, y, _ in all_static_blind_params}
    cross_num = 40
    cross_pts, cross_centers, occupied = gen_cross_invalid_pixels(
        w=w,
        h=h,
        num_crosses=cross_num,
        forbidden=forbidden,
        occupied=occupied,
        margin=8,
        retry=80,
    )
    all_static_blind_params += cross_pts
    for cx, cy in cross_centers:
        forbidden.append((cx, cy, 20))

    # 4.6 闪元候选点：固定 10 个位置，每帧随机激活 5~6 个，输出为单像素（block_size=1）
    flash_anchor_num = 10
    flash_anchors, occupied = gen_flash_anchor_positions(
        w=w,
        h=h,
        num_anchors=flash_anchor_num,
        forbidden=forbidden,
        occupied=occupied,
        margin=8,
        retry=120,
        min_dist=32,
        block_size=1,
    )
    for ax, ay in flash_anchors:
        forbidden.append((ax, ay, 12))

    # 5. 渲染保存（彩色模式）
    mask_img = np.zeros((h, w), dtype=np.uint8)
    csv_records = []
    flash_records = []

    for idx, p in enumerate(img_paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)   # BGR 彩色图像 (H,W,3)
        if img is None:
            print(f"无法读取图像：{p}，跳过")
            continue
        out_img = img.copy()
        for x, y, c in all_static_blind_params:
            if 0 <= y < h and 0 <= x < w:
                # 将盲元像素设置为纯灰色（R=G=B=c）
                out_img[y, x] = [c, c, c]
                if idx == 0:  # 仅记录第一张图的掩码和坐标
                    mask_img[y, x] = 255
                    # 原图该像素的灰度近似（取红色通道值）
                    orig_gray = img[y, x][0]
                    csv_records.append([x, y, orig_gray, c])

        # 闪元：每帧随机激活 5~6 个候选点，位置固定但是否闪、怎么闪都随机
        flash_pts, frame_flash_records = sample_flash_pixels_for_frame(
            img, flash_anchors, min_active=5, max_active=6, block_size=1
        )
        for x, y, c in flash_pts:
            if 0 <= y < h and 0 <= x < w:
                out_img[y, x] = [c, c, c]
        if frame_flash_records:
            frame_name = os.path.basename(p)
            for rec in frame_flash_records:
                flash_records.append([frame_name] + rec)

        # 输出统一为 PNG 格式
        out_filename = os.path.splitext(os.path.basename(p))[0] + '.png'
        cv2.imwrite(os.path.join(dst_dir, out_filename), out_img)

    # 保存掩码和 CSV
    cv2.imwrite(os.path.join(mask_dir, "blind_pixel_mask.png"), mask_img)
    with open(os.path.join(mask_dir, "blind_pixel_coords.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'original_gray', 'simulated_gray'])
        writer.writerows(csv_records)

    # 闪元记录（帧级别）
    if flash_records:
        with open(os.path.join(mask_dir, "flash_pixel_coords.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_name', 'x', 'y', 'original_gray', 'simulated_gray', 'mode'])
            writer.writerows(flash_records)

    print(f"仿真完成：{len(img_paths)} 张图像已处理，盲元掩码位于 {mask_dir}；闪元候选点数：{len(flash_anchors)}")


if __name__ == "__main__":
    # 根据您的数据集路径进行修改
    DATA_BASE = r"D:\project\FMA-Net\data"

    # 示例：处理训练集中的城市类别（001）
    run_consistent_simulation(
        src_dir=os.path.join(DATA_BASE, "test_sharp", "001"),
        dst_dir=os.path.join(DATA_BASE, "test_blur", "001"),
        mask_dir=os.path.join(DATA_BASE, "test_mask", "001")
    )

    # 如果您还希望处理湖泊（002）和农田（003），可取消下面的注释
    # run_consistent_simulation(
    #     src_dir=os.path.join(DATA_BASE, "train_sharp", "002"),
    #     dst_dir=os.path.join(DATA_BASE, "train_blur", "002"),
    #     mask_dir=os.path.join(DATA_BASE, "train_mask", "002")
    # )
    # run_consistent_simulation(
    #     src_dir=os.path.join(DATA_BASE, "train_sharp", "003"),
    #     dst_dir=os.path.join(DATA_BASE, "train_blur", "003"),
    #     mask_dir=os.path.join(DATA_BASE, "train_mask", "003")
    # )

    print("所有仿真任务完成。")
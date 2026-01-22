import cv2
import numpy as np
import random
import os
import glob

# --- 全局变量与原始仿真逻辑 ---
fixed_rect_params = []
fixed_circle_params = []


def generate_fixed_params(width, height, num_spots, small_spot_ratio,
                          min_small_dim, max_small_dim, min_large_dim, max_large_dim):
    """完全保留原始生成逻辑，仅适配灰度输出"""
    global fixed_rect_params, fixed_circle_params
    fixed_rect_params = []
    fixed_circle_params = []

    # 灰度值范围保持不变
    GRAY_DARK_MIN, GRAY_DARK_MAX = 0, 100
    GRAY_BRIGHT_MIN, GRAY_BRIGHT_MAX = 200, 255

    # 1. 生成矩形盲元
    for _ in range(num_spots):
        is_small = random.random() < small_spot_ratio
        D = random.randint(min_small_dim, max_small_dim) if is_small else random.randint(min_large_dim, max_large_dim)

        # 80% 暗点，20% 亮点
        gray_val = random.randint(GRAY_DARK_MIN, GRAY_DARK_MAX) if random.random() < 0.8 else random.randint(
            GRAY_BRIGHT_MIN, GRAY_BRIGHT_MAX)
        color_bgr = (gray_val, gray_val, gray_val)

        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)
        fixed_rect_params.append((center_x, center_y, D, color_bgr))

    # 2. 生成圆形大斑点 (固定2个)
    for _ in range(2):
        gray_val = random.randint(150, 250)
        color_bgr = (gray_val, gray_val, gray_val)
        radius = 5
        center_x = random.randint(radius, width - 1 - radius)
        center_y = random.randint(radius, height - 1 - radius)
        fixed_circle_params.append((center_x, center_y, radius, color_bgr))


def draw_fixed_spots(img):
    """完全保留原始绘制逻辑"""
    height, width = img.shape[:2]
    for cx, cy, D, color in fixed_rect_params:
        x1, y1 = max(0, cx - D // 2), max(0, cy - D // 2)
        x2, y2 = min(width - 1, cx + (D + 1) // 2), min(height - 1, cy + (D + 1) // 2)
        if x1 < x2 and y1 < y2:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    for cx, cy, R, color in fixed_circle_params:
        cv2.circle(img, (cx, cy), R, color, -1, lineType=cv2.LINE_AA)


# --- 适配 D:\mangyuan\FMA-Net\data 的处理逻辑 ---

def process_dataset(root_dir):
    # 处理 train, val, test 三个部分
    for mode in ['train', 'val', 'test']:
        input_root = os.path.join(root_dir, f"{mode}_sharp")
        output_root = os.path.join(root_dir, f"{mode}_blur")

        if not os.path.exists(input_root):
            continue

        # 获取子文件夹 (001, 002, ...)
        seq_folders = sorted([f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))])

        for seq in seq_folders:
            in_seq_path = os.path.join(input_root, seq)
            out_seq_path = os.path.join(output_root, seq)
            os.makedirs(out_seq_path, exist_ok=True)

            img_paths = sorted(glob.glob(os.path.join(in_seq_path, "*.png")))
            if not img_paths: continue

            # 每个序列生成一套全新的固定盲元位置
            generate_fixed_params(640, 512, 200, 0.98, 1, 1, 3, 5)

            print(f"正在处理: {mode}/{seq}，共 {len(img_paths)} 帧")
            for p in img_paths:
                # 以灰度模式读取
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                # 为了使用原始的 BGR 绘制函数并保证效果一致，先转 BGR 再转回灰度
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                draw_fixed_spots(img_bgr)
                img_final = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(os.path.join(out_seq_path, os.path.basename(p)), img_final)


if __name__ == "__main__":
    DATASET_PATH = r"D:\mangyuan\FMA-Net\data"
    process_dataset(DATASET_PATH)
    print("盲元仿真数据生成完成！")
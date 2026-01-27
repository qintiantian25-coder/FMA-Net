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
    global fixed_rect_params, fixed_circle_params
    fixed_rect_params = []
    fixed_circle_params = []

    GRAY_DARK_MIN, GRAY_DARK_MAX = 0, 100
    GRAY_BRIGHT_MIN, GRAY_BRIGHT_MAX = 200, 255

    # 1. 生成矩形盲元
    for _ in range(num_spots):
        is_small = random.random() < small_spot_ratio
        D = random.randint(min_small_dim, max_small_dim) if is_small else random.randint(min_large_dim, max_large_dim)
        gray_val = random.randint(GRAY_DARK_MIN, GRAY_DARK_MAX) if random.random() < 0.8 else random.randint(
            GRAY_BRIGHT_MIN, GRAY_BRIGHT_MAX)
        color_bgr = (gray_val, gray_val, gray_val)
        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)
        fixed_rect_params.append((center_x, center_y, D, color_bgr))

    # 2. 生成圆形大斑点
    num_circles = 10
    for _ in range(num_circles):
        gray_val = random.randint(50, 250)
        color_bgr = (gray_val, gray_val, gray_val)
        radius = random.randint(5, 12)
        center_x = random.randint(radius, width - 1 - radius)
        center_y = random.randint(radius, height - 1 - radius)
        fixed_circle_params.append((center_x, center_y, radius, color_bgr))


def draw_fixed_spots(img):
    height, width = img.shape[:2]
    for cx, cy, D, color in fixed_rect_params:
        x1, y1 = max(0, cx - D // 2), max(0, cy - D // 2)
        x2, y2 = min(width - 1, cx + (D + 1) // 2), min(height - 1, cy + (D + 1) // 2)
        if x1 < x2 and y1 < y2:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    for cx, cy, R, color in fixed_circle_params:
        cv2.circle(img, (cx, cy), R, color, -1, lineType=cv2.LINE_AA)


# --- 核心修改：精准处理单一路径 ---

def process_single_sequence(input_dir, output_dir):
    """
    input_dir: 1.png - 270.png 所在的原始清晰图路径
    output_dir: 结果保存的模糊图路径
    """
    if not os.path.exists(input_dir):
        print(f"错误：找不到路径 {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    if not img_paths:
        print(f"错误：在 {input_dir} 下没找到 PNG 图片")
        return

    # 生成一套固定参数
    # 假设图片尺寸为 640x512
    generate_fixed_params(640, 512, 200, 0.90, 1, 1, 5, 10)

    print(f"正在处理序列: {input_dir}")
    print(f"目标保存路径: {output_dir}")

    for p in img_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        draw_fixed_spots(img_bgr)
        img_final = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(os.path.join(output_dir, os.path.basename(p)), img_final)


if __name__ == "__main__":
    # --- 你只需要修改这里 ---
    # 指定你想要处理的 sharp 文件夹
    SRC_DIR = r"E:\qtt\mangyuan\FMA-Net\data\val_sharp\001"

    # 指定你想要保存的 blur 文件夹
    DST_DIR = r"E:\qtt\mangyuan\FMA-Net\data\val_blur\001"

    process_single_sequence(SRC_DIR, DST_DIR)
    print("指定序列仿真完成！")
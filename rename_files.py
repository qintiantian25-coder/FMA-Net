import os

def rename_recursive(base_dir):
    count = 0
    # os.walk 会遍历所有子目录
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            # 如果是纯数字且没有后缀
            if f.isdigit() and "." not in f:
                old_path = os.path.join(root, f)
                new_path = os.path.join(root, f + ".png")
                os.rename(old_path, new_path)
                count += 1
    print(f"成功在子文件夹中重命名了 {count} 个文件为 .png")

if __name__ == "__main__":
    target_dir = r"E:\qtt\mangyuan\FMA-Net\data\test_blur"
    rename_recursive(target_dir)
import os
import glob
import torch
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class BlindPixelDataset(Dataset):
    def __init__(self, data_root, mode='train', num_seq=5, patch_size=256):
        """
        data_root: D:/mangyuan/FMA-Net/data
        mode: 'train', 'val', or 'test'
        num_seq: 连续帧的数量 (T)，FMA-Net 通常设为 3 或 5
        patch_size: 训练时的随机裁剪尺寸
        """
        self.mode = mode
        self.num_seq = num_seq
        self.patch_size = patch_size

        # 路径设定
        self.blur_root = os.path.join(data_root, f'{mode}_blur')
        self.sharp_root = os.path.join(data_root, f'{mode}_sharp')
        self.flow_root = os.path.join(data_root, f'{mode}_flow')

        # 获取所有文件夹（序列） 001, 002...
        self.seq_list = sorted(
            [d for d in os.listdir(self.blur_root) if os.path.isdir(os.path.join(self.blur_root, d))])

        # 构建样本列表
        self.samples = []
        for seq in self.seq_list:
            # 统计该序列下的图片数量
            frames = sorted(glob.glob(os.path.join(self.blur_root, seq, '*.png')))
            num_frames = len(frames)

            # 滑动窗口取样：确保中心帧前后都有足够的邻帧
            # 例如 num_seq=5, 则从第 3 帧开始，到第 N-2 帧结束
            for i in range(num_frames - num_seq + 1):
                self.samples.append({
                    'seq': seq,
                    'start_idx': i,  # 这一段序列在文件夹中的起始索引
                    'frame_names': [os.path.basename(f) for f in frames[i: i + num_seq]]
                })

    def __len__(self):
        return len(self.samples)

    def _load_gray_img(self, path):
        """读取灰度图并归一化"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        return img.astype(np.float32) / 255.0

    def __getitem__(self, index):
        sample = self.samples[index]
        seq = sample['seq']
        frame_names = sample['frame_names']

        blur_frames = []
        sharp_frames = []

        # 1. 读取连续 T 帧图像
        for name in frame_names:
            b_path = os.path.join(self.blur_root, seq, name)
            s_path = os.path.join(self.sharp_root, seq, name)

            blur_frames.append(self._load_gray_img(b_path))
            sharp_frames.append(self._load_gray_img(s_path))

        # 转换为 [T, H, W]
        blur_seq = np.stack(blur_frames, axis=0)
        sharp_seq = np.stack(sharp_frames, axis=0)

        # 2. 读取光流 (.npy)
        # FMA-Net 期待中心帧与前后帧的光流。假设 T=5，中心帧是 index 2
        # generate_flow.py 生成的文件名格式为: "中心帧_邻帧.npy" (不含扩展名)
        center_idx = self.num_seq // 2
        center_name = frame_names[center_idx].split('.')[0]

        flows = []
        for i in range(self.num_seq):
            if i == center_idx:
                # 中心帧到自身的光流设为 0
                flow = np.zeros((2, blur_seq.shape[1], blur_seq.shape[2]), dtype=np.float32)
            else:
                neighbor_name = frame_names[i].split('.')[0]
                flow_path = os.path.join(self.flow_root, seq, f"{center_name}_{neighbor_name}.npy")
                # 如果光流文件还没生成，先用 0 代替防止崩溃
                if os.path.exists(flow_path):
                    flow = np.load(flow_path).transpose(2, 0, 1)  # [H, W, 2] -> [2, H, W]
                else:
                    flow = np.zeros((2, blur_seq.shape[1], blur_seq.shape[2]), dtype=np.float32)
            flows.append(flow)

        # 组合光流 [T, 2, H, W] -> 最终 FMA-Net 需要 [2*T, H, W] 或类似，这里先按 T 堆叠
        flow_tensor = np.stack(flows, axis=0)

        # 3. 训练集随机裁剪
        if self.mode == 'train':
            t, h, w = blur_seq.shape
            th, tw = self.patch_size, self.patch_size
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            blur_seq = blur_seq[:, y1:y1 + th, x1:x1 + tw]
            sharp_seq = sharp_seq[:, y1:y1 + th, x1:x1 + tw]
            flow_tensor = flow_tensor[:, :, y1:y1 + th, x1:x1 + tw]

            # 随机翻转
            if random.random() < 0.5:
                blur_seq = blur_seq[:, :, ::-1]
                sharp_seq = sharp_seq[:, :, ::-1]
                flow_tensor = flow_tensor[:, :, :, ::-1]
                flow_tensor[:, 0, :, :] *= -1  # 水平翻转后 U 分量取反

        # 4. 封装为 Tensor 并适配 FMA-Net 维度
        # lr_blur_seq: [1, T, H, W]
        lr_blur_seq = torch.from_numpy(blur_seq.copy()).unsqueeze(0)
        # hr_sharp_seq: [1, T, H, W]
        hr_sharp_seq = torch.from_numpy(sharp_seq.copy()).unsqueeze(0)
        # lr_sharp_seq: 在去盲元 Scale=1 任务中，它等于 hr_sharp_seq
        lr_sharp_seq = hr_sharp_seq.clone()
        # flow: [T, 2, H, W] -> 展平为 [2*T, H, W] 以匹配原 model.py 期待的输入
        t, c, h, w = flow_tensor.shape
        flow_final = torch.from_numpy(flow_tensor.copy()).view(-1, h, w)
        if self.mode == 'test':
            # 测试时，我们只需要输入图和它的名字，不需要标签
            center_name = frame_names[self.num_seq // 2]
            relative_path = os.path.join(seq, center_name)
            return lr_blur_seq, relative_path  # 只返回两个东西
        else:
            # 训练和验证时，返回四个东西用于算 Loss
            return lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow_final


def get_dataset(config, type='train'): # <--- 必须明确接收这两个参数
    """
    根据配置返回对应的 DataLoader
    config: Config 对象 (来自 config.py)
    type: 字符串 'train', 'val', 或 'test'
    """
    # 1. 实例化 Dataset
    dataset = BlindPixelDataset(
        data_root=config.dataset_path,
        mode=type,
        num_seq=config.num_seq,
        patch_size=config.patch_size
    )

    # 2. 设定参数
    is_train = (type == 'train')

    # 3. 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size if is_train else 1,
        shuffle=is_train,
        num_workers=int(config.nThreads),
        pin_memory=True,
        drop_last=is_train
    )

    return dataloader
import os
import glob
import torch
import cv2
import numpy as np
import random
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def natural_sort_key(s):
    """
    自然排序算法的 Key 函数：
    将字符串中的数字提取并转为整数，使排序结果为 1, 2, 10 而不是 1, 10, 2。
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


class BlindPixelDataset(Dataset):
    def __init__(self, data_root, mode='train', num_seq=5, patch_size=256):
        """
        data_root: 数据根目录
        mode: 'train', 'val', or 'test'
        num_seq: 连续帧的数量 (T)
        patch_size: 训练时的随机裁剪尺寸
        """
        self.mode = mode
        self.num_seq = num_seq
        self.patch_size = patch_size

        # 路径设定
        self.blur_root = os.path.join(data_root, f'{mode}_blur')
        self.sharp_root = os.path.join(data_root, f'{mode}_sharp')
        self.flow_root = os.path.join(data_root, f'{mode}_flow')

        # 1. 获取所有序列文件夹（如 001, 002...）并进行自然排序
        self.seq_list = sorted(
            [d for d in os.listdir(self.blur_root) if os.path.isdir(os.path.join(self.blur_root, d))],
            key=natural_sort_key
        )

        # 2. 构建样本列表
        self.samples = []
        for seq in self.seq_list:
            # 获取该序列下的所有图片，并进行自然排序 (1.png, 2.png, 10.png...)
            raw_frames = glob.glob(os.path.join(self.blur_root, seq, '*.png'))
            frames = sorted(raw_frames, key=natural_sort_key)
            num_frames = len(frames)

            # 滑动窗口取样：确保中心帧前后都有足够的邻帧
            for i in range(num_frames - num_seq + 1):
                self.samples.append({
                    'seq': seq,
                    'start_idx': i,
                    'frame_names': [os.path.basename(f) for f in frames[i: i + num_seq]]
                })

        print(
            f"===> {mode} dataset initialized. Found {len(self.samples)} samples across {len(self.seq_list)} sequences.")

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
            # 训练/验证需要 sharp，测试时如果没有 sharp 可以填 dummy 数据
            if os.path.exists(s_path):
                sharp_frames.append(self._load_gray_img(s_path))
            else:
                sharp_frames.append(np.zeros_like(blur_frames[-1]))

        blur_seq = np.stack(blur_frames, axis=0)
        sharp_seq = np.stack(sharp_frames, axis=0)

        # 2. 读取光流
        center_idx = self.num_seq // 2
        center_name = frame_names[center_idx].split('.')[0]

        flows = []
        for i in range(self.num_seq):
            if i == center_idx:
                flow = np.zeros((2, blur_seq.shape[1], blur_seq.shape[2]), dtype=np.float32)
            else:
                neighbor_name = frame_names[i].split('.')[0]
                flow_path = os.path.join(self.flow_root, seq, f"{center_name}_{neighbor_name}.npy")
                if os.path.exists(flow_path):
                    flow = np.load(flow_path).transpose(2, 0, 1)  # [H, W, 2] -> [2, H, W]
                else:
                    flow = np.zeros((2, blur_seq.shape[1], blur_seq.shape[2]), dtype=np.float32)
            flows.append(flow)

        flow_tensor = np.stack(flows, axis=0)

        # 3. 训练集随机裁剪与翻转
        if self.mode == 'train':
            t, h, w = blur_seq.shape
            th, tw = self.patch_size, self.patch_size

            # If image is smaller than required patch, pad (reflect) to avoid negative ranges
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                def pad_stack(arr, is_flow=False):
                    # arr shape: [T, H, W] for images, [T, H, W, C] for flow
                    if is_flow:
                        return np.pad(arr, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
                    else:
                        return np.pad(arr, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')

                blur_seq = pad_stack(blur_seq, is_flow=False)
                sharp_seq = pad_stack(sharp_seq, is_flow=False)
                flow_tensor = pad_stack(flow_tensor, is_flow=True)

                # update sizes
                t, h, w = blur_seq.shape

            # now safe to sample
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            blur_seq = blur_seq[:, y1:y1 + th, x1:x1 + tw]
            sharp_seq = sharp_seq[:, y1:y1 + th, x1:x1 + tw]
            flow_tensor = flow_tensor[:, :, y1:y1 + th, x1:x1 + tw]

            if random.random() < 0.5:
                blur_seq = blur_seq[:, :, ::-1]
                sharp_seq = sharp_seq[:, :, ::-1]
                flow_tensor = flow_tensor[:, :, :, ::-1]
                flow_tensor[:, 0, :, :] *= -1

            # 随机垂直翻转
            if random.random() < 0.5:
                blur_seq = blur_seq[:, ::-1, :]
                sharp_seq = sharp_seq[:, ::-1, :]
                flow_tensor = flow_tensor[:, :, ::-1, :]
                flow_tensor[:, 1, :, :] *= -1

            # 随机 90° 旋转 (k ∈ {0, 1, 2, 3})，适配灰度图
            k = random.randint(0, 3)
            if k > 0:
                blur_seq = np.rot90(blur_seq, k=k, axes=(1, 2)).copy()
                sharp_seq = np.rot90(sharp_seq, k=k, axes=(1, 2)).copy()
                # 光流旋转：先旋转空间维度，再调整方向
                flow_tensor = np.rot90(np.ascontiguousarray(flow_tensor), k=k, axes=(2, 3)).copy()
                if k == 1:
                    flow_tensor[:, [0, 1], :, :] = -flow_tensor[:, [1, 0], :, :]
                elif k == 3:
                    flow_tensor[:, [0, 1], :, :] = flow_tensor[:, [1, 0], :, :]
                    flow_tensor[:, 0, :, :] *= -1
                elif k == 2:
                    flow_tensor[:, 0, :, :] *= -1
                    flow_tensor[:, 1, :, :] *= -1

            # 随机亮度/对比度调整 (灰度图)
            if random.random() < 0.3:
                contrast = random.uniform(0.85, 1.15)
                brightness = random.uniform(-0.03, 0.03)
                blur_seq = np.clip(blur_seq * contrast + brightness, 0.0, 1.0)
                sharp_seq = np.clip(sharp_seq * contrast + brightness, 0.0, 1.0)

            # 时序随机反转
            if random.random() < 0.3:
                blur_seq = blur_seq[::-1]
                sharp_seq = sharp_seq[::-1]
                flow_tensor = flow_tensor[::-1]
                flow_tensor[:, 0, :, :] *= -1
                flow_tensor[:, 1, :, :] *= -1

        # 4. 封装 Tensor
        lr_blur_seq = torch.from_numpy(blur_seq.copy()).unsqueeze(0)
        hr_sharp_seq = torch.from_numpy(sharp_seq.copy()).unsqueeze(0)
        lr_sharp_seq = hr_sharp_seq.clone()

        t, c, h, w = flow_tensor.shape
        flow_final = torch.from_numpy(flow_tensor.copy()).view(-1, h, w)

        if self.mode == 'test':
            center_name_with_ext = frame_names[self.num_seq // 2]
            relative_path = os.path.join(seq, center_name_with_ext)
            return lr_blur_seq, relative_path
        else:
            return lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow_final


def get_dataset(config, type='train'):
    """
    根据配置返回对应的 DataLoader
    """
    dataset = BlindPixelDataset(
        data_root=config.dataset_path,
        mode=type,
        num_seq=config.num_seq,
        patch_size=config.patch_size
    )

    is_train = (type == 'train')
    num_workers = int(config.nThreads)
    pin_memory = bool(getattr(config, 'pin_memory', True))
    persistent_workers = bool(getattr(config, 'persistent_workers', True)) and (num_workers > 0)
    prefetch_factor = int(getattr(config, 'prefetch_factor', 2))

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=config.batch_size if is_train else 1,
        shuffle=is_train,  # 训练时打乱，测试时不打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        persistent_workers=persistent_workers,
    )
    # prefetch_factor 仅在多进程 DataLoader 生效。
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    dataloader = DataLoader(
        **loader_kwargs
    )

    return dataloader
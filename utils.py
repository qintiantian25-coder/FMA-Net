import os
import sys
import cv2
import math
import torch
import numpy as np
from pathlib import Path


# --------------------------------------------
# 基础写操作与去归一化
# --------------------------------------------
def write(log, string):
    sys.stdout.flush()
    log.write(string + '\n')
    log.flush()


def denorm(x):
    """将 [0, 1] 的 Tensor 转为 [0, 255] 的 numpy 数组"""
    x = x.cpu().detach().numpy()
    x = np.clip(x, 0, 1) * 255.0
    return np.round(x).astype(np.uint8)


# --------------------------------------------
# 指标计算 (PSNR / SSIM)
# --------------------------------------------
def RGB_PSNR(img1, img2, border=0):
    """适配灰度图与彩色图的 PSNR 计算"""
    # Fast path: keep tensor math on device to avoid CPU sync/NumPy conversion in val loop.
    if torch.is_tensor(img1) and torch.is_tensor(img2):
        if img1.shape != img2.shape:
            raise ValueError(f'Input images must have the same dimensions. {img1.shape} vs {img2.shape}')
        if border > 0:
            img1 = img1[..., border:-border, border:-border]
            img2 = img2[..., border:-border, border:-border]
        img1 = img1.float()
        img2 = img2.float()
        # Be robust to both [0,1] tensors (training/validation) and [0,255] tensors (test utilities).
        scale = 1.0 if (img1.detach().max() > 1.5 or img2.detach().max() > 1.5) else 255.0
        mse = torch.mean(((img1 * scale) - (img2 * scale)) ** 2)
        if mse.item() == 0:
            return float('inf')
        return float(20.0 * torch.log10(torch.tensor(255.0, device=img1.device) / torch.sqrt(mse)).item())

    # 如果输入是 Tensor，先转 numpy 并去掉 batch/channel 维度
    if torch.is_tensor(img1):
        img1 = img1.squeeze().cpu().detach().numpy()
        img2 = img2.squeeze().cpu().detach().numpy()

    if not img1.shape == img2.shape:
        raise ValueError(f'Input images must have the same dimensions. {img1.shape} vs {img2.shape}')

    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border, ...]
    img2 = img2[border:h - border, border:w - border, ...]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def SSIM(img1, img2, border=0):
    """适配灰度图与彩色图的 SSIM 计算"""
    if torch.is_tensor(img1):
        img1 = img1.squeeze().cpu().detach().numpy()
        img2 = img2.squeeze().cpu().detach().numpy()

    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border, ...]
    img2 = img2[border:h - border, border:w - border, ...]

    if img1.ndim == 2:  # 灰度图
        return ssim(img1, img2)
    else:  # 彩色图 [H, W, C]
        ssims = []
        for i in range(img1.shape[2]):
            ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
        return np.array(ssims).mean()


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# --------------------------------------------
# 训练报告类 (Train_Report) - 已修正初始化报错
# --------------------------------------------
class Train_Report:
    def __init__(self, save_dir=None, type='train', stage=1):
        self.reset()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f'stage{stage}_{type}_log.txt')
            self.logFile = open(filename, 'a', encoding='utf-8')
        else:
            self.logFile = None

    def reset(self):
        self.restoration_loss = []
        self.recon_loss = []
        self.hr_warping_loss = []
        self.lr_warping_loss = []
        self.flow_loss = []
        self.D_TA_loss = []
        self.R_TA_loss = []
        self.total_loss = []
        self.psnr = []
        self.recon_psnr = []
        self.num_examples = 0

    def update(self, batch_size, restoration_loss, recon_loss, hr_warping_loss, lr_warping_loss, flow_loss, D_TA_loss,
               R_TA_loss, total_loss):
        self.num_examples += batch_size
        self.restoration_loss.append(restoration_loss * batch_size)
        self.recon_loss.append(recon_loss * batch_size)
        self.hr_warping_loss.append(hr_warping_loss * batch_size)
        self.lr_warping_loss.append(lr_warping_loss * batch_size)
        self.flow_loss.append(flow_loss * batch_size)
        self.D_TA_loss.append(D_TA_loss * batch_size)
        self.R_TA_loss.append(R_TA_loss * batch_size)
        self.total_loss.append(total_loss * batch_size)

    def update_restoration_metric(self, output, y):
        self.psnr.append(RGB_PSNR(output, y))

    def update_recon_metric(self, output, y):
        self.recon_psnr.append(RGB_PSNR(output, y))

    def compute_mean(self, val_list):
        if len(val_list) == 0: return 0
        return np.sum(val_list) / self.num_examples

    def result_str(self, lr_D, lr_R, period_time):
        avg_total = self.compute_mean(self.total_loss)
        avg_recon = self.compute_mean(self.recon_loss)
        if lr_R == 0 or lr_R is None:  # Stage 1
            res = f"Recon: {avg_recon:.6f} | Total: {avg_total:.6f} | LR_D: {lr_D:.7f} | Time: {period_time:.2f}s"
        else:  # Stage 2
            avg_res = self.compute_mean(self.restoration_loss)
            res = f"Res: {avg_res:.6f} | Recon: {avg_recon:.6f} | LR_D: {lr_D:.7f} | LR_R: {lr_R:.7f} | Time: {period_time:.2f}s"
        return res

    def val_result_str(self, period_time):
        # 验证时的 PSNR 取的是平均值，而不是 batch 累加
        avg_psnr = np.mean(self.psnr) if self.psnr else 0
        avg_recon_psnr = np.mean(self.recon_psnr) if self.recon_psnr else 0
        return f"Val PSNR: {avg_psnr:.3f} | Recon PSNR: {avg_recon_psnr:.3f} | Time: {period_time:.2f}s"

    def write(self, string):
        print(string)
        if self.logFile:
            self.logFile.write(string + '\n')
            self.logFile.flush()

    def __del__(self):
        if hasattr(self, 'logFile') and self.logFile:
            self.logFile.close()


# --------------------------------------------
# 测试报告类 (TestReport)
# --------------------------------------------
class TestReport:
    def __init__(self, base_dir=None):
        self.total_rgb_psnr = []
        self.total_ssim = []
        # 如果需要保存到文件，可以在此处仿照 Train_Report 开启文件流

    def update_metric(self, gt, output, filename):
        # Keep uint8 numpy path to avoid accidental value-range mismatch in PSNR.
        psnr = RGB_PSNR(output, gt)
        ssim_val = SSIM(output, gt)

        self.total_rgb_psnr.append(psnr)
        self.total_ssim.append(ssim_val)
        print(f"Image: {filename} | PSNR: {psnr:.3f} | SSIM: {ssim_val:.4f}")

    def print_final_result(self):
        print("=" * 30)
        print(f"Final Average PSNR: {np.mean(self.total_rgb_psnr):.3f}")
        print(f"Final Average SSIM: {np.mean(self.total_ssim):.4f}")
        print("=" * 30)


# --------------------------------------------
# 图像保存管理 (SaveManager)
# --------------------------------------------
class SaveManager:
    def __init__(self, config):
        self.config = config

    def save_batch_images(self, src, batch_size, step):
        num = min(batch_size, 4)
        dir = self.config.log_dir
        os.makedirs(dir, exist_ok=True)

        # 将多个阶段图拼接保存 [Batch, C, H, W]
        for i in range(num):
            combined = []
            for img_tensor in src:
                img = denorm(img_tensor[i])  # [C, H, W]
                if img.shape[0] == 1:  # 灰度图
                    img = img.squeeze(0)  # [H, W]
                else:  # 彩色图
                    img = np.transpose(img, (1, 2, 0))  # [H, W, C]
                combined.append(img)

            final_img = np.concatenate(combined, axis=1)
            cv2.imwrite(os.path.join(dir, f"step_{step:06d}_batch_{i}.png"), final_img)
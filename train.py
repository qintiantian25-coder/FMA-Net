import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import re
from utils import Train_Report, TestReport, SaveManager


class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

        # 图像保存管理器（用于训练中查看中间结果）
        if self.config.save_train_img:
            self.save_manager = SaveManager(config)

        # 基础损失函数
        self.criterion = nn.L1Loss()

        # 学习率里程碑（Epochs）
        milestones = [260, 360, 380, 390]

        # 优化器与调度器：退化学习网络 (Net_D)
        self.optimizer_D = torch.optim.Adam(
            self.model.degradation_learning_network.parameters(),
            lr=self.config.lr
        )
        self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_D, milestones=milestones, gamma=0.5
        )

        # 优化器与调度器：恢复网络 (Net_R) - 仅在 Stage 2 开启
        if self.config.stage == 2:
            self.optimizer_R = torch.optim.Adam(
                self.model.restoration_network.parameters(),
                lr=self.config.lr
            )
            self.scheduler_R = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_R, milestones=milestones, gamma=0.5
            )

        # 权重保存路径
        self.checkpoint_path = os.path.join(self.config.save_dir, f'model_stage{self.config.stage}')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.model.cuda()

    def smart_recon_loss(self, recon, target):
        abs_diff = torch.abs(recon - target)
        with torch.no_grad():
            flat_diff = abs_diff.view(-1)
            k = int(flat_diff.numel() * 0.005)
            threshold, _ = torch.topk(flat_diff, k)
            min_topk_val = threshold[-1]
            blind_mask = (abs_diff >= min_topk_val).float()

        l1_loss = abs_diff
        l2_loss = (recon - target) ** 2
        weighted_loss = torch.where(blind_mask > 0.5, 1000.0 * l2_loss, 1.0 * l1_loss)
        return weighted_loss.mean()

    def train(self, dataloader, train_log, global_step):
        self.model.train()
        report = Train_Report()
        start = time.time()

        for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow) in enumerate(dataloader):
            lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow = lr_blur_seq.cuda(), hr_sharp_seq.cuda(), lr_sharp_seq.cuda(), flow.cuda()
            result_dict = self.model(lr_blur_seq, hr_sharp_seq)
            batch_size, _, t, _, _ = lr_blur_seq.shape

            if self.config.stage == 1:
                recon_loss = self.smart_recon_loss(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                hr_warping_loss = self.config.hr_warping_loss_weight * self.criterion(
                    result_dict['hr_warp'], hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))

                b, _, t, h, w = result_dict['image_flow'].size()
                flow_loss = 10.0 * self.config.flow_loss_weight * self.criterion(result_dict['image_flow'],
                                                                                 flow.view(b, 2, t, h, w))
                D_TA_loss = self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                total_loss = recon_loss + hr_warping_loss + flow_loss + D_TA_loss
                self.optimizer_D.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer_D.step()
                report.update(batch_size, 0, recon_loss.item(), hr_warping_loss.item(), 0, flow_loss.item(),
                              D_TA_loss.item(), 0, total_loss.item())

            elif self.config.stage == 2:
                restoration_loss = self.criterion(result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :])
                recon_loss = self.config.Net_D_weight * self.smart_recon_loss(result_dict['recon'],
                                                                              lr_blur_seq[:, :, t // 2, :, :])
                lr_warping_loss = self.config.lr_warping_loss_weight * self.criterion(result_dict['lr_warp'],
                                                                                      lr_blur_seq[:, :,
                                                                                      t // 2:t // 2 + 1, :, :].repeat(
                                                                                          [1, 1, t, 1, 1]))
                hr_warping_loss = self.config.Net_D_weight * self.config.hr_warping_loss_weight * self.criterion(
                    result_dict['hr_warp'], hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))

                b, _, t, h, w = result_dict['image_flow'].size()
                flow_loss = self.config.Net_D_weight * self.config.flow_loss_weight * self.criterion(
                    result_dict['image_flow'], flow.view(b, 2, t, h, w))
                R_TA_loss = self.config.R_TA_loss_weight * self.criterion(result_dict['F_sharp_R'], lr_sharp_seq)
                D_TA_loss = self.config.Net_D_weight * self.config.D_TA_loss_weight * self.criterion(
                    result_dict['F_sharp_D'], lr_sharp_seq)

                total_loss = restoration_loss + recon_loss + hr_warping_loss + lr_warping_loss + flow_loss + R_TA_loss + D_TA_loss
                self.optimizer_D.zero_grad(set_to_none=True)
                self.optimizer_R.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer_D.step()
                self.optimizer_R.step()
                report.update(batch_size, restoration_loss.item(), recon_loss.item(), hr_warping_loss.item(),
                              lr_warping_loss.item(), flow_loss.item(), D_TA_loss.item(), R_TA_loss.item(),
                              total_loss.item())

            global_step += 1
            if global_step % 100 == 0 or idx == len(dataloader) - 1:
                lr_D = self.optimizer_D.param_groups[0]['lr']
                lr_R = self.optimizer_R.param_groups[0]['lr'] if self.config.stage == 2 else 0.0
                train_log.write(f"[{global_step}]\t" + report.result_str(lr_D, lr_R, time.time() - start))
                start = time.time()
                if self.config.save_train_img:
                    src = [lr_blur_seq[:, :, t // 2, :, :], result_dict['recon']]
                    if self.config.stage == 2: src.extend([result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :]])
                    self.save_manager.save_batch_images(src, batch_size, global_step)
                report = Train_Report()

        self.scheduler_D.step()
        if self.config.stage == 2: self.scheduler_R.step()
        return global_step

    def validate(self, dataloader, val_log, epoch):
        self.model.eval()
        report = Train_Report()
        start = time.time()
        with torch.no_grad():
            for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow) in enumerate(dataloader):
                lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow = lr_blur_seq.cuda(), hr_sharp_seq.cuda(), lr_sharp_seq.cuda(), flow.cuda()
                result_dict = self.model(lr_blur_seq, hr_sharp_seq)
                if self.config.stage == 1:
                    report.update_recon_metric(result_dict['recon'], lr_blur_seq[:, :, lr_blur_seq.shape[2] // 2, :, :])
                else:
                    report.update_restoration_metric(result_dict['output'],
                                                     hr_sharp_seq[:, :, hr_sharp_seq.shape[2] // 2, :, :])
        val_log.write(f"VAL Epoch [{epoch}]\t" + report.val_result_str(time.time() - start))
        return report.psnr if self.config.stage == 2 else report.recon_psnr

    def test(self, dataloader):
        self.model.eval()
        save_triple = os.path.join(self.config.save_dir, 'triple_comparison')
        save_pure = os.path.join(self.config.save_dir, 'test')
        os.makedirs(save_triple, exist_ok=True)
        os.makedirs(save_pure, exist_ok=True)

        gt_root = os.path.join(self.config.dataset_path, 'test_sharp')
        gt_finder = {}
        for root, _, files in os.walk(gt_root):
            for f in files:
                if f.lower().endswith('.png'):
                    gt_finder[f] = os.path.join(root, f)

        print(f"===> 开始精准推理与可视化...")

        with torch.no_grad():
            for idx, (lr_blur_seq, relative_path) in enumerate(dataloader):
                # 1. 确定中心帧索引
                t_idx = lr_blur_seq.shape[2] // 2

                # 2. 核心修正：适配多种 path 返回结构 (解决 IndexError)
                if isinstance(relative_path, (list, tuple)) and len(relative_path) == 1:
                    # 情况 A: 只有一帧名字 ['001\\3.png']
                    current_frame_name = os.path.basename(relative_path[0])
                elif isinstance(relative_path, (list, tuple)) and len(relative_path) > t_idx:
                    # 情况 B: 有完整的序列名字 ['1.png', '2.png', '3.png'...]
                    target_item = relative_path[t_idx]
                    current_frame_name = os.path.basename(
                        target_item[0] if isinstance(target_item, (list, tuple)) else target_item)
                else:
                    # 兜底
                    current_frame_name = os.path.basename(relative_path[0])

                if not current_frame_name.lower().endswith('.png'):
                    current_frame_name += '.png'

                # 3. 推理核心
                lr_blur_seq = lr_blur_seq.cuda()
                result_dict = self.model(lr_blur_seq)
                output = result_dict['output']

                # 4. 提取对应的输入中心帧和真值进行对比
                input_blind = lr_blur_seq[:, :, t_idx, :, :]
                gt_path = gt_finder.get(current_frame_name)

                if gt_path:
                    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    gt_tensor = torch.from_numpy(gt_img).float() / 255.0
                    gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0).cuda()

                    # 统一尺寸对齐 (处理 Padding 差异)
                    if output.shape != gt_tensor.shape:
                        output = F.interpolate(output, size=(gt_tensor.shape[2], gt_tensor.shape[3]), mode='bilinear')
                    if input_blind.shape != gt_tensor.shape:
                        input_blind = F.interpolate(input_blind, size=(gt_tensor.shape[2], gt_tensor.shape[3]),
                                                    mode='bilinear')

                    # 拼接：[带盲元输入 | 模型修复结果 | 真值GT]
                    comparison = torch.cat([input_blind, output, gt_tensor], dim=3)

                    import torchvision
                    torchvision.utils.save_image(comparison, os.path.join(save_triple, f"triple_{current_frame_name}"))
                    torchvision.utils.save_image(output, os.path.join(save_pure, current_frame_name))
                else:
                    print(f"[!] 警告: 找不到 {current_frame_name} 的 GT")

                if (idx + 1) % 10 == 0:
                    print(f"进度: {idx + 1}/{len(dataloader)} | 正在保存: {current_frame_name}")

    def test_quantitative_result(self, gt_dir, output_dir, image_border):
        from utils import TestReport
        report = TestReport()

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        out_imgs = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')], key=natural_sort_key)
        gt_map = {}
        for root, _, files in os.walk(gt_dir):
            for f in files:
                if f.endswith('.png'): gt_map[f] = os.path.join(root, f)

        print(f"===> 开始定量打分，准备比对 {len(out_imgs)} 张图片...")
        for img_name in out_imgs:
            out_path = os.path.join(output_dir, img_name)
            gt_path = gt_map.get(img_name)
            if gt_path and os.path.exists(out_path):
                out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_img is not None and out_img is not None:
                    if out_img.shape != gt_img.shape:
                        out_img = cv2.resize(out_img, (gt_img.shape[1], gt_img.shape[0]))
                    report.update_metric(gt_img, out_img, img_name)
        report.print_final_result()

    def save_checkpoint(self, epoch):
        save_dict = {'epoch': epoch, 'model_D_state_dict': self.model.degradation_learning_network.state_dict(),
                     'optimizer_D_state_dict': self.optimizer_D.state_dict()}
        if self.config.stage == 2:
            save_dict.update({'model_R_state_dict': self.model.restoration_network.state_dict(),
                              'optimizer_R_state_dict': self.optimizer_R.state_dict()})
        torch.save(save_dict, os.path.join(self.checkpoint_path, 'latest.pt'))

    def save_best_model(self, epoch):
        save_dict = {'epoch': epoch, 'model_D_state_dict': self.model.degradation_learning_network.state_dict()}
        if self.config.stage == 2: save_dict['model_R_state_dict'] = self.model.restoration_network.state_dict()
        torch.save(save_dict, os.path.join(self.checkpoint_path, 'model_best.pt'))

    def load_checkpoint(self):
        latest_path = os.path.join(self.checkpoint_path, 'latest.pt')
        if os.path.exists(latest_path):
            ckpt = torch.load(latest_path)
            self.model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
            if self.config.stage == 2 and 'model_R_state_dict' in ckpt:
                self.model.restoration_network.load_state_dict(ckpt['model_R_state_dict'])
            return ckpt['epoch']
        return 0

    def load_best_model(self):
        best_path = os.path.join(self.checkpoint_path, 'model_best.pt')
        ckpt = torch.load(best_path)
        self.model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
        if self.config.stage == 2: self.model.restoration_network.load_state_dict(ckpt['model_R_state_dict'])
        print("[*] Loaded Best Model.")

    def load_best_stage1_model(self):
        stage1_path = self.checkpoint_path.replace('stage2', 'stage1')
        best_path = os.path.join(stage1_path, 'model_best.pt')
        ckpt = torch.load(best_path)
        self.model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
        print(f"[*] Loaded Stage 1 Best Net_D from {best_path}")
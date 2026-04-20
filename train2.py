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

        # 图像保存管理器
        if self.config.save_train_img:
            self.save_manager = SaveManager(config)

        # 基础损失函数
        self.criterion = nn.L1Loss()

        # --- 针对微调 (Fine-tuning) 的动态调整 ---
        # 如果是微调阶段，里程碑应设得更早
        if self.config.stage == 1:
            milestones = [80, 120, 150]
            initial_lr = self.config.lr * 0.5  # 微调建议降低初始学习率
        else:
            milestones = [260, 360, 380, 390]
            initial_lr = self.config.lr

        # 优化器与调度器：退化学习网络 (Net_D)
        self.optimizer_D = torch.optim.Adam(
            self.model.degradation_learning_network.parameters(),
            lr=initial_lr
        )
        self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_D, milestones=milestones, gamma=0.5
        )

        # 优化器与调度器：恢复网络 (Net_R)
        if self.config.stage == 2:
            self.optimizer_R = torch.optim.Adam(
                self.model.restoration_network.parameters(),
                lr=initial_lr
            )
            self.scheduler_R = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_R, milestones=milestones, gamma=0.5
            )

        # 权重保存路径
        self.checkpoint_path = os.path.join(self.config.save_dir, f'model_stage{self.config.stage}')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.model.cuda()

    def smart_recon_loss(self, recon, target):
        """
        核心修正：针对大圆斑点调整了 Top-k 的比例 (0.5% -> 2.0%)
        """
        abs_diff = torch.abs(recon - target)
        with torch.no_grad():
            flat_diff = abs_diff.view(-1)
            # Use configurable top-k fraction (fallback to 0.02 for legacy behaviour in this file)
            topk_frac = float(getattr(self.config, 'smart_blind_topk_frac', 0.02))
            k = max(1, int(flat_diff.numel() * topk_frac))
            threshold, _ = torch.topk(flat_diff, k)
            min_topk_val = threshold[-1]
            blind_mask = (abs_diff >= min_topk_val).float()

        l1_loss = abs_diff
        l2_loss = (recon - target) ** 2
        # Use configurable L2 scale (fallback to 500 for legacy behaviour here)
        l2_scale = float(getattr(self.config, 'smart_blind_l2_scale', 500.0))
        weighted_loss = torch.where(blind_mask > 0.5, l2_scale * l2_loss, 1.0 * l1_loss)
        return weighted_loss.mean()

    def train(self, dataloader, train_log, global_step):
        self.model.train()
        report = Train_Report()
        start = time.time()

        for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow) in enumerate(dataloader):
            lr_blur_seq = lr_blur_seq.cuda()
            hr_sharp_seq = hr_sharp_seq.cuda()
            lr_sharp_seq = lr_sharp_seq.cuda()
            flow = flow.cuda()

            result_dict = self.model(lr_blur_seq, hr_sharp_seq)
            batch_size, _, t, _, _ = lr_blur_seq.shape

            if self.config.stage == 1:
                # Stage 1: 训练退化模型
                recon_loss = self.smart_recon_loss(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                hr_warping_loss = self.config.hr_warping_loss_weight * self.criterion(
                    result_dict['hr_warp'], hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))

                b, _, t, h, w = result_dict['image_flow'].size()
                flow_scale = float(getattr(self.config, 'flow_loss_scale', 10.0))
                flow_loss = flow_scale * self.config.flow_loss_weight * self.criterion(result_dict['image_flow'],
                                                                                 flow.view(b, 2, t, h, w))
                D_TA_loss = self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                total_loss = recon_loss + hr_warping_loss + flow_loss + D_TA_loss

                self.optimizer_D.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer_D.step()

                report.update(batch_size, 0, recon_loss.item(), hr_warping_loss.item(), 0, flow_loss.item(),
                              D_TA_loss.item(), 0, total_loss.item())

            elif self.config.stage == 2:
                # Stage 2: 训练修复模型
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
                    # 实时保存中间结果图，用于观察 Net_D 是否学会了“画圆”
                    src = [lr_blur_seq[:, :, t // 2, :, :], result_dict['recon']]
                    if self.config.stage == 2:
                        src.extend([result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :]])
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
        # ... 原有 test 推理逻辑保持不变 ...
        pass

    def test_quantitative_result(self, gt_dir, output_dir, image_border):
        # ... 原有打分并保存文件的逻辑保持不变 ...
        pass

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
            print(f"[*] Resumed from epoch {ckpt['epoch']}")
            return ckpt['epoch']
        return 0

    def load_best_model(self):
        best_path = os.path.join(self.checkpoint_path, 'model_best.pt')
        if os.path.exists(best_path):
            ckpt = torch.load(best_path)
            self.model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
            if self.config.stage == 2: self.model.restoration_network.load_state_dict(ckpt['model_R_state_dict'])
            print("[*] Loaded Best Model.")

    def load_best_stage1_model(self):
        # 这个方法专门用于 Stage 2 开始前加载 Stage 1 的训练成果
        stage1_path = self.checkpoint_path.replace('stage2', 'stage1')
        best_path = os.path.join(stage1_path, 'model_best.pt')
        if os.path.exists(best_path):
            ckpt = torch.load(best_path)
            self.model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
            print(f"[*] Loaded Stage 1 Best Net_D from {best_path}")
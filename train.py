import os
import time
import torch
import torch.nn as nn
import cv2
import numpy as np
from utils import Train_Report, TestReport, SaveManager


class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

        # 图像保存管理器（用于训练中查看中间结果）
        if self.config.save_train_img:
            self.save_manager = SaveManager(config)

        # 损失函数
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

    # ----------------------------------------------------------------------
    # 核心训练逻辑
    # ----------------------------------------------------------------------
    def train(self, dataloader, train_log, global_step):
        self.model.train()
        report = Train_Report()
        start = time.time()

        for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow) in enumerate(dataloader):
            lr_blur_seq = lr_blur_seq.cuda()
            hr_sharp_seq = hr_sharp_seq.cuda()
            lr_sharp_seq = lr_sharp_seq.cuda()
            flow = flow.cuda()

            # 前向传播
            result_dict = self.model(lr_blur_seq, hr_sharp_seq)
            batch_size, _, t, _, _ = lr_blur_seq.shape

            if self.config.stage == 1:
                # Stage 1: 仅训练退化支路
                recon_loss = self.criterion(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                hr_warping_loss = self.config.hr_warping_loss_weight * self.criterion(
                    result_dict['hr_warp'],
                    hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1])
                )
                flow_loss = self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                D_TA_loss = self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                total_loss = recon_loss + hr_warping_loss + flow_loss + D_TA_loss

                self.optimizer_D.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer_D.step()

                report.update(batch_size, 0, recon_loss.item(), hr_warping_loss.item(), 0,
                              flow_loss.item(), D_TA_loss.item(), 0, total_loss.item())

            elif self.config.stage == 2:
                # Stage 2: 联合训练
                restoration_loss = self.criterion(result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :])
                recon_loss = self.config.Net_D_weight * self.criterion(result_dict['recon'],
                                                                       lr_blur_seq[:, :, t // 2, :, :])
                lr_warping_loss = self.config.lr_warping_loss_weight * self.criterion(
                    result_dict['lr_warp'],
                    lr_blur_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1])
                )
                hr_warping_loss = self.config.Net_D_weight * self.config.hr_warping_loss_weight * self.criterion(
                    result_dict['hr_warp'],
                    hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1])
                )
                flow_loss = self.config.Net_D_weight * self.config.flow_loss_weight * self.criterion(
                    result_dict['image_flow'], flow)
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
                period_time = time.time() - start
                train_log.write(f"[{global_step}]\t" + report.result_str(lr_D, lr_R, period_time))
                start = time.time()

                if self.config.save_train_img:
                    src = [lr_blur_seq[:, :, t // 2, :, :], result_dict['recon']]
                    if self.config.stage == 2:
                        src.extend([result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :]])
                    self.save_manager.save_batch_images(src, batch_size, global_step)
                report = Train_Report()

        self.scheduler_D.step()
        if self.config.stage == 2:
            self.scheduler_R.step()
        return global_step

    # ----------------------------------------------------------------------
    # 验证逻辑 (Validate)
    # ----------------------------------------------------------------------
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
                    report.update_restoration_metric(result_dict['output'], hr_sharp_seq[:, :, hr_sharp_seq.shape[2] // 2, :, :])

        val_log.write(f"VAL Epoch [{epoch}]\t" + report.val_result_str(time.time() - start))
        return report.psnr if self.config.stage == 2 else report.recon_psnr

    # ----------------------------------------------------------------------
    # 测试逻辑 (Test)
    # ----------------------------------------------------------------------
    def test(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for idx, (lr_blur_seq, relative_path) in enumerate(dataloader):
                lr_blur_seq = lr_blur_seq.cuda()
                result_dict = self.model(lr_blur_seq)
                output = result_dict['output'].squeeze(0)
                output = (output * 255.0).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
                path_info = relative_path[0]
                save_fn = os.path.join(self.config.save_dir, 'test', path_info)
                os.makedirs(os.path.dirname(save_fn), exist_ok=True)
                output = output.squeeze()  # 变成 (H, W)
                cv2.imwrite(save_fn, output)

    def test_quantitative_result(self, gt_dir, output_dir, image_border):
        report = TestReport()
        # 注意：此处逻辑需根据你的 val_sharp 实际子目录结构微调
        scenes = sorted(os.listdir(output_dir))
        for scene in scenes:
            scene_path = os.path.join(output_dir, scene)
            if not os.path.isdir(scene_path): continue
            imgs = sorted(os.listdir(scene_path))
            for img_name in imgs:
                out_img = cv2.imread(os.path.join(scene_path, img_name), cv2.IMREAD_GRAYSCALE)
                gt_img = cv2.imread(os.path.join(gt_dir, scene, img_name), cv2.IMREAD_GRAYSCALE)
                if gt_img is not None:
                    report.update_metric(gt_img, out_img, img_name)
        report.print_final_result()

    # ----------------------------------------------------------------------
    # 权重管理
    # ----------------------------------------------------------------------
    def save_checkpoint(self, epoch):
        save_dict = {
            'epoch': epoch,
            'model_D_state_dict': self.model.degradation_learning_network.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }
        if self.config.stage == 2:
            save_dict['model_R_state_dict'] = self.model.restoration_network.state_dict()
            save_dict['optimizer_R_state_dict'] = self.optimizer_R.state_dict()

        torch.save(save_dict, os.path.join(self.checkpoint_path, 'latest.pt'))

    def save_best_model(self, epoch):
        save_dict = {'epoch': epoch, 'model_D_state_dict': self.model.degradation_learning_network.state_dict()}
        if self.config.stage == 2:
            save_dict['model_R_state_dict'] = self.model.restoration_network.state_dict()
        torch.save(save_dict, os.path.join(self.checkpoint_path, 'model_best.pt'))

    def load_checkpoint(self):
        latest_path = os.path.join(self.checkpoint_path, 'latest.pt')
        if os.path.exists(latest_path):
            ckpt = torch.load(latest_path)
            self.model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
            if self.config.stage == 2 and 'model_R_state_dict' in ckpt:
                self.model.restoration_network.load_state_dict(ckpt['model_R_state_dict'])
            print(f"[*] Loaded checkpoint from epoch {ckpt['epoch']}")
            return ckpt['epoch']
        return 0

    def load_best_model(self):
        best_path = os.path.join(self.checkpoint_path, 'model_best.pt')
        ckpt = torch.load(best_path)
        self.model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
        if self.config.stage == 2:
            self.model.restoration_network.load_state_dict(ckpt['model_R_state_dict'])
        print("[*] Loaded Best Model.")

    def load_best_stage1_model(self):
        stage1_path = self.checkpoint_path.replace('stage2', 'stage1')
        best_path = os.path.join(stage1_path, 'model_best.pt')
        ckpt = torch.load(best_path)
        self.model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
        print(f"[*] Loaded Stage 1 Best Net_D from {best_path}")
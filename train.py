import os
import time
import csv
from contextlib import nullcontext
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

        # 学习率调度：
        # 1) 若开启动态模式：drop_epoch前保持base lr，之后自动cosine衰减到dynamic_min_lr。
        # 2) 否则保持原有策略（单次降lr或默认多里程碑）。
        default_milestones = [260, 360, 380, 390]
        milestones = default_milestones
        gamma = 0.5
        use_dynamic_after_drop = bool(getattr(self.config, 'lr_dynamic_after_drop', False))
        lr_drop_epoch = int(getattr(self.config, 'lr_drop_epoch', -1))
        lr_after_drop = float(getattr(self.config, 'lr_after_drop', 0.0))

        if use_dynamic_after_drop and lr_drop_epoch > 0 and self.config.num_epochs > lr_drop_epoch and self.config.lr > 0:
            start_lr = lr_after_drop if lr_after_drop > 0 else self.config.lr
            min_lr = float(getattr(self.config, 'dynamic_min_lr', 1e-6))
            start_factor = max(1e-8, min(1.0, start_lr / self.config.lr))
            min_factor = max(1e-8, min(start_factor, min_lr / self.config.lr))
            denom = max(1, self.config.num_epochs - lr_drop_epoch - 1)

            def _lr_lambda(epoch_idx):
                # epoch_idx是scheduler内部已完成的epoch计数；这样可保证drop前训练轮次保持固定学习率。
                if epoch_idx < lr_drop_epoch:
                    return 1.0
                t = min(1.0, max(0.0, (epoch_idx - lr_drop_epoch) / denom))
                return float(min_factor + 0.5 * (start_factor - min_factor) * (1.0 + np.cos(np.pi * t)))

            self._lr_scheduler_dynamic = True
        else:
            self._lr_scheduler_dynamic = False
            if lr_drop_epoch > 0 and lr_after_drop > 0 and self.config.lr > 0:
                milestones = [lr_drop_epoch]
                gamma = max(1e-8, min(1.0, lr_after_drop / self.config.lr))

        # 优化器与调度器：退化学习网络 (Net_D)
        self.optimizer_D = torch.optim.Adam(
            self.model.degradation_learning_network.parameters(),
            lr=self.config.lr
        )
        if self._lr_scheduler_dynamic:
            self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=_lr_lambda)
        else:
            self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_D, milestones=milestones, gamma=gamma
            )

        # 优化器与调度器：恢复网络 (Net_R) - 仅在 Stage 2 开启
        if self.config.stage == 2:
            self.optimizer_R = torch.optim.Adam(
                self.model.restoration_network.parameters(),
                lr=self.config.lr
            )
            if self._lr_scheduler_dynamic:
                self.scheduler_R = torch.optim.lr_scheduler.LambdaLR(self.optimizer_R, lr_lambda=_lr_lambda)
            else:
                self.scheduler_R = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer_R, milestones=milestones, gamma=gamma
                )

        # 权重保存路径
        self.checkpoint_path = os.path.join(self.config.save_dir, f'model_stage{self.config.stage}')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.model.cuda()

        # fast:Mixed precision can significantly improve throughput on modern GPUs.
        self.use_amp = bool(getattr(self.config, 'use_amp', True))
        amp_dtype = str(getattr(self.config, 'amp_dtype', 'fp16')).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=self.use_amp,
            init_scale=float(getattr(self.config, 'amp_init_scale', 65536.0)),
            growth_interval=int(getattr(self.config, 'amp_growth_interval', 2000)),
            backoff_factor=float(getattr(self.config, 'amp_backoff_factor', 0.5))
        )

        # Stability-first knobs: these protect long stage-2 training from late-epoch NaN divergence.
        self.grad_clip_norm = float(getattr(self.config, 'grad_clip_norm', 0.0))
        self.overflow_patience = int(getattr(self.config, 'overflow_patience', 2))
        self.lr_overflow_decay = float(getattr(self.config, 'lr_overflow_decay', 0.5))
        self.min_lr = float(getattr(self.config, 'min_lr', 1e-6))
        self.overflow_lr_decay_cooldown = int(getattr(self.config, 'overflow_lr_decay_cooldown', 200))
        self.overflow_log_interval = int(getattr(self.config, 'overflow_log_interval', 50))
        self.amp_recovery_steps = int(getattr(self.config, 'amp_recovery_steps', 128))
        self.overflow_streak = 0
        self.skipped_non_finite = 0
        self.force_fp32_steps = 0
        self.last_lr_decay_step = -10 ** 9
        self.last_overflow_log_step = -10 ** 9

    def _clip_gradients(self):
        if self.grad_clip_norm <= 0:
            return
        nn.utils.clip_grad_norm_(self.model.degradation_learning_network.parameters(), self.grad_clip_norm)
        if self.config.stage == 2:
            nn.utils.clip_grad_norm_(self.model.restoration_network.parameters(), self.grad_clip_norm)

    def _all_grads_finite(self):
        modules = [self.model.degradation_learning_network]
        if self.config.stage == 2:
            modules.append(self.model.restoration_network)

        for module in modules:
            for p in module.parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    return False
        return True

    def _decay_optimizer_lr(self, optimizer):
        changed = False
        for group in optimizer.param_groups:
            current_lr = group['lr']
            new_lr = max(self.min_lr, current_lr * self.lr_overflow_decay)
            if new_lr < current_lr:
                changed = True
            group['lr'] = new_lr
        return changed

    def _handle_overflow(self, train_log, global_step, reason):
        self.overflow_streak += 1
        self.skipped_non_finite += 1

        should_log = (global_step - self.last_overflow_log_step) >= self.overflow_log_interval
        if self.overflow_streak < self.overflow_patience:
            if should_log:
                train_log.write(
                    f"[Stability] step={global_step} reason={reason} "
                    f"overflow_streak={self.overflow_streak}/{self.overflow_patience}"
                )
                self.last_overflow_log_step = global_step
            return

        can_decay = (global_step - self.last_lr_decay_step) >= self.overflow_lr_decay_cooldown
        lr_changed = False
        if can_decay:
            lr_changed = self._decay_optimizer_lr(self.optimizer_D)
            if self.config.stage == 2:
                lr_changed = self._decay_optimizer_lr(self.optimizer_R) or lr_changed
            if lr_changed:
                self.last_lr_decay_step = global_step

        if should_log or lr_changed:
            lr_d = self.optimizer_D.param_groups[0]['lr']
            lr_r = self.optimizer_R.param_groups[0]['lr'] if self.config.stage == 2 else 0.0
            msg = (
                f"[Stability] step={global_step} reason={reason} overflow_streak={self.overflow_streak} "
                f"-> LR_D={lr_d:.7f}, LR_R={lr_r:.7f}"
            )
            if not can_decay:
                msg += f" (decay_cooldown={self.overflow_lr_decay_cooldown})"
            elif not lr_changed:
                msg += " (already_at_min_lr)"
            train_log.write(msg)
            self.last_overflow_log_step = global_step

        self.overflow_streak = 0

    def _is_finite_result(self, result_dict):
        for value in result_dict.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                return False
        return True

    def smart_recon_loss(self, recon, target):
        # Run hard blind-pixel weighting in fp32 to reduce overflow risk under AMP.
        recon_f = recon.float()
        target_f = target.float()
        abs_diff = torch.abs(recon_f - target_f)
        with torch.no_grad():
            flat_diff = abs_diff.view(-1)
            # fast2:kthvalue is cheaper than topk for obtaining only the threshold.
            # Use configurable top-k fraction (smart_blind_topk_frac) to determine blind pixels.
            topk_frac = float(getattr(self.config, 'smart_blind_topk_frac', 0.005))
            k = max(1, int(flat_diff.numel() * topk_frac))
            kth = max(1, flat_diff.numel() - k + 1)
            min_topk_val = torch.kthvalue(flat_diff, kth).values
            blind_mask = (abs_diff >= min_topk_val).float()

        l1_loss = abs_diff
        l2_loss = (recon_f - target_f) ** 2
        # Use configurable L2 scaling for blind pixels (smart_blind_l2_scale).
        l2_scale = float(getattr(self.config, 'smart_blind_l2_scale', 1000.0))
        weighted_loss = torch.where(blind_mask > 0.5, l2_scale * l2_loss, 1.0 * l1_loss)
        return weighted_loss.mean()

    def _tensor_to_u8(self, x):
        x = x.detach().float().clamp(0.0, 1.0)
        if x.dim() == 3:
            x = x[0]
        x = (x * 255.0).round().to(torch.uint8).cpu().numpy()
        return x

    def _build_blind_masks(self, lr_center, hr_center):
        # 基于中心帧差异构造盲元/非盲元掩码，用于避免把污染区域当作监督目标。
        if lr_center.shape[-2:] != hr_center.shape[-2:]:
            hr_center = F.interpolate(hr_center, size=lr_center.shape[-2:], mode='bilinear', align_corners=False)

        blind_thr = getattr(self.config, 'blind_mask_threshold', 0.08)
        diff = torch.abs(lr_center - hr_center)
        blind_mask = (diff >= blind_thr).float()
        non_blind_mask = 1.0 - blind_mask
        return blind_mask, non_blind_mask

    def _save_align_vis(self, epoch, idx, ref_seq, warped_seq):
        """保存一张对齐前后差分图，便于快速判断光流对齐是否有效。"""
        if ref_seq.shape[2] < 2:
            return

        t = ref_seq.shape[2]
        c_idx = t // 2
        n_idx = c_idx - 1 if c_idx > 0 else c_idx + 1

        ref = ref_seq[0, :, n_idx, :, :]
        warped = warped_seq[0, :, n_idx, :, :]
        center = ref_seq[0, :, c_idx, :, :]

        before_diff = torch.abs(ref - center)
        after_diff = torch.abs(warped - center)

        ref_img = self._tensor_to_u8(ref)
        warped_img = self._tensor_to_u8(warped)
        center_img = self._tensor_to_u8(center)
        before_img = self._tensor_to_u8(before_diff)
        after_img = self._tensor_to_u8(after_diff)

        panel = np.concatenate([ref_img, warped_img, center_img, before_img, after_img], axis=1)
        save_dir = os.path.join(self.config.save_dir, 'align_vis')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'epoch{epoch + 1:04d}_iter{idx + 1:04d}.png')
        cv2.imwrite(save_path, panel)

    def _compute_align_metrics(self, result_dict, lr_blur_seq, hr_sharp_seq, flow):
        """计算对齐前后误差、改进率和EPE。"""
        if 'lr_warp' in result_dict:
            ref_seq = lr_blur_seq
            warped_seq = result_dict['lr_warp']
        elif 'hr_warp' in result_dict:
            ref_seq = hr_sharp_seq
            warped_seq = result_dict['hr_warp']
        else:
            return None

        t = ref_seq.shape[2]
        c_idx = t // 2
        nbr_idx = [i for i in range(t) if i != c_idx]
        if len(nbr_idx) == 0:
            return None

        center = ref_seq[:, :, c_idx:c_idx + 1, :, :]
        center_rep = center.repeat(1, 1, len(nbr_idx), 1, 1)

        ref_neighbors = ref_seq[:, :, nbr_idx, :, :]
        warped_neighbors = warped_seq[:, :, nbr_idx, :, :]

        before_l1 = torch.mean(torch.abs(ref_neighbors - center_rep))
        after_l1 = torch.mean(torch.abs(warped_neighbors - center_rep))
        gain_pct = (before_l1 - after_l1) / (before_l1 + 1e-8) * 100.0

        blind_before_l1 = None
        blind_after_l1 = None
        blind_gain_pct = None
        blind_ratio = None

        # Stage-2 重点指标：只在中心帧盲元区域评估对齐收益，区分“全图好看”与“修补区有效”。
        if 'lr_warp' in result_dict:
            center_lr = lr_blur_seq[:, :, c_idx:c_idx + 1, :, :]
            center_hr = hr_sharp_seq[:, :, c_idx:c_idx + 1, :, :]
            blind_mask, _ = self._build_blind_masks(center_lr, center_hr)
            blind_ratio = blind_mask.mean().item()

            blind_mask_rep = blind_mask.repeat(1, 1, len(nbr_idx), 1, 1)
            blind_count = blind_mask_rep.sum()
            if blind_count.item() > 0:
                center_hr_rep = center_hr.repeat(1, 1, len(nbr_idx), 1, 1)
                blind_before_l1 = (torch.abs(ref_neighbors - center_hr_rep) * blind_mask_rep).sum() / blind_count
                blind_after_l1 = (torch.abs(warped_neighbors - center_hr_rep) * blind_mask_rep).sum() / blind_count
                blind_gain_pct = (blind_before_l1 - blind_after_l1) / (blind_before_l1 + 1e-8) * 100.0

        epe = None
        if 'image_flow' in result_dict:
            pred_flow = result_dict['image_flow']
            gt_flow = flow.view_as(pred_flow)
            epe = torch.norm(pred_flow - gt_flow, dim=1).mean()

        return {
            'before_l1': before_l1.item(),
            'after_l1': after_l1.item(),
            'gain_pct': gain_pct.item(),
            'epe': None if epe is None else epe.item(),
            'blind_before_l1': None if blind_before_l1 is None else blind_before_l1.item(),
            'blind_after_l1': None if blind_after_l1 is None else blind_after_l1.item(),
            'blind_gain_pct': None if blind_gain_pct is None else blind_gain_pct.item(),
            'blind_ratio': blind_ratio,
            'batch_size': ref_seq.shape[0],
            'ref_seq': ref_seq,
            'warped_seq': warped_seq
        }

    def train(self, dataloader, train_log, global_step):
        self.model.train()
        report = Train_Report()
        start = time.time()

        for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow) in enumerate(dataloader):
            lr_blur_seq = lr_blur_seq.cuda(non_blocking=True)
            hr_sharp_seq = hr_sharp_seq.cuda(non_blocking=True)
            lr_sharp_seq = lr_sharp_seq.cuda(non_blocking=True)
            flow = flow.cuda(non_blocking=True)
            batch_size, _, t, _, _ = lr_blur_seq.shape

            # Stage-2 uses an explicit blind mask from GT difference; stage-1 keeps original forward.
            blind_mask = None
            if self.config.stage == 2:
                center_lr_for_mask = lr_blur_seq[:, :, t // 2:t // 2 + 1, :, :]
                center_hr_for_mask = hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :]
                blind_mask, _ = self._build_blind_masks(center_lr_for_mask, center_hr_for_mask)

            amp_enabled = self.use_amp and self.force_fp32_steps <= 0
            amp_ctx = torch.amp.autocast('cuda', dtype=self.amp_dtype) if amp_enabled else nullcontext()

            with amp_ctx:
                result_dict = self.model(lr_blur_seq, hr_sharp_seq, blind_mask=blind_mask)

                if not self._is_finite_result(result_dict):
                    if amp_enabled:
                        self.force_fp32_steps = max(self.force_fp32_steps, self.amp_recovery_steps)
                    self.optimizer_D.zero_grad(set_to_none=True)
                    if self.config.stage == 2:
                        self.optimizer_R.zero_grad(set_to_none=True)
                    self._handle_overflow(train_log, global_step, 'non_finite_model_output')
                    global_step += 1
                    continue

                if self.force_fp32_steps > 0:
                    self.force_fp32_steps -= 1

                if self.config.stage == 1:
                    recon_loss = self.smart_recon_loss(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                    center_hr = hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :]
                    hr_target = center_hr.expand(-1, -1, t, -1, -1)
                    hr_warping_loss = self.config.hr_warping_loss_weight * self.criterion(
                        result_dict['hr_warp'].float(), hr_target.float())

                    b, _, t, h, w = result_dict['image_flow'].size()
                    # flow loss scale moved to config: flow_loss_scale (default 10.0)
                    flow_scale = float(getattr(self.config, 'flow_loss_scale', getattr(self.config, 'flow_loss_scale', 10.0)))
                    flow_loss = flow_scale * self.config.flow_loss_weight * self.criterion(
                        result_dict['image_flow'].float(), flow.view(b, 2, t, h, w).float()
                    )
                    D_TA_loss = self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'].float(),
                                                                              lr_sharp_seq.float())

                    total_loss = recon_loss + hr_warping_loss + flow_loss + D_TA_loss
                    self.optimizer_D.zero_grad(set_to_none=True)

                elif self.config.stage == 2:
                    center_hr_2d = hr_sharp_seq[:, :, t // 2, :, :]
                    center_out = result_dict['output']

                    # 2D 盲元掩码与计数（用于按盲元像素平均）
                    blind_mask_2d = blind_mask[:, :, 0, :, :].float()
                    blind_pixels = blind_mask_2d.sum().clamp_min(1.0)
                    non_blind_mask_2d = 1.0 - blind_mask_2d
                    non_blind_pixels = non_blind_mask_2d.sum().clamp_min(1.0)

                    # restoration_loss: 仅在非盲元区域计算（按像素平均），避免与盲元分支冲突
                    restoration_loss = (torch.abs(center_out - center_hr_2d) * non_blind_mask_2d).sum() / non_blind_pixels

                    # blind_res_loss: 对 blind_res 分支只在盲区做平均残差监督（按盲元像素平均）
                    with torch.no_grad():
                        target_res = center_hr_2d - result_dict['base_output']
                    masked_res_abs = torch.abs(result_dict['blind_res'] - target_res) * blind_mask_2d
                    blind_res_loss = masked_res_abs.sum() / blind_pixels

                    # recon_loss 保持不变（Net_D 的重建监督）
                    recon_loss = self.config.Net_D_weight * self.smart_recon_loss(
                        result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :]
                    )

                    # blind_restore_loss: final output 在盲区的平均 L1（保持做为评估与损失项）
                    blind_l1 = (torch.abs(center_out - center_hr_2d) * blind_mask_2d).sum() / blind_pixels
                    blind_restore_loss = self.config.blind_restore_loss_weight * blind_l1

                    # lr_warping_loss: 仍使用非盲区作为监督（和原逻辑一致）
                    center_lr = lr_blur_seq[:, :, t // 2:t // 2 + 1, :, :]
                    non_blind_mask = 1.0 - blind_mask
                    target_lr = center_lr.expand(-1, -1, t, -1, -1)
                    non_blind_mask_t = non_blind_mask.expand(-1, -1, t, -1, -1)
                    valid_non_blind = non_blind_mask_t.sum().clamp_min(1.0)
                    masked_l1 = (torch.abs(result_dict['lr_warp'].float() - target_lr.float()) * non_blind_mask_t.float()).sum() / valid_non_blind
                    lr_warping_loss = self.config.lr_warping_loss_weight * masked_l1

                    hr_target = hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].expand(-1, -1, t, -1, -1)
                    hr_warping_loss = self.config.Net_D_weight * self.config.hr_warping_loss_weight * self.criterion(
                        result_dict['hr_warp'].float(), hr_target.float()
                    )

                    b, _, t, h, w = result_dict['image_flow'].size()
                    # Apply flow_loss_scale consistently in stage 2 as well
                    flow_scale = float(getattr(self.config, 'flow_loss_scale', 10.0))
                    flow_loss = self.config.Net_D_weight * flow_scale * self.config.flow_loss_weight * self.criterion(
                        result_dict['image_flow'].float(), flow.view(b, 2, t, h, w).float()
                    )

                    R_TA_loss = self.config.R_TA_loss_weight * self.criterion(result_dict['F_sharp_R'].float(), lr_sharp_seq.float())
                    D_TA_loss = self.config.Net_D_weight * self.config.D_TA_loss_weight * self.criterion(
                        result_dict['F_sharp_D'].float(), lr_sharp_seq.float()
                    )

                    # 使用配置项 blind_res_loss_weight（fallback 到 2.0）替代代码中硬编码的 2.0
                    blind_res_loss_weight = float(getattr(self.config, 'blind_res_loss_weight', 2.0))

                    # 汇总总损失（restoration 在非盲区，盲区由 blind_res + blind_restore_loss 专门驱动）
                    total_loss = (
                        restoration_loss
                        + blind_restore_loss
                        + (blind_res_loss * blind_res_loss_weight)
                        + recon_loss + hr_warping_loss + lr_warping_loss
                        + flow_loss + R_TA_loss + D_TA_loss
                    )

                    self.optimizer_D.zero_grad(set_to_none=True)
                    self.optimizer_R.zero_grad(set_to_none=True)

                # 反向传播逻辑
                if not torch.isfinite(total_loss):
                    self.optimizer_D.zero_grad(set_to_none=True)
                    if self.config.stage == 2: self.optimizer_R.zero_grad(set_to_none=True)
                    if amp_enabled:
                        self.force_fp32_steps = max(self.force_fp32_steps, self.amp_recovery_steps)
                    self._handle_overflow(train_log, global_step, f'non_finite_loss_stage{self.config.stage}')
                    global_step += 1
                    continue

                if amp_enabled:
                    prev_scale = self.scaler.get_scale()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer_D)
                    if self.config.stage == 2: self.scaler.unscale_(self.optimizer_R)
                    self._clip_gradients()
                    if not self._all_grads_finite():
                        self.optimizer_D.zero_grad(set_to_none=True)
                        if self.config.stage == 2: self.optimizer_R.zero_grad(set_to_none=True)
                        self.scaler.update()
                        self._handle_overflow(train_log, global_step, f'non_finite_grad_stage{self.config.stage}')
                        global_step += 1
                        continue
                    self.scaler.step(self.optimizer_D)
                    if self.config.stage == 2: self.scaler.step(self.optimizer_R)
                    self.scaler.update()
                    if self.scaler.get_scale() < prev_scale:
                        self._handle_overflow(train_log, global_step, f'amp_overflow_stage{self.config.stage}')
                    else:
                        self.overflow_streak = 0
                else:
                    total_loss.backward()
                    self._clip_gradients()
                    if not self._all_grads_finite():
                        self.optimizer_D.zero_grad(set_to_none=True)
                        if self.config.stage == 2: self.optimizer_R.zero_grad(set_to_none=True)
                        self._handle_overflow(train_log, global_step, f'non_finite_grad_stage{self.config.stage}')
                        global_step += 1
                        continue
                    self.optimizer_D.step()
                    if self.config.stage == 2: self.optimizer_R.step()
                    self.overflow_streak = 0

                if self.config.stage == 1:
                    report.update(batch_size, 0, recon_loss.item(), hr_warping_loss.item(), 0, flow_loss.item(),
                                  D_TA_loss.item(), 0, total_loss.item())
                else:
                    report.update(batch_size, restoration_loss.item(), recon_loss.item(), hr_warping_loss.item(),
                                  lr_warping_loss.item(), flow_loss.item(), D_TA_loss.item(), R_TA_loss.item(),
                                  total_loss.item())

            # 进度记录与可视化保存
            global_step += 1
            if global_step % 100 == 0 or idx == len(dataloader) - 1:
                lr_D = self.optimizer_D.param_groups[0]['lr']
                lr_R = self.optimizer_R.param_groups[0]['lr'] if self.config.stage == 2 else 0.0
                train_log.write(f"[{global_step}]\t" + report.result_str(lr_D, lr_R, time.time() - start))
                if self.skipped_non_finite > 0:
                    train_log.write(f"[Stability] skipped_non_finite_steps={self.skipped_non_finite}")
                if self.force_fp32_steps > 0:
                    train_log.write(f"[Stability] temporary_fp32_steps_left={self.force_fp32_steps}")
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

        # Heavy alignment metrics can be computed less frequently than base val metrics.
        align_period = max(1, int(getattr(self.config, 'align_metrics_period', 1)))
        compute_align_metrics = bool(getattr(self.config, 'enable_align_metrics', True)) and (
                    (epoch + 1) % align_period == 0)

        align_before_sum = 0.0
        align_after_sum = 0.0
        align_gain_sum = 0.0
        align_epe_sum = 0.0
        align_count = 0
        epe_count = 0
        blind_before_sum = 0.0
        blind_after_sum = 0.0
        blind_gain_sum = 0.0
        blind_ratio_sum = 0.0
        blind_count = 0
        blind_abs_sum = 0.0
        blind_sq_sum = 0.0
        blind_pix_sum = 0.0

        with torch.no_grad():
            for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow) in enumerate(dataloader):
                lr_blur_seq = lr_blur_seq.cuda(non_blocking=True)
                hr_sharp_seq = hr_sharp_seq.cuda(non_blocking=True)
                lr_sharp_seq = lr_sharp_seq.cuda(non_blocking=True)
                flow = flow.cuda(non_blocking=True)
                t = hr_sharp_seq.shape[2]

                blind_mask = None
                if self.config.stage == 2:
                    center_lr_for_mask = lr_blur_seq[:, :, t // 2:t // 2 + 1, :, :]
                    center_hr_for_mask = hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :]
                    blind_mask, _ = self._build_blind_masks(center_lr_for_mask, center_hr_for_mask)

                amp_ctx = torch.amp.autocast('cuda', dtype=self.amp_dtype) if self.use_amp else nullcontext()
                with amp_ctx:
                    result_dict = self.model(lr_blur_seq, hr_sharp_seq, blind_mask=blind_mask)

                if self.config.stage == 1:
                    report.update_recon_metric(result_dict['recon'], lr_blur_seq[:, :, lr_blur_seq.shape[2] // 2, :, :])
                else:
                    report.update_restoration_metric(result_dict['output'],
                                                     hr_sharp_seq[:, :, hr_sharp_seq.shape[2] // 2, :, :])

                    # Blind-region restoration metrics
                    center_hr_2d = hr_sharp_seq[:, :, t // 2, :, :]
                    center_out_2d = result_dict['output']
                    blind_mask_2d = blind_mask[:, :, 0, :, :]
                    blind_pixels = blind_mask_2d.sum().item()
                    if blind_pixels > 0:
                        diff = center_out_2d - center_hr_2d
                        blind_abs_sum += (diff.abs() * blind_mask_2d).sum().item()
                        blind_sq_sum += ((diff ** 2) * blind_mask_2d).sum().item()
                        blind_pix_sum += blind_pixels

                # --- 核心修改：只保留修復效果可視化，刪除對齊可視化 ---
                if self.config.save_train_img and idx == 0:
                    src = [lr_blur_seq[:, :, t // 2, :, :], result_dict['recon']]
                    if self.config.stage == 2:
                        src.extend([result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :]])

                    # 使用 prefix='val' 區分訓練圖片，iter_step 傳入 epoch 方便查看
                    self.save_manager.save_batch_images(src, batch_size=lr_blur_seq.shape[0],
                                                        iter_step=epoch, prefix='val')

                # 指標計算邏輯保持不變
                align_metrics = None
                if compute_align_metrics:
                    align_metrics = self._compute_align_metrics(result_dict, lr_blur_seq, hr_sharp_seq, flow)

                if align_metrics is not None:
                    bsz = align_metrics['batch_size']
                    align_before_sum += align_metrics['before_l1'] * bsz
                    align_after_sum += align_metrics['after_l1'] * bsz
                    align_gain_sum += align_metrics['gain_pct'] * bsz
                    align_count += bsz
                    if align_metrics['epe'] is not None:
                        align_epe_sum += align_metrics['epe'] * bsz
                        epe_count += bsz
                    if align_metrics['blind_before_l1'] is not None:
                        blind_before_sum += align_metrics['blind_before_l1'] * bsz
                        blind_after_sum += align_metrics['blind_after_l1'] * bsz
                        blind_gain_sum += align_metrics['blind_gain_pct'] * bsz
                        blind_ratio_sum += (align_metrics['blind_ratio'] or 0.0) * bsz
                        blind_count += bsz

        # 日誌記錄邏輯
        log_msg = f"VAL Epoch [{epoch}]\t" + report.val_result_str(time.time() - start)
        if align_count > 0:
            align_before = align_before_sum / align_count
            align_after = align_after_sum / align_count
            align_gain = align_gain_sum / align_count
            align_msg = f"\tAlignBefore: {align_before:.6f}\tAlignAfter: {align_after:.6f}\tAlignGain(%): {align_gain:.3f}"
            if epe_count > 0:
                align_msg += f"\tEPE: {align_epe_sum / epe_count:.6f}"
            if blind_count > 0:
                align_msg += (
                    f"\tBlindRatio: {blind_ratio_sum / blind_count:.4f}"
                    f"\tBlindAlignBefore: {blind_before_sum / blind_count:.6f}"
                    f"\tBlindAlignAfter: {blind_after_sum / blind_count:.6f}"
                    f"\tBlindAlignGain(%): {blind_gain_sum / blind_count:.3f}"
                )
            log_msg += align_msg
        elif bool(getattr(self.config, 'enable_align_metrics', True)):
            log_msg += f"\tAlignMetrics: skipped (period={align_period})"

        blind_l1 = None
        blind_psnr = None
        if self.config.stage == 2 and blind_pix_sum > 0:
            blind_l1 = blind_abs_sum / blind_pix_sum
            blind_mse = blind_sq_sum / blind_pix_sum
            blind_psnr = 10.0 * np.log10(1.0 / max(blind_mse, 1e-12))
            log_msg += f"\tBlindL1: {blind_l1:.6f}\tBlindPSNR: {blind_psnr:.3f}"

        val_log.write(log_msg)

        if self.config.stage == 2:
            val_psnr_scalar = float(np.mean(report.psnr)) if len(report.psnr) > 0 else 0.0
            return {
                'val_psnr': val_psnr_scalar,
                'blind_l1': blind_l1,
                'blind_psnr': blind_psnr
            }
        return float(np.mean(report.recon_psnr)) if len(report.recon_psnr) > 0 else 0.0

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

        def load_blind_coords(csv_path):
            # 读取仿真保存的盲元坐标，并去重，确保统计稳定。
            if not os.path.exists(csv_path):
                return None
            coords = []
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
                    return None
                for row in reader:
                    try:
                        coords.append((int(float(row['x'])), int(float(row['y']))))
                    except Exception:
                        continue
            if len(coords) == 0:
                return None
            arr = np.unique(np.array(coords, dtype=np.int32), axis=0)
            return arr

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        out_imgs = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')], key=natural_sort_key)
        gt_map = {}
        for root, _, files in os.walk(gt_dir):
            for f in files:
                if f.endswith('.png'): gt_map[f] = os.path.join(root, f)

        # 盲元专项评估输入：仿真时保存的坐标文件 + 输入盲图（用于计算恢复增益）。
        test_mask_csv = getattr(
            self.config,
            'test_mask_csv',
            os.path.join(self.config.dataset_path, 'test_mask', '001', 'blind_pixel_coords.csv')
        )
        blind_coords = load_blind_coords(test_mask_csv)

        input_root = os.path.join(self.config.dataset_path, 'test_blur')
        input_map = {}
        if os.path.exists(input_root):
            for root, _, files in os.walk(input_root):
                for f in files:
                    if f.endswith('.png'):
                        input_map[f] = os.path.join(root, f)

        blind_abs_sum = 0.0
        blind_sq_sum = 0.0
        blind_abs_in_sum = 0.0
        blind_sq_in_sum = 0.0
        blind_pix_sum = 0
        per_image_logs = []

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
                    full_psnr = float(report.total_rgb_psnr[-1])
                    full_ssim = float(report.total_ssim[-1])

                    # Always log full-frame quality so every test run has complete per-image records.
                    row = {
                        'image': img_name,
                        'psnr': full_psnr,
                        'ssim': full_ssim,
                        'blind_mae': None,
                        'blind_rmse': None,
                        'blind_psnr': None,
                        'blind_mae_input': None,
                        'blind_mae_gain_abs': None,
                        'blind_mae_gain_pct': None,
                        'blind_count': 0
                    }

                    # 盲元专项评估：仅在 CSV 指定坐标统计误差，直接对应你的仿真盲元位置。
                    if blind_coords is not None:
                        h, w = gt_img.shape[:2]
                        x = blind_coords[:, 0]
                        y = blind_coords[:, 1]
                        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                        if np.any(valid):
                            x = x[valid]
                            y = y[valid]

                            gt_vals = gt_img[y, x].astype(np.float64)
                            out_vals = out_img[y, x].astype(np.float64)
                            err = out_vals - gt_vals

                            blind_abs = np.abs(err)
                            blind_sq = err ** 2

                            blind_abs_sum += float(blind_abs.sum())
                            blind_sq_sum += float(blind_sq.sum())
                            blind_pix_sum += int(len(err))

                            # 同时记录输入盲图误差，便于看“修复提升了多少”。
                            in_path = input_map.get(img_name)
                            in_mae = None
                            if in_path and os.path.exists(in_path):
                                in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                                if in_img is not None:
                                    if in_img.shape != gt_img.shape:
                                        in_img = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                                    in_vals = in_img[y, x].astype(np.float64)
                                    in_err = in_vals - gt_vals
                                    in_abs = np.abs(in_err)
                                    in_sq = in_err ** 2
                                    blind_abs_in_sum += float(in_abs.sum())
                                    blind_sq_in_sum += float(in_sq.sum())
                                    in_mae = float(in_abs.mean())

                            row.update({
                                'blind_mae': float(blind_abs.mean()),
                                'blind_rmse': float(np.sqrt(blind_sq.mean())),
                                'blind_psnr': float(10.0 * np.log10((255.0 * 255.0) / max(float(blind_sq.mean()), 1e-12))),
                                'blind_mae_input': in_mae,
                                'blind_count': int(len(err))
                            })
                            if in_mae is not None:
                                row['blind_mae_gain_abs'] = in_mae - row['blind_mae']
                                row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (in_mae + 1e-12)
                    per_image_logs.append(row)
        report.print_final_result()

        # Save full-frame + blind-region per-image metrics for every test run.
        save_blind_dir = os.path.join(self.config.save_dir, 'blind_eval')
        os.makedirs(save_blind_dir, exist_ok=True)
        save_blind_csv = os.path.join(save_blind_dir, 'test_blind_metrics.csv')
        if len(per_image_logs) > 0:
            keys = [
                'image', 'psnr', 'ssim',
                'blind_mae', 'blind_rmse', 'blind_psnr',
                'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count'
            ]
            with open(save_blind_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in per_image_logs:
                    writer.writerow(row)
            print(f"Per-image test metrics saved to: {save_blind_csv}")

        # 汇总输出盲元专项指标，便于不同模型横向比较。
        if blind_coords is not None and blind_pix_sum > 0:
            blind_mae = blind_abs_sum / blind_pix_sum
            blind_mse = blind_sq_sum / blind_pix_sum
            blind_rmse = float(np.sqrt(blind_mse))
            blind_psnr = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse, 1e-12)))

            print("===> Blind-Pixel Focused Metrics")
            print(f"BlindCoordsCSV: {test_mask_csv}")
            print(f"BlindCount(total sampled): {blind_pix_sum}")
            print(f"Blind MAE: {blind_mae:.6f} | Blind RMSE: {blind_rmse:.6f} | Blind PSNR: {blind_psnr:.3f}")

            if blind_abs_in_sum > 0:
                blind_mae_in = blind_abs_in_sum / blind_pix_sum
                blind_mse_in = blind_sq_in_sum / blind_pix_sum
                blind_psnr_in = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse_in, 1e-12)))
                gain_abs = blind_mae_in - blind_mae
                gain_pct = 100.0 * gain_abs / (blind_mae_in + 1e-12)
                print(
                    f"Input Blind MAE: {blind_mae_in:.6f} | Input Blind PSNR: {blind_psnr_in:.3f} | "
                    f"MAE Gain: {gain_abs:.6f} ({gain_pct:.2f}%)"
                )

            if len(per_image_logs) > 0:
                print(f"Blind per-image metrics saved to: {save_blind_csv}")

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
        # Prefer explicit stage1_pretrained_dir so Stage2 can save to a new folder safely.
        if getattr(self.config, 'stage1_pretrained_dir', ''):
            stage1_path = os.path.join(self.config.stage1_pretrained_dir, 'model_stage1')
        else:
            stage1_path = self.checkpoint_path.replace('stage2', 'stage1')
        best_path = os.path.join(stage1_path, 'model_best.pt')
        ckpt = torch.load(best_path)
        self.model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
        print(f"[*] Loaded Stage 1 Best Net_D from {best_path}")
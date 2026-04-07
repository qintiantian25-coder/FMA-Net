import os
import sys
import argparse
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from data_blindpixel import get_dataset
from model import FMANet
from scripts.gate_config import DEFAULT_GATE_CONFIG_PATH, pick_value, read_gate_section
from scripts.fusion_gate_wrapper import GatedRestorer
from utils import RGB_PSNR


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _resolve_amp_dtype(name):
    text = str(name).strip().lower()
    if text in ('bf16', 'bfloat16'):
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def validate(model, dataloader, device, blind_threshold, fill_strength, max_batches=None, use_amp=False, amp_dtype=torch.bfloat16):
    model.eval()
    psnr_vals = []
    blind_l1_vals = []

    amp_enabled = bool(use_amp and (device.type == 'cuda'))

    for batch_idx, (lr_blur_seq, hr_sharp_seq, _, _) in enumerate(dataloader):
        lr_blur_seq = lr_blur_seq.to(device, non_blocking=True)
        hr_sharp_seq = hr_sharp_seq.to(device, non_blocking=True)

        amp_ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if amp_enabled else nullcontext()
        with amp_ctx:
            result = model(lr_blur_seq, fill_strength=fill_strength)
            pred = result['output']
        t = hr_sharp_seq.shape[2] // 2
        target = hr_sharp_seq[:, :, t, :, :]

        psnr_vals.append(RGB_PSNR(pred, target))

        center_lr = lr_blur_seq[:, :, t, :, :]
        blind_mask = (torch.abs(center_lr - target) >= blind_threshold).float()
        blind_count = blind_mask.sum()
        if blind_count.item() > 0:
            blind_l1 = (torch.abs(pred - target) * blind_mask).sum() / blind_count
            blind_l1_vals.append(blind_l1.item())

        # Optional quick validation mode to reduce wall-clock during long finetune.
        if max_batches is not None and max_batches > 0 and (batch_idx + 1) >= max_batches:
            break

    avg_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    avg_blind_l1 = float(np.mean(blind_l1_vals)) if blind_l1_vals else 0.0
    return avg_psnr, avg_blind_l1


def load_base_model(config, ckpt_path, device):
    model = FMANet(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
    if config.stage == 2:
        model.restoration_network.load_state_dict(ckpt['model_R_state_dict'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Fine-tune only a lightweight fusion gate on top of frozen Stage2 FMANet.')
    parser.add_argument('--gate_config_path', type=str, default=DEFAULT_GATE_CONFIG_PATH)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--base_ckpt', type=str, default=None, help='Path to pretrained stage2 model_best.pt')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden_channels', type=int, default=None)
    parser.add_argument('--fill_strength', type=float, default=None)
    parser.add_argument('--blind_threshold', type=float, default=None)
    parser.add_argument('--blind_loss_weight', type=float, default=None)
    parser.add_argument('--log_period', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--val_period', type=int, default=None)
    parser.add_argument('--max_val_batches', type=int, default=None)
    parser.add_argument('--use_amp', type=str, default=None, help='true/false, defaults to true on CUDA')
    parser.add_argument('--amp_dtype', type=str, default=None, help='bf16 or fp16, default bf16')
    args = parser.parse_args()

    # Support external script config while keeping CLI as highest priority.
    common_cfg = read_gate_section(args.gate_config_path, 'common')
    finetune_cfg = read_gate_section(args.gate_config_path, 'finetune')

    config_path = pick_value(args.config_path, common_cfg, 'config_path', str, './experiment.cfg')
    base_ckpt = pick_value(args.base_ckpt, finetune_cfg, 'base_ckpt', str, None)
    save_dir = pick_value(args.save_dir, finetune_cfg, 'save_dir', str, './results/gate_finetune')
    epochs = pick_value(args.epochs, finetune_cfg, 'epochs', int, 30)
    lr = pick_value(args.lr, finetune_cfg, 'lr', float, 1e-4)
    hidden_channels = pick_value(args.hidden_channels, finetune_cfg, 'hidden_channels', int, 16)
    fill_strength = pick_value(args.fill_strength, finetune_cfg, 'fill_strength', float, 1.0)
    blind_threshold_override = pick_value(args.blind_threshold, finetune_cfg, 'blind_threshold', float, None)
    blind_loss_weight = pick_value(args.blind_loss_weight, finetune_cfg, 'blind_loss_weight', float, 1.0)
    log_period = pick_value(args.log_period, finetune_cfg, 'log_period', int, 100)
    num_workers = pick_value(args.num_workers, finetune_cfg, 'num_workers', int, None)
    seed_override = pick_value(args.seed, finetune_cfg, 'seed', int, None)
    val_period = max(1, pick_value(args.val_period, finetune_cfg, 'val_period', int, 1))
    max_val_batches = pick_value(args.max_val_batches, finetune_cfg, 'max_val_batches', int, 0)
    use_amp = pick_value(args.use_amp, finetune_cfg, 'use_amp', bool, True)
    amp_dtype_name = pick_value(args.amp_dtype, finetune_cfg, 'amp_dtype', str, 'bf16')

    if base_ckpt is None:
        parser.error('Missing --base_ckpt. Set it via CLI or [finetune].base_ckpt in gate config.')

    config = Config(config_path)
    config.stage = 2

    if num_workers is not None:
        config.nThreads = int(num_workers)

    seed = config.seed if seed_override is None else seed_override
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = _resolve_amp_dtype(amp_dtype_name)
    amp_enabled = bool(use_amp and (device.type == 'cuda'))
    # GradScaler is only needed for fp16 path; bf16 is typically stable without scaling.
    scaler = torch.amp.GradScaler('cuda', enabled=bool(amp_enabled and amp_dtype == torch.float16))

    os.makedirs(save_dir, exist_ok=True)

    train_loader = get_dataset(config, type='train')
    val_loader = get_dataset(config, type='val')

    base_model = load_base_model(config, base_ckpt, device)
    gated_model = GatedRestorer(
        base_model=base_model,
        in_channels=config.in_channels,
        hidden_channels=hidden_channels,
        init_scale=1.0,
    ).to(device)
    gated_model.freeze_base()

    optimizer = torch.optim.Adam(gated_model.gate.parameters(), lr=lr)

    blind_threshold = config.blind_mask_threshold if blind_threshold_override is None else blind_threshold_override

    best_psnr = -1.0
    best_path = os.path.join(save_dir, 'gate_best.pt')
    latest_path = os.path.join(save_dir, 'gate_latest.pt')

    for epoch in range(epochs):
        gated_model.train()
        running_loss = 0.0

        for step, (lr_blur_seq, hr_sharp_seq, _, _) in enumerate(train_loader, start=1):
            lr_blur_seq = lr_blur_seq.to(device, non_blocking=True)
            hr_sharp_seq = hr_sharp_seq.to(device, non_blocking=True)

            amp_ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if amp_enabled else nullcontext()
            with amp_ctx:
                result = gated_model(lr_blur_seq, fill_strength=fill_strength)
                pred = result['output']

                t = hr_sharp_seq.shape[2] // 2
                target = hr_sharp_seq[:, :, t, :, :]
                center_lr = lr_blur_seq[:, :, t, :, :]

                full_l1 = F.l1_loss(pred, target)

                blind_mask = (torch.abs(center_lr - target) >= blind_threshold).float()
                blind_count = blind_mask.sum()
                if blind_count.item() > 0:
                    blind_l1 = (torch.abs(pred - target) * blind_mask).sum() / blind_count
                else:
                    blind_l1 = torch.tensor(0.0, device=device)

                loss = full_l1 + blind_loss_weight * blind_l1

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            if step % log_period == 0:
                avg_loss = running_loss / log_period
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Step [{step}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.6f} FullL1: {full_l1.item():.6f} BlindL1: {blind_l1.item():.6f} "
                    f"GateScale: {gated_model.gate.gate_scale.item():.4f}"
                )
                running_loss = 0.0

        run_validation = ((epoch + 1) % val_period == 0) or ((epoch + 1) == epochs)
        val_psnr = float('nan')
        val_blind_l1 = float('nan')
        if run_validation:
            val_psnr, val_blind_l1 = validate(
                model=gated_model,
                dataloader=val_loader,
                device=device,
                blind_threshold=blind_threshold,
                fill_strength=fill_strength,
                max_batches=max_val_batches,
                use_amp=amp_enabled,
                amp_dtype=amp_dtype,
            )

        save_payload = {
            'epoch': epoch + 1,
            'gate_state_dict': gated_model.gate.state_dict(),
            'gate_scale': float(gated_model.gate.gate_scale.item()),
            'val_psnr': val_psnr,
            'val_blind_l1': val_blind_l1,
            'base_ckpt': base_ckpt,
            'fill_strength': fill_strength,
            'blind_threshold': blind_threshold,
        }
        torch.save(save_payload, latest_path)

        if run_validation:
            print(
                f"[VAL] Epoch {epoch + 1}: PSNR={val_psnr:.4f}, BlindL1={val_blind_l1:.6f}, "
                f"GateScale={gated_model.gate.gate_scale.item():.4f}"
            )
        else:
            print(
                f"[VAL] Epoch {epoch + 1}: skipped (val_period={val_period}) "
                f"GateScale={gated_model.gate.gate_scale.item():.4f}"
            )

        if run_validation and (val_psnr > best_psnr):
            best_psnr = val_psnr
            torch.save(save_payload, best_path)
            print(f"[*] New best gate checkpoint saved to {best_path}")

    print(f"Done. Best VAL PSNR: {best_psnr:.4f}")


if __name__ == '__main__':
    main()


import os
import sys
import argparse

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from data_blindpixel import get_dataset
from model import FMANet
from scripts.gate_config import DEFAULT_GATE_CONFIG_PATH, pick_value, read_gate_section
from scripts.fusion_gate_wrapper import GatedRestorer
from utils import RGB_PSNR, SSIM


def load_base_model(config, ckpt_path, device):
    model = FMANet(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
    if config.stage == 2:
        model.restoration_network.load_state_dict(ckpt['model_R_state_dict'])
    model.eval()
    return model


def tensor_to_u8(x):
    arr = x.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Evaluate pretrained Stage2 model with optional fusion gate wrapper.')
    parser.add_argument('--gate_config_path', type=str, default=DEFAULT_GATE_CONFIG_PATH)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--base_ckpt', type=str, default=None, help='Path to stage2 model_best.pt')
    parser.add_argument('--split', type=str, default=None, choices=['val', 'test'])
    parser.add_argument('--gate_ckpt', type=str, default=None, help='Optional path to gate_best.pt')
    parser.add_argument('--fill_strength', type=float, default=None)
    parser.add_argument('--blind_threshold', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--max_save', type=int, default=None)
    parser.add_argument('--hidden_channels', type=int, default=None)
    args = parser.parse_args()

    # Support config-driven eval settings while preserving CLI override behavior.
    common_cfg = read_gate_section(args.gate_config_path, 'common')
    eval_cfg = read_gate_section(args.gate_config_path, 'eval')

    config_path = pick_value(args.config_path, common_cfg, 'config_path', str, './experiment.cfg')
    base_ckpt = pick_value(args.base_ckpt, eval_cfg, 'base_ckpt', str, None)
    split = pick_value(args.split, eval_cfg, 'split', str, 'val')
    gate_ckpt = pick_value(args.gate_ckpt, eval_cfg, 'gate_ckpt', str, None)
    fill_strength = pick_value(args.fill_strength, eval_cfg, 'fill_strength', float, 1.0)
    blind_threshold_override = pick_value(args.blind_threshold, eval_cfg, 'blind_threshold', float, None)
    save_dir = pick_value(args.save_dir, eval_cfg, 'save_dir', str, None)
    max_save = pick_value(args.max_save, eval_cfg, 'max_save', int, 20)
    hidden_channels = pick_value(args.hidden_channels, eval_cfg, 'hidden_channels', int, 16)

    if base_ckpt is None:
        parser.error('Missing --base_ckpt. Set it via CLI or [eval].base_ckpt in gate config.')
    if split not in ('val', 'test'):
        parser.error("split must be 'val' or 'test'.")

    config = Config(config_path)
    config.stage = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataset(config, type=split)

    base_model = load_base_model(config, base_ckpt, device)

    use_gate = gate_ckpt is not None
    if use_gate:
        wrapped = GatedRestorer(base_model=base_model, in_channels=config.in_channels, hidden_channels=hidden_channels, init_scale=1.0).to(device)
        gate_state = torch.load(gate_ckpt, map_location=device)
        wrapped.gate.load_state_dict(gate_state['gate_state_dict'])
        wrapped.eval()
        model = wrapped
    else:
        model = base_model

    blind_threshold = config.blind_mask_threshold if blind_threshold_override is None else blind_threshold_override

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    psnr_vals = []
    ssim_vals = []
    blind_l1_vals = []

    for idx, batch in enumerate(dataloader):
        if split == 'test':
            lr_blur_seq, relative_path = batch
            hr_sharp_seq = None
            relative_path = relative_path[0] if isinstance(relative_path, (list, tuple)) else relative_path
        else:
            lr_blur_seq, hr_sharp_seq, _, _ = batch
            relative_path = f'val_{idx:05d}.png'

        lr_blur_seq = lr_blur_seq.to(device)

        if use_gate:
            result = model(lr_blur_seq, fill_strength=fill_strength)
            pred = result['output']
            base_pred = result['output_base']
            gate_map = result['gate_map']
        else:
            result = model(lr_blur_seq)
            pred = result['output']
            base_pred = pred
            gate_map = None

        t = lr_blur_seq.shape[2] // 2
        center_lr = lr_blur_seq[:, :, t, :, :]

        if hr_sharp_seq is None:
            gt_path = os.path.join(config.dataset_path, 'test_sharp', relative_path)
            if os.path.exists(gt_path):
                gt_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_np is not None:
                    gt = torch.from_numpy(gt_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                else:
                    gt = None
            else:
                gt = None
        else:
            hr_sharp_seq = hr_sharp_seq.to(device)
            gt = hr_sharp_seq[:, :, t, :, :]

        if gt is not None:
            psnr_vals.append(RGB_PSNR(pred, gt))
            ssim_vals.append(SSIM(pred, gt))

            blind_mask = (torch.abs(center_lr - gt) >= blind_threshold).float()
            blind_count = blind_mask.sum()
            if blind_count.item() > 0:
                blind_l1 = (torch.abs(pred - gt) * blind_mask).sum() / blind_count
                blind_l1_vals.append(blind_l1.item())

        if save_dir and idx < max_save:
            pred_u8 = tensor_to_u8(pred[0, 0])
            center_u8 = tensor_to_u8(center_lr[0, 0])
            base_u8 = tensor_to_u8(base_pred[0, 0])
            if gt is not None:
                gt_u8 = tensor_to_u8(gt[0, 0])
            else:
                gt_u8 = np.zeros_like(pred_u8)

            if gate_map is not None:
                gate_u8 = tensor_to_u8(gate_map[0, 0])
                panel = np.concatenate([center_u8, base_u8, pred_u8, gt_u8, gate_u8], axis=1)
            else:
                panel = np.concatenate([center_u8, pred_u8, gt_u8], axis=1)

            out_name = os.path.basename(str(relative_path))
            if not out_name.lower().endswith('.png'):
                out_name += '.png'
            cv2.imwrite(os.path.join(save_dir, out_name), panel)

    avg_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    avg_ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    avg_blind_l1 = float(np.mean(blind_l1_vals)) if blind_l1_vals else 0.0

    mode = 'GATED' if use_gate else 'BASE'
    print(f'[{mode}] split={split} PSNR={avg_psnr:.4f} SSIM={avg_ssim:.5f} BlindL1={avg_blind_l1:.6f}')
    if use_gate:
        print(f'[GATED] fill_strength={fill_strength:.3f}')


if __name__ == '__main__':
    main()


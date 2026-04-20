import os
import torch
import random
import argparse
import numpy as np
import numbers

from model import FMANet
from train import Trainer
# 修正：根据 utils.py 实际的类名进行导入
from utils import Train_Report as Report
from data_blindpixel import get_dataset, BlindPixelDataset
from config import Config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config):
    global_step = 0
    train_log = Report(config.save_dir, type='train', stage=config.stage)
    val_log = Report(config.save_dir, type='val', stage=config.stage)

    train_dataloader = get_dataset(config, type='train')
    # 保持变量名统一：使用 val_dataloader
    val_dataloader = get_dataset(config, type='val')

    model = FMANet(config=config)
    trainer = Trainer(config=config, model=model)

    print(f'num parameters: {count_parameters(model)}')

    if config.stage == 2:
        trainer.load_best_stage1_model()

    best_psnr = float('-inf')
    best_blind_l1 = float('inf')
    best_blind_psnr = float('-inf')
    last_epoch = 0
    if config.finetuning:
        last_epoch = trainer.load_checkpoint()

    for epoch in range(last_epoch, config.num_epochs):
        train_log.write(f'========= Epoch {epoch + 1} of {config.num_epochs} =========')
        global_step = trainer.train(train_dataloader, train_log, global_step)

        if (epoch + 1) % config.val_period == 0 or epoch == config.num_epochs - 1:
            val_result = trainer.validate(val_dataloader, val_log, epoch)

            # 每次验证后打印可学习的融合/盲元参数，默认按 val_period 出现（trainer.validate 每 val_period 调用一次）
            try:
                if config.stage == 2 and hasattr(trainer.model, 'restoration_network'):
                    rn = trainer.model.restoration_network
                    # 映射到受约束的可读值
                    alpha = float(torch.sigmoid(rn.base_alpha_param).detach().cpu().item()) if hasattr(rn, 'base_alpha_param') else float(getattr(config, 'base_alpha', 0.0))
                    beta = float(torch.sigmoid(rn.base_beta_param).detach().cpu().item()) if hasattr(rn, 'base_beta_param') else float(getattr(config, 'base_beta', 0.0))
                    blind_res_scale = float(torch.nn.functional.softplus(rn.blind_res_scale_param).detach().cpu().item()) if hasattr(rn, 'blind_res_scale_param') else float(getattr(config, 'blind_res_scale', 0.0))
                    # 读取可学习的损失权重（softplus 映射为正），fallback 到配置
                    # restoration weight is fixed in config (not learned) to protect non-blind regions
                    restoration_w = float(getattr(config, 'restoration_loss_weight', 1.0))
                    try:
                        blind_restore_w = float(torch.nn.functional.softplus(rn.blind_restore_loss_weight_param).detach().cpu().item()) if hasattr(rn, 'blind_restore_loss_weight_param') else float(getattr(config, 'blind_restore_loss_weight', 0.6))
                    except Exception:
                        blind_restore_w = float(getattr(config, 'blind_restore_loss_weight', 0.6))
                    try:
                        blind_res_w = float(torch.nn.functional.softplus(rn.blind_res_loss_weight_param).detach().cpu().item()) if hasattr(rn, 'blind_res_loss_weight_param') else float(getattr(config, 'blind_res_loss_weight', 2.0))
                    except Exception:
                        blind_res_w = float(getattr(config, 'blind_res_loss_weight', 2.0))

                    line = (
                        f"VAL Epoch [{epoch + 1}] alpha: {alpha:.4f} | beta: {beta:.4f} | blind_res_scale: {blind_res_scale:.3f}"
                        f" | restoration_w: {restoration_w:.3f} | blind_restore_w: {blind_restore_w:.3f} | blind_res_w: {blind_res_w:.3f}"
                    )
                    print(line)
                    # 同时写入验证日志文件，便于后续分析
                    try:
                        val_log.write(line)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[!] Warning: failed to print fusion params after validation: {e}")

            # validate 支持返回字典: 主指标(PSNR) + 盲元兜底指标(BlindL1/BlindPSNR)。
            if isinstance(val_result, dict):
                current_psnr = float(val_result.get('val_psnr', float('-inf')))
                current_blind_l1 = val_result.get('blind_l1', None)
                current_blind_psnr = val_result.get('blind_psnr', None)
            elif isinstance(val_result, (list, tuple)):
                current_psnr = val_result[0]
                current_blind_l1 = None
            else:
                current_psnr = val_result
                current_blind_l1 = None

            # Robust scalar conversion: accepts Python/NumPy numbers, tensors, lists, and arrays.
            # This prevents np.isinf/np.isnan from crashing when validate() returns non-scalar containers.
            if isinstance(current_psnr, numbers.Number):
                current_psnr = float(current_psnr)
            elif torch.is_tensor(current_psnr):
                current_psnr = float(current_psnr.detach().float().mean().item())
            else:
                try:
                    current_psnr = float(np.asarray(current_psnr, dtype=np.float64).mean())
                except Exception:
                    print(f"[!] Warning: Unsupported PSNR type at Epoch {epoch + 1}: {type(current_psnr)}")
                    continue

            # 如果 PSNR 是 0 或 Inf，可能是数据问题，打印提醒
            if np.isinf(current_psnr) or np.isnan(current_psnr):
                print(f"[!] Warning: Invalid PSNR detected at Epoch {epoch + 1}")
                continue

            psnr_eps = getattr(config, 'checkpoint_psnr_tolerance', 1e-4)
            blind_eps = getattr(config, 'checkpoint_blind_l1_tolerance', 1e-6)

            # 主指标优先；主指标接近持平时，用 BlindL1 更低者作为兜底优胜。
            better_by_psnr = current_psnr > (best_psnr + psnr_eps)
            tie_on_psnr = abs(current_psnr - best_psnr) <= psnr_eps

            # Prefer higher blind_psnr as first tiebreaker (if available), then lower blind_l1
            better_by_blindpsnr = (
                tie_on_psnr and
                current_blind_psnr is not None and
                (current_blind_psnr > best_blind_psnr + psnr_eps)
            )

            better_by_blindl1 = (
                tie_on_psnr and
                current_blind_psnr is None and
                current_blind_l1 is not None and
                (current_blind_l1 + blind_eps) < best_blind_l1
            )

            # Parallel saving strategy:
            # - keep best by PSNR
            # - keep best by blind PSNR (if available)
            # This ensures we don't lose models that are specialized for blind-pixel recovery.
            saved_any = False

            if better_by_psnr:
                best_psnr = current_psnr
                # legacy save (model_best.pt) and dedicated PSNR copy
                trainer.save_best_model(epoch)
                try:
                    trainer.save_best_model_tagged(epoch, 'psnr')
                except Exception:
                    pass
                saved_any = True
                print(f"[*] New Best PSNR saved at Epoch {epoch + 1} | PSNR: {best_psnr:.3f}")

            if better_by_blindpsnr and current_blind_psnr is not None:
                best_blind_psnr = current_blind_psnr
                try:
                    trainer.save_best_model_tagged(epoch, 'blindpsnr')
                except Exception:
                    pass
                saved_any = True
                print(f"[*] New Best BlindPSNR saved at Epoch {epoch + 1} | BlindPSNR: {best_blind_psnr:.3f}")

            # If PSNR tie but blind_l1 improved and blind_psnr not available, optionally save blindL1-best
            if (not saved_any) and better_by_blindl1:
                best_blind_l1 = current_blind_l1
                try:
                    trainer.save_best_model_tagged(epoch, 'blindl1')
                except Exception:
                    pass
                print(f"[*] New Best BlindL1 saved at Epoch {epoch + 1} | BlindL1: {best_blind_l1:.6f}")


def test(config):
    # 1. 加载测试数据
    test_dataloader = get_dataset(config, type='test')
    model = FMANet(config=config)
    trainer = Trainer(config=config, model=model)

    # 2. 加载最优权重
    trainer.load_best_model()

    # 3. 执行测试（生成并保存图片）
    trainer.test(test_dataloader)

    # 4. 定量评估 (PSNR/SSIM)
    # 修改点：gt_dir 必须指向你真实的 test_sharp 文件夹
    # 假设你的目录是：D:/mangyuan/FMA-Net/data/test_sharp
    gt_root = os.path.join(config.dataset_path, 'test_sharp')
    output_root = os.path.join(config.save_dir, 'test') # 对应 Trainer.test 里的保存位置

    print("===> Starting Quantitative Evaluation...")
    # 注意：如果你的 test_sharp 里没有子文件夹（全是图），
    # 记得配合我之前建议你修改的那个“扁平化”版本的 test_quantitative_result
    trainer.test_quantitative_result(
        gt_dir=gt_root,
        output_dir=output_root,
        image_border=0  # 盲元修复通常看全图，如果不需要裁边可以设为 0
    )


# def test_custom(config):
#     # 此处逻辑基本保持不变
#     from data import Custom_Dataset
#     data = Custom_Dataset(config)
#     test_dataloader = torch.utils.data.DataLoader(
#         data, batch_size=1, drop_last=False, shuffle=False,
#         num_workers=int(config.nThreads), pin_memory=True
#     )
#     model = FMANet(config=config)
#     trainer = Trainer(config=config, model=model)
#     trainer.load_best_model()
#     trainer.test(test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='执行训练')
    parser.add_argument('--test', action='store_true', help='在测试集上评估')
    parser.add_argument('--test_custom', action='store_true', help='处理自定义图片')
    parser.add_argument('--config_path', type=str, default='./experiment.cfg', help='配置文件路径')
    args = parser.parse_args()

    # 载入配置
    config = Config(args.config_path)

    # Runtime backend tuning: default to fast path, keep debug sync optional.
    if getattr(config, 'cuda_launch_blocking', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    else:
        os.environ.pop('CUDA_LAUNCH_BLOCKING', None)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = bool(getattr(config, 'cudnn_benchmark', True))
    torch.backends.cuda.matmul.allow_tf32 = True

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # GPU 设置
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    if args.train:
        train(config)
    if args.test:
        test(config)
    # if args.test_custom:
    #     test_custom(config)
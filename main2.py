import os
import torch
import random
import argparse
import numpy as np

# 针对 Blackwell 等高性能显卡的优化设置
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # 性能电脑建议开启
torch.backends.cuda.matmul.allow_tf32 = True

from model import FMANet
from train import Trainer
from utils import Train_Report as Report
from data_blindpixel import get_dataset
from config import Config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config):
    global_step = 0
    train_log = Report(config.save_dir, type='train', stage=config.stage)
    val_log = Report(config.save_dir, type='val', stage=config.stage)

    train_dataloader = get_dataset(config, type='train')
    val_dataloader = get_dataset(config, type='val')

    model = FMANet(config=config)
    trainer = Trainer(config=config, model=model)

    print(f'[*] 模型参数量: {count_parameters(model)}')

    # --- 核心加载逻辑：新旧结合 ---
    if config.stage == 1:
        print("[*] 正在 Stage 1 微调模式：尝试加载旧权重以加速收敛...")
        trainer.load_best_model()

    elif config.stage == 2:
        print("[*] 正在 Stage 2 增量微调模式...")
        # 第一步：加载你刚练好的「大圆版」Stage 1
        trainer.load_best_stage1_model()
        # 第二步：加载你之前练好的「旧版」Stage 2
        try:
            trainer.load_best_model()
            print("[*] 成功实现缝合：新 Net_D (大圆) + 旧 Net_R (修复基础)")
        except:
            print("[!] 未找到旧 Stage 2 权重，Net_R 将从头开始适应新 Net_D")

    best_psnr = 0
    last_epoch = 0

    # 如果是针对新数据的首次训练，建议 cfg 里的 finetuning 设为 False
    if config.finetuning:
        last_epoch = trainer.load_checkpoint()
        global_step = last_epoch * len(train_dataloader)

    # --- 训练循环 ---
    for epoch in range(last_epoch, config.num_epochs):
        train_log.write(f'========= Epoch {epoch + 1} of {config.num_epochs} =========')
        global_step = trainer.train(train_dataloader, train_log, global_step)

        if (epoch + 1) % config.val_period == 0 or epoch == config.num_epochs - 1:
            val_result = trainer.validate(val_dataloader, val_log, epoch)

            # 兼容性处理验证结果
            current_psnr = val_result[0] if isinstance(val_result, (list, tuple)) else val_result

            if np.isinf(current_psnr) or np.isnan(current_psnr):
                continue

            if current_psnr > best_psnr:
                best_psnr = current_psnr
                trainer.save_best_model(epoch)
                print(f"[*] 发现更佳模型，已保存: {best_psnr:.3f}")


def test(config):
    test_dataloader = get_dataset(config, type='test')
    model = FMANet(config=config)
    trainer = Trainer(config=config, model=model)
    trainer.load_best_model()
    trainer.test(test_dataloader)

    gt_root = os.path.join(config.dataset_path, 'test_sharp')
    output_root = os.path.join(config.save_dir, f'model_stage{config.stage}', 'test')

    trainer.test_quantitative_result(gt_dir=gt_root, output_dir=output_root, image_border=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--config_path', type=str, default='./experiment.cfg')
    args = parser.parse_args()

    config = Config(args.config_path)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    if args.train:
        train(config)
    if args.test:
        test(config)
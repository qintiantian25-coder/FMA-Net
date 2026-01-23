import os
import torch
# 1. 强制同步，报错位置会变得精确
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 2. 禁用 cuDNN 优化，避免黑盒算子错误
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
# 3. 针对 Blackwell 的特殊设置
torch.backends.cuda.matmul.allow_tf32 = True
import random
import argparse
import numpy as np

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

    best_psnr = 0
    last_epoch = 0
    if config.finetuning:
        last_epoch = trainer.load_checkpoint()

    for epoch in range(last_epoch, config.num_epochs):
        train_log.write(f'========= Epoch {epoch + 1} of {config.num_epochs} =========')
        global_step = trainer.train(train_dataloader, train_log, global_step)

        if (epoch + 1) % config.val_period == 0 or epoch == config.num_epochs - 1:
            val_result = trainer.validate(val_dataloader, val_log, epoch)

            # --- 修改建议：根据 Stage 稳健提取指标 ---
            if isinstance(val_result, (list, tuple)):
                current_psnr = val_result[0]  # 通常第一个是主要的 PSNR 指标
            else:
                current_psnr = val_result

            # 如果 PSNR 是 0 或 Inf，可能是数据问题，打印提醒
            if np.isinf(current_psnr) or np.isnan(current_psnr):
                print(f"[!] Warning: Invalid PSNR detected at Epoch {epoch + 1}")
                continue

            if current_psnr > best_psnr:
                best_psnr = current_psnr
                trainer.save_best_model(epoch)
                print(f"[*] New Best PSNR: {best_psnr:.3f} saved at Epoch {epoch + 1}")


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
    trainer.test_quantitative_result(
        gt_dir=gt_root,
        output_dir=output_root,
        image_border=config.num_seq // 2
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
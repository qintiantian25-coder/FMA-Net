import os
import configparser

#配置管理模块
# 功能：使用 Python 的 configparser 库读取并解析外部配置文件（通常是 .ini 或 .cfg 文件），将散乱的配置参数（如学习率、网络深度、数据集路径等）统一封装进一个 Config 对象中，方便在整个项目（训练、推理、测试）中通过点号（如 config.lr）进行访问。
# 输入：配置文件的路径 config_path。
# 输出：一个包含所有实验参数、训练超参数、网络架构设置和路径信息的配置对象。

class Config:
    def __init__(self, config_path):
        parser = configparser.ConfigParser()
        # 使用 utf-8 编码打开文件
        with open(config_path, 'r', encoding='utf-8') as f:
            parser.read_file(f)

        # experiment
        self.seed = int(parser.get('experiment', 'seed'))

        # training
        self.dataset_path = parser.get('training', 'dataset_path')
        self.save_dir = parser.get('training', 'save_dir')
        # Optional: keep Stage1 loading path independent from current save_dir.
        # Useful when starting a new Stage2 run in a separate folder.
        self.stage1_pretrained_dir = parser.get('training', 'stage1_pretrained_dir', fallback='')
        self.stage = int(parser.get('training', 'stage'))
        self.log_dir = parser.get('training', 'log_dir')
        self.log_dir = os.path.join(self.save_dir, f'stage{self.stage}_{self.log_dir}')

        self.nThreads = int(parser.get("training", "nThreads"))
        self.num_epochs = int(parser.get("training", "num_epochs"))
        self.lr = float(parser.get("training", "lr"))
        # Optional one-shot LR switch: keep one-command training from start to finish.
        self.lr_drop_epoch = int(parser.get("training", "lr_drop_epoch", fallback="-1"))
        self.lr_after_drop = float(parser.get("training", "lr_after_drop", fallback="0"))
        # Dynamic LR mode: keep base LR before drop epoch, then decay automatically (cosine) afterwards.
        self.lr_dynamic_after_drop = parser.getboolean("training", "lr_dynamic_after_drop", fallback=False)
        self.dynamic_min_lr = float(parser.get("training", "dynamic_min_lr", fallback="1e-6"))
        self.batch_size = int(parser.get('training', 'batch_size'))
        self.patch_size = int(parser.get('training', 'patch_size'))
        self.finetuning = (parser.get('training', 'finetuning') == 'True')
        self.save_train_img = (parser.get('training', 'save_train_img') == 'True')

        self.scale = int(parser.get('training', 'scale'))
        self.num_seq = int(parser.get('training', 'num_seq'))

        # --- Loss weights: centralized in [loss] section for easy tuning ---
        # These weights are used to compose total_loss in training stage.
        self.lr_warping_loss_weight = float(parser.get("loss", "lr_warping_loss_weight", fallback="0.0"))
        self.hr_warping_loss_weight = float(parser.get("loss", "hr_warping_loss_weight", fallback="0.0"))
        self.flow_loss_weight = float(parser.get("loss", "flow_loss_weight", fallback="0.0"))
        self.D_TA_loss_weight = float(parser.get("loss", "D_TA_loss_weight", fallback="0.0"))
        self.R_TA_loss_weight = float(parser.get("loss", "R_TA_loss_weight", fallback="0.0"))
        self.Net_D_weight = float(parser.get("loss", "Net_D_weight", fallback="0.0"))
        # 盲元判定阈值：用于 stage-2 的 mask 监督与盲元区域对齐统计。
        self.blind_mask_threshold = float(parser.get("loss", "blind_mask_threshold", fallback=parser.get("training", "blind_mask_threshold", fallback="0.08")))
        # Stage-2 盲元专项分支损失权重：仅对盲元掩码区域生效。
        self.blind_restore_loss_weight = float(parser.get("loss", "blind_restore_loss_weight", fallback=parser.get("training", "blind_restore_loss_weight", fallback="0.2")))
        # 盲元残差分支损失权重（可配置）：之前代码中硬编码为 2.0，现在从 [loss] 中读取以便调参。
        self.blind_res_loss_weight = float(parser.get("loss", "blind_res_loss_weight", fallback=parser.get("training", "blind_res_loss_weight", fallback="2.0")))
        # flow_loss_scale: multiplicative scale for flow loss (fallback to 10.0)
        self.flow_loss_scale = float(parser.get("loss", "flow_loss_scale", fallback="10.0"))
        # smart_recon parameters: previously hard-coded inside Trainer.smart_recon_loss
        self.smart_blind_l2_scale = float(parser.get("loss", "smart_blind_l2_scale", fallback="1000.0"))
        self.smart_blind_topk_frac = float(parser.get("loss", "smart_blind_topk_frac", fallback="0.005"))
        # 盲元专项残差分支的输出缩放及盲元推理阈值：集中在 [fusion] 节管理
        self.blind_res_scale = float(parser.get("fusion", "blind_res_scale", fallback="1.0"))
        # 推理时无GT，使用中心帧阈值近似盲元区域（从 [fusion] 读取）。
        self.blind_infer_threshold = float(parser.get("fusion", "blind_infer_threshold", fallback="0.08"))
        # 模型保存判定中的主指标容差与盲元兜底容差。
        self.checkpoint_psnr_tolerance = float(parser.get("training", "checkpoint_psnr_tolerance", fallback="1e-4"))
        self.checkpoint_blind_l1_tolerance = float(parser.get("training", "checkpoint_blind_l1_tolerance", fallback="1e-6"))

        self.gpu = parser.get("training", "gpu")

        # Performance knobs: all have safe fallbacks so old cfg files keep working.
        self.use_amp = parser.getboolean('training', 'use_amp', fallback=True)
        self.amp_dtype = parser.get('training', 'amp_dtype', fallback='fp16')
        self.cudnn_benchmark = parser.getboolean('training', 'cudnn_benchmark', fallback=True)
        self.cuda_launch_blocking = parser.getboolean('training', 'cuda_launch_blocking', fallback=False)
        self.pin_memory = parser.getboolean('training', 'pin_memory', fallback=True)
        self.persistent_workers = parser.getboolean('training', 'persistent_workers', fallback=True)
        self.prefetch_factor = int(parser.get('training', 'prefetch_factor', fallback='2'))
        # Stability knobs: keep defaults backward-compatible while allowing NaN/overflow mitigation.
        self.grad_clip_norm = float(parser.get('training', 'grad_clip_norm', fallback='0.0'))
        self.amp_init_scale = float(parser.get('training', 'amp_init_scale', fallback='65536'))
        self.amp_growth_interval = int(parser.get('training', 'amp_growth_interval', fallback='2000'))
        self.amp_backoff_factor = float(parser.get('training', 'amp_backoff_factor', fallback='0.5'))
        self.overflow_patience = int(parser.get('training', 'overflow_patience', fallback='2'))
        self.lr_overflow_decay = float(parser.get('training', 'lr_overflow_decay', fallback='0.5'))
        self.min_lr = float(parser.get('training', 'min_lr', fallback='1e-6'))
        self.overflow_lr_decay_cooldown = int(parser.get('training', 'overflow_lr_decay_cooldown', fallback='200'))
        self.overflow_log_interval = int(parser.get('training', 'overflow_log_interval', fallback='50'))
        self.amp_recovery_steps = int(parser.get('training', 'amp_recovery_steps', fallback='128'))

        # Network
        self.in_channels = int(parser.get('network', 'in_channels'))
        self.dim = int(parser.get('network', 'dim'))
        self.ds_kernel_size = int(parser.get('network', 'ds_kernel_size'))
        self.us_kernel_size = int(parser.get('network', 'us_kernel_size'))
        self.num_RDB = int(parser.get('network', 'num_RDB'))
        self.growth_rate = int(parser.get('network', 'growth_rate'))
        self.num_dense_layer = int(parser.get('network', 'num_dense_layer'))
        self.num_flow = int(parser.get('network', 'num_flow'))
        self.num_FRMA = int(parser.get('network', 'num_FRMA'))
        self.num_transformer_block = int(parser.get('network', 'num_transformer_block'))
        self.num_heads = int(parser.get('network', 'num_heads'))
        self.LayerNorm_type = parser.get('network', 'LayerNorm_type')
        self.ffn_expansion_factor = float(parser.get('network', 'ffn_expansion_factor'))
        self.bias = (parser.get('network', 'bias') == 'True')

        # --- 融合逻辑权重 (集中从 [fusion] 节读取) ---
        # base_alpha: 邻帧补偿(DUF)的基础权重。
        self.base_alpha = float(parser.get('fusion', 'base_alpha', fallback='0.8'))
        # base_beta: 自身重构(Res)的基础权重。
        self.base_beta = float(parser.get('fusion', 'base_beta', fallback='0.2'))
        # 可学习参数的初始 blind_res_scale（fusion 节）已在上方读取

        # validation
        self.val_period = int(parser.get('validation', 'val_period'))
        self.save_val_align_vis = parser.getboolean('validation', 'save_val_align_vis', fallback=False)
        self.enable_align_metrics = parser.getboolean('validation', 'enable_align_metrics', fallback=True)
        self.align_metrics_period = int(parser.get('validation', 'align_metrics_period', fallback='1'))

        # test
        # custom_path is optional in some cfg files; keep empty when not provided.
        self.custom_path = parser.get('test', 'custom_path', fallback='')
        # 测试期盲元坐标文件（可选），用于盲元专项定量评估。
        self.test_mask_csv = parser.get('test', 'test_mask_csv', fallback='')

        self.need_patch = (parser.get('training', 'need_patch') == 'True')
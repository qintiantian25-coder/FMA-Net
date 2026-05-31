# FMA-Net: Feature-Aligned Multi-Attention Network

用于视频盲元（坏点）修复的两阶段深度学习网络。通过第一阶段学习图像退化模式，第二阶段利用多头注意力和动态上采样恢复清晰视频帧。

## 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [架构概览](#架构概览)
- [优化记录](#优化记录)
- [配置参数参考](#配置参数参考)
- [数据准备](#数据准备)
- [训练与测试](#训练与测试)

---

## 快速开始

**环境依赖**: `torch >= 2.0`, `torchvision`, `opencv-python`, `numpy`, `einops`

```bash
# Stage 1 训练（学习退化模式）
python main.py --train --config_path experiment.stage1.cfg

# Stage 2 训练（盲元修复，需加载 Stage 1 权重）
python main.py --train --config_path experiment.stage2.cfg

# 测试
python main.py --test --config_path experiment.stage2.cfg
```

---

## 项目结构

```
FMA-Net/
├── main.py                  # 入口：训练/测试调度
├── model.py                 # 核心模型 (Net_D + Net_R + FMANet)
├── train.py                 # 训练器（损失、优化、AMP 稳定性）
├── config.py                # 配置文件解析
├── utils.py                 # PSNR/SSIM 计算、日志、保存管理
├── data.py                  # REDS 格式数据集加载器（旧版）
├── data_blindpixel.py       # 盲元数据集加载器（当前使用）
├── experiment.cfg           # 测试/推理配置（旧版兼容）
├── experiment.stage1.cfg    # Stage 1 训练配置
├── experiment.stage2.cfg    # Stage 2 训练配置
├── experiment.stage2.cfg    # Stage 2 训练配置（备选）
├── fangzhen.py              # 盲元仿真工具（基础版）
├── fangzhen_adaptive.py     # 盲元仿真工具（参数化版，推荐）
├── add_new_dataset.py       # 新数据集导入工具
├── preprocessing/           # 数据预处理脚本
│   ├── generate_flow.py     # RAFT 光流生成
│   ├── raft.py              # RAFT 模型
│   └── ...
└── assets/                  # 网络结构图、效果展示图
```

---

## 架构概览

### 两阶段流水线

```
输入: LR 模糊序列 [B, 1, T, H, W]
              │
    ┌─────────▼──────────┐
    │   Net_D (Stage 1)  │  退化学习网络
    │   - RRDB 特征提取   │  学习模糊核、对齐光流、锚点特征
    │   - FRMA ×3        │  使用标准交叉注意力
    │   - 动态下采样     │
    └─────────┬──────────┘
              │ F, KD, flow, anchor
    ┌─────────▼──────────┐
    │   Net_R (Stage 2)  │  恢复网络
    │   - RRDB 特征提取   │  双分支融合:
    │   - FRMA ×3 (DA)   │    Res 分支: 自身重建
    │   - 盲元 Gate/SE    │    DUF 分支: 邻帧补偿
    │   - 动态上采样     │    Blind 分支: 盲元专项修复
    └─────────┬──────────┘
              │
输出: 修复后清晰图像 [B, 1, H, W]
```

### Net_R 盲元修复机制

```
Fw (融合特征)
    │
    ├──► blind_res_conv1 → SE Attention → blind_res_conv2 → raw_blind_res
    │                                                                │
    ├──► blind_gate (3层 Conv + Sigmoid) → blind_gate              │
    │                                                                │
    └──► 融合公式:                                                   │
         output = (1 - gate·mask) × res                              │
                + gate·mask × (α × duf + β × res + blind_res_scale × blind_res)
```

可学习参数:
- `base_alpha_param`: 邻帧补偿(DUF)权重，sigmoid 约束到 [0, 1]
- `base_beta_param`: 自身重建(Res)权重，sigmoid 约束到 [0, 1]
- `blind_res_scale_param`: 盲元残差缩放因子，softplus 约束到 (0, +∞)
- `blind_threshold_param`: 盲元判定阈值，sigmoid 约束到 [0, 1]

---

## 优化记录

以下优化旨在提升 **PSNR**（峰值信噪比）和 **Blind PSNR**（盲元区域 PSNR），同时改善训练速度和稳定性。

### 一、损失函数优化（对 PSNR 影响最大）

#### 1.1 L1 Loss → Charbonnier Loss

| 项目 | 说明 |
|------|------|
| **修改文件** | `train.py` |
| **目的** | L1 Loss 在零点不光滑，梯度恒定，对异常值鲁棒性差 |
| **替换位置** | `restoration_loss`, `blind_restore_loss`, `blind_res_loss`, `lr_warping_loss`, `boundary_loss` |
| **公式** | `Charbonnier(x) = sqrt(x² + ε²)`, ε 默认 1e-3 |
| **预期效果** | 收敛更稳定，PSNR 提升 **0.1~0.3 dB**，减少训练后期的梯度震荡 |

#### 1.2 新增 SSIM 损失

| 项目 | 说明 |
|------|------|
| **修改文件** | `train.py` |
| **目的** | L1/Charbonnier 仅关注像素级误差，忽略结构信息 |
| **实现** | 11×11 高斯窗口的可微 SSIM，计算 `1 - SSIM` 作为损失 |
| **配置** | `ssim_loss_weight` (全图，默认 0.1), `blind_ssim_loss_weight` (盲元区域，默认 0.2) |
| **预期效果** | 提升视觉质量，减少伪影和模糊，PSNR 提升 **0.1~0.3 dB** |

#### 1.3 新增梯度/边缘损失

| 项目 | 说明 |
|------|------|
| **修改文件** | `train.py` |
| **目的** | 恢复图像边缘容易模糊，需要显式约束边缘锐度 |
| **实现** | 对输出和真值分别计算 Sobel 梯度（x/y 方向），用 Charbonnier 约束其一致性 |
| **配置** | `grad_loss_weight` (默认 0.05) |
| **预期效果** | 边缘更锐利，减少过平滑，Blind PSNR 提升 **0.05~0.15 dB** |

#### 1.4 新增频域损失

| 项目 | 说明 |
|------|------|
| **修改文件** | `train.py` |
| **目的** | 盲元修复的关键是恢复高频细节，空间域损失难以针对性优化 |
| **实现** | 对盲元区域的输出和真值做 `rFFT`，用 Charbonnier 约束频谱幅度 |
| **配置** | `blind_fft_loss_weight` (默认 0.05) |
| **预期效果** | 提升盲元区域高频纹理恢复，Blind PSNR 提升 **0.1~0.2 dB** |

#### Stage 2 完整损失函数

```
total_loss = restoration_weight × restoration_loss(Charbonnier)
           + blind_restore_weight × blind_restore_loss(Charbonnier)
           + blind_res_weight × blind_res_loss(Charbonnier)
           + ssim_weight × SSIM_loss
           + blind_ssim_weight × blind_SSIM_loss
           + grad_weight × gradient_loss(Sobel)
           + fft_weight × blind_fft_loss(rFFT)
           + recon_loss + hr_warping_loss + lr_warping_loss
           + flow_loss + D_TA_loss + R_TA_loss
           + boundary_loss + gate_TV_loss + gate_sparsity_loss
```

---

### 二、盲元架构增强（对 Blind PSNR 影响最大）

#### 2.1 多层盲元 Gate 网络

| 项目 | 说明 |
|------|------|
| **修改文件** | `model.py` — `Net_R.__init__` 和 `Net_R.forward` |
| **目的** | 原方案单层 Conv2d(dim, 1) + Sigmoid 表达能力不足，盲元定位精度低 |
| **新架构** | `Conv2d(80→40) → LeakyReLU → Conv2d(40→20) → LeakyReLU → Conv2d(20→1) → Sigmoid` |
| **参数量** | 原 81 参数 → 现 ~5,000 参数（增量可忽略） |
| **预期效果** | 更精确的盲元区域判定，减少误修复和漏修复，Blind PSNR 提升 **0.15~0.3 dB** |

#### 2.2 SE Channel Attention

| 项目 | 说明 |
|------|------|
| **修改文件** | `model.py` — `Net_R` |
| **目的** | 盲元残差分支没有通道维度的自适应加权，不同通道的特征重要性不同 |
| **实现** | SE 块: `AdaptiveAvgPool2d → FC(80→20) → ReLU → FC(20→80) → Sigmoid`，通道重标定 |
| **预期效果** | 抑制无用通道，增强盲元敏感通道，Blind PSNR 提升 **0.05~0.1 dB** |

#### 2.3 可学习盲元判定阈值

| 项目 | 说明 |
|------|------|
| **修改文件** | `model.py` — `Net_R.__init__`, `train.py` — `_build_blind_masks` |
| **目的** | 原阈值 `blind_mask_threshold=0.08` 硬编码，无法适应不同数据集和训练阶段 |
| **实现** | 新增 `blind_threshold_param`，通过 sigmoid 映射到 [0, 1]，训练时从 Net_R 读取 |
| **预期效果** | 阈值自适应调整，训练初期宽松后期收紧，Blind PSNR 提升 **0.05~0.1 dB** |

---

### 三、训练策略优化（速度与稳定性）

#### 3.1 Warmup + Cosine 退火

| 项目 | 说明 |
|------|------|
| **修改文件** | `train.py` — `Trainer.__init__` |
| **目的** | 训练初期学习率过大会导致震荡，后期恒定学习率收敛慢 |
| **实现** | 前 `warmup_epochs` 线性预热 (0 → base_lr)，之后按 cosine 退火 |
| **配置** | `warmup_epochs` (默认 3) |
| **预期效果** | 训练初期更稳定，后期收敛更充分，PSNR 提升 **0.05~0.15 dB** |

#### 3.2 torch.compile 加速

| 项目 | 说明 |
|------|------|
| **修改文件** | `main.py` |
| **目的** | 利用 PyTorch 2.0+ 的图编译加速训练 |
| **实现** | `torch.compile(model, mode='reduce-overhead')` |
| **配置** | `use_compile` (默认 True) |
| **预期效果** | 训练速度提升 **1.3~1.5 倍**，不影响精度 |

#### 3.3 梯度累积

| 项目 | 说明 |
|------|------|
| **修改文件** | `train.py` — `Trainer.train` |
| **目的** | 显存不足时无法增大 batch size，梯度累积可在小显存下模拟大 batch |
| **实现** | 每 `grad_accum_steps` 步累加梯度后才执行 optimizer.step() |
| **配置** | `grad_accum_steps` (默认 1，即不累积) |
| **预期效果** | 等效 batch size 增大 N 倍，梯度估计更准确，训练更稳定 |

---

### 四、数据增强强化（泛化能力）

| 增强类型 | 概率 | 修改文件 | 预期效果 |
|----------|------|----------|----------|
| **水平翻转** | 50% | `data_blindpixel.py` | 原有 |
| **垂直翻转** | 50% | `data_blindpixel.py` | 新增，上下对称性提升 |
| **90° 旋转** | 100% (随机 4 方向) | `data_blindpixel.py` | 新增，含光流同步变换 |
| **亮度/对比度** | 30% | `data_blindpixel.py` | 新增，模拟不同曝光条件 |
| **时序反转** | 30% | `data_blindpixel.py` | 新增，运动方向对称性 |

**预期效果**: 减少过拟合，验证集 PSNR 提升 **0.1~0.3 dB**。

---

### 优化效果预估

| 指标 | 优化前 | 预期提升 | 主要贡献阶段 |
|------|--------|----------|-------------|
| **PSNR** | baseline | **+0.4~1.2 dB** | Phase 1 损失函数 + Phase 3 训练策略 |
| **Blind PSNR** | baseline | **+0.5~1.5 dB** | Phase 2 架构 + Phase 1 损失函数 |
| **训练速度** | baseline | **1.3~1.5 倍** | torch.compile |
| **收敛稳定性** | baseline | 显著改善 | Warmup + Charbonnier + 梯度累积 |
| **参数量** | 6.34M | 几乎不变 | Gate/SE 增量可忽略 |

---

## 配置参数参考

### 新增配置项一览

| 参数 | 所属节 | 默认值 | 说明 |
|------|--------|--------|------|
| `warmup_epochs` | `[training]` | 3 | 学习率预热轮数 |
| `use_compile` | `[training]` | True | 启用 torch.compile 加速 |
| `grad_accum_steps` | `[training]` | 1 | 梯度累积步数 |
| `charbonnier_eps` | `[loss]` | 1e-3 | Charbonnier 平滑系数 |
| `ssim_loss_weight` | `[loss]` | 0.1 | 全图 SSIM 损失权重 |
| `blind_ssim_loss_weight` | `[loss]` | 0.2 | 盲元区域 SSIM 损失权重 |
| `grad_loss_weight` | `[loss]` | 0.05 | Sobel 梯度损失权重 |
| `blind_fft_loss_weight` | `[loss]` | 0.05 | 盲元频域损失权重 |

### 关键超参调优建议

```ini
# 追求 PSNR（全图质量）
ssim_loss_weight = 0.15
grad_loss_weight = 0.08

# 追求 Blind PSNR（盲元修复质量）
blind_ssim_loss_weight = 0.3
blind_fft_loss_weight = 0.1

# 小显存环境（< 8GB）
batch_size = 1
grad_accum_steps = 4    # 等效 batch_size=4
use_compile = True      # 显存优化

# 大显存环境（>= 24GB）
batch_size = 4
grad_accum_steps = 1
amp_dtype = bf16
```

---

## 数据准备

数据集需要包含训练/验证/测试三个子集，目录结构如下：

```
data/
├── train_blur/          # 训练集模糊图像
│   ├── 001/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── 002/
├── train_sharp/         # 训练集清晰图像（真值）
│   └── ...
├── train_flow/          # 训练集光流（RAFT 生成）
│   └── ...
├── val_blur/
├── val_sharp/
├── val_flow/
├── test_blur/
├── test_sharp/
└── test_mask/           # 盲元坐标标注（用于定量评估）
    └── 001/
        ├── blind_pixel_coords.csv
        └── flash_pixel_coords.csv
```

### 生成仿真数据

```bash
# 使用自适应盲元仿真工具
python fangzhen_adaptive.py --data_root ./data --mode train

# 导入新数据集并自动运行仿真
python add_new_dataset.py --src /path/to/videos --dataset-root ./data --apply
```

---

## 训练与测试

### Stage 1: 退化学习

训练 Net_D 学习模糊核和光流对齐，输入为模糊序列 + 清晰序列真值。

```bash
python main.py --train --config_path experiment.stage1.cfg
```

关键输出:
- `results/{exp_name}/model_stage1/model_best.pt` — 最优 Net_D 权重
- `results/{exp_name}/stage1_train_log.txt` — 训练日志
- `results/{exp_name}/stage1_val_log.txt` — 验证日志

### Stage 2: 盲元修复

加载 Stage 1 权重，联合训练 Net_D + Net_R。

```bash
python main.py --train --config_path experiment.stage2.cfg
```

关键输出:
- `results/{exp_name}/model_stage2/model_best_psnr.pt` — PSNR 最优模型
- `results/{exp_name}/model_stage2/model_best_blindpsnr.pt` — Blind PSNR 最优模型
- `results/{exp_name}/model_stage2/model_best.pt` — PSNR + BlindPSNR 同时最优

### 测试

```bash
python main.py --test --config_path experiment.stage2.cfg
```

测试输出:
- `results/{exp_name}/test/` — 修复后图像
- `results/{exp_name}/triple_comparison/` — 输入/输出/真值 三拼对比图
- `results/{exp_name}/blind_eval/test_blind_metrics.csv` — 盲元区域定量指标

### 断点续训

```ini
# experiment.stage1.cfg 或 experiment.stage2.cfg
finetuning = True  # 自动加载 latest.pt
```

从上次中断处继续训练，学习率和优化器状态一并恢复。

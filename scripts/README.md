# Fusion Gate 独立流程说明

这个目录提供了一套独立工作流，用来复用你已经训练好的 Stage-2 checkpoint，且**不需要修改** `model.py`。

## 包含哪些文件

- `eval_stage2_or_gate.py`：评估原始 stage2 模型，或评估加了 fusion gate 的模型。
- `finetune_fusion_gate.py`：冻结基础模型，只微调轻量 gate（通常 20-40 个 epoch）。
- `compare_base_vs_gate.py`：一条命令连续跑 base 与 gate，快速做 A/B 对比。
- `fusion_gate_wrapper.py`：gate 的轻量封装模块。
- `fusion_gate.cfg`：统一管理 finetune/eval/compare 参数的配置文件。

## 0) 先配置一次

编辑 `scripts/fusion_gate.cfg`，重点关注以下 section：
- `[common]`：公共 `config_path`
- `[finetune]`：gate 微调相关参数
- `[eval]`：评估默认参数
- `[compare]`：A/B 对比默认参数

参数优先级规则：**命令行参数 > `fusion_gate.cfg` > 脚本内置默认值**。

## 1) 先评估现有 Stage-2 模型

```powershell
python .\scripts\eval_stage2_or_gate.py --gate_config_path .\scripts\fusion_gate.cfg
```

## 2) 如果填补力度不够，再做 gate 微调（20-40 epoch）

```powershell
python .\scripts\finetune_fusion_gate.py --gate_config_path .\scripts\fusion_gate.cfg
```

微调完成后会保存：
- `./results/gate_finetune/gate_latest.pt`
- `./results/gate_finetune/gate_best.pt`

## 3) 评估 gated 模型

```powershell
python .\scripts\eval_stage2_or_gate.py --gate_config_path .\scripts\fusion_gate.cfg --gate_ckpt .\results\gate_finetune\gate_best.pt --fill_strength 1.1
```

## 4) 快速 A/B 对比

```powershell
python .\scripts\compare_base_vs_gate.py --gate_config_path .\scripts\fusion_gate.cfg
```

## 备注

- 这套流程会保持 FMANet 基础模型冻结。
- 微调时只训练 gate，不改动原始主干参数。
- 如果你不想训练，也可以只在评估阶段扫 `--fill_strength`。


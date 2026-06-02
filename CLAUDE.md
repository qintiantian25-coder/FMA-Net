# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Stage 1 training (degradation learning: Net_D only)
python main.py --train --config_path ./experiment.stage1.cfg

# Stage 2 training (restoration + blind pixel handling: Net_D + Net_R)
python main.py --train --config_path ./experiment.stage2.cfg

# Testing (evaluates on test set, generates images + quantitative metrics)
python main.py --test --config_path ./experiment.cfg

# Add a new dataset (sorts subfolders by image count, assigns train/val/test, runs blind pixel simulation)
python add_new_dataset.py --src /path/to/raw_frames --apply
```

All configuration is centralized in `.cfg` files read by `Config` (`config.py`). The three provided configs (`experiment.cfg`, `experiment.stage1.cfg`, `experiment.stage2.cfg`) represent different training phases.

## Architecture

FMA-Net is a **two-stage blind pixel restoration network** for grayscale video. It works on 5-frame sequences (configurable via `num_seq`).

### Stage 1 — Degradation Learning (Net_D, `model.py:437`)
- Input: LR blur sequence → Feature extractor (Conv3d + RRDB) → FRMA blocks
- Outputs: degradation kernel `KD`, optical flow `f_Y`, anchor features
- Trained with: reconstruction loss (smart_recon), HR warping loss, flow loss, D_TA loss

### Stage 2 — Restoration (Net_R, `model.py:496`)
- Loads pre-trained Net_D from Stage 1 output directory
- FRMA blocks use Restormer-style MDTA (`RestormerAttention`) instead of cross-attention
- **Three-branch output** with learnable fusion:
  1. **res**: self-reconstruction branch (Conv2d + PixelShuffle)
  2. **duf**: neighborhood compensation via Dynamic Upsampling
  3. **blind_res**: blind pixel residual correction (gated by `blind_gate`)
- Learnable parameters: `base_alpha`, `base_beta` (fusion weights, sigmoid-constrained), `blind_res_scale` (softplus), loss weight params
- Blind mask: constructed from GT diff in training; from `blind_infer_threshold` on center frame during inference

### Key modules in `model.py`
| Module | Description |
|--------|-------------|
| `DynamicDownsampling` / `DynamicUpampling` | Learned kernel-based down/up-sampling |
| `ImageBWarp` / `MultiFlowBWarp` | Optical flow warping with cached grids |
| `RDB` / `RRDB` | Residual Dense Blocks (3D convs) |
| `FRMA` | Flow-guided Residual Multi-Attention: RDB → flow warp → multi-head attention |
| `MultiAttentionBlock` | LayerNorm → Attention → FeedForward (GDFN), optionally with dual attention |
| `RestormerAttention` | Channel-wise self-attention (MDTA); used in Stage 2 FRMA |
| `Attention` | Cross-attention (Q from flow, KV from features); used in Stage 1 FRMA |

### Data pipeline
- **Active dataset**: `BlindPixelDataset` in `data_blindpixel.py` — expects `{train,val,test}_{blur,sharp,flow}` directory structure
- **Legacy dataset**: `REDS_Dataset` in `data.py` — expects REDS4-style `{train,val}_blur_bicubic` with X4 scale
- `main.py` currently imports from `data_blindpixel`; the older `data.py` is kept for reference
- Data format: grayscale, [0,1] normalized, shape `[B, C, T, H, W]`

### Training details (`train.py`)
- AMP with bf16/fp16, gradient scaling, overflow recovery with automatic LR decay
- `smart_recon_loss`: top-k hardest pixels get L2 (×1000 scale), rest get L1
- Stage 2 loss: restoration (non-blind area) + blind_restore (blind area L1) + blind_res (residual L1 in blind area) + boundary/gate regularization
- Checkpoint strategy: saves `model_best_psnr.pt`, `model_best_blindpsnr.pt`, and `model_best.pt` (when both improve)

### Blind pixel simulation (`fangzhen.py`, `fangzhen_adaptive.py`)
- `fangzhen.py`: original fixed-parameter simulation (tight blob clusters)
- `fangzhen_adaptive.py`: adaptive simulation scaled by image dimensions; used by `add_new_dataset.py`
- Generates both static blind pixels and per-frame "flash" pixels, with CSV coordinate records

### Preprocessing (`preprocessing/`)
- `generate_flow.py`: generates RAFT pseudo-ground-truth optical flow from sharp sequences
- `extractor.py`: frame extraction from video
- Other utilities for the REDS4 dataset pipeline

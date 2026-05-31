# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FMA-Net (Feature-Aligned Multi-Attention Network) is a two-stage video blind-pixel restoration network. It repairs dead/stuck pixels in video sequences by learning degradation patterns (Stage 1) and then restoring clean frames via multi-head attention and dynamic upsampling (Stage 2).

## Commands

```bash
# Stage 1 training (learn degradation)
python main.py --train --config_path experiment.stage1.cfg

# Stage 2 training (restoration — requires Stage 1 pretrained Net_D)
python main.py --train --config_path experiment.stage2.cfg

# Test on test set
python main.py --test --config_path experiment.stage2.cfg
```

There is no `requirements.txt` or `setup.py`. Dependencies: `torch`, `torchvision`, `opencv-python`, `numpy`, `einops`.

## Architecture

**Two-stage pipeline** (`model.py`):
- **Net_D** (`degradation_learning_network`): Learns degradation kernels, alignment flow, and anchor features from blur sequences. Uses residual dense blocks (RRDB) + FRMA blocks with standard cross-attention.
- **Net_R** (`restoration_network`): Fuses features via two branches — a residual self-reconstruction path and a DUF (dynamic upsampling) neighbor-compensation path. Uses Restormer-style MDTA (channel-wise self-attention). Includes learnable fusion parameters (`base_alpha`, `base_beta`, `blind_res_scale`) and a blind-pixel gate network.
- **FMANet** (top-level): Wraps Net_D + Net_R. Stage 1 only runs Net_D; Stage 2 runs both.

**Net_R blind-pixel handling**:
- **Multi-layer gate** (`blind_gate`): 3-layer Conv2d + Sigmoid for precise blind pixel localization
- **SE channel attention** (`blind_se`): Squeeze-and-Excitation block enhancing blind residual features
- **Learnable threshold** (`blind_threshold_param`): Adaptive blind mask threshold (sigmoid-constrained)
- **Fusion**: `output = (1 - gate*mask) * res + gate*mask * (alpha*duf + beta*res + blind_res)`

**Key modules in `model.py`**:
- `DynamicDownsampling` / `DynamicUpampling`: Learned kernel-based down/up-sampling
- `ImageBWarp` / `MultiFlowBWarp`: Flow-based image warping
- `RRDB` → `RDB` → `DenseLayer`: Residual-in-residual dense blocks for feature extraction
- `FRMA` (Feature Refinement with Multi-Attention): Core iterative block combining RDB, flow warping, and multi-head attention
- `Attention`: Original cross-attention (used in Net_D)
- `RestormerAttention`: Channel-wise self-attention a la Restormer (used in Net_R)

**Training** (`train.py`):
- `Trainer` class owns two optimizers (`optimizer_D`, `optimizer_R`) and two schedulers
- AMP (automatic mixed precision) with `torch.amp.GradScaler` and stability safeguards (NaN detection, overflow LR decay, forced FP32 fallback)
- Smart reconstruction loss: top-k hardest pixels get L2 weighting, rest get L1
- **Enhanced loss functions**: Charbonnier (smooth L1) replaces L1 for restoration/blind/warping; SSIM loss, Sobel gradient loss, and FFT frequency-domain loss supplement reconstruction
- **Warmup + Cosine LR**: Linear warmup for `warmup_epochs` then cosine annealing
- **Gradient accumulation**: `grad_accum_steps` for larger effective batch sizes
- **torch.compile**: Optional `use_compile` for PyTorch 2.0+ speedup

**Stage 2 total loss** = `restoration_weight * restoration_loss(Charbonnier)` + `blind_restore_weight * blind_restore_loss(Charbonnier)` + `blind_res_weight * blind_res_loss(Charbonnier)` + `ssim_weight * SSIM_loss` + `blind_ssim_weight * blind_SSIM_loss` + `grad_weight * gradient_loss(Sobel)` + `fft_weight * blind_fft_loss` + `recon_loss` + `hr/lr_warping_loss(Charbonnier)` + `flow_loss` + `D_TA/R_TA_loss` + `boundary_loss(Charbonnier)` + `gate_tv_loss` + `gate_sparsity_loss`

**Data loading**:
- `data_blindpixel.py`: Current dataset loader (`BlindPixelDataset`), used by `main.py`. Expects `{train,val,test}_{blur,sharp,flow}/` subdirectories with numerically-named sequences. Uses natural sorting.
- Data augmentation: horizontal/vertical flips, random 90° rotations, brightness/contrast jitter, temporal reversal
- `data.py`: Legacy REDS-format dataset loader with bicubic/X4 conventions.

**Configuration** (`config.py`):
- INI-style `.cfg` files parsed by `configparser`. Sections: `[experiment]`, `[training]`, `[network]`, `[fusion]`, `[validation]`, `[test]`, `[loss]`.
- Most hyperparameters have `fallback` defaults for backward compatibility with older config files.

**Simulation tools**:
- `fangzhen.py` / `fangzhen_adaptive.py`: Generate synthetic blind pixels on clean images. `fangzhen_adaptive.py` has parameterized blind-pixel generation and is the preferred version.
- `add_new_dataset.py`: Import new video sequences into the project data structure, optionally running blind-pixel simulation.

**Preprocessing** (`preprocessing/`):
- `generate_flow.py`: Generate RAFT optical flow for training data
- `datasets.py`, `extractor.py`, `raft.py`: RAFT model and inference utilities
- `update.py`, `corr.py`, `generate_reds4.py`: REDS dataset conversion utilities

## Key conventions

- Image tensors are in `[B, C, T, H, W]` format (channel-first, sequence as third dim)
- Images normalized to `[0, 1]` range
- Center frame index is always `T // 2`
- Blind masks are `[B, 1, H, W]` in `[0, 1]` range
- Natural sorting is used for all filenames (1, 2, ..., 10, not 1, 10, 2)

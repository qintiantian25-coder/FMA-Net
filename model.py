import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers
from einops import rearrange


# FMA-Net 核心模型代碼
# 功能:這是一個兩階段的視頻恢復網絡。
# 第一階段(NetD)學習圖像如何變模糊，生成退化內核和對齊光流;
# 第二階段(NetR)利用這些特徵，通過多頭注意力和動態上採樣，將模糊序列還原。

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# Feathering utilities removed per request. Selective feathering and laplacian blending
# logic were deleted to simplify the codebase. If you need them later, they can be
# restored from version control or reimplemented as separate utility modules.


class DynamicDownsampling(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super(DynamicDownsampling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def kernel_normalize(self, kernel):
        return F.softmax(kernel, dim=-1)

    def forward(self, x, kernel, DT):
        b, _, t, h, w = kernel.shape
        kernel = kernel.permute(0, 3, 4, 2, 1).contiguous()
        kernel = kernel.view(b, h, w, t * self.kernel_size * self.kernel_size)
        kernel = self.kernel_normalize(kernel)
        kernel = kernel.unsqueeze(dim=1)

        num_pad = (self.kernel_size - self.stride) // 2
        x = F.pad(x, (num_pad, num_pad, num_pad, num_pad, 0, 0), mode="replicate")
        x = x.unfold(3, self.kernel_size, self.stride)
        x = x.unfold(4, self.kernel_size, self.stride)
        x = x.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
        x = x.view(b, -1, h, w, t * self.kernel_size * self.kernel_size)

        x = x * kernel
        x = torch.sum(x, -1)

        DT = F.pad(DT, (num_pad, num_pad, num_pad, num_pad, 0, 0), mode="replicate")
        DT = DT.unfold(3, self.kernel_size, self.stride)
        DT = DT.unfold(4, self.kernel_size, self.stride)
        DT = DT.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
        DT = DT.view(b, -1, h, w, t * self.kernel_size * self.kernel_size)

        DT = DT * kernel
        DT = torch.sum(DT, -1)
        x = x / (DT + 1e-8)
        return x


class DynamicUpampling(torch.nn.Module):
    def __init__(self, kernel_size, scale):
        super(DynamicUpampling, self).__init__()
        self.kernel_size = kernel_size
        self.scale = scale

    def kernel_normalize(self, kernel):
        K = kernel.shape[-1]
        kernel = kernel - torch.mean(kernel, dim=-1, keepdim=True)
        kernel = kernel + 1.0 / K
        return kernel

    def forward(self, x, kernel):
        b, c, t, h, w = x.shape
        kernel = rearrange(kernel, 'b (s1 s2 k1 k2) t h w -> b h w s1 s2 (t k1 k2)', s1=self.scale, s2=self.scale,
                           k1=self.kernel_size, k2=self.kernel_size)
        kernel = kernel.unsqueeze(dim=1)
        kernel = self.kernel_normalize(kernel)

        num_pad = self.kernel_size // 2
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = F.pad(x, (num_pad, num_pad, num_pad, num_pad), mode="replicate")
        x = F.unfold(x, [self.kernel_size, self.kernel_size], padding=0)
        x = rearrange(x, '(b t) (c k1 k2) (h w) -> b c h w (t k1 k2)', b=b, t=t, c=c, k1=self.kernel_size,
                      k2=self.kernel_size, h=h, w=w)
        x = x.unsqueeze(dim=4).unsqueeze(dim=5)
        x = torch.sum(x * kernel, dim=-1)
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(b, c, self.scale * h, self.scale * w)
        return x


def backwarp(x, flow, objBackwarpcache):
    if 'grid' + str(flow.dtype) + str(flow.device) + str(flow.shape[2]) + str(flow.shape[3]) not in objBackwarpcache:
        tenHor = torch.linspace(start=-1.0, end=1.0, steps=flow.shape[3], dtype=flow.dtype,
                                device=flow.device).view(1, 1, 1, -1).repeat(1, 1, flow.shape[2], 1)
        tenVer = torch.linspace(start=-1.0, end=1.0, steps=flow.shape[2], dtype=flow.dtype,
                                device=flow.device).view(1, 1, -1, 1).repeat(1, 1, 1, flow.shape[3])
        objBackwarpcache[
            'grid' + str(flow.dtype) + str(flow.device) + str(flow.shape[2]) + str(flow.shape[3])] = torch.cat(
            [tenHor, tenVer], 1)

    if flow.shape[3] == flow.shape[2]:
        flow = flow * (2.0 / ((flow.shape[3] and flow.shape[2]) - 1.0))
    elif flow.shape[3] != flow.shape[2]:
        flow = flow * torch.tensor(data=[2.0 / (flow.shape[3] - 1.0), 2.0 / (flow.shape[2] - 1.0)], dtype=flow.dtype,
                                   device=flow.device).view(1, 2, 1, 1)

    return nn.functional.grid_sample(input=x, grid=(objBackwarpcache['grid' + str(flow.dtype) + str(flow.device) + str(
        flow.shape[2]) + str(flow.shape[3])] + flow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros',
                                     align_corners=True)


class ImageBWarp(torch.nn.Module):
    def __init__(self, scale, num_seq):
        super(ImageBWarp, self).__init__()
        self.scale = scale
        self.num_seq = num_seq
        self.objBackwarpcache = {}
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, f):
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        f = rearrange(f, 'b c t h w -> (b t) c h w')
        weight = f[:, 2:3, :, :]
        flow = f[:, :2, :, :]
        weight = self.sigmoid(weight)
        ones = torch.ones_like(x)

        if self.scale != 1:
            flow = self.scale * F.interpolate(flow, scale_factor=(self.scale, self.scale), mode='bilinear',
                                              align_corners=False)
            weight = F.interpolate(weight, scale_factor=(self.scale, self.scale), mode='bilinear', align_corners=False)

        x = backwarp(x, flow, self.objBackwarpcache)
        ones = backwarp(ones, flow, self.objBackwarpcache)
        ones = ones * weight
        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.num_seq)
        ones = rearrange(ones, '(b t) c h w -> b c t h w', t=self.num_seq)
        weight = rearrange(weight, '(b t) c h w -> b c t h w', t=self.num_seq)
        return ones, x * weight


class MultiFlowBWarp(torch.nn.Module):
    def __init__(self, dim, num_seq, num_flow):
        super(MultiFlowBWarp, self).__init__()
        self.dim = dim
        self.num_seq = num_seq
        self.num_flow = num_flow
        self.objBackwarpcache = {}
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, F, f):
        F = rearrange(F, 'b (n c) t h w -> (b n t) c h w', c=self.dim // self.num_flow, n=self.num_flow)
        f = rearrange(f, 'b (n c) t h w -> (b n t) c h w', c=3, n=self.num_flow)
        weight = f[:, 2:3, :, :]
        flow = f[:, :2, :, :]
        weight = self.sigmoid(weight)
        F = backwarp(F, flow, self.objBackwarpcache)
        F = F * weight
        F = rearrange(F, '(b n t) c h w -> b (n c) t h w', t=self.num_seq, n=self.num_flow)
        return F


class PixelShuffleBlock(torch.nn.Module):
    def __init__(self, channels, bias, scale):
        super(PixelShuffleBlock, self).__init__()
        self.scale = scale
        if scale > 1:
            out_c = channels * (scale ** 2)
            self.conv1 = nn.Conv2d(channels, out_c, 3, 1, 1, bias=bias)
            self.shuffle = nn.PixelShuffle(scale)
        else:
            self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=bias)
            self.shuffle = nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=bias)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.shuffle(self.conv1(x)))
        x = self.relu(self.conv2(x))
        return x


class DenseLayer(torch.nn.Module):
    def __init__(self, dim, growth_rate, bias):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(dim, growth_rate, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class RDB(torch.nn.Module):
    def __init__(self, dim, growth_rate, num_dense_layer, bias):
        super(RDB, self).__init__()
        self.layer = [DenseLayer(dim=dim + growth_rate * i, growth_rate=growth_rate, bias=bias) for i in
                      range(num_dense_layer)]
        self.layer = torch.nn.Sequential(*self.layer)
        self.conv = nn.Conv3d(dim + growth_rate * num_dense_layer, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out = self.layer(x)
        out = self.conv(out)
        out = out + x
        return out


class RRDB(nn.Module):
    def __init__(self, dim, num_RDB, growth_rate, num_dense_layer, bias):
        super(RRDB, self).__init__()
        self.RDBs = nn.ModuleList(
            [RDB(dim=dim, growth_rate=growth_rate, num_dense_layer=num_dense_layer, bias=bias) for _ in range(num_RDB)])
        self.conv = nn.Sequential(*[nn.Conv3d(dim * num_RDB, dim, kernel_size=1, padding=0, stride=1, bias=bias),
                                    nn.Conv3d(dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias)])

    def forward(self, x):
        input = x
        RDBs_out = []
        for rdb_block in self.RDBs:
            x = rdb_block(x)
            RDBs_out.append(x)
        x = self.conv(torch.cat(RDBs_out, dim=1))
        return x + input


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    """
    Restormer-style MDTA (Multi-DConv Head Transposed Attention) adapted to accept
    an optional guidance tensor `f`. If `f` is provided we add it to the attention input
    (x + f) so existing cross-attention call sites remain compatible.

    Behavior: compute q/k/v from the same input (self-attention along channel heads),
    use depthwise 3x3 conv for locality, normalize q/k (L2) and scale by a learnable
    per-head temperature before softmax.
    """
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv in one conv followed by depthwise conv (preserves spatial info)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                     groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, f=None):
        # x: context tensor (b, c, h, w)
        # f: optional guidance/query tensor (b, c, h, w) — will be added to x if provided
        if f is None:
            x_in = x
        else:
            # keep shape and device consistent
            x_in = x + f

        b, c, h, w = x_in.shape

        qkv = self.qkv_dwconv(self.qkv(x_in))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # normalize along token dimension (h*w)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Channel-wise attention (Restormer-style): compute attention over channel projections
        # q: [B, head, C, HW], k: [B, head, C, HW]
        # attn = q @ k^T -> [B, head, C, C]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # apply attention to v: [B, head, C, C] @ [B, head, C, HW] -> [B, head, C, HW]
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class MultiAttentionBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias, is_DA):
        super(MultiAttentionBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.co_attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
        if is_DA:
            self.norm3 = LayerNorm(dim, LayerNorm_type)
            self.da_attn = Attention(dim, num_heads, bias)
            self.norm4 = LayerNorm(dim, LayerNorm_type)
            self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, Fw, F0_c, Kd):
        Fw = Fw + self.co_attn(self.norm1(Fw), F0_c)
        Fw = Fw + self.ffn1(self.norm2(Fw))
        if Kd is not None:
            Fw = Fw + self.da_attn(self.norm3(Fw), Kd)
            Fw = Fw + self.ffn2(self.norm4(Fw))
        return Fw


class FRMA(torch.nn.Module):
    def __init__(self, dim, num_seq, growth_rate, num_dense_layer, num_flow, num_multi_attn, num_heads,
                 LayerNorm_type, ffn_expansion_factor, bias, is_DA=False, is_first_f=False, is_first_Fw=False):
        super(FRMA, self).__init__()
        self.rdb = RDB(dim, growth_rate, num_dense_layer, bias)
        self.rdb_KD = RDB(dim, growth_rate, num_dense_layer, bias) if is_DA else None
        self.conv_KD = nn.Conv2d(dim * num_seq, dim, kernel_size=1, padding=0, stride=1, bias=bias) if is_DA else None
        self.bwarp = MultiFlowBWarp(dim, num_seq, num_flow)
        self.conv_Fw = nn.Conv2d(dim * num_seq if is_first_Fw else dim + dim * num_seq, dim, kernel_size=1, padding=0,
                                 stride=1, bias=bias)
        self.conv_f = nn.Sequential(
            nn.Conv3d(dim * 2 if is_first_f else dim * 2 + 3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3],
                      padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))
        self.multi_attn_block = nn.ModuleList(
            [MultiAttentionBlock(dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias, is_DA) for _ in
             range(num_multi_attn)])

    def forward(self, F, Fw, f, F0_c, KD=None):
        B, C, T, H, W = F.shape
        F = self.rdb(F)
        if f is not None:
            warped_F = self.bwarp(F, f)
            f = f + self.conv_f(torch.cat([F0_c.repeat([1, 1, T, 1, 1]), f, warped_F], dim=1))
        else:
            f = self.conv_f(torch.cat([F0_c.repeat([1, 1, T, 1, 1]), F], dim=1))
        warped_F = self.bwarp(F, f)
        warped_F = rearrange(warped_F, 'b c t h w -> b (c t) h w')
        if Fw is not None:
            Fw = self.conv_Fw(torch.cat([Fw, warped_F], dim=1))
        else:
            Fw = self.conv_Fw(warped_F)
        if KD is not None:
            KD = self.rdb_KD(KD)
            KD = rearrange(KD, 'b c t h w -> b (c t) h w')
            KD = self.conv_KD(KD)
        for blk in self.multi_attn_block:
            Fw = blk(Fw, F0_c.squeeze(dim=2), KD)
        return F, Fw, f


class Net_D(torch.nn.Module):
    def __init__(self, config):
        super(Net_D, self).__init__()
        self.dim = config.dim
        in_channels = config.in_channels
        dim = config.dim
        num_seq = config.num_seq
        ds_kernel_size = config.ds_kernel_size
        num_RDB = config.num_RDB
        growth_rate = config.growth_rate
        num_dense_layer = config.num_dense_layer
        num_flow = config.num_flow
        num_FRMA = config.num_FRMA
        num_transformer_block = config.num_transformer_block
        num_heads = config.num_heads
        LayerNorm_type = config.LayerNorm_type
        ffn_expansion_factor = config.ffn_expansion_factor
        bias = config.bias

        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            RRDB(dim=dim, num_RDB=num_RDB, growth_rate=growth_rate, num_dense_layer=num_dense_layer, bias=config.bias))
        self.FRMA_blocks = nn.ModuleList(
            [FRMA(dim, num_seq, growth_rate, num_dense_layer, num_flow, num_transformer_block, num_heads,
                  LayerNorm_type, ffn_expansion_factor, bias, is_DA=False, is_first_f=True if i == 0 else False,
                  is_first_Fw=True if i == 0 else False) for i in range(num_FRMA)])
        self.f_conv = nn.Sequential(
            nn.Conv3d(3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(3 * num_flow, 3, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))
        self.d_conv = nn.Sequential(
            nn.Conv3d(dim // num_seq, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(dim, ds_kernel_size * ds_kernel_size, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1,
                      bias=bias))
        self.a_conv = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv3d(dim, in_channels, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1,
                                              bias=bias))

    def forward(self, x):
        B, C, T, H, W = x.shape
        F = self.feature_extractor(x)
        F0_c = F[:, :, T // 2:T // 2 + 1, :, :]
        Fw = None
        f = None
        for blk in self.FRMA_blocks:
            F, Fw, f = blk(F, Fw, f, F0_c)
        Fw = rearrange(Fw, 'b (c t) h w -> b c t h w', t=T, c=self.dim // T)
        KD = self.d_conv(Fw)
        f_Y = self.f_conv(f)
        anchor = self.a_conv(F)
        return F, KD, f_Y, f, anchor


class Net_R(torch.nn.Module):
    def __init__(self, config):
        super(Net_R, self).__init__()
        in_channels = config.in_channels
        dim = config.dim
        num_seq = config.num_seq
        scale = config.scale

        # --- 配置權重 (可學習) ---
        # 将 base_alpha/base_beta/blind_res_scale 注册为可训练参数。
        # 使用可逆映射保证数值约束：
        #  - alpha, beta -> sigmoid(param) 映射到 (0,1)
        #  - blind_res_scale -> softplus(param) 保证为正
        self.base_alpha_param = nn.Parameter(torch.tensor(float(getattr(config, 'base_alpha', 0.8)), dtype=torch.float32))
        self.base_beta_param = nn.Parameter(torch.tensor(float(getattr(config, 'base_beta', 0.2)), dtype=torch.float32))
        # store initial scale param; use softplus on this param in forward to ensure > 0
        self.blind_res_scale_param = nn.Parameter(torch.tensor(float(getattr(config, 'blind_res_scale', 1.20)), dtype=torch.float32))
        # --- 盲元损失权重（可学习） ---
        # restoration_loss_weight: non-blind region (final output outside blind) 的权重初值
        # blind_restore_loss_weight: final output 在盲元区域的权重初值
        # blind_res_loss_weight: blind residual 分支的权重初值
        # restoration_loss_weight: non-blind region loss weight is controlled by config (fixed),
        # not a learnable parameter to avoid the optimizer disabling non-blind protection.
        self.blind_restore_loss_weight_param = nn.Parameter(torch.tensor(float(getattr(config, 'blind_restore_loss_weight', 0.6)), dtype=torch.float32))
        self.blind_res_loss_weight_param = nn.Parameter(torch.tensor(float(getattr(config, 'blind_res_loss_weight', 2.0)), dtype=torch.float32))

        ds_kernel_size = config.ds_kernel_size
        us_kernel_size = config.us_kernel_size
        num_RDB = config.num_RDB
        growth_rate = config.growth_rate
        num_dense_layer = config.num_dense_layer
        num_flow = config.num_flow
        num_FRMA = config.num_FRMA
        num_transformer_block = config.num_transformer_block
        num_heads = config.num_heads
        LayerNorm_type = config.LayerNorm_type
        ffn_expansion_factor = config.ffn_expansion_factor
        bias = config.bias

        self.feature_extractor = nn.Sequential(
            nn.Conv3d(config.in_channels + dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            RRDB(dim=dim, num_RDB=num_RDB, growth_rate=growth_rate, num_dense_layer=num_dense_layer, bias=config.bias)
        )
        self.FRMA_blocks = nn.ModuleList([FRMA(dim, num_seq, growth_rate, num_dense_layer, num_flow,
                                               num_transformer_block, num_heads, LayerNorm_type, ffn_expansion_factor,
                                               bias, is_DA=True, is_first_Fw=True if i == 0 else False) for i in
                                          range(num_FRMA)])
        self.f_conv1 = nn.Sequential(
            nn.Conv3d(3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))
        self.d_conv = nn.Sequential(
            nn.Conv3d(ds_kernel_size * ds_kernel_size, dim, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.LeakyReLU(0.2, True), nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1, bias=bias))
        self.res_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=bias)
        self.res_conv2 = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1, stride=1, bias=bias)
        self.relu = nn.LeakyReLU(0.2, True)
        self.upsample = PixelShuffleBlock(dim, bias=bias, scale=scale)

        self.blind_res_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=bias)
        self.blind_res_conv2 = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1, stride=1, bias=bias)
        self.blind_gate_conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1, stride=1, bias=bias)
        self.blind_gate_act = nn.Sigmoid()

        # blind_res_scale is now represented by a learnable parameter `blind_res_scale_param` -> remove redundant cfg copy
        self.blind_infer_threshold = float(getattr(config, 'blind_infer_threshold', 0.10))

        self.r_conv = nn.Sequential(
            nn.Conv3d(dim // num_seq, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(dim, us_kernel_size * us_kernel_size * scale * scale, kernel_size=[1, 3, 3], padding=[0, 1, 1],
                      stride=1, bias=bias))
        self.f_conv2 = nn.Sequential(
            nn.Conv3d(3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(3 * num_flow, 3, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))
        self.f_conv3 = nn.Sequential(
            nn.Conv3d(3 * num_flow + 2, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(3 * num_flow, 3, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))
        self.bwarp = ImageBWarp(1, num_seq)
        self.duf = DynamicUpampling(us_kernel_size, scale)
        self.a_conv = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv3d(dim, in_channels, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1,
                                              bias=bias))

    def forward(self, x, F, f, KD, blind_mask=None):
        B, C, T, H, W = x.shape
        F = self.feature_extractor(torch.cat([x, F], dim=1))
        F0_c = F[:, :, T // 2:T // 2 + 1, :, :]
        Fw = None
        f = self.f_conv1(f)
        KD = self.d_conv(KD)

        for blk in self.FRMA_blocks:
            F, Fw, f = blk(F, Fw, f, F0_c, KD)

        # 1. 自身重構分支 (Res)
        res_feat = self.relu(self.res_conv1(Fw))
        res = self.res_conv2(self.upsample(res_feat))

        # 2. 鄰幀補償分支 (DUF)
        KR = rearrange(Fw, 'b (c t) h w -> b c t h w', t=T)
        KR = self.r_conv(KR)
        f_X_initial = self.f_conv2(f)
        _, warped_X_pre = self.bwarp(x, f_X_initial)
        feat_to_cat = torch.cat([f, warped_X_pre, x[:, :, T // 2:T // 2 + 1, :, :].repeat([1, 1, T, 1, 1])], dim=1)
        f_X_final = self.f_conv3(feat_to_cat)
        _, warped_X = self.bwarp(x, f_X_final)
        duf_output = self.duf(warped_X, KR)

        # 3. 盲元預測與掩碼
        blind_feat = self.relu(self.blind_res_conv1(Fw))
        # compute raw blind residual (scale applied below using a learnable positive param)
        raw_blind_res = self.blind_res_conv2(blind_feat)
        blind_gate = self.blind_gate_act(self.blind_gate_conv(blind_feat))

        # Map learnable params to constrained ranges and place on same device as features
        base_alpha = torch.sigmoid(self.base_alpha_param).to(Fw.device)
        base_beta = torch.sigmoid(self.base_beta_param).to(Fw.device)
        blind_res_scale = torch.nn.functional.softplus(self.blind_res_scale_param).to(Fw.device)

        # scaled blind residual
        blind_res = raw_blind_res * blind_res_scale

        if blind_mask is None:
            center_lr = x[:, :, T // 2, :, :]
            blind_mask = (center_lr <= self.blind_infer_threshold).float()
        elif blind_mask.dim() == 5:
            blind_mask = blind_mask[:, :, 0, :, :]

        if blind_mask.shape[-2:] != duf_output.shape[-2:]:
            blind_mask = F.interpolate(blind_mask, size=duf_output.shape[-2:], mode='nearest')

        # final_blind_mask 結合了物理壞點位置和對齊置信度
        final_blind_mask = blind_mask * blind_gate

        # ================================================================
        # 核心改進：局部比例融合邏輯
        # ================================================================
        # 1. 計算盲元區專用的混合內容:
        #    α * 邻帧补偿(duf) + β * 自身重构(res) + blind_res_scale * blind_res
        # Note: base_alpha/base_beta are from sigmoid-mapped learnable params,
        # and blind_res contribution is controlled by blind_res_scale (softplus-mapped param).
        blind_area_fill = (base_alpha * duf_output) + (base_beta * res) + blind_res

        # 2. 局部替換邏輯：
        # 非盲元區 (Mask=0): 直接使用 res (保證背景細節不動)
        # 盲元區 (Mask=1): 使用融合後的 blind_area_fill
        output = (1.0 - final_blind_mask) * res + final_blind_mask * blind_area_fill
        # ================================================================

        base_output = duf_output + res  # 僅供監控
        anchor = self.a_conv(F)
        # 返回额外的 res 与 duf_output，便于验证阶段在外部对部分样本做羽化比较
        return output, warped_X, anchor, blind_res, blind_gate, final_blind_mask, base_output, res, duf_output


class FMANet(torch.nn.Module):
    def __init__(self, config):
        super(FMANet, self).__init__()
        self.stage = config.stage
        self.degradation_learning_network = Net_D(config)
        self.bwarp = ImageBWarp(config.scale, config.num_seq)
        self.ddf = DynamicDownsampling(config.ds_kernel_size, config.scale)
        if self.stage == 2:
            self.restoration_network = Net_R(config)

    def forward(self, x, y=None, blind_mask=None):
        result_dict = {}
        F, KD, f_Y, f, anchor_D = self.degradation_learning_network(x)
        if y is not None:
            ones, warped_Y = self.bwarp(y, f_Y)
            recon = self.ddf(warped_Y, KD, ones)
            result_dict['recon'] = recon
            result_dict['hr_warp'] = warped_Y
            result_dict['image_flow'] = f_Y[:, :2, :, :, :]
            result_dict['F_sharp_D'] = anchor_D

        if self.stage == 1:
            return result_dict
        elif self.stage == 2:
            output, warped_X, anchor_R, blind_res, blind_gate, blind_mask_used, base_output, res, duf_output = self.restoration_network(
                x, F, f, KD, blind_mask=blind_mask
            )
            result_dict['output'] = output
            result_dict['lr_warp'] = warped_X
            result_dict['F_sharp_R'] = anchor_R
            result_dict['blind_res'] = blind_res
            result_dict['blind_gate'] = blind_gate
            result_dict['blind_mask'] = blind_mask_used
            result_dict['base_output'] = base_output
            # expose internal branches for optional evaluation/post-processing (e.g. selective feathering)
            result_dict['res'] = res
            result_dict['duf_output'] = duf_output
            return result_dict
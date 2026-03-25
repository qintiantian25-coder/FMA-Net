import torch
import torch.nn as nn


class FusionGate(nn.Module):
    """Lightweight learnable gate to boost/attenuate base restoration fill strength."""

    def __init__(self, in_channels=1, hidden_channels=16, init_scale=1.0):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels * 3, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.gate_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(self, base_output, center_input, fill_strength=1.0):
        # Use base, center, and residual magnitude to estimate per-pixel fusion confidence.
        residual = base_output - center_input
        features = torch.cat([base_output, center_input, torch.abs(residual)], dim=1)
        gate = torch.sigmoid(self.gate_net(features))

        strength = self.gate_scale * float(fill_strength)
        enhanced = center_input + strength * gate * residual
        return enhanced, gate


class GatedRestorer(nn.Module):
    """Wraps a pretrained FMANet (stage2) and applies only a learnable post-fusion gate."""

    def __init__(self, base_model, in_channels=1, hidden_channels=16, init_scale=1.0):
        super().__init__()
        self.base_model = base_model
        self.gate = FusionGate(in_channels=in_channels, hidden_channels=hidden_channels, init_scale=init_scale)

    def freeze_base(self):
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

    def forward(self, x, y=None, fill_strength=1.0):
        with torch.no_grad():
            base_dict = self.base_model(x, y)

        t = x.shape[2] // 2
        center = x[:, :, t, :, :]

        enhanced, gate_map = self.gate(base_dict['output'], center, fill_strength=fill_strength)

        out = dict(base_dict)
        out['output_base'] = base_dict['output']
        out['output'] = enhanced
        out['gate_map'] = gate_map
        out['gate_scale'] = self.gate.gate_scale.detach()
        return out


from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Video stub encoder ----------------
class VideoBackboneStub(nn.Module):
    """
    Tiny CNN over individual frames, then average across N frames in a clip.
    Input: (B, T, N, 3, H, W)
    Output: (B, T, Dv)
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (C,1,1)
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C, H, W = x.shape
        x = x.view(B * T * N, C, H, W)
        h = self.conv(x).view(B * T * N, -1)  # (BTN, 128)
        h = self.proj(h)  # (BTN, D)
        h = h.view(B, T, N, -1).mean(dim=2)  # average over N frames → (B, T, D)
        return h

# ---------------- Audio stub encoder ----------------
class AudioBackboneStub(nn.Module):
    """
    Small 2D CNN over Mel (M, Lclip)
    Input: (B, T, M, L)
    Output: (B, T, Da)
    """
    def __init__(self, n_mels: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (C,1,1)
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        B, T, M, L = mel.shape
        x = mel.unsqueeze(2)  # (B,T,1,M,L) -> treat M as H and L as W
        x = x.view(B * T, 1, M, L)
        h = self.net(x).view(B * T, -1)  # (BT, 64)
        h = self.proj(h)                 # (BT, D)
        h = h.view(B, T, -1)             # (B, T, D)
        return h

# ---------------- Gated fusion ----------------
class GatedFusion(nn.Module):
    def __init__(self, dv: int, da: int, df: int):
        super().__init__()
        self.proj_v = nn.Linear(dv, df)
        self.proj_a = nn.Linear(da, df)
        self.gate = nn.Sequential(
            nn.Linear(dv + da, df),
            nn.ReLU(inplace=True),
            nn.Linear(df, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_v: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([z_v, z_a], dim=-1))  # (B, T, 1)
        zv = self.proj_v(z_v)
        za = self.proj_a(z_a)
        z = g * za + (1.0 - g) * zv
        return z  # (B, T, df)

# ---------------- Temporal head (ED-TCN-ish) ----------------
class TemporalHeadEDTCN(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d, d, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(d, d, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Conv1d(d, 1, kernel_size=1)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # z: (B,T,d) -> (B,d,T)
        x = z.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.head(x).squeeze(1)  # (B, T)
        # We don't enforce exact length matching for odd T; crop/pad to mask length
        T_mask = mask.shape[1]
        if x.shape[1] != T_mask:
            if x.shape[1] > T_mask:
                x = x[:, :T_mask]
            else:
                pad = torch.zeros(x.shape[0], T_mask - x.shape[1], device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
        return x  # logits (B,T)

# ---------------- Full model ----------------
class MultiModalMILNet(nn.Module):
    def __init__(self, n_mels: int, dv: int = 256, da: int = 256, df: int = 256):
        super().__init__()
        self.enc_v = VideoBackboneStub(out_dim=dv)
        self.enc_a = AudioBackboneStub(n_mels=n_mels, out_dim=da)
        self.fuse = GatedFusion(dv, da, df)
        self.temporal = TemporalHeadEDTCN(df)

    def forward(self, video, audio, mask):
        # video: (B,T,N,3,H,W), audio: (B,T,M,L)
        z_v = self.enc_v(video)  # (B,T,dv)
        z_a = self.enc_a(audio)  # (B,T,da)
        z = self.fuse(z_v, z_a)  # (B,T,df)
        logits = self.temporal(z, mask)  # (B,T)
        return logits

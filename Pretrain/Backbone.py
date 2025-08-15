import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils import SinusoidalEmbedding




# UNet
class ResBlock1D(nn.Module):
    def __init__(self, ch: int, time_dim: int, pos_dim: int, cond_dim: int = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(ch)
        self.ff1 = nn.Linear(ch, ch)
        self.norm2 = nn.LayerNorm(ch)
        self.ff2 = nn.Linear(ch, ch)
        self.time_proj = nn.Linear(time_dim, ch)
        self.pos_proj = nn.Linear(pos_dim, ch) if pos_dim > 0 else None
        self.cond_proj = nn.Linear(cond_dim, ch) if cond_dim > 0 else None
        self.act = nn.SiLU()

    def forward(self, x, t_emb, pos_emb=None, cond_emb=None):
        h = self.ff1(self.norm1(x))
        add = self.time_proj(t_emb)[:, None, :]
        if self.pos_proj is not None and pos_emb is not None:
            add = add + self.pos_proj(pos_emb)[None, :, :]
        if self.cond_proj is not None and cond_emb is not None:
            add = add + self.cond_proj(cond_emb)[:, None, :]
        h = h + add
        h = self.act(h)
        h = self.ff2(self.norm2(h))
        return x + h

class Downsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.proj = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.proj(x)
        x = x.transpose(1,2)
        return x

class Upsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.proj = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.proj(x)
        x = x.transpose(1,2)
        return x

class TrajectoryUNet1D(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, time_dim: int = 128, pos_dim: int = 128, cond_dim: int = 0):
        super().__init__()
        self.input = nn.Linear(feat_dim, hidden)
        self.time_emb = SinusoidalEmbedding(time_dim)
        self.pos_dim = pos_dim
        self.cond_dim = cond_dim

        self.enc1 = ResBlock1D(hidden, time_dim, pos_dim, cond_dim)
        self.down1 = Downsample1D(hidden)
        self.enc2 = ResBlock1D(hidden, time_dim, pos_dim, cond_dim)
        self.down2 = Downsample1D(hidden)

        self.mid = ResBlock1D(hidden, time_dim, pos_dim, cond_dim)

        self.up1 = Upsample1D(hidden)
        self.dec1 = ResBlock1D(hidden, time_dim, pos_dim, cond_dim)
        self.up2 = Upsample1D(hidden)
        self.dec2 = ResBlock1D(hidden, time_dim, pos_dim, cond_dim)

        self.output = nn.Linear(hidden, feat_dim)
        self.act = nn.SiLU()

    def forward(self, traj, t, pos_emb, cond: Optional[torch.Tensor] = None):
        B,H,_ = traj.shape
        t_emb = self.time_emb(t)
        x = self.input(traj)
        e1 = self.enc1(x, t_emb, pos_emb, cond)
        d1 = self.down1(e1)
        # Ensure pos_emb has enough length for each layer
        pos_emb_d1 = pos_emb[:min(d1.size(1), pos_emb.size(0)), :]
        e2 = self.enc2(d1, t_emb, pos_emb_d1, cond)
        pos_emb_e2 = pos_emb[:min(e2.size(1), pos_emb.size(0)), :]
        m = self.mid(e2, t_emb, pos_emb_e2, cond)
        u1 = self.up1(m)
        u1 = u1[:, :e1.size(1), :]
        pos_emb_u1 = pos_emb[:min(u1.size(1), pos_emb.size(0)), :]
        d = self.dec1(u1, t_emb, pos_emb_u1, cond)
        u2 = self.up2(d)
        u2 = u2[:, :H, :]
        d = self.dec2(u2, t_emb, pos_emb, cond)
        out = self.output(self.act(d))
        return out


#Transformer
class TemporalTransformer(nn.Module):
    def __init__(self, feat_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6, time_dim: int = 128, pos_dim: int = 128, cond_dim: int = 0):
        super().__init__()
        self.input = nn.Linear(feat_dim, d_model)
        self.time_emb = SinusoidalEmbedding(time_dim)
        self.pos_proj = nn.Linear(pos_dim, d_model) if pos_dim>0 else None
        self.cond_proj = nn.Linear(cond_dim, d_model) if cond_dim>0 else None

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, feat_dim)
        self.d_model = d_model

    def forward(self, traj, t, pos_emb, cond: Optional[torch.Tensor] = None):
        # traj: [B,H,D], pos_emb: [H,pos_dim]
        B,H,_ = traj.shape
        x = self.input(traj)  # [B,H,d_model]
        # time embedding broadcasted to each step
        t_emb = self.time_emb(t)[:, None, :].expand(B, H, -1)  # [B,H,time_dim]
        # Project time embedding to d_model dimension if needed
        if t_emb.size(-1) != self.d_model:
            if t_emb.size(-1) < self.d_model:
                # Pad with zeros
                t_emb = F.pad(t_emb, (0, self.d_model - t_emb.size(-1)))
            else:
                # Truncate
                t_emb = t_emb[:, :, :self.d_model]
        x = x + t_emb
        if self.pos_proj is not None and pos_emb is not None:
            pos = self.pos_proj(pos_emb)[None,:,:]
            x = x + pos
        if self.cond_proj is not None and cond is not None:
            condp = self.cond_proj(cond)[:, None, :]
            x = x + condp
        x = self.transformer(x)  # [B,H,d_model]
        out = self.output(x)
        return out


# Task-Specific Models (No Conditioning)
class ResBlock1D_TaskSpecific(nn.Module):
    def __init__(self, ch: int, time_dim: int, pos_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(ch)
        self.ff1 = nn.Linear(ch, ch)
        self.norm2 = nn.LayerNorm(ch)
        self.ff2 = nn.Linear(ch, ch)
        self.time_proj = nn.Linear(time_dim, ch)
        self.pos_proj = nn.Linear(pos_dim, ch) if pos_dim > 0 else None
        self.act = nn.SiLU()

    def forward(self, x, t_emb, pos_emb=None):
        h = self.ff1(self.norm1(x))
        add = self.time_proj(t_emb)[:, None, :]
        if self.pos_proj is not None and pos_emb is not None:
            add = add + self.pos_proj(pos_emb)[None, :, :]
        h = h + add
        h = self.act(h)
        h = self.ff2(self.norm2(h))
        return x + h

class TrajectoryUNet1D_TaskSpecific(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, time_dim: int = 128, pos_dim: int = 128):
        super().__init__()
        self.input = nn.Linear(feat_dim, hidden)
        self.time_emb = SinusoidalEmbedding(time_dim)
        self.pos_dim = pos_dim

        self.enc1 = ResBlock1D_TaskSpecific(hidden, time_dim, pos_dim)
        self.down1 = Downsample1D(hidden)
        self.enc2 = ResBlock1D_TaskSpecific(hidden, time_dim, pos_dim)
        self.down2 = Downsample1D(hidden)

        self.mid = ResBlock1D_TaskSpecific(hidden, time_dim, pos_dim)

        self.up1 = Upsample1D(hidden)
        self.dec1 = ResBlock1D_TaskSpecific(hidden, time_dim, pos_dim)
        self.up2 = Upsample1D(hidden)
        self.dec2 = ResBlock1D_TaskSpecific(hidden, time_dim, pos_dim)

        self.output = nn.Linear(hidden, feat_dim)
        self.act = nn.SiLU()

    def forward(self, traj, t, pos_emb):
        B,H,_ = traj.shape
        t_emb = self.time_emb(t)
        x = self.input(traj)
        e1 = self.enc1(x, t_emb, pos_emb)
        d1 = self.down1(e1)
        # Ensure pos_emb has enough length for each layer
        pos_emb_d1 = pos_emb[:min(d1.size(1), pos_emb.size(0)), :]
        e2 = self.enc2(d1, t_emb, pos_emb_d1)
        pos_emb_e2 = pos_emb[:min(e2.size(1), pos_emb.size(0)), :]
        m = self.mid(e2, t_emb, pos_emb_e2)
        u1 = self.up1(m)
        u1 = u1[:, :e1.size(1), :]
        pos_emb_u1 = pos_emb[:min(u1.size(1), pos_emb.size(0)), :]
        d = self.dec1(u1, t_emb, pos_emb_u1)
        u2 = self.up2(d)
        u2 = u2[:, :H, :]
        d = self.dec2(u2, t_emb, pos_emb)
        out = self.output(self.act(d))
        return out

class TemporalTransformer_TaskSpecific(nn.Module):
    def __init__(self, feat_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6, time_dim: int = 128, pos_dim: int = 128):
        super().__init__()
        self.input = nn.Linear(feat_dim, d_model)
        self.time_emb = SinusoidalEmbedding(time_dim)
        self.pos_proj = nn.Linear(pos_dim, d_model) if pos_dim>0 else None

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, feat_dim)
        self.d_model = d_model

    def forward(self, traj, t, pos_emb):
        # traj: [B,H,D], pos_emb: [H,pos_dim]
        B,H,_ = traj.shape
        x = self.input(traj)  # [B,H,d_model]
        # time embedding broadcasted to each step
        t_emb = self.time_emb(t)[:, None, :].expand(B, H, -1)  # [B,H,time_dim]
        # Project time embedding to d_model dimension if needed
        if t_emb.size(-1) != self.d_model:
            if t_emb.size(-1) < self.d_model:
                # Pad with zeros
                t_emb = F.pad(t_emb, (0, self.d_model - t_emb.size(-1)))
            else:
                # Truncate
                t_emb = t_emb[:, :, :self.d_model]
        x = x + t_emb
        if self.pos_proj is not None and pos_emb is not None:
            pos = self.pos_proj(pos_emb)[None,:,:]
            x = x + pos
        x = self.transformer(x)  # [B,H,d_model]
        out = self.output(x)
        return out




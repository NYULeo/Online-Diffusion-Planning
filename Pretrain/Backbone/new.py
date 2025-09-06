from typing import List, Optional
import einops
import numpy as np
import torch
import torch.nn as nn
import math

# Timestep embedding used in the DDPM++ and ADM architectures,
# from https://github.com/NVlabs/edm/blob/main/training/networks.py#L269
class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class UntrainablePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = torch.einsum('...i,j->...ij', x, freqs.to(x.dtype))
        # x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x



# Timestep embedding used in Transformer
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.einsum('...i,j->...ij', x, emb.to(x.dtype))
        # emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Timestep embedding used in the DDPM++ and ADM architectures
class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 8) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor):
        emb = torch.einsum('...i,j->...ij', x, (2 * np.pi * self.freqs).to(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return self.mlp(emb)


class UntrainableFourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor):
        emb = torch.einsum('...i,j->...ij', x, (2 * np.pi * self.freqs).to(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return emb


SUPPORTED_TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "fourier": FourierEmbedding,
    "untrainable_fourier": UntrainableFourierEmbedding,
    "untrainable_positional": UntrainablePositionalEmbedding,
}


#from cleandiffuser.utils import GroupNorm1d
class GroupNorm1d(nn.Module):
    def __init__(self, dim, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, dim // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x.unsqueeze(2),
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x.squeeze(2)


def get_norm(dim: int, norm_type: str = "groupnorm"):
    if norm_type == "groupnorm":
        return GroupNorm1d(dim, 8, 4)
    elif norm_type == "layernorm":
        return LayerNorm(dim)
    else:
        return nn.Identity()


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, emb_dim: int, kernel_size: int = 3, norm_type: str = "groupnorm"):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            get_norm(out_dim, norm_type), nn.Mish())
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            get_norm(out_dim, norm_type), nn.Mish())
        self.emb_mlp = nn.Sequential(
            nn.Mish(), nn.Linear(emb_dim, out_dim))
        self.residual_conv = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, emb):
        out = self.conv1(x) + self.emb_mlp(emb).unsqueeze(-1)
        out = self.conv2(out)
        return out + self.residual_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        out = self.to_out(out)
        return out + x



class BaseNNDiffusion(nn.Module):
    """
    The neural network backbone for the Diffusion model used for score matching
     (or training a noise predictor) should take in three inputs.
     The first input is the noisy data.
     The second input is the denoising time step, which can be either as a discrete variable
     or a continuous variable, specified by the parameter `discrete_t`.
     The third input is the condition embedding that has been processed through the `nn_condition`.
     In the general case, we assume that there may be multiple conditions,
     which are inputted as a tensor dictionary, or a single condition, directly inputted as a tensor.
    """

    def __init__(
        self, emb_dim: int, 
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None
    ):
        assert timestep_emb_type in SUPPORTED_TIMESTEP_EMBEDDING.keys()
        super().__init__()
        timestep_emb_params = timestep_emb_params or {}
        self.map_noise = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](emb_dim, **timestep_emb_params)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        raise NotImplementedError
    

class JannerUNet1d(BaseNNDiffusion):
    def __init__(
            self,
            in_dim: int,
            model_dim: int = 32,
            emb_dim: int = 32,
            kernel_size: int = 3,
            dim_mult: List[int] = [1, 2, 2, 2],
            #dim_mult : List[int] = [1, 2, 4, 8],
            norm_type: str = "groupnorm",
            attention: bool = False,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):  
        #dim_mult: List[int] = [1, 2, 2, 2],
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        dims = [in_dim] + [model_dim * m for m in np.cumprod(dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.map_emb = nn.Sequential(
            #PositionalEmbedding(emb_dim),
            #SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, model_dim * 4), 
            nn.Mish(),
            nn.Linear(model_dim * 4, model_dim))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out, model_dim, kernel_size, norm_type),
                ResidualBlock(dim_out, dim_out, model_dim, kernel_size, norm_type),
                LinearAttention(dim_out) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, model_dim, kernel_size, norm_type)
        self.mid_attn = LinearAttention(mid_dim) if attention else nn.Identity()
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, model_dim, kernel_size, norm_type)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualBlock(dim_out * 2, dim_in, model_dim, kernel_size, norm_type),
                ResidualBlock(dim_in, dim_in, model_dim, kernel_size, norm_type),
                LinearAttention(dim_in) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 5, padding=2),
            get_norm(model_dim, norm_type), nn.Mish(),
            nn.Conv1d(model_dim, in_dim, 1))

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        # check horizon dimension
        assert x.shape[1] & (x.shape[1] - 1) == 0, "Ta dimension must be 2^n"

        x = x.permute(0, 2, 1)
        emb = self.map_noise(noise)
        """
        if condition is not None:
            emb = emb + condition
        else:
            emb = emb + torch.zeros_like(emb)
        """
        emb = self.map_emb(emb)

        h = []

        for resnet1, resnet2, attn, downsample in self.downs:
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for resnet1, resnet2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = x.permute(0, 2, 1)
        return x

model = JannerUNet1d(in_dim = 68)
B, H, D = 4, 64, 68
x = torch.rand(B, H, D).float()
t = torch.rand(B).float()
model(x, t)



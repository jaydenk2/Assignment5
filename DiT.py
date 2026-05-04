import torch
import torch.nn as nn
from einops import rearrange
import math


class Patchify(nn.Module):
    def __init__(self, patch_size=8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size

    def forward(self, x):
        p = self.patch_size
        return rearrange(x, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=p, pw=p)


class Unpatchify(nn.Module):
    def __init__(self, patch_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size

    def forward(self, x):
        b, n, pdim = x.shape
        p = self.patch_size
        g = int(n**0.5)
        c = pdim // (p * p)
        return rearrange(
            x,
            "b (h w) (ph pw c) -> b c (h ph) (w pw)",
            h=g,
            w=g,
            ph=p,
            pw=p,
            c=c,
        )


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, inner_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, inner_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.q = nn.Linear(hidden_dim, inner_dim)
        self.k = nn.Linear(hidden_dim, inner_dim)
        self.v = nn.Linear(hidden_dim, inner_dim)
        self.scale = inner_dim**-0.5

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ v


class MultiHeadSelfAttn(nn.Module):
    def __init__(self, hidden_dim, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        head_dim = hidden_dim // num_heads
        self.heads = nn.ModuleList(
            [SelfAttention(hidden_dim, head_dim) for _ in range(num_heads)]
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        o = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.out(o)


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, time_emb_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = MultiHeadSelfAttn(hidden_dim, num_heads)
        self.ff = FeedForward(hidden_dim, ff_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim // 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim // 4, hidden_dim * 6),
        )
        for m in self.cond_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, cond):
        p = self.cond_mlp(cond)
        a1, b1, g1, a2, b2, g2 = p.chunk(6, dim=-1)
        a1 = a1.unsqueeze(1)
        b1 = b1.unsqueeze(1)
        g1 = g1.unsqueeze(1)
        a2 = a2.unsqueeze(1)
        b2 = b2.unsqueeze(1)
        g2 = g2.unsqueeze(1)
        h = (g1 + 1) * self.norm1(x) + b1
        h = self.attn(h)
        h = a1 * h
        x = x + h
        h = (g2 + 1) * self.norm2(x) + b2
        h = self.ff(h)
        h = a2 * h
        x = x + h
        return x


class ContinuousTimestepEmbedder(nn.Module):
    def __init__(self, emb_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.view(t.size(0))
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT(nn.Module):
    def get_position_embedding(self, num_patches, patch_size, hidden_dim):
        grid_size = int(num_patches**0.5)
        dim = hidden_dim // 2

        pos = torch.arange(num_patches, dtype=torch.float32)
        r = pos // grid_size
        c = pos % grid_size

        omega = torch.arange(0, dim, 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / dim))

        out_r = r[:, None] @ omega[None, :]
        out_c = c[:, None] @ omega[None, :]

        row_emb = torch.zeros((1, num_patches, dim))
        col_emb = torch.zeros((1, num_patches, dim))

        row_emb[:, :, 0::2] = torch.sin(out_r)
        row_emb[:, :, 1::2] = torch.cos(out_r)
        col_emb[:, :, 0::2] = torch.sin(out_c)
        col_emb[:, :, 1::2] = torch.cos(out_c)

        pos_emb = torch.cat([row_emb, col_emb], dim=-1)
        pos_emb = nn.Parameter(pos_emb, requires_grad=False)
        return pos_emb

    def __init__(
        self,
        patch_size,
        num_blocks,
        num_heads,
        ff_dim,
        time_emb_dim,
        num_timesteps,
        hidden_dim,
        num_patches,
        num_channels=3,
        training_type="ddpm",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.training_type = training_type
        patch_dim = patch_size * patch_size * num_channels
        self.patchify = Patchify(patch_size)
        self.unpatchify = Unpatchify(patch_size)
        self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        self.pos_emb = self.get_position_embedding(num_patches, patch_size, hidden_dim)
        if training_type == "ddpm":
            self.time_embed = nn.Embedding(num_timesteps, time_emb_dim)
        else:
            self.time_embed = ContinuousTimestepEmbedder(time_emb_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads, ff_dim, time_emb_dim)
                for _ in range(num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_dim, patch_dim)
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, image, timestep):
        if timestep.dim() > 1:
            timestep = timestep.view(timestep.shape[0])
        x = self.patchify(image)
        x = self.patch_proj(x) + self.pos_emb.to(x.device)
        if self.training_type == "ddpm":
            te = self.time_embed(timestep.long())
        else:
            te = self.time_embed(timestep)
        for blk in self.blocks:
            x = blk(x, te)
        x = self.out_norm(x)
        x = self.out_proj(x)
        return self.unpatchify(x)

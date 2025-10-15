from __future__ import annotations

import math

import numpy as np
import torch
from einops import rearrange
from torch import einsum, nn

from act_rec.model.utils import bn_init, conv_branch_init, conv_init


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.scale = hidden_dim**-0.5
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Linear(in_channels, inner_dim * 2)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = rearrange(x, "n c t v -> n t v c").contiguous()
        y = self.ln(y)
        y = self.to_qk(y)
        qk = y.chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, "b t v (h d) -> (b t) h v d", h=self.n_heads), qk)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1).float()
        return attn


class SA_GC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray) -> None:
        super().__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head = A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)))

        self.conv_d = nn.ModuleList()
        for _ in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = nn.Identity()

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for conv in self.conv_d:
            conv_branch_init(conv, self.num_head)

        rel_channels = max(1, in_channels // 8)
        self.attn = SelfAttention(in_channels, rel_channels, self.num_head)

    def forward(self, x: torch.Tensor, attn: torch.Tensor | None = None) -> torch.Tensor:
        N, C, T, V = x.size()
        out = None
        if attn is None:
            attn = self.attn(x)
        A = attn * self.shared_topology.unsqueeze(0)
        for h in range(self.num_head):
            A_h = A[:, h, :, :]
            feature = rearrange(x, "n c t v -> (n t) v c")
            z = A_h @ feature
            z = rearrange(z, "(n t) v c -> n c t v", t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        out = out + self.down(x)
        out = self.relu(out)
        return out


class GCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray) -> None:
        super().__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head = A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)))

        self.conv_d = nn.ModuleList()
        for _ in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = nn.Identity()

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for conv in self.conv_d:
            conv_branch_init(conv, self.num_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        out = None
        A = self.shared_topology.unsqueeze(0)
        for h in range(self.num_head):
            A_h = A[:, h, :, :]
            feature = rearrange(x, "n c t v -> (n t) v c")
            z = A_h @ feature
            z = rearrange(z, "(n t) v c -> n c t v", t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        out = out + self.down(x)
        out = self.relu(out)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_mask: bool = False,
        SAGC_proj: bool = False,
        A: np.ndarray | int = 1,
        num_point: int = 25,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.use_mask = use_mask
        self.num_point = num_point

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        if SAGC_proj:
            self.to_qkv = SA_GC(dim, inner_dim * 3, A)
        else:
            self.to_qkv = GCN(dim, inner_dim * 3, A)

        mask = torch.ones(seq_len, seq_len).tril().view(1, 1, seq_len, seq_len)
        self.register_buffer("mask", mask, persistent=False)

        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        else:
            self.to_out = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        V = self.num_point

        x = rearrange(x, "(b v) t c -> b c t v", v=V)
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b c t v -> (b v) t c", v=V)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.use_mask:
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.attend(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        max_seq_len: int,
        dropout: float = 0.0,
        use_mask: bool = True,
        A: np.ndarray | int = 1,
        num_point: int = 25,
        SAGC_proj: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                max_seq_len,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_mask=use_mask,
                                SAGC_proj=SAGC_proj,
                                A=A,
                                num_point=num_point,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )
        self._attns: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attns = []
        for attn, ff in self.layers:
            res = x
            x, sa = attn(x)
            x = x + res
            x = ff(x) + x
            attns.append(sa.clone())
        self._attns = attns
        return x

    def get_attns(self) -> list[torch.Tensor]:
        return self._attns


def positional_encoding(d_model: int, max_len: int) -> torch.Tensor:
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        A: np.ndarray | int = 1,
        num_point: int = 25,
        SAGC_proj: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.register_buffer("pe", positional_encoding(dim, seq_len), persistent=False)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            seq_len,
            dropout,
            use_mask=True,
            A=A,
            num_point=num_point,
            SAGC_proj=SAGC_proj,
        )
        self.to_latent = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, V = x.shape
        x = rearrange(x, "b c t v -> (b v) t c")
        x = x + self.pe[:, :T, :].to(x.device)
        x = self.transformer(x)
        x = self.to_latent(x)
        x = rearrange(x, "(b v) t c -> b c t v", v=V)
        return x

    def get_attention(self) -> list[torch.Tensor]:
        return self.transformer.get_attns()

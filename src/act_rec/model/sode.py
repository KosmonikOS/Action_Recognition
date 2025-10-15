from __future__ import annotations

import math
import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.distributions import Normal, kl_divergence
from torchdiffeq import odeint

from act_rec.model.modules import GCN, TemporalEncoder
from act_rec.model.utils import import_class


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func: nn.Module, method: str, odeint_rtol: float = 1e-4, odeint_atol: float = 1e-5) -> None:
        super().__init__()
        self.ode_method = method
        self.ode_func = ode_func
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point: torch.Tensor, time_steps_to_predict: torch.Tensor) -> torch.Tensor:
        return odeint(
            self.ode_func,
            first_point,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
        )


class ODEFunc(nn.Module):
    def __init__(self, dim: int, A: torch.Tensor, N: int, T: int = 64) -> None:
        super().__init__()
        self.register_buffer("A", A)
        self.T = T
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.proj = nn.Conv1d(dim, dim, 1)
        with torch.no_grad():
            self.register_buffer("temporal_pe", self.init_pe(T + N, dim), persistent=False)
            index = torch.arange(T).unsqueeze(-1).expand(T, dim)
            self.register_buffer("index", index, persistent=False)

    def init_pe(self, length: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe.view(length, d_model)

    def add_pe(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, V = x.size()
        T = self.T
        index = self.index.to(x.device) + int(t.item())  # scalar t
        pe = torch.gather(self.temporal_pe.to(x.device), dim=0, index=index)
        pe = repeat(pe, "t c -> b t c v", v=V, b=B // T)
        x = rearrange(x, "(b t) c v -> b t c v", t=T)
        x = x + pe
        x = rearrange(x, "b t c v -> (b t) c v")
        return x

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.add_pe(x, t)
        A = self.A.to(x)
        x = torch.einsum("vu,ncu->ncv", A, x)
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.einsum("vu,ncu->ncv", A, x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.proj(x)
        return x


class SODE(nn.Module):
    def __init__(
        self,
        num_class: int = 60,
        num_point: int = 25,
        num_person: int = 2,
        ode_method: str = "rk4",
        graph: str | None = None,
        in_channels: int = 3,
        num_head: int = 3,
        k: int = 0,
        base_channel: int = 64,
        depth: int = 4,
        device: torch.device | str = "cpu",
        T: int = 64,
        n_step: int = 1,
        dilation: int = 1,
        SAGC_proj: bool = True,
        num_cls: int = 10,
        n_sample: int = 1,
        backbone: str = "transformer",
    ) -> None:
        super().__init__()

        if graph is None:
            raise ValueError("graph must be provided as import path string.")
        GraphCls = import_class(graph)
        self.Graph = GraphCls()

        if n_step < 1:
            raise ValueError("n_step must be >= 1 for the current SODE implementation.")

        with torch.no_grad():
            A = np.stack([self.Graph.A_norm] * num_head, axis=0).astype(np.float32)
            shift_idx = torch.arange(0, T, dtype=torch.long).view(1, T, 1, 1)
            shift_idx = repeat(shift_idx, "b t c v -> n b t c v", n=n_step)
            shift_idx = shift_idx - torch.arange(1, n_step + 1, dtype=torch.long).view(n_step, 1, 1, 1, 1)
            mask = torch.triu(torch.ones(n_step, T), diagonal=1).view(n_step, 1, T, 1, 1)

        self.T = T
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.n_step = n_step
        self.n_sample = n_sample

        self.register_buffer("shift_idx", (shift_idx % T), persistent=False)
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("arange_n_step", torch.arange(n_step + 1, dtype=torch.float32), persistent=False)

        z0_mean = torch.tensor([0.0], dtype=torch.float32)
        z0_std = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("z0_prior_mean", z0_mean)
        self.register_buffer("z0_prior_std", z0_std)
        self.register_buffer("zero", torch.tensor(0.0, dtype=torch.float32))

        self.cls_idx = [int(math.ceil(T * i / num_cls)) for i in range(num_cls + 1)]
        self.cls_idx[0] = 0
        self.cls_idx[-1] = T
        self.num_cls = num_cls

        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))

        if backbone != "transformer":
            raise NotImplementedError("Only transformer backbone is currently supported.")

        self.temporal_encoder = TemporalEncoder(
            seq_len=T,
            dim=base_channel,
            depth=depth,
            heads=4,
            mlp_dim=base_channel * 2,
            dim_head=base_channel // 4,
            A=A,
            num_point=num_point,
            SAGC_proj=SAGC_proj,
        )

        ode_A = torch.from_numpy(self.Graph.A_norm).float()
        self.ode_func = ODEFunc(base_channel, ode_A, N=n_step, T=T)
        self.diffeq_solver = DiffeqSolver(self.ode_func, method=ode_method)

        self.recon_decoder = nn.Sequential(
            GCN(base_channel, base_channel, A),
            GCN(base_channel, base_channel, A),
            nn.Conv2d(base_channel, in_channels, 1),
        )

        in_dim = base_channel * (n_step + 1)
        mid_dim = base_channel * (n_step + 1) // 2
        out_dim = base_channel * (n_step + 1) // 2

        self.cls_decoder = nn.Sequential(
            GCN(in_dim, mid_dim, A),
            GCN(mid_dim, out_dim, A),
        )

        self.classifiers = nn.ModuleList([nn.Conv1d(out_dim, num_class, 1) for _ in range(num_cls)])
        self.spatial_pooling = torch.mean

    def KL_div(self, z_mu: torch.Tensor, z_std: torch.Tensor, kl_coef: float = 1.0) -> torch.Tensor:
        prior = Normal(self.z0_prior_mean.to(z_mu.device), self.z0_prior_std.to(z_mu.device))
        z_distr = Normal(z_mu, z_std)
        kldiv_z0 = kl_divergence(z_distr, prior)
        loss = kldiv_z0.mean()
        return loss * kl_coef

    def extrapolate(self, z_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, T, V = z_0.size()
        z_flat = rearrange(z_0, "b c t v -> (b t) c v")
        zs = self.diffeq_solver(z_flat, t)
        zs = rearrange(zs, "n (b t) c v -> n b t c v", t=T)
        z_hat = zs[1:]

        shift_idx = self.shift_idx.to(z_hat.device)
        mask = self.mask.to(z_hat.device, dtype=z_hat.dtype)
        z_hat_shifted = torch.gather(z_hat.clone(), dim=2, index=shift_idx.expand_as(z_hat).long())
        z_hat_shifted = mask * z_hat_shifted
        z_hat_shifted = rearrange(z_hat_shifted, "n b t c v -> (n b) c t v")
        z_hat = rearrange(z_hat, "n b t c v -> (n b) c t v")
        z_0 = rearrange(z_flat, "(b t) c v -> b c t v", t=T)
        return z_0, z_hat, z_hat_shifted

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N, C, T, V, M = x.size()
        x_emb = rearrange(x, "n c t v m -> (n m t) v c", n=N, m=M, v=V)

        x_emb = self.to_joint_embedding(x_emb)
        x_emb = x_emb + self.pos_embedding[:, : self.num_point]

        x_emb = rearrange(x_emb, "(n m t) v c -> (n m) c t v", m=M, n=N)
        z = self.temporal_encoder(x_emb)

        t = self.arange_n_step.to(z.dtype).to(z.device)
        z_0, z_hat, z_hat_shifted = self.extrapolate(z, t)

        x_hat = self.recon_decoder(z_hat_shifted)
        x_hat = rearrange(x_hat, "(n m l) c t v -> n l c t v m", m=M, l=self.n_sample).mean(1)

        z_hat_cls = rearrange(z_hat, "(n b) c t v -> b (n c) t v", n=self.n_step)
        z_cls = self.cls_decoder(torch.cat([z_0, z_hat_cls], dim=1))
        z_cls = rearrange(z_cls, "(n m l) c t v -> (n l) m c t v", m=M, l=self.n_sample).mean(1)
        z_cls = self.spatial_pooling(z_cls, dim=-1)

        y_lst = []
        for i, classifier in enumerate(self.classifiers):
            start, end = self.cls_idx[i], self.cls_idx[i + 1]
            y_lst.append(classifier(z_cls[:, :, start:end]))
        y = torch.cat(y_lst, dim=-1)
        y = rearrange(y, "(n l) c t -> n l c t", l=self.n_sample).mean(1)
        return y, x_hat, z_0, z_hat_shifted, self.zero.to(x.device)

    def get_attention(self) -> list[torch.Tensor]:
        return self.temporal_encoder.get_attention()

    def get_A(self, k: int) -> torch.Tensor:
        A_outward = self.Graph.A_outward_binary
        I = np.eye(self.Graph.num_node)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k)).float()

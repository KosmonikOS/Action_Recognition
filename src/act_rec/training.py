from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Dict

import torch
from torch.utils.data import DataLoader
from einops import rearrange, repeat

from act_rec.model.losses import LabelSmoothingCrossEntropy, masked_recon_loss


class AverageMeter:
    """Tracks running average of scalar metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1, 5)) -> Dict[int, float]:
    """Compute top-k accuracies from logits of shape (batch, num_class)."""
    maxk = max(topk)
    _, pred = torch.topk(logits, k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        res[k] = correct_k / target.size(0)
    return res


@dataclass
class TrainConfig:
    device: torch.device
    cls_loss: LabelSmoothingCrossEntropy
    lambda_cls: float = 1.0
    lambda_recon: float = 0.0
    lambda_feature: float = 0.0
    lambda_kl: float = 0.0
    n_step: int = 1
    recon_loss_fn: Callable = masked_recon_loss
    feature_loss_fn: Callable | None = masked_recon_loss


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> dict[str, float]:
    model.train()
    cls_meter = AverageMeter()
    recon_meter = AverageMeter()
    feature_meter = AverageMeter()
    kl_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_meter = AverageMeter()

    for data, labels, mask, _ in dataloader:
        data = data.to(config.device, dtype=torch.float32)
        labels = labels.to(config.device, dtype=torch.long)
        mask = mask.to(config.device)

        logits, recon, z_0, z_hat_shifted, kl_term = model(data)

        batch_size = data.size(0)
        num_class = logits.size(1)
        num_frames = logits.size(-1)

        n_cls = max(logits.size(0) // batch_size, 1)
        logits_view = logits.view(n_cls, batch_size, num_class, num_frames)
        cls_loss = torch.tensor(0.0, device=config.device)
        if config.lambda_cls > 0:
            labels_exp = labels.view(1, batch_size, 1).expand(n_cls, batch_size, num_frames)
            logits_flat = rearrange(logits_view, "n b c t -> (n b t) c")
            cls_loss = config.lambda_cls * config.cls_loss(logits_flat, labels_exp.reshape(-1))

        recon_loss = torch.tensor(0.0, device=config.device)
        if (
            config.lambda_recon > 0
            and config.recon_loss_fn is not None
            and recon is not None
            and recon.numel() > 0
        ):
            n_rec = max(recon.size(0) // batch_size, 1)
            x_gt = repeat(data, "b c t v m -> (n b) c t v m", n=n_rec)
            mask_recon = repeat(mask.float(), "b c t v m -> n b c t v m", n=n_rec).clone()
            for i in range(n_rec):
                if n_rec == config.n_step:
                    mask_recon[i, :, :, : i + 1, :, :] = 0.0
                else:
                    mask_recon[i, :, :, :i, :, :] = 0.0
            mask_recon = rearrange(mask_recon, "n b c t v m -> (n b) c t v m")
            recon_loss = config.lambda_recon * config.recon_loss_fn(recon, x_gt, mask_recon)

        feature_loss = torch.tensor(0.0, device=config.device)
        if (
            config.lambda_feature > 0
            and config.feature_loss_fn is not None
            and z_hat_shifted is not None
            and z_hat_shifted.numel() > 0
        ):
            n_step = config.n_step
            if n_step > 0:
                B_, C, T, V = z_0.shape
                z0_tiled = repeat(z_0, "b c t v -> n b c t v", n=n_step)
                z_hat_feat = z_hat_shifted.view(n_step, B_, C, T, V)
                mask_feature = (z_hat_feat != 0.0).float()
                feature_loss = config.lambda_feature * config.feature_loss_fn(
                    z_hat_feat.reshape(-1, C, T, V, 1),
                    z0_tiled.reshape(-1, C, T, V, 1),
                    mask_feature.reshape(-1, C, T, V, 1),
                )

        kl_loss = torch.tensor(0.0, device=config.device)
        if config.lambda_kl > 0 and kl_term is not None:
            kl_loss = config.lambda_kl * kl_term

        loss = cls_loss + recon_loss + feature_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        cls_meter.update(cls_loss.item(), batch_size)
        recon_meter.update(recon_loss.item(), batch_size)
        feature_meter.update(feature_loss.item(), batch_size)
        kl_meter.update(kl_loss.item(), batch_size)
        total_meter.update(loss.item(), batch_size)

        logits_last = logits_view[-1, :, :, -1]
        acc = topk_accuracy(logits_last.detach(), labels, topk=(1,))[1]
        acc_meter.update(acc, batch_size)

    return {
        "train_cls_loss": cls_meter.avg,
        "train_recon_loss": recon_meter.avg,
        "train_feature_loss": feature_meter.avg,
        "train_kl_loss": kl_meter.avg,
        "train_total_loss": total_meter.avg,
        "train_acc": acc_meter.avg,
    }


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: TrainConfig,
) -> dict[str, float]:
    model.eval()
    cls_meter = AverageMeter()
    recon_meter = AverageMeter()
    feature_meter = AverageMeter()
    kl_meter = AverageMeter()
    total_meter = AverageMeter()
    acc_meter_top1 = AverageMeter()
    acc_meter_top5 = AverageMeter()

    with torch.no_grad():
        for data, labels, mask, _ in dataloader:
            data = data.to(config.device, dtype=torch.float32)
            labels = labels.to(config.device, dtype=torch.long)
            mask = mask.to(config.device)

            logits, recon, z_0, z_hat_shifted, kl_term = model(data)

            batch_size = data.size(0)
            num_class = logits.size(1)
            num_frames = logits.size(-1)

            n_cls = max(logits.size(0) // batch_size, 1)
            logits_view = logits.view(n_cls, batch_size, num_class, num_frames)
            cls_loss = torch.tensor(0.0, device=config.device)
            if config.lambda_cls > 0:
                labels_exp = labels.view(1, batch_size, 1).expand(n_cls, batch_size, num_frames)
                logits_flat = rearrange(logits_view, "n b c t -> (n b t) c")
                cls_loss = config.lambda_cls * config.cls_loss(logits_flat, labels_exp.reshape(-1))
            cls_meter.update(cls_loss.item(), batch_size)

            logits_last = logits_view[-1, :, :, -1]
            accs = topk_accuracy(logits_last, labels, topk=(1, 5))
            acc_meter_top1.update(accs[1], batch_size)
            acc_meter_top5.update(accs.get(5, accs[1]), batch_size)

            recon_loss = torch.tensor(0.0, device=config.device)
            if (
                config.lambda_recon > 0
                and config.recon_loss_fn is not None
                and recon is not None
                and recon.numel() > 0
            ):
                n_rec = max(recon.size(0) // batch_size, 1)
                x_gt = repeat(data, "b c t v m -> (n b) c t v m", n=n_rec)
                mask_recon = repeat(mask.float(), "b c t v m -> n b c t v m", n=n_rec).clone()
                for i in range(n_rec):
                    if n_rec == config.n_step:
                        mask_recon[i, :, :, : i + 1, :, :] = 0.0
                    else:
                        mask_recon[i, :, :, :i, :, :] = 0.0
                mask_recon = rearrange(mask_recon, "n b c t v m -> (n b) c t v m")
                recon_loss = config.lambda_recon * config.recon_loss_fn(recon, x_gt, mask_recon)
            recon_meter.update(recon_loss.item(), batch_size)

            feature_loss = torch.tensor(0.0, device=config.device)
            if (
                config.lambda_feature > 0
                and config.feature_loss_fn is not None
                and z_hat_shifted is not None
                and z_hat_shifted.numel() > 0
            ):
                n_step = config.n_step
                if n_step > 0:
                    B_, C, T, V = z_0.shape
                    z0_tiled = repeat(z_0, "b c t v -> n b c t v", n=n_step)
                    z_hat_feat = z_hat_shifted.view(n_step, B_, C, T, V)
                    mask_feature = (z_hat_feat != 0.0).float()
                    feature_loss = config.lambda_feature * config.feature_loss_fn(
                        z_hat_feat.reshape(-1, C, T, V, 1),
                        z0_tiled.reshape(-1, C, T, V, 1),
                        mask_feature.reshape(-1, C, T, V, 1),
                    )
            feature_meter.update(feature_loss.item(), batch_size)

            kl_loss = torch.tensor(0.0, device=config.device)
            if config.lambda_kl > 0 and kl_term is not None:
                kl_loss = config.lambda_kl * kl_term
            kl_meter.update(kl_loss.item(), batch_size)

            total_meter.update((cls_loss + recon_loss + feature_loss + kl_loss).item(), batch_size)

    return {
        "val_cls_loss": cls_meter.avg,
        "val_recon_loss": recon_meter.avg,
        "val_feature_loss": feature_meter.avg,
        "val_kl_loss": kl_meter.avg,
        "val_total_loss": total_meter.avg,
        "val_top1": acc_meter_top1.avg,
        "val_top5": acc_meter_top5.avg,
    }

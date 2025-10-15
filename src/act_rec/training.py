from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence, Callable

import torch
from torch.utils.data import DataLoader

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
    recon_loss_fn: Callable = masked_recon_loss


def _prepare_reconstruction_targets(
    x: torch.Tensor,
    mask: torch.Tensor,
    recon_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile inputs/masks to match reconstruction predictions."""
    batch_size = x.size(0)
    n_rec = recon_pred.size(0) // batch_size

    x_gt = x.unsqueeze(0).expand(n_rec, *x.shape).reshape(-1, *x.shape[1:])
    mask_full = mask.float().expand(-1, x.size(1), -1, x.size(3), x.size(4))
    mask_recon = mask_full.unsqueeze(0).expand(n_rec, *mask_full.shape).reshape(-1, *mask_full.shape[1:])
    return x_gt, mask_recon


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> dict[str, float]:
    model.train()
    cls_meter = AverageMeter()
    recon_meter = AverageMeter()
    acc_meter = AverageMeter()

    for data, labels, mask, _ in dataloader:
        data = data.to(config.device, dtype=torch.float32)
        labels = labels.to(config.device, dtype=torch.long)
        mask = mask.to(config.device)

        logits, recon, *_ = model(data)
        logits_last = logits[:, :, -1]
        cls_loss = config.lambda_cls * config.cls_loss(logits_last, labels)
        loss = cls_loss

        if config.lambda_recon > 0 and config.recon_loss_fn is not None and recon is not None and recon.numel() > 0:
            x_gt, mask_recon = _prepare_reconstruction_targets(data, mask, recon)
            recon_loss_val = config.recon_loss_fn(recon, x_gt, mask_recon)
            loss = loss + config.lambda_recon * recon_loss_val
            recon_meter.update(recon_loss_val.item(), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        cls_meter.update(cls_loss.item(), data.size(0))
        acc = topk_accuracy(logits_last.detach(), labels, topk=(1,))[1]
        acc_meter.update(acc, data.size(0))

    return {
        "train_cls_loss": cls_meter.avg,
        "train_recon_loss": recon_meter.avg,
        "train_acc": acc_meter.avg,
    }


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: TrainConfig,
) -> dict[str, float]:
    model.eval()
    cls_meter = AverageMeter()
    acc_meter_top1 = AverageMeter()
    acc_meter_top5 = AverageMeter()

    with torch.no_grad():
        for data, labels, mask, _ in dataloader:
            data = data.to(config.device, dtype=torch.float32)
            labels = labels.to(config.device, dtype=torch.long)

            logits, recon, *_ = model(data)
            logits_last = logits[:, :, -1]
            cls_loss = config.lambda_cls * config.cls_loss(logits_last, labels)

            cls_meter.update(cls_loss.item(), data.size(0))
            accs = topk_accuracy(logits_last, labels, topk=(1, 5))
            acc_meter_top1.update(accs[1], data.size(0))
            acc_meter_top5.update(accs.get(5, accs[1]), data.size(0))

    return {
        "val_cls_loss": cls_meter.avg,
        "val_top1": acc_meter_top1.avg,
        "val_top5": acc_meter_top5.avg,
    }

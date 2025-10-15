from __future__ import annotations

import importlib
import math
from typing import Any

import torch
import torch.nn as nn


def import_class(import_str: str) -> Any:
    module_name, _, class_name = import_str.rpartition(".")
    if not module_name:
        raise ImportError(f"Invalid import path: {import_str}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Cannot import {class_name} from {module_name}") from exc


def bn_init(bn: nn.BatchNorm2d | nn.BatchNorm1d, scale: float) -> None:
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0.0)


def conv_branch_init(conv: nn.Conv2d, branches: int) -> None:
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)


def conv_init(conv: nn.Conv2d | nn.Conv1d) -> None:
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)

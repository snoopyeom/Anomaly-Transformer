import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def my_kl_loss(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Compute KL divergence used throughout the project.

    Parameters
    ----------
    p : torch.Tensor
        Input probability distribution.
    q : torch.Tensor
        Reference probability distribution.
    eps : float, optional
        Small constant for numerical stability, by default 1e-4.

    Returns
    -------
    torch.Tensor
        Mean KL divergence across the last dimension and batch.
    """
    res = p * (torch.log(p + eps) - torch.log(q + eps))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def filter_short_segments(changes: list[int], min_gap: int) -> list[int]:
    """Remove change points that occur too close together.

    Parameters
    ----------
    changes : list[int]
        Sorted change point indices as returned by ``ruptures``.
    min_gap : int
        Minimum number of samples required between consecutive change points.

    Returns
    -------
    list[int]
        Filtered list with short segments removed.
    """

    if not changes:
        return changes

    filtered = [changes[0]]
    for cp in changes[1:]:
        if cp - filtered[-1] >= min_gap:
            filtered.append(cp)
    return filtered

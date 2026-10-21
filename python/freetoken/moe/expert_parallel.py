"""Expert parallelism: which routed experts this process owns, and how the others are masked.

Tensor parallelism stays size 1. ``ep_info`` selects a striped subset (``id % size == rank``)
and the MoE layer all-reduces only the routed output on :func:`get_ep_comm`. The router
still emits global ids. Bank rows are the dense local index, plus one zero pad row so an
unowned route has a legal index.
"""

from __future__ import annotations

import torch
from freetoken.distributed.info import get_ep_info, try_get_tp_info

# Larger than any LRU step this process will count, and small enough that
# ``usage << slot_bits`` still fits a positive int64 for caches up to 2**30 slots.
# A slot with this usage is never the least-recently-used victim.
EP_PAD_USAGE = 1 << 32


class _Identity:
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x


class EpCommunicator:
    """All-reduce for expert outputs. Not installed on ``DistributedCommunicator.plugins``."""

    def __init__(self) -> None:
        self.impl = _Identity()
        self.size = 1

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.size <= 1:
            return x
        return self.impl.all_reduce(x)


_EP_COMM = EpCommunicator()


def get_ep_comm() -> EpCommunicator:
    return _EP_COMM


def set_ep_comm(impl, size: int) -> None:
    _EP_COMM.impl = impl
    _EP_COMM.size = size


def clear_ep_comm() -> None:
    set_ep_comm(_Identity(), 1)


def is_rank0() -> bool:
    """The process that owns progress bars and the tokenizer sockets."""
    tp = try_get_tp_info()
    return (tp is None or tp.is_primary()) and get_ep_info().is_primary()


def bank_geometry(num_experts: int, size: int | None = None) -> tuple[int, int]:
    """``(owned, bank rows)``. Size 1 returns ``(num_experts, num_experts)``.

    Bank rows are the owned experts plus one zero pad row. The pad is a real index
    for prefill and the resident zero slot for decode.
    """
    if size is None:
        size = get_ep_info().size
    if size <= 1:
        return num_experts, num_experts
    if num_experts <= 0 or num_experts % size:
        raise ValueError(
            f"expert parallel size {size} must divide the model's {num_experts} routed experts"
        )
    owned = num_experts // size
    return owned, owned + 1


def remap_expert(expert: int, *, rank: int | None = None, size: int | None = None) -> int | None:
    """Global expert id to a dense local row, or None when this rank does not own it."""
    if rank is None or size is None:
        info = get_ep_info()
        rank, size = info.rank, info.size
    if size <= 1:
        return expert
    if expert % size != rank:
        return None
    return expert // size


def mask_routes(
    ids: torch.Tensor, weights: torch.Tensor, pad_row: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the weight of unowned routes and point them at ``pad_row``.

    Fixed shape, no host sync. ``pad_row`` is the local zero bank row (prefill indexes
    it directly; decode has that row pinned in a zero slot so ``ensure_experts`` hits).
    """
    info = get_ep_info()
    if info.size <= 1:
        return weights, ids
    owned = (ids % info.size) == info.rank
    local = torch.where(owned, ids // info.size, torch.full_like(ids, pad_row))
    return torch.where(owned, weights, torch.zeros_like(weights)), local


__all__ = [
    "EP_PAD_USAGE",
    "EpCommunicator",
    "bank_geometry",
    "clear_ep_comm",
    "get_ep_comm",
    "is_rank0",
    "mask_routes",
    "remap_expert",
    "set_ep_comm",
]

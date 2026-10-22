"""Routed-expert banks straight from a GGUF's ``blk.N.ffn_{gate,up,down}_exps`` tensors, in their packed ggml rows: pinned host banks for the offload cache, or device banks for resident experts.

Each layer keeps its own ggml types, so each layer gets its own bank shapes: ``gate_up`` [E, 2I, row_bytes] (expert e's gate rows, then its up rows) and ``down`` [E, H, row_bytes]; float experts keep their dtype ([E, 2I, H] and [E, H, I]).
A GGUF that stores ``ffn_gate_up_exps`` (gemma4) already holds the gate_up rows in that order.
When NVFP4 experts carry per-expert ``.scale`` tensors, every layer also gets ``global`` [E, 4] fp32: the (gate, up, down, 1) weight_scale_2, 1 where a tensor has none.
Expert parallel uses the same stripe as the other formats (``id % size == rank``). E above is then this rank's bank rows: the owned experts plus one zero pad row. The file still holds every expert; each rank copies only its own.
"""

from __future__ import annotations

import contextlib
import os

import torch

from freetoken.layers.quantization import QuantKind
from freetoken.layers.quantization.ggml_types import float_dtype, has_expert_scales, mmq_tail_bytes, pad_rows, type_name
from freetoken.utils import init_logger

from .reader import GgufTensorInfo, gguf_tensor_index
from .tensors import weight_scale

logger = init_logger(__name__)


def _expert_rows(info: GgufTensorInfo) -> torch.Tensor:
    """[E, rows, row_bytes] packed ggml rows, or [E, rows, K] for a float tensor."""
    k, rows, experts = info.ne
    raw = torch.from_numpy(info.data()).view(experts, rows, info.nbytes // (experts * rows))
    dtype = float_dtype(info.ggml_type)
    return raw if dtype is None else raw.view(dtype)


def _rank_rows(experts: int) -> tuple[int, int]:
    """Owned experts and bank rows. Size 1 keeps every expert and adds no pad row."""
    from freetoken.distributed import get_ep_info
    from freetoken.moe.expert_parallel import bank_geometry

    return bank_geometry(experts, get_ep_info().size)


def _owned(tensor: torch.Tensor) -> torch.Tensor:
    """Rows this rank owns, in local-row order (``local = id // size`` for ``id % size == rank``)."""
    from freetoken.distributed import get_ep_info

    info = get_ep_info()
    if info.size <= 1:
        return tensor
    return tensor[info.rank :: info.size]


def _copy_owned(dest: torch.Tensor, src: torch.Tensor, owned: int, bank_rows: int) -> None:
    dest[:owned].copy_(_owned(src))
    if bank_rows != owned:
        dest[owned:].zero_()


def _drop_page_cache(info: GgufTensorInfo) -> None:
    try:
        fd = os.open(info.path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, info.offset, info.nbytes, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _device_bank(shape: tuple[int, ...], ggml_type: int, k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """An [E, rows, row] device bank; a zero row follows the last one when ggml's MMQ tiles read past it."""
    experts, rows, row = shape
    flat = torch.empty(experts * rows + pad_rows(ggml_type, k), row, dtype=dtype, device=device)
    flat[experts * rows :].zero_()
    return flat[: experts * rows].view(shape)


def _expert_scales(gate_up: list[GgufTensorInfo], down: GgufTensorInfo, experts: int) -> torch.Tensor:
    """[E, 4] fp32 (gate, up, down, 1) weight_scale_2 of one layer's experts."""
    gate, up = gate_up if len(gate_up) == 2 else (gate_up[0], gate_up[0])
    out = torch.ones(experts, 4, dtype=torch.float32)
    for col, info in enumerate((gate, up, down)):
        s = weight_scale(info)
        if s is None:
            continue
        if s.numel() != experts:
            raise NotImplementedError(f"{info.name}: {s.numel()} scales for {experts} experts")
        out[:, col] = s
    return out


def load_gguf_expert_banks(model_path: str, num_layers: int, *, first_layer: int = 0, dummy: bool = False, device: torch.device | None = None):
    """``ExpertBanks`` with one bank per (layer, role): pinned host banks for the offload cache, filled from the file and pinned while later layers load, or resident banks on ``device``."""
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.host_banks import HostBank, PinPipeline

    index = gguf_tensor_index(model_path)
    scaled = has_expert_scales({name: info.ggml_type for name, info in index.items()})
    roles = ("gate_up", "down", "global") if scaled else ("gate_up", "down")
    sources: dict[str, list[torch.Tensor]] = {role: [] for role in roles}
    tails: dict[str, list[int]] = {role: [] for role in roles}
    tags = []
    owned = experts = 0
    with PinPipeline() if device is None else contextlib.nullcontext() as pins:
        for layer in range(num_layers):
            blk = f"blk.{first_layer + layer}"
            down = index[f"{blk}.ffn_down_exps.weight"]
            if f"{blk}.ffn_gate_up_exps.weight" in index:
                gate = index[f"{blk}.ffn_gate_up_exps.weight"]
                parts = [gate]
            else:
                gate, up = index[f"{blk}.ffn_gate_exps.weight"], index[f"{blk}.ffn_up_exps.weight"]
                if gate.ggml_type != up.ggml_type:
                    raise NotImplementedError(f"{blk}: gate and up experts have different ggml types")
                parts = [gate, up]
            tags.append(("ggml", type_name(gate.ggml_type), type_name(down.ggml_type)))
            tails["gate_up"].append(mmq_tail_bytes(gate.ggml_type, gate.ne[0]))
            tails["down"].append(mmq_tail_bytes(down.ggml_type, down.ne[0]))
            if scaled:
                tails["global"].append(0)
            rows = [_expert_rows(info) for info in parts]
            d = _expert_rows(down)
            experts = rows[0].shape[0]
            owned, bank_rows = _rank_rows(experts)
            if bank_rows != owned and any(t.shape[0] != experts for t in (*rows, d)):
                raise ValueError(f"{blk}: expert tensors disagree on the expert count")
            gate_up_shape = (bank_rows, sum(r.shape[1] for r in rows), rows[0].shape[2])
            down_shape = (bank_rows, *d.shape[1:])
            if device is None:
                host = {"gate_up": HostBank(gate_up_shape, rows[0].dtype), "down": HostBank(down_shape, d.dtype)}
                if scaled:
                    host["global"] = HostBank((bank_rows, 4), torch.float32)
                banks = {role: bank.tensor for role, bank in host.items()}
            else:
                banks = {
                    "gate_up": _device_bank(gate_up_shape, gate.ggml_type, gate.ne[0], rows[0].dtype, device),
                    "down": _device_bank(down_shape, down.ggml_type, down.ne[0], d.dtype, device),
                }
                if scaled:
                    banks["global"] = torch.empty(bank_rows, 4, dtype=torch.float32, device=device)
            if scaled:
                # pad scale stays 1, matching a tensor that has no weight_scale_2
                banks["global"].fill_(1)
                banks["global"][:owned].copy_(_owned(_expert_scales(parts, down, experts)))
            if dummy:
                banks["gate_up"].zero_()
                banks["down"].zero_()
            else:
                start = 0
                for r in rows:
                    _copy_owned(banks["gate_up"][:, start : start + r.shape[1]], r, owned, bank_rows)
                    start += r.shape[1]
                _copy_owned(banks["down"], d, owned, bank_rows)
                for info in (*parts, down):
                    _drop_page_cache(info)
            if device is None:
                pins(layer, host)
            for role, bank in banks.items():
                sources[role].append(bank)
    total = sum(t.numel() * t.element_size() for ts in sources.values() for t in ts)
    where = f"on {device}" if device is not None else "in host RAM"
    shard = ""
    if sources["gate_up"] and sources["gate_up"][0].shape[0] != owned:
        shard = f", {owned} of {experts} experts"
    logger.info_rank0(f"GGUF expert banks: {num_layers} layers{shard}, {total / 2**30:.1f} GiB {where}")
    return ExpertBanks("gguf_scaled" if scaled else "gguf", sources, kind=QuantKind.GGUF, kernel="ggml", layout=None, layer_tags=tags, row_tail_bytes=tails)


__all__ = ["load_gguf_expert_banks"]

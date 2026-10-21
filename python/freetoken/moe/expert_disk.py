"""Checkpoint-backed expert rows for ``--moe-disk``.

The GPU slot cache still decides which rows live on the device. Its host source is one
pinned bounce, aliased as every layer, so the existing copy plan and slot ids stay valid.
A pinned window in front of the checkpoint holds rows the GPU has evicted but a later
token still wants. A miss that is also absent from that window is read from the safetensors
shard and packed with the block-fp8 kernel. The pad row stays zero and is never stored.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass

import torch

from freetoken.moe.expert_banks import ExpertBanks
from freetoken.moe.expert_parallel import bank_geometry, is_rank0, remap_expert
from freetoken.moe.legacy_format import legacy_format_for
from freetoken.utils import init_logger

logger = init_logger(__name__)

# GLM-5.3 routed experts. The bank layer is the checkpoint layer minus the leading dense layers.
_EXPERT_KEY = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)$"
)
_PROJECTIONS = ("gate", "up", "down")
_KIND_SUFFIX = {"weight": "", "weight_scale_inv": "_scale"}
_PIECES = 6

_SAFETENSORS_DTYPE = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


@dataclass(frozen=True)
class TensorLoc:
    """One tensor's payload in a safetensors shard. The offset is a file offset."""

    path: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


def packed_row_bytes(layout) -> int:
    """Bytes of one packed expert: every non-resident bank row in ``layout``."""
    total = 0
    for spec in layout.values():
        if spec.resident:
            continue
        total += math.prod(spec.shape) * spec.dtype.itemsize
    return total


def build_expert_index(folder: str, model_config, *, rank: int, size: int) -> dict[tuple[int, int, str], TensorLoc]:
    """``(bank layer, local row, role) -> payload location`` for the experts this rank owns.

    Reads shard headers only. Unowned expert ids are left out, and so is the pad row.
    """
    from freetoken.models.loader import safetensors_weight_map

    first_k = int(getattr(model_config, "first_k_dense_replace", 0) or 0)
    num_moe_layers = model_config.num_moe_layers
    owned, _rows = bank_geometry(model_config.num_experts, size)
    wanted: dict[str, list[tuple[str, int, int, str]]] = {}
    for name, shard in safetensors_weight_map(folder).items():
        match = _EXPERT_KEY.match(name)
        if match is None:
            continue
        bank = int(match["layer"]) - first_k
        if bank < 0 or bank >= num_moe_layers:
            continue
        row = remap_expert(int(match["expert"]), rank=rank, size=size)
        if row is None:
            continue
        role = match["proj"] + _KIND_SUFFIX[match["kind"]]
        wanted.setdefault(shard, []).append((name, bank, row, role))

    index: dict[tuple[int, int, str], TensorLoc] = {}
    for shard, items in wanted.items():
        path = os.path.join(folder, shard)
        data_base, header = _shard_header(path)
        for name, bank, row, role in items:
            meta = header.get(name)
            if meta is None:
                raise ValueError(f"{path} is listed as holding {name}, but the shard header has no such tensor")
            dtype = _SAFETENSORS_DTYPE.get(meta["dtype"])
            if dtype is None:
                raise ValueError(f"{name} has unsupported safetensors dtype {meta['dtype']!r}")
            start, end = meta["data_offsets"]
            shape = tuple(int(n) for n in meta["shape"])
            nbytes = int(end) - int(start)
            expect = math.prod(shape) * dtype.itemsize
            if nbytes != expect:
                raise ValueError(f"{name} header length {nbytes} does not match shape {shape} {dtype}")
            key = (bank, row, role)
            if key in index:
                raise ValueError(f"duplicate expert tensor for layer {bank} row {row} role {role}: {name}")
            index[key] = TensorLoc(path, data_base + int(start), nbytes, shape, dtype)

    expected = num_moe_layers * owned * _PIECES
    if len(index) != expected:
        raise ValueError(
            f"--moe-disk indexed {len(index)} expert tensors, expected {expected} "
            f"({num_moe_layers} layers x {owned} owned experts x {_PIECES}). "
            "The reader looks for model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj."
            "{weight,weight_scale_inv}."
        )
    return index


class HostExpertWindow:
    """Pinned (or plain) rows keyed by ``(layer, local_row)``, evicted least-recently-used."""

    def __init__(self, layout, capacity: int, *, pin: bool):
        if capacity < 1:
            raise ValueError(f"host expert window needs at least one row, got {capacity}")
        self.capacity = capacity
        self.slots = {
            role: _alloc((capacity, *spec.shape), spec.dtype, pin=pin)
            for role, spec in layout.items()
            if not spec.resident
        }
        self._order: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._free = list(range(capacity))

    def hit(self, layer: int, row: int) -> int | None:
        key = (layer, row)
        slot = self._order.get(key)
        if slot is None:
            return None
        self._order.move_to_end(key)
        return slot

    def claim(self, layer: int, row: int) -> int:
        """A slot for ``(layer, row)``. Evicts the least recent resident when the window is full."""
        key = (layer, row)
        if key in self._order:
            self._order.move_to_end(key)
            return self._order[key]
        if self._free:
            slot = self._free.pop()
        else:
            _old, slot = self._order.popitem(last=False)
            for tensor in self.slots.values():
                tensor[slot].zero_()
        self._order[key] = slot
        return slot


class DiskExpertStore:
    """Host window plus the one-layer bounce ``copy_missing`` DMAs from."""

    def __init__(self, index, method, *, rows: int, capacity: int, pin: bool):
        self.index = index
        self.method = method
        self.rows = rows
        self.window = HostExpertWindow(method.layout(), capacity, pin=pin)
        self.bounce = {
            role: _alloc((rows, *spec.shape), spec.dtype, pin=pin)
            for role, spec in method.layout().items()
            if not spec.resident
        }
        self.hits = 0
        self.misses = 0
        self._fds: dict[str, int] = {}

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def stage(self, cache) -> None:
        """Fill ``bounce[row]`` for each id in ``cache.src_indices``. A GPU hit is not in that list."""
        n = int(cache.num_indices.item())
        if n <= 0:
            return
        rows = cache.src_indices[:n].detach().to(dtype=torch.int64, device="cpu").tolist()
        self.stage_rows(int(cache._pending_src_layer), rows, pad=cache.ep_pad)

    def stage_rows(self, layer: int, rows: list[int], *, pad: int | None) -> None:
        for row in rows:
            if pad is not None and row == pad:
                continue
            slot = self.window.hit(layer, row)
            if slot is None:
                slot = self.window.claim(layer, row)
                self._load(layer, row, slot)
                self.misses += 1
            else:
                self.hits += 1
            for role, bank in self.bounce.items():
                bank[row].copy_(self.window.slots[role][slot])

    def _load(self, layer: int, row: int, slot: int) -> None:
        piece = {}
        for proj in _PROJECTIONS:
            weight = self._read(self._loc(layer, row, proj))
            scale = self._read(self._loc(layer, row, proj + "_scale"))
            piece[proj] = weight.unsqueeze(0)
            piece[proj + "_scale"] = scale.to(torch.bfloat16).unsqueeze(0)
        out = {role: tensor[slot : slot + 1] for role, tensor in self.window.slots.items()}
        self.method.pack(piece, out)

    def _loc(self, layer: int, row: int, role: str) -> TensorLoc:
        try:
            return self.index[(layer, row, role)]
        except KeyError:
            raise KeyError(f"no checkpoint tensor for MoE layer {layer} local expert {row} role {role}") from None

    def _read(self, loc: TensorLoc) -> torch.Tensor:
        buf = os.pread(self._fd(loc.path), loc.nbytes, loc.offset)
        if len(buf) != loc.nbytes:
            raise RuntimeError(f"short read of {loc.path} at {loc.offset}: {len(buf)} != {loc.nbytes}")
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(self._fds[loc.path], loc.offset, loc.nbytes, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        raw = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone()
        return raw.view(loc.dtype).reshape(loc.shape)

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd


def open_disk_experts(config, method, *, pin: bool = True) -> tuple[DiskExpertStore, ExpertBanks]:
    """Index this rank's experts and allocate the bounce that stands in for every layer's banks."""
    model = config.model_config
    info = config.ep_info
    index = build_expert_index(config.model_path, model, rank=info.rank, size=info.size)
    owned, rows = bank_geometry(model.num_experts, info.size)
    layout = method.layout()
    per = packed_row_bytes(layout)
    budget = int(config.moe_disk_cache_gb * (1 << 30))
    if per <= 0 or budget < per:
        raise ValueError(
            f"--moe-disk-cache-gb {config.moe_disk_cache_gb:g} is smaller than one packed expert ({per} bytes)"
        )
    capacity = budget // per
    store = DiskExpertStore(index, method, rows=rows, capacity=capacity, pin=pin)
    sources = {role: [tensor] * model.num_moe_layers for role, tensor in store.bounce.items()}
    banks = ExpertBanks(
        quant_format=legacy_format_for(method.kind, method.kernel.name),
        sources=sources,
        kind=method.kind,
        kernel=method.kernel.name,
        layout=layout,
    )
    if is_rank0():
        logger.info(
            f"--moe-disk: {owned * model.num_moe_layers} owned experts, "
            f"host window {capacity} rows ({capacity * per / 2**30:.2f} GiB), "
            f"bounce {rows * per / 2**30:.2f} GiB"
        )
    return store, banks


def _shard_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return 8 + n, header


def _alloc(shape: tuple[int, ...], dtype: torch.dtype, *, pin: bool) -> torch.Tensor:
    if not pin:
        return torch.zeros(shape, dtype=dtype)
    from freetoken.kernel.pinned import alloc_pinned_tensor

    tensor = alloc_pinned_tensor(*shape, dtype=dtype)
    tensor.zero_()
    return tensor

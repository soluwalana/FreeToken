"""Checkpoint-backed expert rows for ``--moe-disk``.

The GPU slot cache still decides which rows live on the device. Its host source is one
pinned bounce, aliased as every layer, so the existing copy plan and slot ids stay valid.
A pinned window in front of the checkpoint holds rows the GPU has evicted but a later
token still wants. A miss that is also absent from that window is read from the safetensors
shard and packed into the kernel's native rows. Block-fp8 and NVFP4 (``triton``
and ``w4a4``) are the formats this reader knows. Which checkpoint keys those
rows live under comes from the model's ``disk_expert_keys``. The pad row stays
zero and is never stored. NVFP4 rows stay packed; the inline-dequant GEMM is a
kernel choice, not a second copy in the window.
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass

import torch

from freetoken.layers.quantization import QuantKind
from freetoken.moe.expert_banks import ExpertBanks
from freetoken.moe.expert_parallel import bank_geometry, is_rank0, remap_expert
from freetoken.moe.legacy_format import legacy_format_for
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Kernels whose banks are the checkpoint rows. Marlin and b12x retile inside pack.
DISK_NVFP4_KERNELS = ("triton", "w4a4")
# A disk prefill that already touches this much of the owned rows keeps the sequential read.
_WHOLE_LAYER_NUMERATOR = 9
_WHOLE_LAYER_DENOMINATOR = 10
# Bank roles the native-row kernels pack. The model's key spec maps checkpoint projections onto these.
_PROJECTIONS = ("gate", "up", "down")

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


def _disk_keys(model_config, kind: QuantKind):
    """The family's checkpoint key spec. The reader does not know model tensor names."""
    from freetoken.models.register import _load_attr, get_model_spec

    archs = getattr(model_config, "architectures", None) or []
    if not archs:
        raise ValueError("--moe-disk needs the checkpoint architecture to find expert keys")
    spec = get_model_spec(archs[0])
    try:
        hook = _load_attr(spec.module, "disk_expert_keys")
    except AttributeError:
        hook = None
    keys = hook(kind) if hook is not None else None
    if keys is None:
        raise ValueError(f"--moe-disk has no expert index for {archs[0]} ({kind})")
    return keys


def build_expert_index(
    folder: str,
    model_config,
    *,
    rank: int,
    size: int,
    kind: QuantKind = QuantKind.FP8_BLOCK,
) -> dict[tuple[int, int, str], TensorLoc]:
    """``(bank layer, local row, role) -> payload location`` for the experts this rank owns.

    Reads shard headers only. Unowned expert ids are left out, and so is the pad row.
    """
    index, _inputs = _scan_expert_index(folder, model_config, rank=rank, size=size, kind=kind)
    return index


def _scan_expert_index(
    folder: str,
    model_config,
    *,
    rank: int,
    size: int,
    kind: QuantKind,
) -> tuple[dict[tuple[int, int, str], TensorLoc], dict[tuple[int, int, str], TensorLoc]]:
    from freetoken.models.loader import safetensors_weight_map

    keys = _disk_keys(model_config, kind)
    pattern, suffixes, pieces = keys.pattern, keys.suffixes, keys.pieces
    first_k = int(getattr(model_config, "first_k_dense_replace", 0) or 0)
    num_moe_layers = model_config.num_moe_layers
    owned, _rows = bank_geometry(model_config.num_experts, size)
    # (name, bank, row, role, side). side is the activation-scale map, which is not a host-window row.
    wanted: dict[str, list[tuple[str, int, int, str, bool]]] = {}
    for name, shard in safetensors_weight_map(folder).items():
        match = pattern.match(name)
        side = False
        if match is None and keys.input_pattern is not None:
            match = keys.input_pattern.match(name)
            side = match is not None
        if match is None:
            continue
        bank = int(match["layer"]) - first_k
        if bank < 0 or bank >= num_moe_layers:
            continue
        row = remap_expert(int(match["expert"]), rank=rank, size=size)
        if row is None:
            continue
        if side:
            role = match["proj"] + "_input"
        else:
            role = match["proj"] + suffixes[match["kind"]]
        wanted.setdefault(shard, []).append((name, bank, row, role, side))

    index: dict[tuple[int, int, str], TensorLoc] = {}
    inputs: dict[tuple[int, int, str], TensorLoc] = {}
    for shard, items in wanted.items():
        path = os.path.join(folder, shard)
        data_base, header = _shard_header(path)
        for name, bank, row, role, side in items:
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
            dest = inputs if side else index
            if key in dest:
                raise ValueError(f"duplicate expert tensor for layer {bank} row {row} role {role}: {name}")
            dest[key] = TensorLoc(path, data_base + int(start), nbytes, shape, dtype)

    expected = num_moe_layers * owned * pieces
    if len(index) != expected:
        raise ValueError(
            f"--moe-disk indexed {len(index)} expert tensors, expected {expected} "
            f"({num_moe_layers} layers x {owned} owned experts x {pieces}). "
            f"The reader looks for {keys.where}{keys.names}."
        )
    return index, inputs


def routed_prefill_rows(topk_ids, *, num_experts: int, pad: int | None) -> list[int] | None:
    """Unique owned local rows for a disk prefill.

    ``None`` means the set covers at least 90% of the owned rows, so the caller
    keeps the sequential whole-layer read. The pad and any id outside the owned
    rows are dropped.
    """
    owned = num_experts - 1 if pad is not None else num_experts
    if isinstance(topk_ids, torch.Tensor):
        flat = topk_ids.detach().reshape(-1).to(dtype=torch.int64, device="cpu").tolist()
    else:
        flat = list(topk_ids)
    rows = sorted({int(i) for i in flat if 0 <= int(i) < owned})
    if owned > 0 and len(rows) * _WHOLE_LAYER_DENOMINATOR >= owned * _WHOLE_LAYER_NUMERATOR:
        return None
    return rows


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

    def __init__(
        self,
        index,
        method,
        *,
        rows: int,
        capacity: int,
        pin: bool,
        inputs=None,
        gate_up_alpha: torch.Tensor | None = None,
        down_alpha: torch.Tensor | None = None,
    ):
        self.index = index
        self.inputs = inputs or {}
        self.method = method
        self.rows = rows
        self.gate_up_alpha = gate_up_alpha
        self.down_alpha = down_alpha
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
        layer = int(cache._pending_src_layer)
        n = int(cache.num_indices.item())
        if n <= 0:
            return
        rows = cache.src_indices[:n].detach().to(dtype=torch.int64, device="cpu").tolist()
        self.stage_rows(layer, rows, pad=cache.ep_pad)

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
        if self.method.kind is QuantKind.NVFP4:
            self._load_nvfp4(layer, row, slot)
            return
        piece = {}
        for proj in _PROJECTIONS:
            weight = self._read(self._loc(layer, row, proj))
            scale = self._read(self._loc(layer, row, proj + "_scale"))
            piece[proj] = weight.unsqueeze(0)
            piece[proj + "_scale"] = scale.to(torch.bfloat16).unsqueeze(0)
        out = {role: tensor[slot : slot + 1] for role, tensor in self.window.slots.items()}
        self.method.pack(piece, out)

    def _load_nvfp4(self, layer: int, row: int, slot: int) -> None:
        """Pack checkpoint rows. Scales stay fp8. Activation scales are not window rows."""
        want_input = self.method.kernel.name == "w4a4"
        piece = {}
        for proj in _PROJECTIONS:
            piece[proj] = self._read(self._loc(layer, row, proj)).unsqueeze(0)
            piece[proj + "_scale"] = self._read(self._loc(layer, row, proj + "_scale")).unsqueeze(0)
            piece[proj + "_global"] = self._read(self._loc(layer, row, proj + "_global")).reshape(1, -1)
            if want_input:
                piece[proj + "_input"] = self._read(self._input_loc(layer, row, proj)).reshape(1, -1)
        out = {role: tensor[slot : slot + 1] for role, tensor in self.window.slots.items()}
        got = self.method.pack(piece, out)
        if not got or self.gate_up_alpha is None:
            return
        at = layer * self.rows + row
        self.gate_up_alpha[at] = got["gate_up_alpha"].reshape(-1)[0]
        self.down_alpha[at] = got["down_alpha"].reshape(-1)[0]

    def _input_loc(self, layer: int, row: int, proj: str) -> TensorLoc:
        try:
            return self.inputs[(layer, row, proj + "_input")]
        except KeyError:
            raise KeyError(
                f"no input_scale for MoE layer {layer} local expert {row} {proj}_proj; "
                "w4a4 needs the checkpoint's static activation scales"
            ) from None

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
    index, inputs = _scan_expert_index(config.model_path, model, rank=info.rank, size=info.size, kind=method.kind)
    owned, rows = bank_geometry(model.num_experts, info.size)
    layout = method.layout()
    per = packed_row_bytes(layout)
    budget = int(config.moe_disk_cache_gb * (1 << 30))
    if per <= 0 or budget < per:
        raise ValueError(
            f"--moe-disk-cache-gb {config.moe_disk_cache_gb:g} is smaller than one packed expert ({per} bytes)"
        )
    capacity = budget // per
    gate_up_alpha, down_alpha = _resident_alphas(layout, model.num_moe_layers, rows)
    store = DiskExpertStore(
        index, method, rows=rows, capacity=capacity, pin=pin, inputs=inputs,
        gate_up_alpha=gate_up_alpha, down_alpha=down_alpha,
    )
    sources = {role: [tensor] * model.num_moe_layers for role, tensor in store.bounce.items()}
    banks = ExpertBanks(
        quant_format=legacy_format_for(method.kind, method.kernel.name),
        sources=sources,
        gate_up_alpha=gate_up_alpha,
        down_alpha=down_alpha,
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


def _resident_alphas(layout, layers: int, rows: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Per-expert activation scales for a native W4A4 kernel. Triton keeps none.

    Allocated on the current CUDA device when there is one, so ``set_alphas`` keeps
    this tensor and a later disk pack writes the scale the GEMM reads.
    """
    if "gate_up_alpha" not in layout:
        return None, None
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    n = layers * rows
    gate = torch.ones(n, dtype=layout["gate_up_alpha"].dtype, device=device)
    down = torch.ones(n, dtype=layout["down_alpha"].dtype, device=device)
    return gate, down


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

"""Weight loading for GLM-5.2 / GLM-5.3 (``glm_moe_dsa``).

Resident projections load as stored: bf16, or fp8-e4m3 codes with their block scales.
``kv_b_proj`` and the indexer's ``wk`` / ``weights_proj`` are served bf16 (MLA absorption
reads ``kv_b`` as a bmm; the indexer top-k stays in bf16), so an fp8 copy is dequantized
here. The router selection bias is remapped ``mlp.gate.e_score_correction_bias ->
mlp.e_score_correction_bias``. Only the trailing MTP layer is skipped. Routed experts
are NVFP4 (glm4_moe key layout) or block-fp8 and go to the offload cache.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import safetensors
import torch
from freetoken.layers.quantization import QuantKind
from freetoken.models.glm4_moe.weight import (
    nvfp4_expert_spec,
)
from freetoken.models.loader import drop_page_cache
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = device
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=str(self._device)
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best effort
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _projection_scheme(name: str):
    """The installed quant config's scheme for ``name``, or None when weights load without one."""
    from freetoken.layers.quantization import get_quant_config

    try:
        quant = get_quant_config()
    except RuntimeError:
        return None
    if quant is None:
        return None
    return quant.scheme_for_name(name)


def _per_row_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar or per-channel scale -> fp32 ``[rows]``."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(
            f"fp8 weight_scale has {flat.numel()} elements for a weight with {rows} output rows "
            f"(shape {tuple(scale.shape)}); expected 1 or {rows}"
        )
    return flat.contiguous()


def _proj(reader, src: str, dst: str, *, bf16: bool = False) -> Iterator[tuple[str, torch.Tensor]]:
    """One projection as the module stores it.

    A bf16 checkpoint tensor stays bf16. Per-tensor FP8 (``weight_scale`` and, when the
    shard has it, ``input_scale``) is yielded for an ``FP8_TENSOR`` scheme, which Ada's
    fp8 tensor cores can run. Block-fp8 stays ``weight`` + ``weight_scale_inv``. ``bf16``
    dequantizes either fp8 form for modules the architecture serves unquantized.
    """
    w = reader.get(f"{src}.weight")
    if w.dtype != torch.float8_e4m3fn:
        yield f"{dst}.weight", w.to(torch.bfloat16)
        return
    scheme = _projection_scheme(src)
    if scheme is not None and scheme.kind is QuantKind.FP8_TENSOR:
        scale = _per_row_scale(reader.get(f"{src}.weight_scale"), w.shape[0])
        if bf16:
            yield f"{dst}.weight", (w.to(torch.float32) * scale[:, None]).to(torch.bfloat16)
            return
        yield f"{dst}.weight", w
        yield f"{dst}.weight_scale", scale
        if scheme.has("input_scale"):
            key = f"{src}.input_scale"
            if not reader.has(key):
                raise KeyError(f"{src} is per-tensor FP8 with an activation scale, but the checkpoint has no {key}")
            yield f"{dst}.input_scale", reader.get(key).reshape(()).to(torch.float32)
        return
    scale = reader.get(f"{src}.weight_scale_inv")
    if not bf16:
        yield f"{dst}.weight", w
        yield f"{dst}.weight_scale_inv", scale
        return
    from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8

    yield f"{dst}.weight", dequant_block_fp8(w, scale)


_FP8_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)$"
)


def _layer_to_bank(layer: int, config):
    if layer < config.first_k_dense_replace or layer >= config.num_layers:
        return None
    return layer - config.first_k_dense_replace


def _bf16_scales(tensors):
    """Expert scale banks are bf16. Checkpoints store the block scales as fp32."""
    for name, tensor in tensors:
        if name.endswith("weight_scale_inv") and tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        yield name, tensor


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM routed experts only serve from the offload cache; they are loaded from their expert pieces."
    )
    assert include_non_moe
    from freetoken.moe.expert_parallel import is_rank0

    config = parse_config(cached_load_hf_config(model_path))
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = is_rank0()
    dense = config.first_k_dense_replace
    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading GLM-5.2 dense weights",
            disable=not primary,
        ):
            a = f"model.layers.{layer}.self_attn"
            for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                yield from _proj(reader, f"{a}.{proj}", f"{a}.{proj}", bf16=proj == "kv_b_proj")
            for norm in ("q_a_layernorm", "kv_a_layernorm"):
                yield f"{a}.{norm}.weight", reader.get(f"{a}.{norm}.weight")
            # DSA lightning indexer ("full" layers only; "shared" layers reuse their
            # group leader's selection and ship no indexer tensors).
            idx_types = config.glm_dsa_args.indexer_types
            if idx_types and idx_types[layer] == "full":
                yield from _proj(reader, f"{a}.indexer.wq_b", f"{a}.indexer.wq_b")
                # wk and weights_proj stay bf16 in the module (top-k). An fp8 wk is dequantized.
                for proj in ("wk", "weights_proj"):
                    yield from _proj(reader, f"{a}.indexer.{proj}", f"{a}.indexer.{proj}", bf16=True)
                yield f"{a}.indexer.k_norm.weight", reader.get(f"{a}.indexer.k_norm.weight")
                yield f"{a}.indexer.k_norm.bias", reader.get(f"{a}.indexer.k_norm.bias")
            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield (
                    f"model.layers.{layer}.{norm}.weight",
                    reader.get(f"model.layers.{layer}.{norm}.weight"),
                )

            m = f"model.layers.{layer}.mlp"

            if layer < dense:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _proj(reader, f"{m}.{proj}", f"{m}.{proj}")
            else:
                yield f"{m}.gate.weight", reader.get(f"{m}.gate.weight")
                yield (
                    f"{m}.e_score_correction_bias",
                    reader.get(f"{m}.gate.e_score_correction_bias").to(torch.bfloat16),
                )
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _proj(reader, f"{m}.shared_experts.{proj}", f"{m}.shared_experts.{proj}")

        yield "model.embed_tokens.weight", reader.get("model.embed_tokens.weight")
        yield "model.norm.weight", reader.get("model.norm.weight")
        yield "lm_head.weight", reader.get("lm_head.weight")
    finally:
        reader.close()


def iter_expert_pieces(model_path, config, kind: QuantKind, *, layers=None, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Block-fp8 routed experts, one piece per owned expert. Other kinds use the generic readers."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
    from freetoken.distributed import get_tp_info
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_parallel import is_rank0, remap_expert
    from freetoken.moe.expert_pieces import per_expert_pieces

    if get_tp_info().size > 1:
        raise NotImplementedError("glm_moe_dsa fp8 expert banks support TP=1 only")

    suffix = {"weight": "", "weight_scale_inv": "_scale"}
    keep = None if layers is None else set(layers)

    def locate(raw_name: str):
        m = _FP8_EXPERT_RE.match(raw_name)
        if m is None:
            return None
        bank = _layer_to_bank(int(m["layer"]), config)
        if bank is None or (keep is not None and bank not in keep):
            return None
        row = remap_expert(int(m["expert"]))
        if row is None:
            return None
        return bank, row, m["proj"] + suffix[m["kind"]]

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(_bf16_scales(tensors), locate, tensors_per_expert=6)

    def _serial():
        folder = download_hf_weight(model_path)
        with open(os.path.join(folder, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        reader = _ShardReader(folder, weight_map, torch.device("cpu"))
        try:
            ckpt_layers = [
                layer for layer in range(config.first_k_dense_replace, config.num_layers)
                if keep is None or _layer_to_bank(layer, config) in keep
            ]
            for layer in tqdm(ckpt_layers, desc="Loading GLM fp8 experts (serial)", disable=not is_rank0()):
                for e in range(config.num_experts):
                    if remap_expert(e) is None:
                        continue
                    base = f"model.layers.{layer}.mlp.experts.{e}"
                    for proj in ("gate", "up", "down"):
                        for kind_name in suffix:
                            name = f"{base}.{proj}_proj.{kind_name}"
                            yield name, reader.get(name)
        finally:
            reader.close()

    return per_expert_pieces(_bf16_scales(_serial()), locate, tensors_per_expert=6)


__all__ = ["iter_weights", "iter_expert_pieces", "nvfp4_expert_spec"]

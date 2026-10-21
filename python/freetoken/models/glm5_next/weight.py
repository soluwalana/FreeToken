"""Weight loading for GLM-5.3-Flash (``glm5_next``).

Supported checkpoints, all in the multimodal-wrapper layout (``model.language_model.*``):
NVFP4 exports (ModelOpt tensor kinds from LibertAIDAI, compressed-tensors kinds from
RedHatAI, selected by ``quantization_config``) and the zai-org block-fp8 release. Not
supported: text-only key layouts, TP > 1, resident routed experts.

Routed experts go to the offload cache from their NVFP4 or block-fp8 pieces. A dense
projection the quant config marks NVFP4 (the leading MLPs on nvidia/GLM-5.3-Flash-NVFP4)
is yielded as the NVFP4 linear's buffers: packed ``weight``, fp8 ``weight_scale``,
per-row ``weight_global`` from the scalar ``weight_scale_2``, and ``input_scale`` when
the scheme has one. Every other projection loads as stored (bf16, or fp8 codes with
their block scales) with keys renamed ``model.language_model.X`` -> ``model.X``. The
vision tower loads as stored under ``model.visual.X`` -> ``visual.X`` when an encoder
is built; the trailing MTP layer is never read.

Load-time fusions (must mirror the module split orders):

* KDA ``in_proj``  = q|k|v|b|f_a|g_a projections concatenated on the output axis
* KDA ``conv1d``   = q|k|v depthwise conv weights concatenated on the channel axis

fp32-kept tensors: ``A_log`` / ``dt_bias``, the mHC ``hc_*`` tensors, the indexer
APE, and the router ``e_score_correction_bias``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import torch
from freetoken.distributed import get_ep_info, get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.glm_moe_dsa.weight import _ShardReader
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .args import Glm5NextArgs
from .config import parse_config

# Checkpoint prefix (multimodal wrapper) -> model prefix.
_CKPT = "model.language_model"
_MODEL = "model"

# MTP-layer experts (layer == num_layers under the full checkpoint) map to None
# alongside the dense prefix; the bank loader skips them.
def _layer_to_bank(layer, config):
    return (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    )


# ModelOpt export (LibertAIDAI/GLM-5.3-Flash-NVFP4): weight | weight_scale |
# weight_scale_2 (dequant-side global).
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts",
)

# llm-compressor export (RedHatAI/GLM-5.3-Flash-NVFP4): weight_packed |
# weight_scale | weight_global_scale (quant-side global -> reciprocal at ingest).
# ``input_global_scale`` (the calibrated W4A4 activation scale) deliberately does
# not match: our routed-expert paths are W4A16 and never quantize activations.
_NVFP4_CT_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight_packed|weight_global_scale|weight_scale)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts (compressed-tensors)",
    kind_map={"weight_packed": "weight", "weight_global_scale": "weight_scale_2"},
    global_reciprocal=True,
)


def _select_expert_source_spec(model_path: str) -> Nvfp4ExpertSourceSpec:
    quant = getattr(cached_load_hf_config(model_path), "quantization_config", None) or {}
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    method = str(get("quant_method") or "").lower()
    return _NVFP4_CT_SOURCE_SPEC if method == "compressed-tensors" else _NVFP4_SOURCE_SPEC

# KDA in_proj fusion order; MUST match Glm5NextKDA._in_proj_split.
_KDA_IN_PROJ = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")

# zai-org fp8 release: fp8 codes + fp32 128x128 block scales per expert projection
_FP8_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)$"
)


def _projection_scheme(name: str):
    """Scheme of the module ``name``. ``scheme_for`` applies the language_model root."""
    from freetoken.layers.quantization import get_quant_config

    try:
        quant = get_quant_config()
    except RuntimeError:
        return None
    if quant is None:
        return None
    return quant.scheme_for(name)


def _nvfp4_global(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor ``weight_scale_2`` -> fp32 ``[rows]``, the shape the NVFP4 linear stores."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(
            f"weight_scale_2 has {flat.numel()} elements for a weight with {rows} output rows "
            f"(shape {tuple(scale.shape)}); expected 1 or {rows}"
        )
    return flat.contiguous()


def _iter_nvfp4_linear(reader, src: str, dst: str, weight: torch.Tensor, scheme):
    """Packed NVFP4 buffers for one dense linear. Scales stay in the checkpoint dtypes."""
    if weight.dtype is not torch.uint8:
        raise ValueError(f"{src}.weight is {weight.dtype}, expected packed uint8 for an NVFP4 module")
    scale = reader.get(f"{src}.weight_scale")
    if scale.dtype is not torch.float8_e4m3fn:
        raise ValueError(f"{src}.weight_scale is {scale.dtype}, expected float8_e4m3fn")
    yield f"{dst}.weight", weight
    yield f"{dst}.weight_scale", scale
    yield f"{dst}.weight_global", _nvfp4_global(reader.get(f"{src}.weight_scale_2"), weight.shape[0])
    if scheme.has("input_scale"):
        key = f"{src}.input_scale"
        if not reader.has(key):
            raise KeyError(f"{src} is NVFP4 with an activation scale, but the checkpoint has no {key}")
        yield f"{dst}.input_scale", reader.get(key).reshape(()).to(torch.float32)


def _proj(reader, src: str, dst: str) -> Iterator[tuple[str, torch.Tensor]]:
    """One projection as the module stores it.

    NVFP4 (packed uint8 weight, fp8 block scale, per-row global, optional activation
    scale) is yielded when the quant config says so. Otherwise the tensor is bf16, or
    fp8 codes with their block scales. A packed uint8 weight on any other module is an
    error: casting those bytes to bf16 would not recover the projection.
    """
    w = reader.get(f"{src}.weight")
    scheme = _projection_scheme(dst)
    if scheme is not None and scheme.kind is QuantKind.NVFP4:
        yield from _iter_nvfp4_linear(reader, src, dst, w, scheme)
        return
    if w.dtype is torch.uint8:
        raise ValueError(f"{src}.weight is packed uint8 but {dst} is not an NVFP4 module")
    if w.dtype == torch.float8_e4m3fn:
        yield f"{dst}.weight", w
        yield f"{dst}.weight_scale_inv", reader.get(f"{src}.weight_scale_inv")
    else:
        yield f"{dst}.weight", w.to(torch.bfloat16)


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    return _select_expert_source_spec(model_path)


def _iter_kda_layer(reader, layer: int) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    # One fused input GEMM: q|k|v|b|f_a|g_a (output-axis concat).
    parts = [reader.get(f"{src}.{p}.weight") for p in _KDA_IN_PROJ]
    if any(p.dtype == torch.float8_e4m3fn for p in parts):
        raise NotImplementedError("fp8 KDA input projections are not fused by this reader")
    yield f"{dst}.in_proj.weight", torch.cat([p.to(torch.bfloat16) for p in parts], dim=0)
    # One merged depthwise conv over the q|k|v stream (channel-axis concat).
    conv = torch.cat(
        [reader.get(f"{src}.{p}_conv1d.weight").to(torch.bfloat16) for p in ("q", "k", "v")],
        dim=0,
    )
    yield f"{dst}.conv1d.weight", conv
    for p in ("f_b_proj", "g_b_proj", "o_proj"):
        yield from _proj(reader, f"{src}.{p}", f"{dst}.{p}")
    # Gate params stay fp32 (the recurrent kernels read them as fp32).
    yield f"{dst}.A_log", reader.get(f"{src}.A_log").to(torch.float32)
    yield f"{dst}.dt_bias", reader.get(f"{src}.dt_bias").to(torch.float32)
    yield f"{dst}.o_norm.weight", reader.get(f"{src}.o_norm.weight").to(torch.bfloat16)


def _iter_dsa_layer(reader, layer: int) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        yield from _proj(reader, f"{src}.{proj}", f"{dst}.{proj}")
    for norm in ("q_a_layernorm", "kv_a_layernorm"):
        yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(torch.bfloat16)
    # kpool indexer (every DSA layer owns one). Kept bf16; the APE is fp32.
    for proj in ("wq_b", "wk", "weights_proj"):
        yield f"{dst}.indexer.{proj}.weight", reader.get(
            f"{src}.indexer.{proj}.weight"
        ).to(torch.bfloat16)
    for part, dtype in (
        ("k_norm.weight", torch.bfloat16),
        ("k_norm.bias", torch.bfloat16),
        ("index_kpool_compress_gate", torch.bfloat16),
        ("index_kpool_compress_ape", torch.float32),
    ):
        yield f"{dst}.indexer.{part}", reader.get(f"{src}.indexer.{part}").to(dtype)


def _iter_vision(reader, weight_map: dict) -> Iterator[tuple[str, torch.Tensor]]:
    for name in weight_map:
        if name.startswith("model.visual."):
            yield "visual." + name[len("model.visual.") :], reader.get(name).to(torch.bfloat16)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.3 routed experts only serve from the offload cache; they are loaded from their expert pieces."
    )
    assert include_non_moe
    if get_tp_info().size > 1:
        # The loader emits full fused KDA/DSA tensors; TP sharding (per-head q|k|v|b
        # splits, replicated f_a|g_a, row-parallel o_proj) is not implemented yet --
        # same status as every other linear-hybrid / offload-family model in tree.
        raise NotImplementedError("glm5_next weight loading currently supports TP=1 only")
    config = parse_config(cached_load_hf_config(model_path))
    args: Glm5NextArgs = config.glm5_args
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary() and get_ep_info().is_primary()
    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading GLM-5.3 dense weights",
            disable=not primary,
        ):
            src = f"{_CKPT}.layers.{layer}"
            dst = f"{_MODEL}.layers.{layer}"
            if args.is_kda_layer(layer):
                yield from _iter_kda_layer(reader, layer)
            else:
                yield from _iter_dsa_layer(reader, layer)

            # mHC mixing tensors, fp32 on every layer.
            for hc in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                       "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
                yield f"{dst}.{hc}", reader.get(f"{src}.{hc}").to(torch.float32)

            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(
                    torch.bfloat16
                )

            if layer < config.first_k_dense_replace:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _proj(reader, f"{src}.mlp.{proj}", f"{dst}.mlp.{proj}")
            else:
                yield f"{dst}.mlp.gate.weight", reader.get(f"{src}.mlp.gate.weight").to(
                    torch.bfloat16
                )
                yield (
                    f"{dst}.mlp.e_score_correction_bias",
                    # fp32 like HF's router math (the module declares fp32; a bf16
                    # cast would perturb top-8 selection on fp32-bias checkpoints).
                    reader.get(f"{src}.mlp.gate.e_score_correction_bias").to(torch.float32),
                )
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _proj(reader, f"{src}.mlp.shared_experts.{proj}", f"{dst}.mlp.shared_experts.{proj}")

        yield f"{_MODEL}.embed_tokens.weight", reader.get(
            f"{_CKPT}.embed_tokens.weight"
        ).to(torch.bfloat16)
        yield f"{_MODEL}.norm.weight", reader.get(f"{_CKPT}.norm.weight").to(torch.bfloat16)
        yield "lm_head.weight", reader.get("lm_head.weight").to(torch.bfloat16)
        if include_vision:
            yield from _iter_vision(reader, weight_map)
    finally:
        reader.close()


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    try:
        yield from _iter_vision(reader, weight_map)
    finally:
        reader.close()


def iter_expert_pieces(model_path, config, kind: QuantKind, *, layers=None, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Block-fp8 routed experts, one piece per expert: ``{gate, up, down}`` fp8 codes and their ``_scale`` companions; other kinds use the generic readers."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("glm5_next fp8 expert banks support TP=1 only")
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_parallel import is_rank0, remap_expert
    from freetoken.moe.expert_pieces import per_expert_pieces

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
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

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
            for layer in tqdm(ckpt_layers, desc="Loading GLM-5.3 fp8 experts (serial)", disable=not is_rank0()):
                for e in range(config.num_experts):
                    if remap_expert(e) is None:
                        continue
                    base = f"{_CKPT}.layers.{layer}.mlp.experts.{e}"
                    for proj in ("gate", "up", "down"):
                        for kind_name in suffix:
                            name = f"{base}.{proj}_proj.{kind_name}"
                            yield name, reader.get(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


__all__ = ["iter_weights", "iter_expert_pieces", "nvfp4_expert_spec"]

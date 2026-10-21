"""Expert parallelism in one process: ownership, bank rows, masked prefill and decode.

No second process and no NCCL. The gloo join, the request broadcast, and the captured
all-reduce are outside this file.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import (
    DistributedInfo,
    get_tp_info,
    set_ep_info,
    set_tp_info,
    try_get_tp_info,
)
from freetoken.layers.quantization import QuantKind
from freetoken.moe.expert_parallel import bank_geometry, clear_ep_comm, mask_routes, set_ep_comm


@pytest.fixture(scope="module", autouse=True)
def _tp():
    if try_get_tp_info() is None:
        set_tp_info(0, 1)


@pytest.fixture(autouse=True)
def _reset_ep():
    yield
    set_ep_info(0, 1)
    clear_ep_comm()


def _hf():
    return SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": "bfloat16"})


def test_ep_info_does_not_shard_tensor_parallel_ranks():
    from freetoken.server.args import ServerArgs
    from freetoken.server.launch import scheduler_rank_args

    parent = ServerArgs(
        model_path="/models/anon",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        ep_info=DistributedInfo(0, 2),
    )
    rank0 = scheduler_rank_args(parent, 0)
    rank1 = scheduler_rank_args(parent, 1)
    assert rank0.tp_info == DistributedInfo(0, 1) and rank0.ep_info == DistributedInfo(0, 2)
    assert rank1.tp_info == DistributedInfo(0, 1) and rank1.ep_info == DistributedInfo(1, 2)
    assert get_tp_info() == DistributedInfo(0, 1)


def test_two_gpus_require_expert_parallel_size(capsys):
    from freetoken.server.args import parse_args

    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _hf()):
        args, _ = parse_args([
            "--model", "/models/anon", "--expert-parallel-size", "2", "--gpu", "0,1",
        ])
        assert args.ep_info == DistributedInfo(0, 2)
        assert args.tp_info == DistributedInfo(0, 1)
        with pytest.raises(SystemExit):
            parse_args(["--model", "/models/anon", "--gpu", "0,1"])
        err = capsys.readouterr().err
        assert "expert-parallel-size" in err
        with pytest.raises(SystemExit):
            parse_args([
                "--model", "/models/anon", "--expert-parallel-size", "2",
                "--tensor-parallel-size", "2", "--gpu", "0,1",
            ])
        with pytest.raises(SystemExit):
            parse_args([
                "--model", "/models/anon", "--expert-parallel-size", "2",
                "--moe-strategy", "cpu",
            ])


def test_engine_refuses_a_size_that_does_not_divide_and_a_full_expert_profile():
    from freetoken.engine.engine import _check_expert_parallel

    def cfg(**over):
        base = dict(
            ep_info=DistributedInfo(0, 2),
            tp_info=DistributedInfo(0, 1),
            model_config=SimpleNamespace(is_moe=True, num_experts=288),
            moe_strategy="offload",
            moe_cpu_layers=None,
            moe_cache_split="equal",
        )
        base.update(over)
        return SimpleNamespace(**base)

    _check_expert_parallel(cfg(), strategy_ready=True)
    assert bank_geometry(288, 2) == (144, 145)
    with pytest.raises(ValueError, match="divide"):
        _check_expert_parallel(cfg(ep_info=DistributedInfo(0, 7)), strategy_ready=False)
    with pytest.raises(ValueError, match="profile"):
        _check_expert_parallel(cfg(moe_cache_split="profile:/p.json"), strategy_ready=True)
    with pytest.raises(ValueError, match="offload"):
        _check_expert_parallel(cfg(moe_strategy="cpu"), strategy_ready=True)
    with pytest.raises(ValueError, match="tensor-parallel"):
        _check_expert_parallel(cfg(tp_info=DistributedInfo(0, 2)), strategy_ready=False)


def test_mask_writes_a_legal_local_row_and_a_zero_weight():
    set_ep_info(0, 2)
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.4, 0.6], [0.2, 0.8]])
    masked_w, masked_ids = mask_routes(ids, weights, pad_row=2)
    assert torch.equal(masked_ids, torch.tensor([[0, 2], [1, 2]], dtype=torch.int32))
    assert torch.equal(masked_w, torch.tensor([[0.4, 0.0], [0.2, 0.0]]))
    assert int(masked_ids.max()) < 3


def _bf16_layer(num_experts: int):
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig

    return OffloadMoELayer(
        0, num_experts, 2, 4, 4, quant_config=NoQuantConfig(), prefix="model.layers.0.mlp.experts",
    )


def test_banks_are_dense_local_rows_and_a_global_id_does_not_fit():
    from freetoken.moe.expert_banks import build_expert_banks

    set_ep_info(0, 2)
    layer = _bf16_layer(4)
    assert layer.num_experts == 4 and layer.quant_method.cfg.num_experts == 4

    def pieces():
        for row, value in ((0, 1.0), (1, 3.0)):
            yield 0, row, row + 1, {
                "gate_up": torch.full((1, 8, 4), value, dtype=torch.bfloat16),
                "down": torch.full((1, 4, 4), value, dtype=torch.bfloat16),
            }

    banks = build_expert_banks(layer.quant_method, 1, pieces(), device=torch.device("cpu"))
    gate = banks.sources["gate_up"][0]
    assert gate.shape[0] == 3
    assert torch.all(gate[0].float() == 1.0) and torch.all(gate[1].float() == 3.0)
    assert torch.count_nonzero(gate[2]) == 0

    def global_id():
        yield 0, 2, 3, {
            "gate_up": torch.zeros((1, 8, 4), dtype=torch.bfloat16),
            "down": torch.zeros((1, 4, 4), dtype=torch.bfloat16),
        }

    with pytest.raises(ValueError, match="out of range"):
        build_expert_banks(layer.quant_method, 1, global_id(), device=torch.device("cpu"))


def test_dummy_pad_row_is_zero_and_the_router_width_stays_global():
    from freetoken.moe.expert_banks import build_expert_banks

    set_ep_info(1, 2)
    layer = _bf16_layer(4)
    banks = build_expert_banks(layer.quant_method, 2, None, device=torch.device("cpu"), dummy=True)
    assert layer.quant_method.cfg.num_experts == 4
    for per_layer in banks.sources.values():
        for tensor in per_layer:
            assert tensor.shape[0] == 3
            assert torch.count_nonzero(tensor[-1]) == 0


def _expert_config():
    return SimpleNamespace(num_experts=4, num_layers=1, num_moe_layers=1, first_k_dense_replace=0)


def test_fp8_pieces_are_the_owned_subset_with_dense_rows(tmp_path):
    from freetoken.models.glm5_next.weight import iter_expert_pieces

    tensors = {}
    for expert in range(4):
        for proj in ("gate", "up", "down"):
            base = f"model.language_model.layers.0.mlp.experts.{expert}.{proj}_proj"
            tensors[f"{base}.weight"] = torch.zeros(4, dtype=torch.float8_e4m3fn)
            tensors[f"{base}.weight_scale_inv"] = torch.tensor([float(expert)])
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors},
    }))
    seen = {}
    for rank in (0, 1):
        set_ep_info(rank, 2)
        pieces = list(iter_expert_pieces(str(tmp_path), _expert_config(), QuantKind.FP8_BLOCK, parallel=False))
        assert len(pieces) == 2
        rows = {}
        for _layer, e0, e1, piece in pieces:
            assert e1 == e0 + 1 and e0 in (0, 1)
            rows[e0] = int(piece["gate_scale"].reshape(-1)[0])
        seen[rank] = rows
    assert seen[0] == {0: 0, 1: 2}
    assert seen[1] == {0: 1, 1: 3}
    assert set(seen[0].values()).isdisjoint(seen[1].values())
    assert set(seen[0].values()) | set(seen[1].values()) == {0, 1, 2, 3}


def test_glm_dsa_fp8_pieces_use_the_model_layers_layout(tmp_path):
    from freetoken.models.glm_moe_dsa.weight import iter_expert_pieces

    tensors = {}
    for expert in range(4):
        for proj in ("gate", "up", "down"):
            base = f"model.layers.0.mlp.experts.{expert}.{proj}_proj"
            tensors[f"{base}.weight"] = torch.zeros(4, dtype=torch.float8_e4m3fn)
            tensors[f"{base}.weight_scale_inv"] = torch.tensor([float(expert)])
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors},
    }))
    set_ep_info(1, 2)
    pieces = list(iter_expert_pieces(str(tmp_path), _expert_config(), QuantKind.FP8_BLOCK, parallel=False))
    rows = {}
    for _layer, e0, e1, piece in pieces:
        assert e1 == e0 + 1
        rows[e0] = int(piece["gate_scale"].reshape(-1)[0])
        assert piece["gate_scale"].dtype == torch.bfloat16
    assert rows == {0: 1, 1: 3}


def test_nvfp4_piece_count_accepts_the_owned_subset(tmp_path):
    from freetoken.models.glm5_next.weight import _NVFP4_SOURCE_SPEC
    from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces

    tensors = {}
    for expert in range(4):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.language_model.layers.0.mlp.experts.{expert}.{proj}"
            tensors[f"{base}.weight"] = torch.full((8,), expert, dtype=torch.uint8)
            tensors[f"{base}.weight_scale"] = torch.ones(4, dtype=torch.float8_e4m3fn)
            tensors[f"{base}.weight_scale_2"] = torch.tensor(1.0 + expert)
    save_file(tensors, tmp_path / "model.safetensors")
    set_ep_info(0, 2)
    pieces = list(iter_nvfp4_expert_pieces(str(tmp_path), _expert_config(), _NVFP4_SOURCE_SPEC))
    assert len(pieces) == 2
    rows = {e0: int(piece["gate"][0, 0]) for _layer, e0, _e1, piece in pieces}
    assert rows == {0: 0, 1: 2}
    set_ep_info(1, 2)
    other = list(iter_nvfp4_expert_pieces(str(tmp_path), _expert_config(), _NVFP4_SOURCE_SPEC))
    other_rows = {e0: int(piece["gate"][0, 0]) for _layer, e0, _e1, piece in other}
    assert other_rows == {0: 1, 1: 3}


def test_stacked_and_ftw_banks_are_refused(monkeypatch):
    from freetoken.moe.expert_banks import load_expert_banks
    from freetoken.moe.expert_pieces import stacked_expert_pieces

    set_ep_info(0, 2)
    with pytest.raises(ValueError, match="per-expert"):
        list(stacked_expert_pieces([], _expert_config()))
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda _path: True)
    with pytest.raises(ValueError, match="FTW"):
        load_expert_banks("/unused", _expert_config(), device=torch.device("cpu"))


class _Identity:
    def all_reduce(self, x):
        return x


def _silu_moe(hidden, gate_up, down, ids, weights):
    out = torch.zeros(hidden.shape, dtype=torch.float32)
    h = hidden.float()
    gu, dn, w = gate_up.float(), down.float(), weights.float()
    for token in range(h.shape[0]):
        acc = torch.zeros(h.shape[1])
        for k in range(ids.shape[1]):
            row = int(ids[token, k])
            assert 0 <= row < gu.shape[0]
            gate, up = gu[row].chunk(2, dim=0)
            mid = torch.nn.functional.silu(gate @ h[token]) * (up @ h[token])
            acc = acc + w[token, k] * (dn[row] @ mid)
        out[token] = acc
    return out


def _full_banks():
    torch.manual_seed(0)
    gate = torch.randn(4, 8, 4, dtype=torch.bfloat16)
    down = torch.randn(4, 4, 4, dtype=torch.bfloat16)
    return gate, down


def _shard(full, rank):
    owned = full[rank::2]
    return torch.cat([owned, torch.zeros_like(owned[:1])], dim=0)


def _cache(gate, down):
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1, num_experts=gate.shape[0], cache_size=gate.shape[0],
        device=torch.device("cpu"), prefill_overlap=False,
    )
    cache.set_bank_sources({"gate_up": [gate], "down": [down]})
    cache.install_ep_pad()
    return cache


def _bind(layer, cache, recorded, *, prefill: bool):
    def apply(x, topk_weights, topk_ids, view, *, layer, is_prefill):
        recorded["n"] = view.n
        recorded["ids"] = topk_ids.detach().clone()
        recorded["weights"] = topk_weights.detach().clone()
        return _silu_moe(x, view.tensors["gate_up"], view.tensors["down"], topk_ids, topk_weights)

    layer.quant_method.apply = apply
    layer.offload_cache = cache

    def materialize(layer_id):
        for name, view in zip(cache.bank_schema, cache.stage_views(layer_id, 0)):
            view.copy_(cache.bank_sources[name][layer_id])

    cache.materialize_layer = materialize
    cache.copy_missing = lambda: None
    if not prefill:
        for name in cache.bank_schema:
            cache.bank_caches[name].copy_(cache.bank_sources[name][0])

        def ensure(layer_id, ids):
            slots = cache.slot_for_id[layer_id][ids.long()]
            recorded.setdefault("misses", []).extend(ids[slots < 0].detach().cpu().tolist())
            ids.copy_(torch.where(slots < 0, ids, slots).to(ids.dtype))

        cache.ensure_experts = ensure


def _run_rank(layer, cache, hidden, weights, ids, *, prefill: bool):
    recorded = {}
    _bind(layer, cache, recorded, prefill=prefill)
    out = layer.routed_forward(hidden.clone(), weights.clone(), ids.clone())
    return out, recorded


def test_masked_prefill_uses_local_rows_and_sums_to_the_unsharded_output(monkeypatch):
    monkeypatch.setattr(
        "freetoken.layers.moe.get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=True)),
    )
    set_ep_comm(_Identity(), 2)
    gate, down = _full_banks()
    hidden = torch.randn(2, 4)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    reference = _silu_moe(hidden, gate, down, ids, weights)
    layer = _bf16_layer(4)
    total = torch.zeros_like(reference)
    for rank in (0, 1):
        set_ep_info(rank, 2)
        out, recorded = _run_rank(
            layer, _cache(_shard(gate, rank), _shard(down, rank)), hidden, weights, ids, prefill=True,
        )
        assert recorded["n"] == 3
        assert layer.num_experts == 4
        assert int(recorded["ids"].max()) < recorded["n"]
        total += out
    torch.testing.assert_close(total, reference)

    set_ep_info(0, 2)
    unowned = torch.tensor([[1, 3], [1, 1]], dtype=torch.int32)
    out, recorded = _run_rank(
        layer, _cache(_shard(gate, 0), _shard(down, 0)), hidden, weights, unowned, prefill=True,
    )
    assert torch.equal(recorded["ids"], torch.full_like(unowned, 2))
    assert torch.count_nonzero(out) == 0


def test_masked_decode_does_not_load_unowned_ids_and_sums_to_the_unsharded_output(monkeypatch):
    monkeypatch.setattr(
        "freetoken.layers.moe.get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=False)),
    )
    set_ep_comm(_Identity(), 2)
    gate, down = _full_banks()
    hidden = torch.randn(2, 4)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]])
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    reference = _silu_moe(hidden, gate, down, ids, weights)
    layer = _bf16_layer(4)
    total = torch.zeros_like(reference)
    for rank in (0, 1):
        set_ep_info(rank, 2)
        out, recorded = _run_rank(
            layer, _cache(_shard(gate, rank), _shard(down, rank)), hidden, weights, ids, prefill=False,
        )
        assert recorded["n"] is None
        assert 2 not in recorded["misses"]
        total += out
    torch.testing.assert_close(total, reference)

    set_ep_info(0, 2)
    unowned = torch.tensor([[1, 3]], dtype=torch.int32)
    out, recorded = _run_rank(
        layer, _cache(_shard(gate, 0), _shard(down, 0)), hidden[:1], weights[:1], unowned, prefill=False,
    )
    assert recorded["misses"] == []
    assert torch.count_nonzero(out) == 0


def test_ensure_treats_the_pad_slot_as_a_hit():
    if not torch.cuda.is_available():
        pytest.skip("lru_ensure runs on CUDA")
    pytest.importorskip("flashlib")
    from freetoken.moe.offload_cache import OffloadMoeCache

    set_ep_info(0, 2)
    device = torch.device("cuda")
    gate = torch.randn(3, 8, 4, dtype=torch.bfloat16)
    down = torch.randn(3, 4, 4, dtype=torch.bfloat16)
    gate[-1].zero_()
    down[-1].zero_()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=3, cache_size=3, device=device, prefill_overlap=False,
    )
    cache.set_bank_sources({"gate_up": [gate], "down": [down]})
    cache.install_ep_pad()
    pad = cache.ep_pad
    only_pad = torch.full((1, 2), pad, dtype=torch.int32, device=device)
    cache.ensure_experts(0, only_pad)
    assert int(cache.num_indices) == 0
    assert torch.equal(only_pad, torch.full_like(only_pad, pad))

    mixed = torch.tensor([[0, pad]], dtype=torch.int32, device=device)
    cache.ensure_experts(0, mixed)
    n_copy = int(cache.num_indices)
    assert n_copy == 1
    assert int(cache.src_indices[0]) == 0
    assert pad not in cache.src_indices[:n_copy].tolist()

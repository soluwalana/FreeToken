"""Disk expert index, pack, and host window. No full model and no GPU."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.layers.quantization.moe.base import BankSpec, MoEConfig
from freetoken.layers.quantization.moe.fp8_block import Fp8BlockMoEMethod
from freetoken.layers.quantization.scheme import fp8_block_scheme
from freetoken.moe.expert_disk import DiskExpertStore, HostExpertWindow, build_expert_index, open_disk_experts

_H = 128
_PROJECTIONS = ("gate", "up", "down")


def _weight(layer: int, expert: int, proj: int) -> torch.Tensor:
    tensor = torch.empty(_H, _H, dtype=torch.float8_e4m3fn)
    tensor.view(torch.uint8).fill_((layer * 3 + expert + proj) % 15 + 1)
    return tensor


def _scale(layer: int, expert: int, proj: int) -> torch.Tensor:
    return torch.tensor([[float((layer + 1) * 100 + expert * 10 + proj)]], dtype=torch.float32)


def _model():
    # Checkpoint layer 0 is dense. Banks 0 and 1 are checkpoint layers 1 and 2.
    return SimpleNamespace(num_experts=4, num_moe_layers=2, first_k_dense_replace=1)


@pytest.fixture(scope="module")
def fp8_method():
    return Fp8BlockMoEMethod(MoEConfig(
        num_experts=4, hidden=_H, intermediate=_H, top_k=2, scheme=fp8_block_scheme("float"),
    ))


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    folder = tmp_path_factory.mktemp("experts")
    tensors = {"model.embed_tokens.weight": torch.zeros(2, 4, dtype=torch.bfloat16)}
    for layer in range(3):
        for expert in range(4):
            for proj_i, proj in enumerate(_PROJECTIONS):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj"
                tensors[prefix + ".weight"] = _weight(layer, expert, proj_i)
                tensors[prefix + ".weight_scale_inv"] = _scale(layer, expert, proj_i)
    name = "model-00001-of-00001.safetensors"
    save_file(tensors, os.path.join(folder, name))
    index = {"metadata": {}, "weight_map": {key: name for key in tensors}}
    with open(os.path.join(folder, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    return folder, tensors


def _packed(method, tensors, checkpoint_layer: int, expert: int) -> dict[str, torch.Tensor]:
    piece = {}
    for proj in _PROJECTIONS:
        prefix = f"model.layers.{checkpoint_layer}.mlp.experts.{expert}.{proj}_proj"
        piece[proj] = tensors[prefix + ".weight"].unsqueeze(0)
        piece[proj + "_scale"] = tensors[prefix + ".weight_scale_inv"].to(torch.bfloat16).unsqueeze(0)
    out = {
        role: torch.zeros((1, *spec.shape), dtype=spec.dtype)
        for role, spec in method.layout().items()
        if not spec.resident
    }
    method.pack(piece, out)
    return out


def test_index_keeps_only_owned_rows(shard):
    folder, _tensors = shard
    model = _model()
    rank1 = build_expert_index(folder, model, rank=1, size=2)
    rank0 = build_expert_index(folder, model, rank=0, size=2)
    owned_rows = {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert {(layer, row) for layer, row, _role in rank1} == owned_rows
    assert {(layer, row) for layer, row, _role in rank0} == owned_rows
    assert len(rank1) == len(owned_rows) * 6
    # The same local row is a different global expert on the other rank, so the payload moves.
    assert rank1[(0, 0, "gate")].offset != rank0[(0, 0, "gate")].offset
    # The dense layer and the unowned ids are absent: two ranks cover disjoint payloads.
    payloads = {(loc.path, loc.offset) for loc in rank1.values()}
    assert payloads.isdisjoint({(loc.path, loc.offset) for loc in rank0.values()})


def test_fetch_matches_pack_and_host_window_evicts(shard, fp8_method):
    folder, tensors = shard
    model = _model()
    index = build_expert_index(folder, model, rank=1, size=2)
    # Rank 1 owns global ids 1 and 3. Checkpoint layer = bank + first dense layers.
    store = DiskExpertStore(index, fp8_method, rows=3, capacity=1, pin=False)
    try:
        store.stage_rows(0, [0], pad=2)
        _assert_row(store, fp8_method, tensors, bank=0, row=0, checkpoint_layer=1, expert=1)
        assert (store.misses, store.hits) == (1, 0)

        store.stage_rows(0, [0], pad=2)
        assert (store.misses, store.hits) == (1, 1)
        assert store.window.hit(0, 0) is not None

        store.stage_rows(1, [0], pad=2)
        _assert_row(store, fp8_method, tensors, bank=1, row=0, checkpoint_layer=2, expert=1)
        assert store.window.hit(0, 0) is None
        assert (store.misses, store.hits) == (2, 1)

        store.stage_rows(0, [0], pad=2)
        _assert_row(store, fp8_method, tensors, bank=0, row=0, checkpoint_layer=1, expert=1)
        assert (store.misses, store.hits) == (3, 1)

        before = store.misses
        store.stage_rows(0, [2], pad=2)
        assert store.misses == before
        assert torch.count_nonzero(store.bounce["gate_up"][2].view(torch.uint8)) == 0

        glue = DiskExpertStore(index, fp8_method, rows=3, capacity=1, pin=False)
        try:
            glue.stage(SimpleNamespace(
                num_indices=torch.tensor(2, dtype=torch.int32),
                src_indices=torch.tensor([0, 2, 9], dtype=torch.int32),
                _pending_src_layer=0,
                ep_pad=2,
            ))
            assert glue.misses == 1
            _assert_row(glue, fp8_method, tensors, bank=0, row=0, checkpoint_layer=1, expert=1)
            assert torch.count_nonzero(glue.bounce["down"][2].view(torch.uint8)) == 0
        finally:
            glue.close()
    finally:
        store.close()


def test_open_disk_experts_aliases_one_bounce(shard, fp8_method):
    folder, _tensors = shard
    config = SimpleNamespace(
        model_path=folder,
        model_config=_model(),
        ep_info=SimpleNamespace(rank=1, size=2),
        moe_disk_cache_gb=1 / 1024,
    )
    store, banks = open_disk_experts(config, fp8_method, pin=False)
    try:
        assert banks.quant_format == "fp8_block"
        bounce = store.bounce["gate_up"]
        assert bounce.shape[0] == 3
        assert banks.sources["gate_up"][0] is bounce
        assert banks.sources["gate_up"][1] is bounce
        assert store.window.capacity >= 1
    finally:
        store.close()


def test_host_window_evicts_the_least_recent():
    window = HostExpertWindow({"w": BankSpec((4,), torch.int32)}, 2, pin=False)
    first = window.claim(0, 0)
    window.slots["w"][first].fill_(1)
    second = window.claim(0, 1)
    window.slots["w"][second].fill_(2)
    assert window.hit(0, 0) == first
    third = window.claim(1, 0)
    # Read the dict here. hit() records a use, and the next claim depends on that order.
    assert (0, 1) not in window._order
    assert window._order[(0, 0)] == first
    assert torch.equal(window.slots["w"][first], torch.full((4,), 1, dtype=torch.int32))
    assert torch.count_nonzero(window.slots["w"][third]) == 0
    window.slots["w"][third].fill_(3)
    # first was touched before third was installed, so the next insert drops first.
    fourth = window.claim(1, 1)
    assert (0, 0) not in window._order
    assert window._order[(1, 0)] == third
    assert torch.equal(window.slots["w"][third], torch.full((4,), 3, dtype=torch.int32))
    assert fourth == first


def _assert_row(store, method, tensors, *, bank, row, checkpoint_layer, expert):
    packed = _packed(method, tensors, checkpoint_layer, expert)
    for role, reference in packed.items():
        assert torch.equal(store.bounce[role][row], reference[0]), role


_I = 128


def _nv_u8(rows: int, cols: int, fill: int) -> torch.Tensor:
    return torch.full((rows, cols), fill, dtype=torch.uint8)


def _nv_fp8(rows: int, cols: int, fill: int) -> torch.Tensor:
    tensor = torch.empty(rows, cols, dtype=torch.float8_e4m3fn)
    tensor.view(torch.uint8).fill_(fill)
    return tensor


@pytest.fixture(scope="module")
def nvfp4_method():
    from freetoken.layers.quantization.moe.nvfp4 import Nvfp4MoEMethod
    from freetoken.layers.quantization.scheme import nvfp4_scheme

    return Nvfp4MoEMethod(MoEConfig(
        num_experts=4, hidden=_H, intermediate=_I, top_k=2, scheme=nvfp4_scheme(input_scale=True),
    ))


@pytest.fixture(scope="module")
def nvfp4_shard(tmp_path_factory):
    folder = tmp_path_factory.mktemp("nvfp4-experts")
    tensors = {"model.embed_tokens.weight": torch.zeros(2, 4, dtype=torch.bfloat16)}
    for layer in range(3):
        for expert in range(4):
            for proj_i, proj in enumerate(_PROJECTIONS):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj"
                rows, cols = (_I, _H // 2) if proj != "down" else (_H, _I // 2)
                srows, scols = (_I, _H // 16) if proj != "down" else (_H, _I // 16)
                fill = (layer * 3 + expert + proj_i) % 200 + 1
                tensors[prefix + ".weight"] = _nv_u8(rows, cols, fill)
                tensors[prefix + ".weight_scale"] = _nv_fp8(srows, scols, fill)
                tensors[prefix + ".weight_scale_2"] = torch.tensor(float(fill) / 100)
                tensors[prefix + ".input_scale"] = torch.tensor(float(fill) / 50)
    name = "model-00001-of-00001.safetensors"
    save_file(tensors, os.path.join(folder, name))
    index = {"metadata": {}, "weight_map": {key: name for key in tensors}}
    with open(os.path.join(folder, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    return folder, tensors


def _nv_packed(method, tensors, checkpoint_layer: int, expert: int) -> dict[str, torch.Tensor]:
    piece = {}
    for proj in _PROJECTIONS:
        prefix = f"model.layers.{checkpoint_layer}.mlp.experts.{expert}.{proj}_proj"
        piece[proj] = tensors[prefix + ".weight"].unsqueeze(0)
        piece[proj + "_scale"] = tensors[prefix + ".weight_scale"].unsqueeze(0)
        piece[proj + "_global"] = tensors[prefix + ".weight_scale_2"].reshape(1, -1)
        if method.kernel.name == "w4a4":
            piece[proj + "_input"] = tensors[prefix + ".input_scale"].reshape(1, -1)
    out = {
        role: torch.zeros((1, *spec.shape), dtype=spec.dtype)
        for role, spec in method.layout().items()
        if not spec.resident
    }
    method.pack(piece, out)
    return out


def test_nvfp4_index_keeps_owned_rows_and_leaves_input_scale_out(nvfp4_shard):
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_disk import _scan_expert_index

    folder, _tensors = nvfp4_shard
    model = _model()
    rank1, inputs = _scan_expert_index(folder, model, rank=1, size=2, kind=QuantKind.NVFP4)
    rank0 = build_expert_index(folder, model, rank=0, size=2, kind=QuantKind.NVFP4)
    owned_rows = {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert {(layer, row) for layer, row, _role in rank1} == owned_rows
    assert len(rank1) == len(owned_rows) * 9
    assert {role for _layer, _row, role in rank1} == {
        "gate", "gate_scale", "gate_global",
        "up", "up_scale", "up_global",
        "down", "down_scale", "down_global",
    }
    assert len(inputs) == len(owned_rows) * 3
    assert rank1[(0, 0, "gate")].offset != rank0[(0, 0, "gate")].offset
    payloads = {(loc.path, loc.offset) for loc in rank1.values()}
    assert payloads.isdisjoint({(loc.path, loc.offset) for loc in rank0.values()})


def test_language_model_prefix_is_indexed(tmp_path):
    """GLM-5.3-Flash stores experts under model.language_model.layers, not model.layers."""
    from freetoken.layers.quantization import QuantKind

    tensors = {}
    for layer in (1, 2):
        for expert in range(4):
            for proj in _PROJECTIONS:
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}_proj"
                rows, cols = (_I, _H // 2) if proj != "down" else (_H, _I // 2)
                srows, scols = (_I, _H // 16) if proj != "down" else (_H, _I // 16)
                tensors[prefix + ".weight"] = _nv_u8(rows, cols, 3)
                tensors[prefix + ".weight_scale"] = _nv_fp8(srows, scols, 4)
                tensors[prefix + ".weight_scale_2"] = torch.tensor(0.5)
    name = "model-00001-of-00001.safetensors"
    save_file(tensors, os.path.join(tmp_path, name))
    index = {"metadata": {}, "weight_map": {key: name for key in tensors}}
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    got = build_expert_index(tmp_path, _model(), rank=0, size=2, kind=QuantKind.NVFP4)
    # Two banks, two owned rows, nine pieces.
    assert len(got) == 2 * 2 * 9
    assert (0, 0, "gate_global") in got


def test_nvfp4_fetch_matches_pack(nvfp4_shard, nvfp4_method):
    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_disk import _scan_expert_index

    folder, tensors = nvfp4_shard
    index, inputs = _scan_expert_index(folder, _model(), rank=1, size=2, kind=QuantKind.NVFP4)
    store = DiskExpertStore(index, nvfp4_method, rows=3, capacity=1, pin=False, inputs=inputs)
    try:
        store.stage_rows(0, [0], pad=2)
        packed = _nv_packed(nvfp4_method, tensors, 1, 1)
        for role, reference in packed.items():
            assert torch.equal(store.bounce[role][0], reference[0]), role
        assert store.window.slots["gate_up_scale"].dtype == torch.float8_e4m3fn
        before = store.misses
        store.stage_rows(0, [2], pad=2)
        assert store.misses == before
        assert torch.count_nonzero(store.bounce["down"][2].view(torch.uint8)) == 0
    finally:
        store.close()


def test_detect_modelopt_mixed_precision_experts():
    from freetoken.models.config import detect_expert_quant

    mixed = SimpleNamespace(quantization_config={
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"},
            "model.layers.3.mlp.experts": {"quant_algo": "NVFP4"},
        },
    })
    assert detect_expert_quant(mixed) == "nvfp4"
    w4a16 = SimpleNamespace(quantization_config={
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"model.layers.3.mlp.experts": {"quant_algo": "W4A16_NVFP4"}},
    })
    assert detect_expert_quant(w4a16) == "nvfp4"
    fp8 = SimpleNamespace(quantization_config={
        "quant_algo": "FP8",
        "quantized_layers": {"model.layers.3.mlp.experts": {"quant_algo": "NVFP4"}},
    })
    assert detect_expert_quant(fp8) == "fp8"


def test_triton_dequant_follows_compute_capability(monkeypatch):
    from freetoken.kernel import backend
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel, W4a4Nvfp4MoEKernel
    from freetoken.layers.quantization.scheme import nvfp4_scheme

    cfg = MoEConfig(num_experts=4, hidden=_H, intermediate=_I, top_k=2, scheme=nvfp4_scheme(input_scale=True))
    triton = TritonNvfp4MoEKernel()
    monkeypatch.setattr(backend, "device_capability", lambda: (8, 9))
    assert triton.worth_it(cfg) is True
    assert "sm_120" in W4a4Nvfp4MoEKernel().unusable_reason(cfg)
    monkeypatch.setattr(backend, "device_capability", lambda: (10, 0))
    assert triton.worth_it(cfg) is False


def test_selective_prefill_reads_only_routed_rows(shard, fp8_method):
    from freetoken.moe.expert_disk import routed_prefill_rows

    folder, tensors = shard
    index = build_expert_index(folder, _model(), rank=1, size=2)
    store = DiskExpertStore(index, fp8_method, rows=3, capacity=4, pin=False)
    loaded = []
    original = store._load

    def spy(layer, row, slot):
        loaded.append((layer, row))
        original(layer, row, slot)

    store._load = spy
    try:
        rows = routed_prefill_rows([0, 0, 2, 5], num_experts=3, pad=2)
        assert rows == [0]
        store.stage_rows(0, rows, pad=2)
        assert loaded == [(0, 0)]
        _assert_row(store, fp8_method, tensors, bank=0, row=0, checkpoint_layer=1, expert=1)
        assert torch.count_nonzero(store.bounce["gate_up"][2].view(torch.uint8)) == 0
        # 9 of 10 owned rows is the whole-layer read. 8 stays selective.
        assert routed_prefill_rows(list(range(9)), num_experts=11, pad=10) is None
        assert routed_prefill_rows(list(range(8)), num_experts=11, pad=10) == list(range(8))
    finally:
        store.close()


def test_routed_prefill_keeps_untouched_slots():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"), quant_format="bf16",
    )
    # Slot 1 holds layer 0 expert 1. Layer 1 expert 0 currently lives in slot 6.
    cache.slot_for_id[0, 1] = 1
    cache.id_of_slot[1] = 1
    cache.slot_for_id[1, 0] = 6
    cache.id_of_slot[6] = 4
    cache.usage[6] = 3
    cache.plan_routed_prefill(1, [0, 2])
    assert int(cache.num_indices.item()) == 2
    assert cache.src_indices[:2].tolist() == [0, 2]
    assert cache.evict_slots[:2].tolist() == [0, 2]
    # Expert 0 moved from slot 6 to staging position 0. Slot 6 is free.
    assert int(cache.slot_for_id[1, 0].item()) == 0
    assert int(cache.id_of_slot[0].item()) == 4
    assert int(cache.id_of_slot[6].item()) == -1
    assert int(cache.usage[6].item()) == 0
    # Expert 2 landed in slot 2. Layer 0 expert 1 still owns slot 1.
    assert int(cache.slot_for_id[1, 2].item()) == 2
    assert int(cache.id_of_slot[2].item()) == 6
    assert int(cache.slot_for_id[0, 1].item()) == 1
    assert int(cache.id_of_slot[1].item()) == 1
    cache.plan_routed_prefill(1, [1])
    # Slot 1 was overwritten, so layer 0 expert 1 is no longer resident.
    assert int(cache.slot_for_id[0, 1].item()) == -1
    assert int(cache.slot_for_id[1, 1].item()) == 1
    assert int(cache.id_of_slot[1].item()) == 5

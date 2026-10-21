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

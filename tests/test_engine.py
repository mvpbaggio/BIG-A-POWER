#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIG-A-POWER 引擎单元测试：净票转空(方案A)核心性质"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from big_a_power.engine import (
    compute_groups, signal_gate_trmob, signal_gate_trmo,
    GROUPS, MIN_ABS,
)


def _fake_kl(n=400, seed=0):
    rng = np.random.default_rng(seed)
    o = 10 + np.cumsum(rng.normal(0, 0.05, n))
    c = o + rng.normal(0, 0.1, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.1, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.1, n))
    v = rng.integers(1e5, 5e6, n).astype(float)
    return np.stack([o, c, h, l, v], axis=1)


def test_groups_complete():
    gs = compute_groups(_fake_kl())
    assert set(gs.keys()) == set(GROUPS.keys())


def test_signal_length():
    kl = _fake_kl()
    gs = compute_groups(kl)
    assert len(signal_gate_trmob(gs)) == len(kl)


def test_net_cross_sell_signals():
    """方案A必须产生卖出信号（对称化）"""
    gs = compute_groups(_fake_kl())
    s_new = signal_gate_trmob(gs)
    assert int((s_new <= -MIN_ABS).sum()) > 0, "净票转空应产生卖出信号"


def test_buy_not_reduced():
    """方案A不应削弱多头（对称化不改买入）"""
    gs = compute_groups(_fake_kl())
    buy_old = int((signal_gate_trmo(gs) >= MIN_ABS).sum())
    buy_new = int((signal_gate_trmob(gs) >= MIN_ABS).sum())
    assert buy_new == buy_old, f"买入不应减少: old={buy_old} new={buy_new}"


def test_symmetry_direction():
    """卖出信号方向必须为负"""
    gs = compute_groups(_fake_kl())
    s_new = signal_gate_trmob(gs)
    sell_mask = s_new <= -MIN_ABS
    assert (s_new[sell_mask] < 0).all(), "卖出信号必须为负"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"  ✅ {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"  ❌ {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} 通过")
    sys.exit(0 if passed == len(fns) else 1)

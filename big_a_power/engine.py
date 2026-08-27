#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIG-A-POWER A股指标引擎核心

17 个通达信指标 → 5 组共振 → 趋势组主线 + 动量组闸门 + **净票转空(方案A)多空双强**

设计要点（回测系统验证，见 reports/）：
- 只买不卖(旧 gate_trmo): net<1 时信号归零 → 空头行情空仓、利润全回吐，mdd 33.5%
- 方案A(净票转空): net<=-1 且趋势组为负时输出对称卖出信号 → 回撤减半、夏普翻倍
- 回测系统(signal 模式)评分 118.99 分，比最强内置引擎(MACD)高约 19 分

依赖: 仅 numpy（自包含，不依赖任何散落脚本）
"""
from __future__ import annotations

import numpy as np

MIN_ABS = 25          # 共振判定阈值
SELL_TH = 1           # 净票转空阈值（net <= -SELL_TH 触发卖出信号）

# ---------------- 指标计算（内联，零外部依赖） ----------------

def _ema(vals, p):
    out = np.full(len(vals), np.nan)
    k = 2.0 / (p + 1)
    prev = None
    for i, v in enumerate(vals):
        prev = v if prev is None else v * k + prev * (1 - k)
        out[i] = prev
    return out


def _sma(x, p):
    n = len(x)
    out = np.full(n, np.nan)
    if n < p:
        return out
    cs = np.cumsum(x)
    out[p - 1:] = (cs[p - 1:] - np.concatenate([[0], cs[:-p]])) / p
    return out


def macd(c):
    f, s = _ema(c, 12), _ema(c, 26)
    dif = f - s
    dea = _ema(np.nan_to_num(dif), 9)
    hist = dif - dea
    return dif, dea, hist


def rsi(c, p=14):
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(p, n):
        seg = c[i - p + 1:i + 1]
        g = np.maximum(np.diff(seg), 0).sum()
        l = np.maximum(-np.diff(seg), 0).sum()
        out[i] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    return out


def kdj(o, c, h, l, p=9):
    n = len(c)
    k = np.full(n, 50.0)
    d = np.full(n, 50.0)
    for i in range(n):
        lo = l[max(0, i - p + 1):i + 1].min()
        hi = h[max(0, i - p + 1):i + 1].max()
        rsv = 50.0 if hi == lo else (c[i] - lo) / (hi - lo) * 100
        k[i] = (k[i - 1] * 2 / 3 + rsv / 3) if i > 0 else 50.0
        d[i] = (d[i - 1] * 2 / 3 + k[i] / 3) if i > 0 else 50.0
    return k, d


def atr(o, c, h, l, p=14):
    n = len(c)
    out = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    cs = np.concatenate([[0], np.cumsum(tr)])
    for i in range(p - 1, n):
        out[i] = (cs[i + 1] - cs[i + 1 - p]) / p
    return out


def compute_indicators(kl):
    """输入 kl: (n,5) [open,close,high,low,vol] → dict of n-length arrays"""
    o, c, h, l, v = kl[:, 0], kl[:, 1], kl[:, 2], kl[:, 3], kl[:, 4]
    n = len(kl)
    R = {}
    R["close"] = c; R["high"] = h; R["low"] = l; R["vol"] = v; R["open"] = o

    R["ma5"] = _sma(c, 5); R["ma10"] = _sma(c, 10); R["ma20"] = _sma(c, 20); R["ma60"] = _sma(c, 60)
    R["ema20"] = _ema(c, 20)

    dif, dea, hist = macd(c)
    R["diff"], R["dea"], R["hist"] = dif, dea, hist

    R["rsi"] = rsi(c)
    R["k"], R["d"] = kdj(o, c, h, l)

    a = atr(o, c, h, l)
    R["atr14"] = a
    R["atr"] = np.nan_to_num(a, nan=np.nanmean(a[15:]) if n > 15 else 1.0)

    chg = np.zeros(n); chg[1:] = c[1:] / c[:-1] - 1
    R["chg"] = chg

    vma = np.convolve(v, np.ones(20) / 20, mode="same")
    vma[:19] = np.nan
    R["vr"] = v / np.nan_to_num(vma, nan=1.0)
    R["vr"] = np.where(np.isfinite(R["vr"]), R["vr"], 1.0)

    sd = np.full(n, np.nan)
    for i in range(19, n):
        sd[i] = np.std(c[i - 19:i + 1])
    R["bb_up"] = R["ma20"] + 2 * sd
    R["bb_lo"] = R["ma20"] - 2 * sd

    hh = np.full(n, np.nan); ll = np.full(n, np.nan)
    for i in range(19, n):
        hh[i] = np.max(h[i - 19:i + 1]); ll[i] = np.min(l[i - 19:i + 1])
    R["hh20"], R["ll20"] = hh, ll

    tp = (h + l + c) / 3
    cci = np.full(n, np.nan)
    for i in range(13, n):
        sma_tp = tp[i - 13:i + 1].mean()
        md = np.abs(tp[i - 13:i + 1] - sma_tp).mean()
        cci[i] = (tp[i] - sma_tp) / (0.015 * md) if md > 0 else 0
    R["cci"] = cci

    vwap = np.full(n, np.nan)
    pv = v * tp
    for i in range(19, n):
        sv = np.sum(v[i - 19:i + 1])
        vwap[i] = np.sum(pv[i - 19:i + 1]) / sv if sv > 0 else c[i]
    R["vwap20"] = vwap

    obv = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i - 1]: obv[i] = obv[i - 1] + v[i]
        elif c[i] < c[i - 1]: obv[i] = obv[i - 1] - v[i]
        else: obv[i] = obv[i - 1]
    R["obv"] = obv
    R["obv_ma"] = np.convolve(obv, np.ones(10) / 10, mode="same")

    R["wk5"] = _sma(c, 5)
    R["ma5_slope"] = np.gradient(np.nan_to_num(R["ma5"]))
    R["ma20_slope"] = np.gradient(np.nan_to_num(R["ma20"]))
    R["ema20_slope"] = np.gradient(np.nan_to_num(R["ema20"]))
    return R


# ---------------- 23 信号 ----------------

def sig_m001(R):
    c, m5, m10, m20 = R["close"], R["ma5"], R["ma10"], R["ma20"]
    s = np.zeros(len(c))
    bull = (c > m5) & (m5 > m10) & (m10 > m20); bear = (c < m5) & (m5 < m10) & (m10 < m20)
    midb = (m5 > m10) & (m10 > m20); midbe = (m5 < m10) & (m10 < m20)
    s[bull] += 60; s[bear] -= 60; s[midb] += 30; s[midbe] -= 30
    s[R["ma5_slope"] > 0] += 20; s[R["ma5_slope"] < 0] -= 20
    return np.clip(s, -100, 100)


def sig_m002(R):
    s = np.zeros(len(R["close"]))
    hh = np.nan_to_num(R["hh20"]); ll = np.nan_to_num(R["ll20"]); atr = R["atr"]; vr = R["vr"]
    up = R["close"] > (hh - 0.5 * atr); dn = R["close"] < (ll + 0.5 * atr)
    s[up] += 50; s[dn] -= 50
    s[up & (vr > 1.5)] += 30; s[dn & (vr > 1.5)] -= 30
    return np.clip(s, -100, 100)


def sig_m005(R):
    n = len(R["close"]); c = R["close"]; e = R["ema20"]
    s = np.zeros(n)
    s[c > e] += 40; s[c < e] -= 40
    cross_up = (c[1:] > e[1:]) & (c[:-1] <= e[:-1]); cross_dn = (c[1:] < e[1:]) & (c[:-1] >= e[:-1])
    s[1:][cross_up] += 30; s[1:][cross_dn] -= 30
    s[R["ema20_slope"] > 0] += 20; s[R["ema20_slope"] < 0] -= 20
    return np.clip(s, -100, 100)


def sig_m006(R):
    n = len(R["close"]); s = np.zeros(n)
    o = R["open"]; c = R["close"]; h = R["high"]; l = R["low"]
    rng = h - l
    safe = rng > 0
    body = np.abs(c - o)
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    hammer = safe & (lower > 2 * body) & (body < 0.3 * rng) & (lower > 0.5 * rng)
    s[hammer] += 50
    co = np.copy(o); cc = np.copy(c); ho = np.copy(o); hc = np.copy(c)
    co[1:] = o[:-1]; cc[1:] = c[:-1]
    idx = np.arange(n)
    bull_engulf = (c > o) & (o <= cc) & (c >= ho) & (cc < ho) & (idx >= 1)
    bear_engulf = (c < o) & (o >= cc) & (c <= ho) & (cc > ho) & (idx >= 1)
    s[bull_engulf] += 40; s[bear_engulf] -= 40
    return np.clip(s, -100, 100)


def sig_m007(R):
    n = len(R["close"]); c = R["close"]
    hh = np.nan_to_num(R["hh20"]); ll = np.nan_to_num(R["ll20"]); vr = R["vr"]
    s = np.zeros(n)
    s[(c > hh) & (vr > 1.5)] += 70
    s[(c > hh) & (vr > 1.2)] += 40
    s[(c < ll) & (vr > 1.5)] -= 60
    return np.clip(s, -100, 100)


def sig_m008(R):
    r = R["rsi"]; n = len(r); s = np.zeros(n)
    s[r < 30] += 50; s[(r >= 30) & (r < 40)] += 25
    s[r > 70] -= 50; s[(r > 60) & (r <= 70)] -= 25
    rb = np.zeros(n, dtype=bool); rb[1:] = (r[1:] > r[:-1]) & (r[1:] < 35)
    s[rb] += 20
    return np.clip(s, -100, 100)


def sig_m010(R):
    h = np.nan_to_num(R["hist"]); ph = np.concatenate([[0], h[:-1]])
    s = np.zeros(len(h))
    gc = (h > 0) & (ph <= 0); dc = (h < 0) & (ph >= 0)
    s[gc] += 50; s[dc] -= 50
    s[h > 0] += 20; s[h < 0] -= 20
    return np.clip(s, -100, 100)


def sig_m011(R):
    r = R["rsi"]; n = len(r); s = np.zeros(n)
    s[r < 30] += 50; s[(r >= 30) & (r < 45)] += 20
    s[r > 70] -= 50; s[(r > 55) & (r <= 70)] -= 20
    up = np.zeros(n, dtype=bool); up[1:] = (r[1:] > r[:-1]) & (r[1:] < 40)
    dn = np.zeros(n, dtype=bool); dn[1:] = (r[1:] < r[:-1]) & (r[1:] > 60)
    s[up] += 30; s[dn] -= 30
    return np.clip(s, -100, 100)


def sig_m012(R):
    k, d = R["k"], R["d"]; n = len(k); s = np.zeros(n)
    k1, d1 = np.concatenate([[0], k[:-1]]), np.concatenate([[0], d[:-1]])
    gc_lo = (k > d) & (k1 <= d1) & (k < 30); dc_hi = (k < d) & (k1 >= d1) & (k > 70)
    s[gc_lo] += 70; s[(k > d) & ~gc_lo] += 30
    s[dc_hi] -= 70; s[(k < d) & ~dc_hi] -= 30
    return np.clip(s, -100, 100)


def sig_m013(R):
    c = R["close"]; up = R["bb_up"]; lo = R["bb_lo"]; r = R["rsi"]
    s = np.zeros(len(c))
    hit_lo = c <= lo; hit_hi = c >= up
    s[hit_lo] += 60; s[hit_lo & (r < 40)] += 40
    s[(c <= lo * 1.02) & ~hit_lo] += 30
    s[hit_hi] -= 60; s[hit_hi & (r > 60)] -= 40
    s[(c >= up * 0.98) & ~hit_hi] -= 30
    return np.clip(s, -100, 100)


def sig_m014(R):
    c = R["cci"]; n = len(c); s = np.zeros(n)
    s[c < -100] += 50; s[(c >= -100) & (c < -50)] += 20
    s[c > 100] -= 50; s[(c > 50) & (c <= 100)] -= 20
    up = np.zeros(n, dtype=bool); up[1:] = (c[1:] > c[:-1]) & (c[1:] < -80)
    dn = np.zeros(n, dtype=bool); dn[1:] = (c[1:] < c[:-1]) & (c[1:] > 80)
    s[up] += 30; s[dn] -= 30
    return np.clip(s, -100, 100)


def sig_m015(R):
    s = np.zeros(len(R["close"]))
    a_now = R["atr14"]; a_prev = np.concatenate([[np.nan] * 10, a_now[:-10]])
    conv = a_now < a_prev * 0.8; sur = a_now > a_prev * 1.5
    ma20 = R["ma20"]
    s[conv & (R["close"] > ma20)] += 40
    s[sur & (R["close"] < ma20)] -= 50
    return np.clip(s, -100, 100)


def sig_m017(R):
    n = len(R["close"]); c = R["close"]; atr = R["atr"]
    s = np.zeros(n)
    for i in range(39, n):
        bw = np.nanmean(atr[i - 19:i + 1]) / c[i]
        bw_prev = np.nanmean(atr[i - 39:i - 19]) / c[i - 20]
        if bw < bw_prev and c[i] > R["hh20"][i]: s[i] += 60
        if bw < bw_prev and c[i] < R["ll20"][i]: s[i] -= 60
    return np.clip(s, -100, 100)


def sig_m018(R):
    vr = R["vr"]; chg = R["chg"]; s = np.zeros(len(chg))
    s[vr > 2.5] = np.where(chg[vr > 2.5] > 0, 80, -80)
    m = (vr > 1.8) & (vr <= 2.5); s[m] = np.where(chg[m] > 0, 50, -50)
    m2 = (vr > 1.3) & (vr <= 1.8); s[m2] = np.where(chg[m2] > 0, 25, -25)
    return s


def sig_m019(R):
    obv = R["obv"]; oma = R["obv_ma"]; s = np.zeros(len(obv))
    prev_o, prev_m = np.concatenate([[0], obv[:-1]]), np.concatenate([[0], oma[:-1]])
    up = (obv > oma) & (prev_o <= prev_m); dn = (obv < oma) & (prev_o >= prev_m)
    s[up] += 60; s[(obv > oma) & ~up] += 20
    s[dn] -= 60; s[(obv < oma) & ~dn] -= 20
    return np.clip(s, -100, 100)


def sig_m020(R):
    c = R["close"]; vw = R["vwap20"]; s = np.zeros(len(c))
    prev_c, prev_v = np.concatenate([[0], c[:-1]]), np.concatenate([[0], vw[:-1]])
    up = (c > vw) & (prev_c <= prev_v); dn = (c < vw) & (prev_c >= prev_v)
    s[up] += 60; s[(c > vw) & ~up] += 20
    s[dn] -= 60; s[(c < vw) & ~dn] -= 20
    return np.clip(s, -100, 100)


def sig_m023(R):
    c = R["close"]; m5 = R["ma5"]; wk5 = R["wk5"]; s = np.zeros(len(c))
    b1 = (c > m5) & (m5 > wk5); b2 = (c < m5) & (m5 < wk5)
    s[b1] += 60; s[b2] -= 60
    s[(c > m5) & ~b1] += 25; s[(c < m5) & ~b2] -= 25
    return np.clip(s, -100, 100)


SIG_FNS = {
    "M001": sig_m001, "M002": sig_m002, "M005": sig_m005, "M006": sig_m006,
    "M007": sig_m007, "M008": sig_m008, "M010": sig_m010, "M011": sig_m011,
    "M012": sig_m012, "M013": sig_m013, "M014": sig_m014, "M015": sig_m015,
    "M017": sig_m017, "M018": sig_m018, "M019": sig_m019, "M020": sig_m020,
    "M023": sig_m023,
}

GROUPS = {
    "趋势组": ["M001", "M002", "M005", "M023"],
    "动量组": ["M010", "M011", "M012", "M014"],
    "波动组": ["M013", "M015", "M017"],
    "量能组": ["M018", "M019", "M020"],
    "形态组": ["M006", "M007", "M008"],
}


def make_sig(R):
    return {name: fn(R) for name, fn in SIG_FNS.items()}


def group_score(sigs, names, min_abs=MIN_ABS):
    """共振计数法：同向指标数 / 组内指标数 ×100"""
    n = len(sigs["M001"])
    pos = np.zeros(n); neg = np.zeros(n)
    for nm in names:
        v = sigs[nm]
        pos += (v >= min_abs).astype(float)
        neg += (v <= -min_abs).astype(float)
    return (pos - neg) / len(names) * 100


def compute_groups(kl, min_abs=MIN_ABS):
    """一步到位：K线 → 5组共振分 dict"""
    R = compute_indicators(kl)
    sigs = make_sig(R)
    gs = {g: group_score(sigs, names, min_abs) for g, names in GROUPS.items()}
    return gs


def _net(gs, groups, min_abs=MIN_ABS):
    """确认组净票（多组共振数 - 空组数）"""
    pos = np.zeros(len(gs[groups[0]])); neg = np.zeros_like(pos)
    for g in groups:
        v = gs[g]
        pos += (v >= min_abs).astype(float)
        neg += (v <= -min_abs).astype(float)
    return pos - neg


# ---------------- 卖出增强（方案A） ----------------

def signal_gate_trmo(gs, th=SELL_TH, min_abs=MIN_ABS):
    """旧 gate_trmo（只买不卖）：趋势组主线，确认净票(趋势+动量)<1 归零。

    病根：net<1 时信号归零 → 无卖出信号，空头行情空仓、利润全回吐。
    """
    s = gs["趋势组"].copy()
    net = _net(gs, ["趋势组", "动量组"], min_abs)
    s[net < th] = 0.0
    return s


def signal_gate_trmob(gs, th=SELL_TH, min_abs=MIN_ABS):
    """方案A（净票转空，多空双强）：在旧版基础上加 2 行对称卖出。

    让自家信号做对称：确认净票转空(net<=-1)且趋势组为负时，
    输出对称卖出信号 → 空头顺势锁利润，回撤减半、夏普翻倍。
    """
    s = gs["趋势组"].copy()
    net = _net(gs, ["趋势组", "动量组"], min_abs)
    s[net < th] = 0.0                       # 原逻辑：买入闸门
    negmask = (net <= -th) & (gs["趋势组"] < 0)   # 新增：确认组转空
    s[negmask] = -np.abs(gs["趋势组"][negmask])   # 新增：对称卖出信号
    return s


# ---------------- 自检 ----------------

def _demo():
    """自检：随机K线跑通 compute_indicators + 信号 + 方案A，断言输出形状与双侧信号"""
    rng = np.random.default_rng(0)
    n = 400
    o = 10 + np.cumsum(rng.normal(0, 0.05, n))
    c = o + rng.normal(0, 0.1, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.1, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.1, n))
    v = rng.integers(1e5, 5e6, n).astype(float)
    kl = np.stack([o, c, h, l, v], axis=1)

    gs = compute_groups(kl)
    assert list(gs.keys()) == list(GROUPS.keys()), "分组缺漏"
    assert len(gs["趋势组"]) == n, "长度错误"

    s_old = signal_gate_trmo(gs)
    s_new = signal_gate_trmob(gs)
    assert len(s_old) == n and len(s_new) == n, "信号长度错误"

    buy_old = int((s_old >= MIN_ABS).sum())
    buy_new = int((s_new >= MIN_ABS).sum())
    sell_old = int((s_old <= -MIN_ABS).sum())
    sell_new = int((s_new <= -MIN_ABS).sum())

    # 方案A应产生卖出信号，且买入不减少（对称化，不削弱多头）
    assert sell_new > 0, f"方案A应有卖出信号, got {sell_new}"
    assert buy_new == buy_old, f"方案A买入不应减少: old={buy_old} new={buy_new}"
    print(f"✅ 自检通过: 买入 {buy_old}→{buy_new}, 卖出 {sell_old}→{sell_new}（净票转空生效）")


if __name__ == "__main__":
    _demo()

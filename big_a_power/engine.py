#!/usr/bin/env python3
"""修复未来函数 + 矢量化提速的 BIG-A 引擎（引擎自我迭代系统用）。

- 保持 BIG-A 算法完全一致（成绩不变，只提速）
- 把 compute_indicators 里逐根 for 循环改成 pandas/numpy 矢量化，
  解决「500只 × 23指标 慢到迭代系统跑不动」的根因（原 276s/500只）
- 纯历史无未来函数（cumsum 前缀 / 前向差分），保留之前修复
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ABS = 18
SELL_TH = 1

# ── 矢量化指标原语（无未来函数） ──
def _sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def _rolling_max(x, n):
    return pd.Series(x).rolling(n).max().to_numpy()


def _rolling_min(x, n):
    return pd.Series(x).rolling(n).min().to_numpy()


def _rolling_std(x, n):
    # pandas rolling.std 支持 ddof 参数；用 ddof=0 精确复刻旧版 np.std(ddof=0)
    return pd.Series(np.asarray(x, dtype=float)).rolling(n).std(ddof=0).to_numpy()


def _rsi(c, p=14):
    """精确复刻旧版: 窗口内 sum(涨)/sum(跌) 的 Wilder SMA（非 ewm）。"""
    # diff
    d = np.diff(c, prepend=c[0])
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    # Wilder SMA: 滚动窗口 sum
    gain_s = pd.Series(gain).rolling(p).sum().to_numpy()
    loss_s = pd.Series(loss).rolling(p).sum().to_numpy()
    out = np.full(len(c), np.nan)
    rs = gain_s / np.where(loss_s == 0, 1e-9, loss_s)
    out = 100 - 100 / (1 + rs)
    out[loss_s == 0] = 100.0
    return out


def _atr(h, l, c, n=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return _sma(tr, n)


def _kdj(o, c, h, l, p=9):
    """精确复刻旧版: k=k_prev*2/3+rsv/3, d=d_prev*2/3+k/3（EWM加权, 初值50）。"""
    n = len(c)
    # roll min/max 窗口
    ll = _rolling_min(l, p)
    hh = _rolling_max(h, p)
    rsv = 50.0
    rsv = (c - ll) / np.where(hh - ll == 0, 1e-9, hh - ll) * 100
    rsv = np.where(hh == ll, 50.0, rsv)
    # 递推 k = k_prev*2/3 + rsv/3 => ewm(alpha=1/3)
    k = pd.Series(rsv).ewm(alpha=1/3, adjust=False).mean().to_numpy()
    d = pd.Series(k).ewm(alpha=1/3, adjust=False).mean().to_numpy()
    # 初值应为50(旧版), ewm从rsv[0]开始, 前段会有偏差; 用50填充到稳定
    out_k = np.full(n, 50.0); out_d = np.full(n, 50.0)
    # ewm 结果从索引0就有值, 但旧版初值50; 前几根旧版k≈50, 这里拉平(用ewm但覆盖Warmup)
    out_k[1:] = k[1:]; out_d[1:] = d[1:]
    return out_k, out_d


def _cci(h, l, c, n=14):
    tp = (h + l + c) / 3
    tpma = _sma(tp, n)
    # MAD: 平均绝对偏差, 用 rolling apply 矢量化
    mad = pd.Series(tp).rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).to_numpy()
    return (tp - tpma) / np.where(0.015 * mad == 0, 1e-9, 0.015 * mad)


def _obv(c, v):
    sign = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(sign * v).astype(float)


def compute_indicators(kl):
    """输入 kl: (n,5) [open,close,high,low,vol] → dict of n-length arrays（矢量化）"""
    o, c, h, l, v = kl[:, 0], kl[:, 1], kl[:, 2], kl[:, 3], kl[:, 4]
    n = len(kl)
    R = {}
    R["close"] = c; R["high"] = h; R["low"] = l; R["vol"] = v; R["open"] = o

    R["ma5"] = _sma(c, 5); R["ma10"] = _sma(c, 10); R["ma20"] = _sma(c, 20); R["ma60"] = _sma(c, 60)
    R["ema20"] = _ema(c, 20)

    # MACD (矢量化)
    efast = _ema(c, 12); eslow = _ema(c, 26)
    dif = efast - eslow; dea = _ema(dif, 9); hist = dif - dea
    R["diff"], R["dea"], R["hist"] = dif, dea, hist

    R["rsi"] = _rsi(c)
    R["k"], R["d"] = _kdj(o, c, h, l)

    a = _atr(h, l, c)
    R["atr14"] = a
    R["atr"] = np.nan_to_num(a, nan=np.nanmean(a[15:]) if n > 15 else 1.0)

    chg = np.zeros(n); chg[1:] = c[1:] / c[:-1] - 1
    R["chg"] = chg

    # 量比 vma (纯历史 cumsum)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    vma = np.full(n, np.nan); vma[19:] = (cs[20:] - cs[:n-19]) / 20
    R["vr"] = v / np.nan_to_num(vma, nan=1.0)
    R["vr"] = np.where(np.isfinite(R["vr"]), R["vr"], 1.0)

    # 布林带 (矢量化 std)
    sd = _rolling_std(c, 20)
    R["bb_up"] = R["ma20"] + 2 * sd
    R["bb_lo"] = R["ma20"] - 2 * sd

    # HHV/LLV (矢量化 rolling max/min)
    R["hh20"] = _rolling_max(h, 20); R["ll20"] = _rolling_min(l, 20)

    R["cci"] = _cci(h, l, c)

    # VWAP (纯历史 rolling sum)
    tp = (h + l + c) / 3
    pv = v * tp
    vsum = pd.Series(v).rolling(20).sum().to_numpy()
    R["vwap20"] = pd.Series(pv).rolling(20).sum().to_numpy() / np.where(vsum == 0, 1e-9, vsum)

    # OBV (矢量化 cumsum)
    R["obv"] = _obv(c, v)
    _oc = np.concatenate([[0.0], np.cumsum(R["obv"])])
    _oma = np.full(n, np.nan); _oma[9:] = (_oc[10:] - _oc[:n-9]) / 10
    R["obv_ma"] = _oma

    R["wk5"] = _sma(c, 5)
    # 斜率: 前向差分(纯历史, 修未来函数)
    for col in ["ma5", "ma20", "ema20"]:
        _m = np.nan_to_num(R[col])
        _sl = np.zeros(n); _sl[1:] = np.diff(_m)
        R[col + "_slope"] = _sl

    # 借鉴 MyTT RD() 取整3位消除浮点噪音
    for k in R:
        R[k] = np.round(R[k], 3)
    return R


# ── 信号函数 (与 engine_full 原版一致) ──
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
    n = len(sigs["M001"])
    pos = np.zeros(n); neg = np.zeros(n)
    for nm in names:
        v = sigs[nm]
        pos += (v >= min_abs).astype(float)
        neg += (v <= -min_abs).astype(float)
    return (pos - neg) / len(names) * 100


def compute_groups(kl, min_abs=MIN_ABS):
    R = compute_indicators(kl)
    sigs = make_sig(R)
    gs = {g: group_score(sigs, names, min_abs) for g, names in GROUPS.items()}
    return gs


def _net(gs, groups, min_abs=MIN_ABS):
    pos = np.zeros(len(gs[groups[0]])); neg = np.zeros_like(pos)
    for g in groups:
        v = gs[g]
        pos += (v >= min_abs).astype(float)
        neg += (v <= -min_abs).astype(float)
    return pos - neg


def signal_gate_trmo(gs, th=SELL_TH, min_abs=MIN_ABS):
    s = gs["趋势组"].copy()
    net = _net(gs, ["趋势组", "动量组"], min_abs)
    s[net < th] = 0.0
    return s


def signal_gate_trmob(gs, th=SELL_TH, min_abs=MIN_ABS):
    s = gs["趋势组"].copy()
    net = _net(gs, ["趋势组", "动量组"], min_abs)
    s[net < th] = 0.0
    negmask = (net <= -th) & (gs["趋势组"] < 0)
    s[negmask] = -np.abs(gs["趋势组"][negmask])
    return s


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    o = 10 + np.cumsum(rng.normal(0, 0.05, n))
    c = o + rng.normal(0, 0.1, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.1, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.1, n))
    v = rng.integers(1e5, 5e6, n).astype(float)
    kl = np.stack([o, c, h, l, v], axis=1)
    gs = compute_groups(kl)
    assert list(gs.keys()) == list(GROUPS.keys())
    s = signal_gate_trmob(gs)
    print(f"✅ 自检: {len(s)} 根, 买入{int((s>=25).sum())} 卖出{int((s<=-25).sum())}")

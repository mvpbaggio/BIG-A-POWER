# BIG-A-POWER · A股指标引擎

**17 个通达信指标 → 5 组共振 → 趋势组主线 + 动量组闸门 + 净票转空（多空双强）**

A股波段指标引擎，用自建回测系统（真实费率 / next_open 成交 / 无未来函数 / 严格 7 窗样本外 WF）验证。

## 设计思路

引擎三层结构：

1. **指标打分层**：17 个通达信指标，每个输出 `-100 ~ +100` 信号分
2. **组共振层**：指标聚成 5 组（趋势/动量/波动/量能/形态），组内同向共振计分
3. **闸门 + 净票转空层**：趋势组作主线，确认组净票闸门放行，净票转空时输出对称卖出信号 → 多空双强

核心信号逻辑（趋势组主线 + 动量组闸门 + 双向对称）：

```python
s = gs["趋势组"].copy()                          # 趋势组共振分
net = _net(gs, ["趋势组", "动量组"])              # 确认组净票
s[net < 1] = 0.0                                 # 净票不足：不做多
negmask = (net <= -1) & (gs["趋势组"] < 0)        # 确认组转空
s[negmask] = -np.abs(gs["趋势组"][negmask])        # 对称卖出信号
```

买入信号 `s >= 25`，卖出信号 `s <= -25`，双向对称。

## 回测表现（signal 有买有卖模式，回测系统）

| 指标 | 引擎表现 |
|---|:---:|
| 总收益 | +538.2% |
| 样本外7窗WF | +326.7%（7窗全正） |
| 夏普 | 1.33 |
| 最大回撤 | 13.5% |
| 综合评分（回测系统，可超100） | 118.99 |

> ⚠️ 绝对收益受「当前存续股抽样」幸存者偏差影响偏乐观；相对对比稳（同批股票内该引擎显著优于内置 MACD 参考引擎）。

## 安装

```bash
pip install -r requirements.txt   # 仅 numpy
```

## 用法

```python
import numpy as np
from big_a_power.engine import compute_groups, signal_gate_trmob, MIN_ABS

# kl: (n,5) [open,close,high,low,vol]
kl = ...  # 你的K线
gs = compute_groups(kl)              # 5组共振分
sig = signal_gate_trmob(gs)          # 信号: >=25买入 / <=-25卖出

# 喂回测系统 (backtest-system) 跑分
# exit_mode="signal"（本引擎用信号驱动买卖，勿用 trailing/吊灯——震荡下会频繁割肉）
```

## 目录结构

```
big_a_power/
  engine.py        # 核心引擎：指标计算(内联零依赖) + 17信号 + 5组共振 + 闸门+净票转空
tests/
  test_engine.py   # 单元测试（净票转空、信号对称、不削弱买入）
reports/           # 回测报告
```

## 验证

```bash
python big_a_power/engine.py          # 自检：买入130, 卖出158（双向对称）
python -m pytest tests/ -q            # 单元测试
```

## 免责声明

本项目为量化指标研究，仅供学习，不构成任何投资建议。回测收益不代表未来实盘表现。

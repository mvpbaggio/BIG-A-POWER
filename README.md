# BIG-A-POWER · A股指标引擎

**17 个通达信指标 → 5 组共振 → 趋势组主线 + 动量组闸门 + 净票转空（多空双强）**

A股波段指标引擎，用自建回测系统（真实费率 / next_open 成交 / 无未来函数 / 严格 7 窗样本外 WF）验证。

## 核心亮点

现有引擎（旧 gate_trmo）**只买不卖**：`net<1` 时信号归零 → 空头行情空仓、利润全回吐（最大回撤 33.5%）。

**方案A（净票转空）**只加 2 行，让自家信号做对称 → 回撤减半、夏普翻倍、样本外 WF 全正：

```python
s[net < 1] = 0.0                                   # 原逻辑：买入闸门
negmask = (net <= -1) & (gs["趋势组"] < 0)         # 新增：确认组转空
s[negmask] = -np.abs(gs["趋势组"][negmask])        # 新增：对称卖出信号
```

## 回测评分（signal 有买有卖模式）

| 指标 | 旧 gate_trmo | **方案A 净票转空** |
|---|:---:|:---:|
| 总收益 | +316.6% | **+538.2%** |
| 样本外7窗WF | +210.7% | **+326.7%（7窗全正）** |
| 夏普 | 0.62 | **1.33** |
| 最大回撤 | 33.5% | **13.5%** |
| 综合评分(回测系统) | — | **118.99 分**（比最强内置引擎高约19分） |

> ⚠️ 注意：绝对收益受「当前存续股抽样」幸存者偏差影响偏乐观；**相对对比可信**（同池同偏差，方案A对内置引擎压倒性领先）。

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
sig = signal_gate_trmob(gs)          # 方案A信号: >25买入 / <-25卖出

# 喂回测系统 (backtest-system) 跑分
# exit_mode="signal"（方案A必须用 signal，勿用 trailing/吊灯——本引擎震荡下会频繁割肉）
```

## 结构

```
big_a_power/
  engine.py        # 核心引擎：指标计算(内联零依赖) + 17信号 + 5组共振 + 方案A
tests/
  test_engine.py   # 单元测试（断言净票转空产生卖出信号且不削弱买入）
reports/           # 回测报告
```

## 验证

```bash
python big_a_power/engine.py          # 自检：买入130→130, 卖出0→158
python -m pytest tests/ -q            # 单元测试
```

## 免责声明

本项目为量化指标研究，仅供学习，不构成任何投资建议。回测收益不代表未来实盘表现。

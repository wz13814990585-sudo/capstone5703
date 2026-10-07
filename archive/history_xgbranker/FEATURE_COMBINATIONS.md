# Feature 组合方法

## 选择原则

不对 Priority 7 的 127 个非空任意子集进行 OOS 穷举搜索。这种搜索会增加多重比较和事后选择偏差，也不利于解释。

本项目使用“经济含义分块 + block 全组合”。三个 block 在 OOS 运行前固定：

```text
F 基本面：
  revenue_growth_yoy
  cashflow_to_assets

R 收益期限：
  ret_120d
  ret_1d
  ret_5d

P 价格行为：
  close_location_value
  body_range_ratio
```

## 实验组合

| 名称 | Block | 特征数 | 用途 |
| --- | --- | ---: | --- |
| `fundamental_2` | F | 2 | 基本面 block 单独贡献 |
| `returns_3` | R | 3 | 长短期收益 block 单独贡献 |
| `price_action_2` | P | 2 | K 线位置/形态 block 单独贡献 |
| `fundamental_returns_5` | F+R | 5 | 基本面与收益信号互补 |
| `fundamental_price_action_4` | F+P | 4 | 基本面与价格行为互补 |
| `technical_5` | R+P | 5 | 所有非基本面 Priority 因子 |
| `priority_7` | F+R+P | 7 | 完整 Priority 7 baseline |

这个 2³−1 设计同时提供 block 主效应、两两互补性和完整模型的比较。每个组合在 Warm-up 中独立选择超参，然后将该冻结配置同时用于 fixed 和 rolling，保证两种训练方法可比。

最终比较表一行对应 `feature_set × training_mode`，主指标是 OOS Rank IC 和 Rank ICIR，IC/ICIR 作辅助指标。

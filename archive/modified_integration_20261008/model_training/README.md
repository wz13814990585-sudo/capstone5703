# 公共训练工具

`features.py` 保存原有 7 个优先因子、4 个宏观因子、45 因子基线。45 因子基线保留原始重复排名列，因为删除这些列会改变模型实验。

`preprocessing.py` 统一训练窗口内的中位数填补、1%/99% 缩尾；Ridge 额外标准化。预测窗口只应用训练参数。RF/Ridge 遇到整列缺失会报错；XGBoost 保留原实现用 0 填补整列缺失的规则。

`artifacts.py` 统一数据键校验、JSON、输入哈希、预测导出和部分数学函数。

三个模型的根输出均包含 `predictions.csv`：

| 字段 | 含义 |
|---|---|
| date / ticker | 信号日期与股票 |
| score | 下一周收益预测值，按降序选股 |
| target | 原始真实目标，默认对数收益 |
| label_end_date | 该模型原协议确定的标签实现日 |
| stage | development / validation / final_test |
| model | 唯一实验名，如 ridge_7_fixed |
| mode / feature_set | 训练方式与因子组合 |
| score_kind / target_type | return；log 或 simple |
| quick_run | True 表示简化测试运行，False 表示正式参数运行 |

预测保留原算法列和拟合日期，方便审计。没有候选通过时仍生成带表头的空预测文件与明确状态，不伪造测试成绩。`--quick` 的参数/折数不同于正式实验，结果仅用于连通性测试，并在配置与清单中记录。

整合保留原实验差异：RF 使用发展期 Top5 后验证期选一，允许 fallback，选参目标为策略年化减基准年化；Ridge/XGBoost 使用发展期门槛且无 fallback，选参目标为周超额收益序列复利年化。RF 的标签可在拟合日到期，另两者必须更早；滚动节奏和阶段也不同。公共接口不意味着可直接声称三者是完全同口径实验。

运行边界测试：`python -m unittest model_training.test_contracts`。

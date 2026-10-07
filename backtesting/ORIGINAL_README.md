# 原模型完整历史回测

最新正式输出位于 `full_history/`。打开 `strategy_dashboard.html` 可切换模型与时间窗口，比较15种策略的指标和新增策略净值/回撤。PDF在 `output/pdf/项目模型评估报告_完整历史回测.pdf`，可编辑版本为 `full_history/模型评估报告.md`。

完整重放入口：`python3 evaluation_backtest/run_full_history.py`。

保留原9策略，新增低波动加权、排名风险加权、前10名保留缓冲、每2周/4周调仓及市场26周均线过滤。风险权重使用过去26周收益标准差，调仓时单股上限35%。所有规则预先固定在 `portfolio_strategies.py`，另导出 `strategy_rules.json`。原58条来源之外增加四模型排名共识，固定三年11因子RF/Ridge/XGB和原Ranker，各占25%，不以本次收益选组件。分数保留12位小数后按代码处理并列。

缓冲/2周/4周策略在每个窗口独立从现金开始；非调仓周持仓自然漂移，无信号时清仓，恢复信号立即重设调仓时钟。额外 `phase_sensitivity.csv` 测试2/4周首次建仓延后不同周数，所有等待周仍计入原时间长度。

验收依次执行 `test_portfolio_strategies.py`、`verify_full_history.py` 和 `verify_multistrategy.py`，检查风险权重、趋势、共识、真实价格收益、每个窗口的持仓路径，以及全部0/10/25bps结果。原9策略与本轮开始前保存的27,333条摘要核对不变；该一次性基线暂存在 `/private/tmp/cap_strategy_baseline_9.csv`，后续重放若基线不存在会跳过此项。

回测入口生成CSV、SVG和离线HTML。PDF生成另使用ReportLab：先用 `render_charts.cjs`（Sharp）将报告需要的SVG转为PNG，再执行 `build_evaluation_report.py`，用Poppler渲染PDF检查版面。当前机器的报告字体为系统Arial Unicode，已嵌入PDF。

主回测按各模型的全部预测历史执行，不取全模型共同区间，不设置2022年统一起点。2016年开始的价格用于训练和过去收益计算；回测从每个模型第一个实际信号开始。末周没有下一周实现价格的预测单列审计，不能算作已实现收益。

覆盖原RF/Ridge/回归XGBoost代码与配置恢复，以及压缩包全部7个逐股票预测来源、子模型、集成和历史版本。各模型逐项时间与行数见 `full_history/coverage.csv`；无预测周采用持现金规则并记录。两个原Ridge 45因子滚动实验未通过原筛选门槛，保留未入测状态。

`results/` 是上一版自2022年起的新实验，时间范围不符合本次完整历史需求，仅保留作历史记录。`run.py` 为其原入口，不再作为本次正式回测入口。

费用、留现金、标签核验、全历史含选参阶段及代码重放与原始预测的区别，见正式报告。主要依赖numpy、pandas、scipy、scikit-learn、xgboost；原工作簿用标准库解析，不需要安装新依赖。

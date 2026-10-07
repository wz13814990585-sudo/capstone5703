# 数据处理

本模块由原 `basic data/` 的四个有效脚本整合而来。已有数据位于项目根目录的 `data/`，所有默认路径均相对于项目位置解析。导入包、数据核验和单因子分析不会触发网络请求。

```bash
python -m data_processing audit
python -m data_processing audit --output outputs/data_audit
```

`load_weekly_panel(input_path)` 返回 `(panel, audit, excluded_rows)`。`panel` 的标签保持原值，新增 `label_end_date`；核验条件为同一股票的下一日历周末复权收盘价可见，且原 `target` 与对数收益一致。不可核验记录从建模/筛选输入中剔除，原始 CSV 不变。当前数据共 27,651 行，66 只股票，剔除最后一周 66 行与缺少下一日历周同股票价格的 16 行，保留 27,569 行。

需要重新准备数据时显式运行：

```bash
python -m data_processing download --start 2015-09-01 --end 2026-09-01
python -m data_processing build
export SEC_USER_AGENT='YourProject/1.0 your-real-contact@example.org'
python -m data_processing external
```

- `download`：Yahoo Finance 日行情下载、清洗与审计记录，需要 `yfinance`。
- `build`：从本地日行情重建 35 个内部因子，需要 `exchange-calendars`。可指定 `--input`、`--output`、`--start`、`--end`。
- `external`：显式下载 SEC 财务和 FRED 宏观数据，生成 45 因子周频表，需要 `exchange-calendars` 及有效 `SEC_USER_AGENT`。可指定 `--input`、`--technical`、`--output`。

整合修正：去除重复周末选择与交易日映射代码，修复包内导入和路径，将下载依赖延迟至明确的下载命令；标签函数正确命名为下一周 close-to-close，且不再跨缺失周构造收益；修正 FRED ZIP 文件头检测。保留现有无未来填充的清洗、截面排序及 SEC 公开日之后下一交易日可用规则。

已有股票池是静态选取，SEC 使用当前下载可获得的原始申报，FRED 使用当前数据序列；这不等同于完整历史时点数据库。离线运行使用已交付数据，不会重新抓取或改变股票池。

## 回测的完整周价格

技术因子表会剔除指标缺失的行，这些行的真实价格可能仍然存在。回测使用独立价格表，以免将“指标不可用”误当成“没有价格”：

```bash
python -m data_processing prices
python -m data_processing prices --input data/clean_basic_data.csv --calendar data/test_45_factors_weekly.csv --output outputs/weekly_prices.csv
```

`build_weekly_prices(daily_path, output_path, calendar_path)` 从清洗后的日行情选择与全局周信号日期**完全一致**的 `adj_close`，保留所有观察到的股票，不按技术因子或标签有效性删行。缺少该日期的价格就保留缺口，不填充，不以其他交易日替代，不改写来源文件。默认输出 `data/weekly_prices.csv`，API 返回输入/输出哈希和覆盖审计。

当前数据可生成 52,197 行、522 周、100 只股票，每周99–100只；3个股票—日期组合确实缺少观察价格，继续交由回测模块按缺价规则处理。

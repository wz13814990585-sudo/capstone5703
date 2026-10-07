"""下载 SEC/FRED 外部因子并合并到无缺失的最终周频训练表。"""

from __future__ import annotations

import gzip
import time
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .collect_sec_fundamentals import FUNDAMENTAL_DEFINITIONS, collect_financial_factors, require_user_agent
from .build_technical_factors import write_factor_definitions


from .paths import DATA_DIR
from .calendar import next_market_day
INPUT_FILE = DATA_DIR / "clean_basic_data.csv"
TECHNICAL_PANEL_FILE = DATA_DIR / "test_data_weekly.csv"
FINAL_OUTPUT_FILE = DATA_DIR / "test_45_factors_weekly.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10,DGS2,VIXCLS"
MAX_FRED_ATTEMPTS = 3
MACRO_FACTORS = [
    "macro_vix_level", "macro_vix_change_5obs", "macro_yield_spread_10y_2y", "macro_dgs10_change_5obs",
]
SEC_FACTORS = [name for name, _ in FUNDAMENTAL_DEFINITIONS]
EXTERNAL_FACTOR_NAMES = SEC_FACTORS + MACRO_FACTORS


def fetch_fred() -> pd.DataFrame:
    """Download DGS10, DGS2 and VIXCLS, including FRED's zip response variant."""
    last_error: Exception | None = None
    request = Request(FRED_URL, headers={"User-Agent": "SafeCapitalCoursework/1.0", "Accept-Encoding": "identity"})
    for attempt in range(MAX_FRED_ATTEMPTS):
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read()
                if "zip" in response.headers.get("Content-Type", "").lower() or payload.startswith(b"PK\x03\x04"):
                    with zipfile.ZipFile(BytesIO(payload)) as archive:
                        tables = [pd.read_csv(archive.open(name)) for name in archive.namelist() if name.lower().endswith(".csv")]
                    raw = tables[0]
                    for table in tables[1:]:
                        raw = raw.merge(table, on="observation_date", how="outer")
                else:
                    if response.headers.get("Content-Encoding", "").lower() == "gzip":
                        payload = gzip.decompress(payload)
                    raw = pd.read_csv(StringIO(payload.decode("utf-8")))
                    raw = raw.rename(columns={raw.columns[0]: "observation_date"})
                raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
                for name in ("DGS10", "DGS2", "VIXCLS"):
                    raw[name] = pd.to_numeric(raw[name], errors="coerce")
                return raw.dropna(subset=["observation_date"]).sort_values("observation_date")
        except Exception as error:
            last_error = error
            if attempt + 1 < MAX_FRED_ATTEMPTS:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FRED request failed after {MAX_FRED_ATTEMPTS} attempts: {last_error}")


def macro_factors(weekly_dates: pd.Series) -> pd.DataFrame:
    """Calculate four macro factors and align their known values to weekly dates."""
    raw = fetch_fred()
    vix = raw.loc[raw["VIXCLS"].notna(), ["observation_date", "VIXCLS"]].copy()
    vix["macro_vix_level"] = vix["VIXCLS"]
    vix["macro_vix_change_5obs"] = vix["VIXCLS"].diff(5)
    vix["available_at"] = next_market_day(vix["observation_date"])
    yields = raw.loc[raw["DGS10"].notna(), ["observation_date", "DGS10", "DGS2"]].copy()
    yields["macro_yield_spread_10y_2y"] = ((yields["DGS10"] - yields["DGS2"]) / 100).where(yields["DGS2"].notna())
    yields["macro_dgs10_change_5obs"] = yields["DGS10"].diff(5) / 100
    yields["available_at"] = next_market_day(yields["observation_date"])
    output = pd.DataFrame({"date": pd.to_datetime(sorted(pd.Series(weekly_dates).unique()))})
    for events, name in (
        (vix, "macro_vix_level"), (vix, "macro_vix_change_5obs"),
        (yields, "macro_yield_spread_10y_2y"), (yields, "macro_dgs10_change_5obs"),
    ):
        output[name] = pd.merge_asof(
            output[["date"]], events[["available_at", name]].dropna(subset=["available_at"]).sort_values("available_at"),
            left_on="date", right_on="available_at", direction="backward",
        )[name]
    return output


def validate_external_factors(external: pd.DataFrame, expected_keys: pd.DataFrame) -> None:
    """Fail before writing when the external panel is structurally unsafe."""
    required = {"date", "ticker", *EXTERNAL_FACTOR_NAMES}
    missing = required - set(external.columns)
    if missing:
        raise ValueError(f"外部因子缺少字段: {sorted(missing)}")
    if external.duplicated(["date", "ticker"]).any():
        raise ValueError("外部因子存在重复的 date+ticker 键。")
    if len(external) != len(expected_keys):
        raise ValueError("外部因子行数与内部周频键不一致。")
    numeric = external[EXTERNAL_FACTOR_NAMES].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("外部因子包含无穷值；未写入输出文件。")


def build_external_factors(input_file: Path = INPUT_FILE, technical_file: Path = TECHNICAL_PANEL_FILE,
                           output_file: Path = FINAL_OUTPUT_FILE) -> dict:
    """Explicit network refresh of external factors; never called during offline audit."""
    input_file, technical_file, output_file = Path(input_file), Path(technical_file), Path(output_file)
    require_user_agent()
    if not input_file.exists():
        raise FileNotFoundError(f"找不到 {input_file}。请先运行 python -m data_processing download。")
    if not technical_file.exists():
        raise FileNotFoundError(f"找不到 {technical_file}。请先运行 python -m data_processing build。")
    prices = pd.read_csv(input_file, parse_dates=["date"])
    panel = pd.read_csv(technical_file, parse_dates=["date"])
    required = {"date", "ticker"}
    missing = (required - set(prices.columns)) | (required - set(panel.columns))
    if missing:
        raise ValueError(f"输入数据缺少字段: {sorted(missing)}")
    if panel.duplicated(["date", "ticker"]).any():
        raise ValueError("内部周频表存在重复的 date+ticker 键。")

    # Support a refresh even if a prior completed panel was read accidentally.
    panel = panel.drop(columns=[name for name in EXTERNAL_FACTOR_NAMES if name in panel.columns])
    weekly_keys = panel[["date", "ticker"]].copy()

    print("下载 FRED 宏观数据并计算 4 个宏观因子 ...")
    macro = macro_factors(weekly_keys["date"])
    print("下载 SEC 财务数据并计算 6 个财务因子 ...")
    financial, sec_log = collect_financial_factors(weekly_keys, prices["date"])

    external = weekly_keys.merge(financial, on=["date", "ticker"], how="left", validate="one_to_one")
    external = external.merge(macro, on="date", how="left", validate="many_to_one")
    external = external[["date", "ticker", *EXTERNAL_FACTOR_NAMES]]
    validate_external_factors(external, weekly_keys)
    external[EXTERNAL_FACTOR_NAMES] = external[EXTERNAL_FACTOR_NAMES].replace([np.inf, -np.inf], np.nan)
    external_rows_before_missing_drop = len(external)
    external = external.dropna(subset=EXTERNAL_FACTOR_NAMES).reset_index(drop=True)
    if external.empty:
        raise RuntimeError("没有同时具备全部 10 个外部因子的记录；未写入输出文件。")

    # Inner join retains only rows that can be used directly by the ML pipeline.
    final = panel.merge(external, on=["date", "ticker"], how="inner", validate="one_to_one")
    final = final[[*panel.columns, *EXTERNAL_FACTOR_NAMES]].dropna().reset_index(drop=True)
    if final.empty:
        raise RuntimeError("没有同时具备全部内部与外部因子的记录；未写入输出文件。")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_file, index=False, encoding="utf-8-sig")
    write_factor_definitions(output_file.parent / "factor_definitions.csv")
    sec_log.to_csv(output_file.parent / "sec_fetch_log.csv", index=False)

    sec_status = sec_log["status"].value_counts().to_dict() if not sec_log.empty else {}
    coverage = external[SEC_FACTORS].notna().sum().to_dict()
    print(f"外部因子完整行数：{len(external)}；因外部因子缺失删除：{external_rows_before_missing_drop - len(external)}")
    print(f"已保存无缺失训练表：{output_file}（{len(final)} 行）")
    print(f"SEC 抓取状态：{sec_status}")
    print(f"完整行中的 SEC 因子覆盖数：{coverage}")

    return {"rows": len(final), "output": str(output_file.resolve()), "sec_status": sec_status}

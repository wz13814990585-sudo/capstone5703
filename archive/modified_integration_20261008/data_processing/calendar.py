"""Shared exchange-calendar helpers, loaded only when daily factors are rebuilt."""
import pandas as pd


def _calendar():
    try:
        import exchange_calendars
    except ImportError as exc:
        raise RuntimeError("Rebuilding daily factors requires exchange-calendars; install requirements-download.txt.") from exc
    return exchange_calendars.get_calendar("XNYS")


def nyse_session_ids(dates: pd.Series) -> pd.Series:
    """Map dates to official XNYS sessions and reject non-session observations."""
    values = pd.to_datetime(dates).dt.normalize()
    sessions = _calendar().sessions_in_range(values.min(), values.max())
    session_dates = pd.DatetimeIndex(sessions).tz_localize(None).normalize()
    identifiers = session_dates.get_indexer(pd.DatetimeIndex(values))
    if (identifiers < 0).any():
        invalid = sorted(values[identifiers < 0].dt.strftime("%Y-%m-%d").unique())
        raise ValueError(f"Non-XNYS daily observations: {invalid[:5]}")
    return pd.Series(identifiers, index=dates.index, dtype="int64")


def next_market_day(dates: pd.Series, market_dates: pd.Series | None = None) -> pd.Series:
    """Return the next official XNYS session after public release (conservative lag)."""
    values = pd.to_datetime(dates, errors="coerce").dt.normalize()
    result = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    if not values.notna().any():
        return result
    calendar = _calendar()
    first = pd.Timestamp(calendar.first_session).tz_localize(None).normalize()
    last = pd.Timestamp(calendar.last_session).tz_localize(None).normalize()
    valid = values.notna() & values.between(first, last)
    if not valid.any():
        return result
    sessions = pd.DatetimeIndex(calendar.sessions_in_range(
        values.loc[valid].min(), min(values.loc[valid].max() + pd.Timedelta(days=14), last)
    )).tz_localize(None).normalize()
    positions = sessions.searchsorted(pd.DatetimeIndex(values.loc[valid]), side="right")
    result.loc[valid] = [sessions[position] if position < len(sessions) else pd.NaT for position in positions]
    return result

"""Daily revenue forecast: ridge regression on trend + weekday + annual Fourier terms + retail event flags
(White Friday, Ramadan, back-to-school). Backtested against seasonal-naive on a 28-day holdout."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

MIN_DAYS = 56  # 28-day backtest + 28 days to learn from

RAMADAN = [(date(2025, 3, 1), date(2025, 3, 30)), (date(2026, 2, 18), date(2026, 3, 19)),
           (date(2027, 2, 8), date(2027, 3, 9))]


def _features(idx: pd.DatetimeIndex, t0: pd.Timestamp) -> np.ndarray:
    t = (idx - t0).days.to_numpy(dtype=float)
    doy = idx.dayofyear.to_numpy()
    cols = [t / 365]
    cols += [(idx.dayofweek == d).astype(float) for d in range(6)]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    wf = ((idx.month == 11) & (idx.day >= 13)).astype(float)
    wf_peak = ((idx.month == 11) & (idx.day >= 24) & (idx.day <= 30)).astype(float)
    bts = (((idx.month == 8) & (idx.day >= 15)) | ((idx.month == 9) & (idx.day <= 15))).astype(float)
    ram = np.array([any(s <= d.date() <= e for s, e in RAMADAN) for d in idx], dtype=float)
    cols += [wf, wf_peak, bts, ram]
    return np.column_stack(cols)


def daily_series(orders: pd.DataFrame) -> pd.Series:
    o = orders[orders["status"] != "cancelled"]
    s = o.groupby(pd.to_datetime(o["created_at"]).dt.normalize())["total"].sum()
    return s.asfreq("D", fill_value=0.0)


def fit_forecast(series: pd.Series, horizon: int = 14, holdout: int = 28) -> dict:
    t0 = series.index[0]
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    m = Ridge(alpha=1.0).fit(_features(train.index, t0), np.log1p(train.values))
    pred = np.expm1(m.predict(_features(test.index, t0)))
    naive = series.shift(7).iloc[-holdout:].to_numpy()
    mean28 = np.full(holdout, train.iloc[-28:].mean())
    # WAPE (sum|err| / sum actual) - robust to lumpy B2B days, the standard retail forecast metric
    wape = lambda a, p: float(np.abs(a - p).sum() / max(a.sum(), 1))  # noqa: E731
    wk = lambda x: np.add.reduceat(np.asarray(x, dtype=float), np.arange(0, holdout, 7))  # noqa: E731
    metrics = {"wape_model": round(wape(test.values, pred), 4),
               "wape_seasonal_naive": round(wape(test.values, naive), 4),
               "wape_mean28": round(wape(test.values, mean28), 4),
               "weekly_wape_model": round(wape(wk(test.values), wk(pred)), 4),
               "weekly_wape_seasonal_naive": round(wape(wk(test.values), wk(naive)), 4)}
    metrics["wape_gain_vs_naive"] = round(metrics["wape_seasonal_naive"] - metrics["wape_model"], 4)
    full = Ridge(alpha=1.0).fit(_features(series.index, t0), np.log1p(series.values))
    fut = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc = np.expm1(full.predict(_features(fut, t0)))
    resid = np.log1p(series.values) - full.predict(_features(series.index, t0))
    sd = float(np.std(resid[-90:]))
    return {"metrics": metrics, "forecast": [
        {"date": d.date().isoformat(), "yhat": round(float(v), 0), "lo": round(float(v * np.exp(-1.28 * sd)), 0),
         "hi": round(float(v * np.exp(1.28 * sd)), 0)} for d, v in zip(fut, fc)], "model": full, "t0": t0}

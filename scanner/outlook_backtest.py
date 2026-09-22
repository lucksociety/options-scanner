#!/usr/bin/env python3
"""
Score the 3-4 week outlook against what actually happened. Replays outlook.outlook_frame over ~8 years of daily
data and, for every session, records SPY and IWM 15 sessions later. Per band: how often the market was up, the
average and median forward return, and the same for 10 and 20 sessions. Writes data/backtest/outlook.json, which
market.py folds into data/market.json so the banner can say "in this band SPY was higher 15 sessions later X% of
the time" instead of asserting an outlook. Also reports the overall base rate so the bands can be judged against it.
"""
import json, logging, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import outlook

log = logging.getLogger("outlook_bt"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "data" / "backtest"
H = (10, 15, 20)


def block(d):
    o = dict(n=int(len(d)))
    for h in H:
        for t in ("spy", "iwm"):
            c = d[f"{t}{h}"].dropna()
            if not len(c): continue
            o[f"{t}_up{h}"] = round(float((c > 0).mean() * 100), 1)
            o[f"{t}_mean{h}"] = round(float(c.mean()), 2); o[f"{t}_med{h}"] = round(float(c.median()), 2)
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hist = yf.download(outlook.TICKERS, period="8y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    F = outlook.outlook_frame(hist)
    spy = hist["SPY"]["Close"].reindex(F.index).ffill(); iwm = hist["IWM"]["Close"].reindex(F.index).ffill()
    for h in H:
        F[f"spy{h}"] = (spy.shift(-h) / spy - 1) * 100
        F[f"iwm{h}"] = (iwm.shift(-h) / iwm - 1) * 100
    F = F.iloc[260:]                                          # warm-up: 50-day math, rolling windows, term structure
    F = F[F["spy15"].notna()]
    # --- calibration: fit the bucket table on the first 70% of sessions, judge the bands on the last 30%, then refit on
    # everything for the live table. The held-out numbers are the honest ones; the banner shows them.
    cut = int(len(F) * 0.7); train, test = F.iloc[:cut], F.iloc[cut:]
    tab_train = outlook.calibrate(train, train["spy15"])
    test = test.copy(); test["exp"] = outlook.expected(test, tab_train); test["xband"] = [outlook.exp_band_of(v)[0] for v in test["exp"]]
    F["exp"] = outlook.expected(F, tab_train); F["xband"] = [outlook.exp_band_of(v)[0] for v in F["exp"]]
    tab_all = outlook.calibrate(F, F["spy15"])
    tab_all["fit"] = dict(train_first=str(train.index[0].date()), train_last=str(train.index[-1].date()),
                          test_first=str(test.index[0].date()), test_last=str(test.index[-1].date()), test_sessions=int(len(test)))
    json.dump(tab_all, open(OUT / "outlook_model.json", "w"), separators=(",", ":"))
    res = dict(asof=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
               first=str(F.index[0].date()), last=str(F.index[-1].date()), sessions=int(len(F)),
               all=block(F), bands={k: block(F[F["band"] == k]) for _, k, _ in outlook.BANDS},
               xbands_test={k: block(test[test["xband"] == k]) for _, k, _ in outlook.EXP_BANDS}, test_all=block(test),
               xbands_all={k: block(F[F["xband"] == k]) for _, k, _ in outlook.EXP_BANDS},
               by_bias={f"{lo}..{lo+20}": block(F[(F["bias"] >= lo) & (F["bias"] < lo + 20)]) for lo in range(-100, 100, 20)},
               parts={p: {f"{lo}..{lo+10}": block(F[(F[p] >= lo) & (F[p] < lo + 10)]) for lo in range(-30, 30, 10)}
                      for p in ("trend", "breadth", "vol", "appetite", "reversion")})
    json.dump(res, open(OUT / "outlook.json", "w"), separators=(",", ":"))
    F[["trend", "breadth", "vol", "appetite", "reversion", "bias", "band", "exp", "xband", "spy10", "spy15", "spy20", "iwm15"]].round(2).to_csv(OUT / "outlook_daily.csv")
    log.info("base rate SPY up 15d: %s%% · raw bands: %s", res["all"].get("spy_up15"),
             {k: (v["n"], v.get("spy_up15"), v.get("spy_mean15")) for k, v in res["bands"].items()})
    log.info("HELD-OUT calibrated bands (base %s%%): %s", res["test_all"].get("spy_up15"),
             {k: (v["n"], v.get("spy_up15"), v.get("spy_mean15")) for k, v in res["xbands_test"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

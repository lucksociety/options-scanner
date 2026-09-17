#!/usr/bin/env python3
"""
Backtest the trigger on real history — replaces folklore thresholds with measured ones.

What it does
  1. Pulls the same universe the scanner screens (short float >= 15%, $1-10, optionable-ish liquidity) — TODAY's
     constituents, which is survivorship-biased: names that already squeezed and left the $1-10 band, or that got
     delisted, are missing. Say so whenever quoting the numbers.
  2. Downloads ~2 years of daily bars for every name plus SPY / IWM / ^VIX.
  3. Walks every name/day and records every "pattern day" (10-day-high breakout or failed-breakdown reversal)
     with the confirmation inputs the live trigger uses (RVOL, 5-day RS vs SPY, close location, opening gap) and
     a simple SPY regime, then the forward path: best high / worst low / close over the next 5, 10, 20 sessions
     measured from the NEXT day's open (you cannot buy the signal close). Also samples random days as a baseline.
  4. Writes data/backtest/signals.csv (every row) and data/backtest/summary.json (hit-rates by bucket), so the
     question "does RVOL >= 1.5 actually beat < 1.5 on these names" has an answer instead of an opinion.

Nothing here uses short-interest history (not free), so it measures the IGNITION half of the model on a
FUEL-selected universe. That is exactly the half whose thresholds were guessed.
"""
import json, logging, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as EQ

log = logging.getLogger("backtest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "backtest"
HORIZONS = (5, 10, 20)
TARGETS = (10, 20, 30, 50)


def universe():
    q = EQ("and", [EQ("gt", ["short_percentage_of_float.value", 15.0]),
                   EQ("btwn", ["intradayprice", 1.0, 10.0]),
                   EQ("gt", ["avgdailyvol3m", 300_000]), EQ("eq", ["region", "us"])])
    out, offset = [], 0
    while True:
        res = yf.screen(q, offset=offset, size=250, sortField="short_percentage_of_float.value", sortAsc=False)
        quotes = res.get("quotes", []) if res else []
        out += [x["symbol"] for x in quotes if x.get("symbol")]
        if len(quotes) < 250 or offset > 3000: break
        offset += 250; time.sleep(0.5)
    return sorted(set(out))


def frame(hist, t, multi):
    h = hist[t] if multi else hist
    h = h.dropna(subset=["Close"])
    return h if len(h) >= 90 else None


def regime_band(spy_close, i):
    """Simple, unfit regime the advice docs agree on: SPY vs 50-day and the 20-day's slope."""
    if i < 55: return "neutral"
    c = spy_close
    s50 = c[i - 49:i + 1].mean(); s20 = c[i - 19:i + 1].mean(); s20p = c[i - 24:i - 4].mean()
    if c[i] > s50 and s20 > s20p: return "bull"
    if c[i] < s50 and s20 < s20p: return "bear"
    return "neutral"


def walk(t, h, spy, rng, sample_every=25):
    cl = h["Close"].to_numpy(float); hi = h["High"].to_numpy(float); lo = h["Low"].to_numpy(float)
    op = h["Open"].to_numpy(float); vol = h["Volume"].to_numpy(float); idx = h.index
    spy_c = spy.reindex(idx).ffill().to_numpy(float)
    rows = []
    n = len(cl)
    for i in range(65, n - 21):
        brk10 = cl[i] > hi[i - 10:i].max()
        prior_low = lo[i - 25:i - 5].min(); rec = lo[i - 5:i + 1]; rec_i = int(np.argmin(rec))
        failbd = rec[rec_i] < prior_low and cl[i] > prior_low and cl[i] > hi[i - 1] and rec_i < 5
        is_pattern = brk10 or failbd
        is_sample = (i % sample_every == 0) and rng.random() < 0.5
        if not (is_pattern or is_sample): continue
        av = vol[i - 63:i].mean()
        rvol = vol[i] / av if av else np.nan
        ret5 = (cl[i] / cl[i - 5] - 1) * 100; spy5 = (spy_c[i] / spy_c[i - 5] - 1) * 100
        rng_ = hi[i] - lo[i]; clpos = (cl[i] - lo[i]) / rng_ if rng_ > 0 else np.nan
        gap = (op[i] / cl[i - 1] - 1) * 100
        entry = op[i + 1]                                     # next day's open: the price you could actually pay
        if not entry or entry <= 0: continue
        r = dict(t=t, date=str(idx[i].date()), kind=("failbd" if failbd else ("brk10" if brk10 else "sample")),
                 pattern=bool(is_pattern), rvol=round(float(rvol), 2), rs5=round(float(ret5 - spy5), 1), ret5=round(float(ret5), 1),
                 clpos=round(float(clpos), 2) if clpos == clpos else None, gap=round(float(gap), 1),
                 regime=regime_band(spy_c, i), px=round(float(cl[i]), 2), entry=round(float(entry), 2),
                 vr20=round(float(vol[i - 2:i + 1].mean() / (vol[i - 22:i - 2].mean() or 1)), 2),
                 h20=round(float(cl[i] / hi[i - 20:i].max() - 1) * 100, 1))
        for H in HORIZONS:
            fh = hi[i + 1:i + 1 + H]; fl = lo[i + 1:i + 1 + H]; fc = cl[i + H] if i + H < n else cl[-1]
            best = (fh.max() / entry - 1) * 100
            ib = int(np.argmax(fh)); mae = (fl[:ib + 1].min() / entry - 1) * 100
            r[f"best{H}"] = round(float(best), 1); r[f"mae{H}"] = round(float(mae), 1); r[f"close{H}"] = round(float(fc / entry - 1) * 100, 1)
        # the live trigger's confirmation, bull vs bear rules
        bear = r["regime"] == "bear"
        r["fired"] = bool(is_pattern and rvol >= (2.0 if bear else 1.5) and r["rs5"] >= (8.0 if bear else 5.0)
                          and (clpos == clpos and clpos >= 0.75) and gap < 15 and (ret5 > 0 if bear else True))
        rows.append(r)
    return rows


def summarize(df):
    def block(d):
        n = len(d)
        if n == 0: return dict(n=0)
        o = dict(n=int(n))
        for H in HORIZONS:
            o[f"med_best{H}"] = round(float(d[f"best{H}"].median()), 1)
            o[f"med_close{H}"] = round(float(d[f"close{H}"].median()), 1)
            o[f"mean_close{H}"] = round(float(d[f"close{H}"].mean()), 1)
            o[f"med_mae{H}"] = round(float(d[f"mae{H}"].median()), 1)
            o[f"hit{H}"] = {str(T): round(float((d[f"best{H}"] >= T).mean() * 100), 1) for T in TARGETS}
        return o
    S = {"asof": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
         "names": int(df["t"].nunique()), "rows": int(len(df)),
         "caveat": "Today's shorted universe replayed backwards — survivorship-biased (names that squeezed out of the $1-10 band or delisted are missing). Entry = next day's open. No short-interest history: this measures ignition on a fuel-selected universe.",
         "baseline": block(df[df["kind"] == "sample"]),
         "pattern_any": block(df[df["pattern"]]),
         "fired": block(df[df["fired"]]),
         "pattern_not_fired": block(df[df["pattern"] & ~df["fired"]]),
         "by_kind": {k: block(df[df["kind"] == k]) for k in ("brk10", "failbd")},
         "fired_by_regime": {k: block(df[df["fired"] & (df["regime"] == k)]) for k in ("bull", "neutral", "bear")},
         "pattern_by_regime": {k: block(df[df["pattern"] & (df["regime"] == k)]) for k in ("bull", "neutral", "bear")},
         "baseline_by_regime": {k: block(df[(df["kind"] == "sample") & (df["regime"] == k)]) for k in ("bull", "neutral", "bear")}}
    pat = df[df["pattern"]]
    S["by_rvol"] = {lab: block(pat[m]) for lab, m in (("<1.0", pat.rvol < 1), ("1.0-1.5", (pat.rvol >= 1) & (pat.rvol < 1.5)),
                                                    ("1.5-2.0", (pat.rvol >= 1.5) & (pat.rvol < 2)), ("2.0-3.0", (pat.rvol >= 2) & (pat.rvol < 3)), (">=3.0", pat.rvol >= 3))}
    S["by_rs5"] = {lab: block(pat[m]) for lab, m in (("<0", pat.rs5 < 0), ("0-5", (pat.rs5 >= 0) & (pat.rs5 < 5)),
                                                   ("5-10", (pat.rs5 >= 5) & (pat.rs5 < 10)), ("10-20", (pat.rs5 >= 10) & (pat.rs5 < 20)), (">=20", pat.rs5 >= 20))}
    S["by_clpos"] = {lab: block(pat[m]) for lab, m in (("<0.5", pat.clpos < .5), ("0.5-0.75", (pat.clpos >= .5) & (pat.clpos < .75)), (">=0.75", pat.clpos >= .75))}
    S["by_gap"] = {lab: block(pat[m]) for lab, m in (("<5", pat.gap < 5), ("5-15", (pat.gap >= 5) & (pat.gap < 15)), (">=15", pat.gap >= 15))}
    S["by_vr20"] = {lab: block(pat[m]) for lab, m in (("<1.3", pat.vr20 < 1.3), ("1.3-2", (pat.vr20 >= 1.3) & (pat.vr20 < 2)), (">=2", pat.vr20 >= 2))}
    return S


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    syms = universe(); log.info("universe: %d names", len(syms))
    hist = yf.download(syms, period="2y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    bench = yf.download(["SPY", "IWM", "^VIX"], period="2y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    spy = bench["SPY"]["Close"].dropna()
    rng = np.random.default_rng(7); rows = []; ok = 0
    for t in syms:
        try:
            h = frame(hist, t, len(syms) > 1)
            if h is None: continue
            rows += walk(t, h, spy, rng); ok += 1
        except Exception as e:
            log.warning("%s: %s", t, e)
    df = pd.DataFrame(rows)
    log.info("%d names walked, %d rows (%d pattern days, %d fired)", ok, len(df), int(df["pattern"].sum()), int(df["fired"].sum()))
    df.to_csv(OUT / "signals.csv", index=False)
    S = summarize(df)
    json.dump(S, open(OUT / "summary.json", "w"), separators=(",", ":"))
    log.info("baseline hit20 %s · fired hit20 %s", S["baseline"].get("hit20"), S["fired"].get("hit20"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

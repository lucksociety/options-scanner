#!/usr/bin/env python3
"""
Miss audit: which $1-10 optionable stocks actually squeezed in the last N sessions, what did they look like the day
BEFORE the move, and did our scanner have them (universe -> deep 40 -> top 16) on the day it mattered?

This is the evidence a redesign has to answer to. Output: data/audit/misses.json + misses.csv
  movers   every name whose close rose >= MIN_GAIN% from a close to a high within 5 sessions, inside the window
  pre      features on the session BEFORE the first big day: short float / days-to-cover (today's report, the best
           free proxy), float size, price, RSI, distance from 20-day low/high and 52-week low, 5/20-day return,
           RVOL on the first big day and the day before, 3-day volume vs 20-day, gap on the first big day
  ours     for each mover: was SI >= 15% (our universe gate), was it in a history file's stage1 / top / rest on
           any of the 3 sessions before the move, and what score we gave it
"""
import json, logging, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as EQ

log = logging.getLogger("audit"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data"; OUT = DATA / "audit"
WINDOW = 15; MIN_GAIN = 40.0


def rsi(c, n=14):
    d = np.diff(c); g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    ag = g[:n].mean(); al = l[:n].mean(); out = [100 - 100 / (1 + ag / (al or 1e-9))]
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n; al = (al * (n - 1) + l[i]) / n; out.append(100 - 100 / (1 + ag / (al or 1e-9)))
    return out


def universe():
    q = EQ("and", [EQ("btwn", ["intradayprice", 1.0, 10.0]), EQ("gt", ["avgdailyvol3m", 200_000]), EQ("eq", ["region", "us"])])
    out, offset = {}, 0
    while True:
        res = yf.screen(q, offset=offset, size=250, sortField="avgdailyvol3m", sortAsc=False)
        quotes = res.get("quotes", []) if res else []
        for x in quotes:
            if x.get("symbol"): out[x["symbol"]] = x
        if len(quotes) < 250 or offset > 4000: break
        offset += 250; time.sleep(0.4)
    return out


def history_files():
    """Every scan-day file we have, by side, as {date: data}."""
    H = {}
    for side, d in (("calls", DATA / "history"), ("breakout", DATA / "breakout" / "history")):
        H[side] = {}
        for f in sorted(d.glob("????-??-??.json")):
            try: H[side][f.stem] = json.load(open(f))
            except Exception: pass
    return H


def ours(H, t, day):
    """Where was ticker t in our files on the 3 scan days at or before `day`?"""
    best = {}
    for side, files in H.items():
        days = [d for d in files if d <= day][-3:]
        for d in days:
            f = files[d]
            tops = {o["t"]: o for o in f.get("top", [])}
            rest = {r[0]: r for r in f.get("rest", [])}
            s1 = {o["t"]: o for o in f.get("stage1", [])}
            if t in tops:
                o = tops[t]; best[side] = dict(day=d, where="top16", score=o.get("score"), fired=((o.get("trig") or {}).get("state")),
                                              play=bool(o.get("play")), s1=o.get("s1")); break
            if t in rest: best[side] = dict(day=d, where="deep40", score=rest[t][4]); break
            if t in s1: best[side] = dict(day=d, where="stage1", s1=s1[t].get("s1")); break
        best.setdefault(side, dict(where="absent"))
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    U = universe(); syms = sorted(U); log.info("universe: %d names", len(syms))
    hist = yf.download(syms, period="6mo", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    H = history_files()
    movers = []
    for t in syms:
        try:
            h = (hist[t] if len(syms) > 1 else hist).dropna(subset=["Close"])
            if len(h) < 70: continue
            cl = h["Close"].to_numpy(float); hi = h["High"].to_numpy(float); lo = h["Low"].to_numpy(float)
            op = h["Open"].to_numpy(float); vol = h["Volume"].to_numpy(float); idx = h.index
            n = len(cl); start = n - WINDOW
            best = None
            for i in range(max(start, 25), n):                      # i = first session of the move (the "big day")
                base = cl[i - 1]
                peak = hi[i:i + 5].max(); gain = (peak / base - 1) * 100
                if gain >= MIN_GAIN and (best is None or gain > best[1]): best = (i, gain)
            if best is None: continue
            i, gain = best; b = i - 1
            av20 = vol[b - 20:b].mean() or 1; av63 = vol[max(0, b - 63):b].mean() or 1
            r = rsi(cl[: b + 1])[-1] if b > 20 else None
            pre = dict(px=round(cl[b], 2), rsi=round(r, 1) if r else None,
                       lo20=round((cl[b] / lo[b - 20:b + 1].min() - 1) * 100, 1), hi20=round((cl[b] / hi[b - 20:b + 1].max() - 1) * 100, 1),
                       lo52=round((cl[b] / cl[:b + 1].min() - 1) * 100, 1), s20=round((cl[b] / cl[b - 20:b + 1].mean() - 1) * 100, 1),
                       ret5=round((cl[b] / cl[b - 5] - 1) * 100, 1), ret20=round((cl[b] / cl[b - 20] - 1) * 100, 1),
                       rvol_prev=round(vol[b] / av20, 2), vr3=round(vol[b - 2:b + 1].mean() / av20, 2),
                       rvol_day1=round(vol[i] / av20, 2), gap_day1=round((op[i] / cl[b] - 1) * 100, 1), day1=round((cl[i] / cl[b] - 1) * 100, 1),
                       above20=bool(cl[b] > cl[b - 20:b + 1].mean()), dv=round(av63 * cl[b] / 1e6, 2))
            movers.append(dict(t=t, co=U[t].get("longName") or U[t].get("shortName"), day1=str(idx[i].date()), pre_day=str(idx[b].date()),
                               gain5=round(gain, 1), peak=round(float(hi[i:i + 5].max()), 2), pre=pre, mc=U[t].get("marketCap")))
        except Exception as e:
            log.debug("%s: %s", t, e)
    movers.sort(key=lambda m: -m["gain5"])
    log.info("%d movers >= %s%% in the last %d sessions", len(movers), MIN_GAIN, WINDOW)
    for m in movers:
        try:
            info = yf.Ticker(m["t"]).info
            m["sf"] = round(float(info.get("shortPercentOfFloat") or 0) * 100, 1); m["sr"] = info.get("shortRatio")
            m["flt"] = info.get("floatShares"); m["ss"] = info.get("sharesShort")
            if not m["sf"] and m["ss"] and m["flt"]: m["sf"] = round(m["ss"] / m["flt"] * 100, 1)
            m["earn"] = info.get("earningsTimestampStart")
        except Exception: pass
        m["ours"] = ours(H, m["t"], m["pre_day"]); time.sleep(0.15)
    def has_options(t):
        try: return bool(yf.Ticker(t).options)
        except Exception: return False
    for m in movers[:80]: m["optionable"] = has_options(m["t"]); time.sleep(0.1)
    S = dict(asof=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"), window=WINDOW, min_gain=MIN_GAIN,
             universe=len(syms), movers=len(movers),
             in_si_universe=sum(1 for m in movers if (m.get("sf") or 0) >= 15),
             ours_any=sum(1 for m in movers if any(v.get("where") != "absent" for v in m["ours"].values())),
             ours_top16=sum(1 for m in movers if any(v.get("where") == "top16" for v in m["ours"].values())),
             rows=movers)
    json.dump(S, open(OUT / "misses.json", "w"), separators=(",", ":"))
    flat = [dict(t=m["t"], date=m["day1"], gain5=m["gain5"], sf=m.get("sf"), flt=m.get("flt"), optionable=m.get("optionable"),
                 calls=m["ours"]["calls"].get("where"), calls_score=m["ours"]["calls"].get("score"),
                 brk=m["ours"]["breakout"].get("where"), **m["pre"]) for m in movers]
    pd.DataFrame(flat).to_csv(OUT / "misses.csv", index=False)
    log.info("SI>=15: %d/%d · in any of our files: %d · in a top16: %d", S["in_si_universe"], S["movers"], S["ours_any"], S["ours_top16"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

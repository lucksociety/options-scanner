#!/usr/bin/env python3
"""
Luck Society Option Scanner — outcome tracker.

Runs after the close. For every full-scan history file (data/history/*.json and data/puts/history/*.json) it looks at the
ranked picks that had a tradeable contract and records what happened next:

  stock  +5 / +10 / +20 session returns, best move in the window, whether the contract's breakeven was reached
  option current mark (bid, or last if no bid) while the contract is alive; intrinsic value at expiry once it has expired

Outputs
  data/track/<side>/<YYYY-MM-DD>.json   per-pick outcomes for that scan day
  data/track/summary.json               hit-rates by score bucket (what the dashboard shows)
  data/track/open.json                  every pick from the last 45 days with its latest mark (paper-trade log uses this)

A "hit" = the option is (or was at expiry) worth >= 2x the entry ask, OR the stock reached the breakeven price inside the
window. A "bust" = contract expired worthless or the option is down 70%+ with < 5 days left. Everything else is "open".
Usage: python scanner/track.py
"""
import json, sys, time, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("track"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data"; OUT = DATA / "track"
LOOKBACK_DAYS = 45
SIDES = ("calls", "puts", "breakout")

BUCKETS = [(0, 50, "<50"), (50, 60, "50-59"), (60, 70, "60-69"), (70, 101, "70+")]

def bucket(sc):
    for lo, hi, name in BUCKETS:
        if lo <= sc < hi: return name
    return "<50"

def load_history(side):
    d = DATA / ("history" if side == "calls" else f"{side}/history")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    out = []
    for f in sorted(d.glob("????-??-??.json")):
        if f.stem < cutoff: continue
        try: out.append((f.stem, json.load(open(f))))
        except Exception as e: log.warning("%s: %s", f, e)
    return out

def entry_price(play):
    a, b, l = play.get("ask") or 0, play.get("bid") or 0, play.get("last") or 0
    return round(a if a > 0 else (l if l > 0 else b), 2)

def option_mark(tk, play, side):
    """Current value of the contract: bid if there is one, else last. None if the chain can't be read."""
    try:
        chain = tk.option_chain(play["exp"]); tab = chain.puts if side == "puts" else chain.calls
        row = tab[np.isclose(tab["strike"], play["strike"])]
        if row.empty: return None
        r = row.iloc[0]; bid = float(r.bid or 0); last = float(r.lastPrice or 0)
        return round(bid if bid > 0 else last, 2)
    except Exception: return None

def main():
    OUT.mkdir(parents=True, exist_ok=True); today = datetime.now(timezone.utc).date()
    summary = {"asof": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"), "sides": {}}
    open_all = []
    for side in SIDES:
        hist = load_history(side)
        if not hist: continue
        (OUT / side).mkdir(exist_ok=True)
        picks = []   # (date, pick)
        for date, d in hist:
            for o in d.get("top", []):
                if o.get("play") and o.get("px"): picks.append((date, o))
        syms = sorted({o["t"] for _, o in picks})
        if not syms: continue
        log.info("%s: %d picks across %d days, %d tickers", side, len(picks), len(hist), len(syms))
        px = yf.download(syms, period="3mo", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
        tks = {}; per_day = {}
        stats = {name: dict(n=0, hit=0, bust=0, open=0, stock_best=[], opt_ret=[], m10=0, m20=0, m50=0) for _, _, name in BUCKETS}
        for date, o in picks:
            t = o["t"]; play = o["play"]; entry = entry_price(play); px0 = float(o["px"]); be = float(play.get("be") or 0)
            k = float(play["strike"]); exp = datetime.strptime(play["exp"], "%Y-%m-%d").date()
            try:
                h = (px[t] if len(syms) > 1 else px).dropna(subset=["Close"])
                h = h[h.index.date > datetime.strptime(date, "%Y-%m-%d").date()]     # sessions AFTER the scan day
            except Exception: h = pd.DataFrame()
            cl = h["Close"].to_numpy() if len(h) else np.array([]); hi = h["High"].to_numpy() if len(h) else cl; lo = h["Low"].to_numpy() if len(h) else cl
            ret = lambda n: round((float(cl[n - 1]) / px0 - 1) * 100, 1) if len(cl) >= n else None
            win = cl[:20]; win_hi = hi[:20]; win_lo = lo[:20]
            mae = None
            if side != "puts":
                best = round((float(win_hi.max()) / px0 - 1) * 100, 1) if len(win) else None
                if len(win):
                    ib = int(np.argmax(win_hi)); mae = round((float(win_lo[:ib + 1].min()) / px0 - 1) * 100, 1)   # worst dip BEFORE the best print
                be_hit = bool(len(win) and float(win_hi.max()) >= k + entry)
                intrinsic_exp = max(0.0, float(cl[-1]) - k) if (exp <= today and len(cl)) else None
            else:
                best = round((1 - float(win_lo.min()) / px0) * 100, 1) if len(win) else None
                be_hit = bool(len(win) and float(win_lo.min()) <= k - entry)
                intrinsic_exp = max(0.0, k - float(cl[-1])) if (exp <= today and len(cl)) else None
            expired = exp < today
            if expired: mark = round(intrinsic_exp, 2) if intrinsic_exp is not None else None
            else:
                tk = tks.setdefault(t, yf.Ticker(t)); mark = option_mark(tk, play, side); time.sleep(0.3)
            opt_ret = round((mark / entry - 1) * 100, 0) if (mark is not None and entry > 0) else None
            dte_left = (exp - today).days
            if (opt_ret is not None and opt_ret >= 100) or be_hit: status = "hit"
            elif expired and (mark or 0) <= 0.01: status = "bust"
            elif opt_ret is not None and opt_ret <= -70 and dte_left < 5: status = "bust"
            else: status = "open"
            rec = dict(t=t, side=side, scan=date, score=o.get("score"), px0=px0, entry=entry, exp=play["exp"], strike=k, be=be, mae=mae,
                       fired=((o.get("trig") or {}).get("state") == "fired"), earn_in=bool(o.get("earn_in")), regime=(o.get("mkt")),
                       r5=ret(5), r10=ret(10), r20=ret(20), best=best, be_hit=be_hit, mark=mark, opt_ret=opt_ret,
                       sessions=int(min(len(cl), 20)), expired=expired, status=status)
            per_day.setdefault(date, []).append(rec)
            if dte_left >= -1 or status != "open": open_all.append(rec)
            b = stats[bucket(float(o.get("score") or 0))]; b["n"] += 1; b[status] += 1
            if best is not None:
                b["stock_best"].append(best)
                for m in (10, 20, 50):                      # measured frequency, not a modelled probability
                    if best >= m: b[f"m{m}"] += 1
            if opt_ret is not None: b["opt_ret"].append(opt_ret)
        for date, recs in per_day.items():
            json.dump(recs, open(OUT / side / f"{date}.json", "w"), separators=(",", ":"))
        summ = {}
        for name, b in stats.items():
            decided = b["hit"] + b["bust"]
            summ[name] = dict(n=b["n"], hit=b["hit"], bust=b["bust"], open=b["open"],
                              hit_rate=round(b["hit"] / decided * 100) if decided else None,
                              avg_best=round(float(np.mean(b["stock_best"])), 1) if b["stock_best"] else None,
                              moved={str(m): dict(n=b[f"m{m}"], pct=round(b[f"m{m}"] / len(b["stock_best"]) * 100) if b["stock_best"] else None) for m in (10, 20, 50)},
                              measured=len(b["stock_best"]),
                              med_opt=round(float(np.median(b["opt_ret"]))) if b["opt_ret"] else None)
        summary["sides"][side] = dict(days=len(hist), picks=len(picks), buckets=summ,
                                      first=hist[0][0], last=hist[-1][0])
        log.info("%s summary: %s", side, {k: (v["n"], v["hit_rate"]) for k, v in summ.items()})
    json.dump(summary, open(OUT / "summary.json", "w"), separators=(",", ":"))
    json.dump(open_all, open(OUT / "open.json", "w"), separators=(",", ":"))
    log.info("tracked %d picks", len(open_all))
    return 0

if __name__ == "__main__":
    sys.exit(main())

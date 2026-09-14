#!/usr/bin/env python3
"""
Luck Society Option Scanner — finds heavily shorted, sub-$10, optionable stocks that look like they've bottomed,
and picks the best call contract under $0.25 expiring 2–6 weeks out.

Modes
  full     Finviz screener → filter → deep scan (price history + option chains) → score.   (~3–5 min)
  refresh  Re-quote the names from the last full scan (prices, RSI, option quotes) → rescore. (~1–2 min)
  auto     full if no full scan yet today (ET) and it's after 08:25 ET, refresh during 08:25–16:10 ET, else skip.

Data: Yahoo Finance via yfinance (screener + history + option chains).
Output: data/latest.json (+ data/history/YYYY-MM-DD.json after a full scan)
"""
import json, os, re, sys, time, math, io, logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import yfinance as yf

log = logging.getLogger("scan")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CFG = dict(
    price_min=1.0, price_max=10.0, si_min=15.0, avgvol_min=300_000,   # stage 1
    deep_n=40,                                                          # names to deep-scan
    min_dte=14, max_dte=45, max_ask=0.25, min_oi=25, max_be=60.0,       # contract rules
)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9"}

# --------------------------------------------------------------------------- stage 1: Yahoo screener (no Finviz — cloud IPs get a stripped page there)
from yfinance import EquityQuery as EQ

def screener_universe():
    """Every US stock $price_min–$price_max with short float ≥ si_min and 3-month avg volume ≥ avgvol_min."""
    q = EQ("and", [
        EQ("gt", ["short_percentage_of_float.value", CFG["si_min"]]),
        EQ("btwn", ["intradayprice", CFG["price_min"], CFG["price_max"]]),
        EQ("gt", ["avgdailyvol3m", CFG["avgvol_min"]]),
        EQ("eq", ["region", "us"]),
    ])
    out, offset = [], 0
    while True:
        res = yf.screen(q, offset=offset, size=250, sortField="short_percentage_of_float.value", sortAsc=False)
        quotes = res.get("quotes", []) if res else []
        out += quotes
        if len(quotes) < 250 or offset > 2000: break
        offset += 250; time.sleep(0.5)
    seen, uniq = set(), []
    for x in out:
        if x.get("symbol") and x["symbol"] not in seen: seen.add(x["symbol"]); uniq.append(x)
    log.info("screener: %d names", len(uniq))
    return uniq

def stage1():
    quotes = screener_universe()
    syms = [x["symbol"] for x in quotes]
    if not syms: raise RuntimeError("screener returned nothing")
    # one batched download for a year of daily bars
    hist = yf.download(syms, period="1y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    rows = []
    for x in quotes:
        t = x["symbol"]
        try:
            h = hist[t].dropna(subset=["Close"]) if len(syms) > 1 else hist.dropna(subset=["Close"])
            if len(h) < 60: continue
            cl = h["Close"].to_numpy(); vol = h["Volume"].to_numpy(); p = float(cl[-1])
            def perf(n): return round((p / float(cl[-1 - n]) - 1) * 100, 2) if len(cl) > n else float("nan")
            def sma(n): return round((p / float(cl[-n:].mean()) - 1) * 100, 2) if len(cl) >= n else float("nan")
            lo52 = float(cl[-252:].min()); hi52 = float(cl[-252:].max())
            rs = rsi_series(cl[-60:])
            av = float(vol[-63:].mean())
            rows.append(dict(t=t, co=x.get("longName") or x.get("shortName") or t, mc=x.get("marketCap"),
                             sf=float("nan"), sr=float("nan"), pw=perf(5), pm=perf(21), pq=perf(63),
                             s20=sma(20), s50=sma(50), s200=sma(200), hi=round((p / hi52 - 1) * 100, 2), lo=round((p / lo52 - 1) * 100, 2),
                             rsi=round(float(rs[-1]), 2), av=av, rv=round(float(vol[-1]) / (av or 1), 2), p=round(p, 2), ch=x.get("regularMarketChangePercent"), v=float(vol[-1]), earn="-"))
        except Exception as e:
            log.warning("stage1 %s: %s", t, e)
    # short float / days-to-cover / earnings date per name (one small request each)
    for r in rows:
        try:
            info = yf.Ticker(r["t"]).info
            r["sf"] = round(float(info.get("shortPercentOfFloat") or 0) * 100, 2)
            r["sr"] = round(float(info.get("shortRatio") or 0), 2)
            ts = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
            if ts:
                d = datetime.fromtimestamp(ts, ET)
                r["earn"] = d.strftime("%b %d") + ("/b" if d.hour < 12 else "/a")
        except Exception as e:
            log.warning("info %s: %s", r["t"], e)
        time.sleep(0.15)
    keep = [x for x in rows if CFG["price_min"] <= x["p"] <= CFG["price_max"] and x["av"] >= CFG["avgvol_min"] and x["sf"] >= CFG["si_min"]]
    for x in keep:
        s = min(x["sf"], 60) / 60 * 35 + min(x["sr"] if x["sr"] == x["sr"] else 0, 10) / 10 * 10
        if 25 <= x["rsi"] <= 45: s += 15
        elif 45 < x["rsi"] <= 55: s += 8
        elif x["rsi"] < 25: s += 5
        if x["lo"] <= 25: s += 15
        elif x["lo"] <= 50: s += 8
        if x["pq"] < -20: s += 8
        if x["pw"] > -3: s += 6
        if x["s20"] > 0: s += 6
        if x["rv"] >= 1.2: s += 5
        x["s1"] = round(s, 1)
    keep.sort(key=lambda x: -x["s1"])
    # keep only names that actually have listed options, up to deep_n
    top = []
    for x in keep:
        if len(top) >= CFG["deep_n"]: break
        try:
            if yf.Ticker(x["t"]).options: top.append(x)
        except Exception: pass
        time.sleep(0.1)
    top = [{k: (None if isinstance(x[k], float) and math.isnan(x[k]) else x[k]) for k in
            ("t", "co", "p", "sf", "sr", "rsi", "lo", "pw", "pm", "pq", "s20", "rv", "av", "mc", "earn", "s1")} for x in top]
    log.info("stage1: universe=%d pass=%d deep=%d", len(rows), len(keep), len(top))
    return len(rows), len(keep), top

# --------------------------------------------------------------------------- stage 2: Yahoo via yfinance
def rsi_series(closes, n=14):
    c = np.asarray(closes, dtype=float); d = np.diff(c)
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    out = []; ag = g[:n].mean(); al = l[:n].mean()
    out.append(100 - 100 / (1 + ag / (al or 1e-9)))
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n; al = (al * (n - 1) + l[i]) / n
        out.append(100 - 100 / (1 + ag / (al or 1e-9)))
    return out

def clamp(x, a, b): return max(a, min(b, x))

def deep(x):
    o = dict(x); o["err"] = ""
    try:
        tk = yf.Ticker(x["t"])
        h = tk.history(period="3mo", interval="1d", auto_adjust=False)
        h = h.dropna(subset=["Close"])
        if len(h) < 25: raise RuntimeError("not enough price history")
        cl = h["Close"].to_numpy(); vol = h["Volume"].to_numpy()
        o["px"] = round(float(cl[-1]), 2)
        rs = rsi_series(cl)
        o["rsi"] = round(rs[-1], 1); o["rsi3"] = round(rs[-4], 1); o["rsiMin10"] = round(min(rs[-10:]), 1)
        o["rsiUp"] = o["rsi"] > o["rsi3"]; o["wasOversold"] = o["rsiMin10"] < 35
        v3 = vol[-3:].mean(); v20 = vol[-23:-3].mean() or 1
        o["vr"] = round(float(v3 / v20), 2)
        o["low20"] = round(float(cl[-1] / cl[-20:].min() - 1) * 100, 1)
        o["up3d"] = round(float(cl[-1] / cl[-4] - 1) * 100, 1)
        o["closes"] = [round(float(v), 2) for v in cl[-30:]]
        # options
        now = datetime.now(timezone.utc); lst = []; cands = 0
        exps = []
        for e in tk.options:
            dte = (datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) - now).days + 1
            if CFG["min_dte"] <= dte <= CFG["max_dte"]: exps.append((e, dte))
        for e, dte in exps[:3]:
            calls = tk.option_chain(e).calls
            for c in calls.itertuples():
                ask = float(c.ask or 0); bid = float(c.bid or 0); last = float(c.lastPrice or 0)
                if not (ask > 0 and ask <= CFG["max_ask"] and c.strike >= o["px"] * 0.98): continue
                cands += 1
                oi = int(c.openInterest) if c.openInterest == c.openInterest else 0
                abs_sp = ask - bid; spread = abs_sp / ask
                if oi < CFG["min_oi"] or (spread > 0.75 and abs_sp > 0.10): continue
                be = (c.strike + ask) / o["px"] - 1
                iv = round(float(c.impliedVolatility) * 100) if c.impliedVolatility == c.impliedVolatility else None
                lst.append(dict(exp=e, dte=dte, strike=float(c.strike), bid=round(bid, 2), ask=round(ask, 2), last=round(last, 2), oi=oi,
                                vol=int(c.volume) if c.volume == c.volume else 0, iv=iv,
                                be=round(be * 100, 1), spread=round(spread * 100)))
            time.sleep(0.25)
        trad = [c for c in lst if c["be"] <= CFG["max_be"] and (c["bid"] > 0 or c["last"] > 0)]
        for c in trad:
            c["q"] = round(clamp(c["oi"], 0, 2000) / 2000 * 10 + (1 - c["spread"] / 100) * 8
                           + clamp(1 - (c["be"] - 15) / 45, 0, 1) * 12 + (0 if c["dte"] >= 21 else -2), 1)
        trad.sort(key=lambda c: -c["q"])
        o["play"] = trad[0] if trad else None; o["alts"] = trad[1:3]; o["cands"] = cands
        # score
        fuel = clamp(o["sf"] or 0, 0, 50) / 50 * 25 + clamp(o["sr"] or 0, 0, 10) / 10 * 10
        bot = 0
        if 25 <= o["rsi"] <= 45: bot += 10
        elif o["rsi"] < 25 or o["rsi"] <= 52: bot += 4
        if o["rsiUp"]: bot += 10
        if o["wasOversold"]: bot += 5
        if o["low20"] <= 8 or (o["lo"] is not None and o["lo"] <= 15): bot += 5
        if o["vr"] >= 1.3: bot += 5
        opt = o["play"]["q"] if o["play"] else 0
        s = fuel + bot + opt - (5 if o["up3d"] < -10 else 0)
        flags = []
        if not o["play"]: flags.append("no tradeable contract ≤$0.25")
        if o["up3d"] < -10: flags.append(f"still falling ({o['up3d']}% / 3d)")
        if o["rsiUp"]: flags.append("RSI turning up")
        if o["vr"] >= 1.3: flags.append(f"volume pickup {o['vr']}x")
        m = datetime.now(ET).month; mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        if re.match(f"^({mn[m-1]}|{mn[m % 12]})", o.get("earn") or ""): flags.append(f"earnings {o['earn']}")
        o.update(flags=flags, fuel=round(fuel, 1), bot=bot, opt=round(opt, 1), score=round(s if o["play"] else min(s, 40), 1))
    except Exception as e:  # keep going; one bad ticker shouldn't sink the scan
        log.warning("%s: %s", x["t"], e); o["err"] = str(e)[:120]; o["score"] = -1
    return o

def stage2(top):
    out = []
    for i, x in enumerate(top):
        out.append(deep(x)); time.sleep(0.4)
        if i % 10 == 9: log.info("deep %d/%d", i + 1, len(top))
    out.sort(key=lambda o: -o["score"])
    return out

def _clean(v):
    import numpy as _np
    if isinstance(v, dict): return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list): return [_clean(x) for x in v]
    if isinstance(v, (_np.floating,)): return None if _np.isnan(v) else float(v)
    if isinstance(v, (_np.integer,)): return int(v)
    if isinstance(v, float) and v != v: return None
    return v

def assemble(mode, universe, pass1, top_in, deep_out, full_asof):
    keys = ("t","co","px","sf","sr","rsi","rsi3","lo","low20","up3d","vr","pq","pm","av","mc","earn","fuel","bot","opt","score","flags","play","alts","closes","err")
    top = [{k: o.get(k) for k in keys} for o in deep_out[:16]]
    for o in top: o["lo52"] = o.pop("lo")
    rest = [[o["t"], o.get("px"), o.get("sf"), o.get("rsi"), o.get("score"), 1 if o.get("play") else 0] for o in deep_out[16:]]
    return _clean(dict(asof=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                full_asof=full_asof, mode=mode, universe=universe, pass1=pass1, deep=len(deep_out),
                stage1=top_in, top=top, rest=rest, cfg=CFG))

def load_previous():
    p = DATA / "latest.json"
    if p.exists():
        try: return json.load(open(p))
        except Exception: pass
    url = os.environ.get("PAGES_DATA_URL")  # e.g. https://user.github.io/repo/data/latest.json
    if url:
        try:
            r = requests.get(url, timeout=20, headers={"Cache-Control": "no-cache"}); r.raise_for_status(); return r.json()
        except Exception as e: log.warning("no previous data at %s: %s", url, e)
    return None

def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "auto").lower()
    now_et = datetime.now(ET); today = now_et.strftime("%Y-%m-%d")
    prev = load_previous()
    if mode == "auto" and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        mode = "full"  # a person pressed Run workflow — always do the real thing
    if mode == "auto":
        hhmm = now_et.hour * 100 + now_et.minute
        if now_et.weekday() >= 5 or not (825 <= hhmm <= 1610):
            log.info("auto: outside 08:25–16:10 ET weekday window — skipping"); return 0
        have_full_today = bool(prev and (prev.get("full_asof") or "")[:10] == today)
        mode = "refresh" if have_full_today else "full"
    if mode == "refresh" and not (prev and prev.get("stage1")):
        log.info("refresh requested but no previous scan — doing full"); mode = "full"
    DATA.mkdir(parents=True, exist_ok=True); (DATA / "history").mkdir(exist_ok=True)
    if mode == "full":
        universe, pass1, top = stage1()
        full_asof = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    else:
        universe, pass1, top = prev["universe"], prev["pass1"], prev["stage1"]
        full_asof = prev.get("full_asof")
    out = assemble(mode, universe, pass1, top, stage2(top), full_asof)
    json.dump(out, open(DATA / "latest.json", "w"), separators=(",", ":"))
    if mode == "full":
        json.dump(out, open(DATA / "history" / f"{today}.json", "w"), separators=(",", ":"))
        idx = sorted(p.stem for p in (DATA / "history").glob("*.json"))
        json.dump(idx, open(DATA / "history" / "index.json", "w"))
    log.info("%s scan done: %d deep, top: %s", mode, out["deep"], ", ".join(f"{o['t']} {o['score']}" for o in out["top"][:5]))
    return 0

if __name__ == "__main__":
    sys.exit(main())


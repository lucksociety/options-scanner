#!/usr/bin/env python3
"""
Luck Society Option Scanner — two sides, one engine.

  calls  Heavily shorted, sub-$10 stocks that look bottomed → best OTM call ≤ $0.25, 2–6 weeks out.
  puts   Overextended $5–$30 stocks with a forced seller behind them (trapped momentum buyers, dilution,
         lockup expiry) that look like they're topping → best OTM put ≤ $0.35, 3–6 weeks out.

Usage:  python scanner/scan.py <mode> [side]
  mode   full | refresh | auto      (auto: full first run after 08:25 ET, refresh until 16:10 ET, else skip)
  side   calls | puts | both        (default both)

Data: Yahoo Finance via yfinance (screener, batched history, key stats, option chains) + SEC EDGAR (filings).
Output: data/latest.json (calls), data/puts/latest.json (puts), plus data/<side>/history/YYYY-MM-DD.json after a full scan.
"""
import json, os, re, sys, time, math, logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from yfinance import EquityQuery as EQ

log = logging.getLogger("scan")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
SEC_UA = {"User-Agent": "LuckSocietyOptionScanner/1.0 (github.com/lucksociety/options-scanner)"}

CFG = {
    "calls": dict(price_min=1.0, price_max=10.0, si_min=15.0, avgvol_min=300_000, deep_n=40,
                  min_dte=14, max_dte=45, max_ask=0.25, min_oi=25, max_be=60.0),
    "puts":  dict(price_min=3.0, price_max=50.0, si_max=15.0, avgvol_min=500_000, deep_n=40,
                  run5_min=15.0, run10_min=25.0, ext20_min=20.0, gap_min=12.0,       # "ran too far" triggers (any one) — RSI alone is NOT a ticket in
                  min_dte=21, max_dte=45, max_ask=0.35, min_oi=25, max_be=35.0),
}

# --------------------------------------------------------------------------- helpers
def clamp(x, a, b): return max(a, min(b, x))

def rsi_series(closes, n=14):
    c = np.asarray(closes, dtype=float); d = np.diff(c)
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    out = []; ag = g[:n].mean(); al = l[:n].mean()
    out.append(100 - 100 / (1 + ag / (al or 1e-9)))
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n; al = (al * (n - 1) + l[i]) / n
        out.append(100 - 100 / (1 + ag / (al or 1e-9)))
    return out

def screen(query, sort_field):
    out, offset = [], 0
    while True:
        res = yf.screen(query, offset=offset, size=250, sortField=sort_field, sortAsc=False)
        quotes = res.get("quotes", []) if res else []
        out += quotes
        if len(quotes) < 250 or offset > 3000: break
        offset += 250; time.sleep(0.5)
    seen, uniq = set(), []
    for x in out:
        if x.get("symbol") and x["symbol"] not in seen: seen.add(x["symbol"]); uniq.append(x)
    return uniq

def batch_history(syms, period="1y"):
    return yf.download(syms, period=period, interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)

def frame_for(hist, t, n):
    h = hist[t] if n > 1 else hist
    return h.dropna(subset=["Close"])

def earn_label(info):
    ts = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
    if not ts: return "-"
    d = datetime.fromtimestamp(ts, ET)
    return d.strftime("%b %d") + ("/b" if d.hour < 12 else "/a")

def has_options(t):
    try: return bool(yf.Ticker(t).options)
    except Exception: return False

_cik = None
def edgar_dilution(t):
    """Recent shelf / offering filings = the company is (or can be) selling stock into strength."""
    global _cik
    try:
        if _cik is None:
            r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_UA, timeout=20); r.raise_for_status()
            _cik = {v["ticker"].upper(): v["cik_str"] for v in r.json().values()}
        cik = _cik.get(t.upper())
        if not cik: return None
        r = requests.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", headers=SEC_UA, timeout=20); r.raise_for_status()
        f = r.json()["filings"]["recent"]; cutoff = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%d")
        hits = [(f["form"][i], f["filingDate"][i]) for i in range(len(f["form"]))
                if re.match(r"^(S-3|S-1|F-3|F-1|424B5|424B4|424B7)", f["form"][i]) and f["filingDate"][i] >= cutoff]
        time.sleep(0.12)
        return hits[:3]
    except Exception as e:
        log.warning("edgar %s: %s", t, e); return None

# --------------------------------------------------------------------------- stage 1 · calls
def stage1_calls():
    c = CFG["calls"]
    q = EQ("and", [EQ("gt", ["short_percentage_of_float.value", c["si_min"]]),
                   EQ("btwn", ["intradayprice", c["price_min"], c["price_max"]]),
                   EQ("gt", ["avgdailyvol3m", c["avgvol_min"]]), EQ("eq", ["region", "us"])])
    quotes = screen(q, "short_percentage_of_float.value"); syms = [x["symbol"] for x in quotes]
    log.info("calls screener: %d names", len(syms))
    if not syms: raise RuntimeError("screener returned nothing")
    hist = batch_history(syms); rows = []
    for x in quotes:
        t = x["symbol"]
        try:
            h = frame_for(hist, t, len(syms))
            if len(h) < 60: continue
            cl = h["Close"].to_numpy(); vol = h["Volume"].to_numpy(); p = float(cl[-1])
            perf = lambda n: round((p / float(cl[-1 - n]) - 1) * 100, 2) if len(cl) > n else None
            sma = lambda n: round((p / float(cl[-n:].mean()) - 1) * 100, 2) if len(cl) >= n else None
            rs = rsi_series(cl[-60:]); av = float(vol[-63:].mean())
            rows.append(dict(t=t, co=x.get("longName") or x.get("shortName") or t, mc=x.get("marketCap"), sf=0.0, sr=0.0,
                             pw=perf(5), pm=perf(21), pq=perf(63), s20=sma(20), s50=sma(50), s200=sma(200),
                             hi=round((p / float(cl[-252:].max()) - 1) * 100, 2), lo=round((p / float(cl[-252:].min()) - 1) * 100, 2),
                             rsi=round(float(rs[-1]), 2), av=av, rv=round(float(vol[-1]) / (av or 1), 2), p=round(p, 2), earn="-"))
        except Exception as e: log.warning("calls stage1 %s: %s", t, e)
    for r in rows:
        try:
            info = yf.Ticker(r["t"]).info
            r["sf"] = round(float(info.get("shortPercentOfFloat") or 0) * 100, 2); r["sr"] = round(float(info.get("shortRatio") or 0), 2)
            r["earn"] = earn_label(info)
        except Exception as e: log.warning("info %s: %s", r["t"], e)
        time.sleep(0.15)
    keep = [x for x in rows if x["av"] >= c["avgvol_min"] and x["sf"] >= c["si_min"]]
    for x in keep:
        s = min(x["sf"], 60) / 60 * 35 + min(x["sr"] or 0, 10) / 10 * 10
        if 25 <= x["rsi"] <= 45: s += 15
        elif 45 < x["rsi"] <= 55: s += 8
        elif x["rsi"] < 25: s += 5
        if x["lo"] <= 25: s += 15
        elif x["lo"] <= 50: s += 8
        if (x["pq"] or 0) < -20: s += 8
        if (x["pw"] or 0) > -3: s += 6
        if (x["s20"] or 0) > 0: s += 6
        if x["rv"] >= 1.2: s += 5
        x["s1"] = round(s, 1)
    keep.sort(key=lambda x: -x["s1"])
    top = []
    for x in keep:
        if len(top) >= c["deep_n"]: break
        if has_options(x["t"]): top.append(x)
        time.sleep(0.1)
    log.info("calls stage1: universe=%d pass=%d deep=%d", len(rows), len(keep), len(top))
    return len(rows), len(keep), top

# --------------------------------------------------------------------------- stage 1 · puts
def stage1_puts():
    c = CFG["puts"]
    q = EQ("and", [EQ("btwn", ["intradayprice", c["price_min"], c["price_max"]]),
                   EQ("gt", ["avgdailyvol3m", c["avgvol_min"]]),
                   EQ("lt", ["short_percentage_of_float.value", c["si_max"]]),   # low short interest = no squeeze against you
                   EQ("eq", ["region", "us"])])
    quotes = screen(q, "avgdailyvol3m"); syms = [x["symbol"] for x in quotes]
    log.info("puts screener: %d names", len(syms))
    if not syms: raise RuntimeError("screener returned nothing")
    hist = batch_history(syms); rows = []
    for x in quotes:
        t = x["symbol"]
        try:
            h = frame_for(hist, t, len(syms))
            if len(h) < 60: continue
            cl = h["Close"].to_numpy(); hi = h["High"].to_numpy(); lo = h["Low"].to_numpy(); vol = h["Volume"].to_numpy(); p = float(cl[-1])
            perf = lambda n: round((p / float(cl[-1 - n]) - 1) * 100, 2) if len(cl) > n else None
            sma = lambda n: round((p / float(cl[-n:].mean()) - 1) * 100, 2) if len(cl) >= n else None
            rs = rsi_series(cl[-60:]); av = float(vol[-63:].mean())
            run5, run10, ext20, rsi = perf(5) or 0, perf(10) or 0, sma(20) or 0, float(rs[-1])
            # failed gap: gapped up ≥ gap_min% within the last 6 sessions and now trades below that day's close
            op = h["Open"].to_numpy(); gapfail = None
            for i in range(max(1, len(cl) - 6), len(cl)):
                if op[i] >= cl[i - 1] * (1 + c["gap_min"] / 100) and p < cl[i]:
                    gapfail = round((op[i] / cl[i - 1] - 1) * 100, 1)
            # "ran too far, too fast" — any trigger
            if not (run5 >= c["run5_min"] or run10 >= c["run10_min"] or ext20 >= c["ext20_min"] or gapfail): continue
            rows.append(dict(t=t, co=x.get("longName") or x.get("shortName") or t, mc=x.get("marketCap"), sf=0.0, sr=0.0,
                             pw=perf(5), p10=run10, pm=perf(21), pq=perf(63), s20=sma(20), s50=sma(50), s200=sma(200),
                             hi=round((p / float(cl[-252:].max()) - 1) * 100, 2), lo=round((p / float(cl[-252:].min()) - 1) * 100, 2),
                             rsi=round(rsi, 2), av=av, rv=round(float(vol[-1]) / (av or 1), 2), p=round(p, 2), earn="-",
                             runway=None, dil=None, ipo_days=None, gapfail=gapfail))
        except Exception as e: log.warning("puts stage1 %s: %s", t, e)
    log.info("puts: %d names passed the run filter", len(rows))
    for r in rows:
        try:
            info = yf.Ticker(r["t"]).info
            r["sf"] = round(float(info.get("shortPercentOfFloat") or 0) * 100, 2); r["sr"] = round(float(info.get("shortRatio") or 0), 2)
            r["earn"] = earn_label(info)
            cash = info.get("totalCash"); fcf = info.get("freeCashflow")
            if cash and fcf is not None and fcf < 0: r["runway"] = round(cash / (-fcf), 1)      # years of cash at current burn
            ft = info.get("firstTradeDateEpochUtc") or info.get("firstTradeDateMilliseconds")
            if ft:
                ft = ft / 1000 if ft > 1e11 else ft
                r["ipo_days"] = int((datetime.now(timezone.utc) - datetime.fromtimestamp(ft, timezone.utc)).days)
        except Exception as e: log.warning("info %s: %s", r["t"], e)
        time.sleep(0.15)
    keep = [x for x in rows if x["sf"] < c["si_max"]]
    for x in keep:
        s = min(max(x["pw"] or 0, x["p10"] or 0), 100) / 100 * 15 + clamp(x["s20"] or 0, 0, 40) / 40 * 8   # size of the run + extension
        if x["sf"] < 8: s += 4
        if x["runway"] is not None and x["runway"] < 1: s += 6
        elif x["runway"] is not None and x["runway"] < 2: s += 3
        if x["ipo_days"] is not None and 150 <= x["ipo_days"] <= 200: s += 6                      # lockup expiry window
        if x["rsi"] >= 70: s += 10
        elif x["rsi"] >= 60: s += 5
        if x["hi"] is not None and x["hi"] >= -5: s += 4                                          # at/near 52-wk high
        if x["rv"] >= 1.5: s += 4
        if x.get("gapfail"): s += 8
        x["s1"] = round(s, 1)
    keep.sort(key=lambda x: -x["s1"])
    top = []
    for x in keep:
        if len(top) >= c["deep_n"]: break
        if has_options(x["t"]): top.append(x)
        time.sleep(0.1)
    for x in top: x["dil"] = edgar_dilution(x["t"])
    log.info("puts stage1: universe=%d pass=%d deep=%d", len(syms), len(keep), len(top))
    return len(syms), len(keep), top

# --------------------------------------------------------------------------- stage 2 · deep scan (both sides)
def pick_contracts(tk, side, px, c):
    now = datetime.now(timezone.utc); lst = []
    exps = []
    for e in tk.options:
        dte = (datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) - now).days + 1
        if c["min_dte"] <= dte <= c["max_dte"]: exps.append((e, dte))
    for e, dte in exps[:3]:
        chain = tk.option_chain(e); table = chain.calls if side == "calls" else chain.puts
        for r in table.itertuples():
            ask = float(r.ask or 0); bid = float(r.bid or 0); last = float(r.lastPrice or 0); k = float(r.strike)
            otm = k >= px * 0.98 if side == "calls" else k <= px * 1.02
            if not (ask > 0 and ask <= c["max_ask"] and otm): continue
            oi = int(r.openInterest) if r.openInterest == r.openInterest else 0
            abs_sp = ask - bid; spread = abs_sp / ask
            if oi < c["min_oi"] or (spread > 0.75 and abs_sp > 0.10): continue
            be = ((k + ask) / px - 1) * 100 if side == "calls" else (1 - (k - ask) / px) * 100   # % move needed, positive number
            if be > c["max_be"] or not (bid > 0 or last > 0): continue
            iv = round(float(r.impliedVolatility) * 100) if r.impliedVolatility == r.impliedVolatility else None
            q = clamp(oi, 0, 2000) / 2000 * 10 + (1 - spread) * 8 + clamp(1 - (be - (15 if side == "calls" else 10)) / (45 if side == "calls" else 25), 0, 1) * 12 + (0 if dte >= 21 else -2)
            lst.append(dict(exp=e, dte=dte, strike=k, bid=round(bid, 2), ask=round(ask, 2), last=round(last, 2), oi=oi,
                            vol=int(r.volume) if r.volume == r.volume else 0, iv=iv, be=round(be, 1), spread=round(spread * 100), q=round(q, 1)))
        time.sleep(0.25)
    lst.sort(key=lambda z: -z["q"])
    return lst

def deep(x, side):
    c = CFG[side]; o = dict(x); o["err"] = ""
    try:
        tk = yf.Ticker(x["t"])
        h = tk.history(period="3mo", interval="1d", auto_adjust=False).dropna(subset=["Close"])
        if len(h) < 25: raise RuntimeError("not enough price history")
        cl = h["Close"].to_numpy(); hi = h["High"].to_numpy(); lo = h["Low"].to_numpy(); vol = h["Volume"].to_numpy()
        px = round(float(cl[-1]), 2); o["px"] = px
        rs = rsi_series(cl)
        o["rsi"] = round(rs[-1], 1); o["rsi3"] = round(rs[-4], 1)
        o["vr"] = round(float(vol[-3:].mean() / (vol[-23:-3].mean() or 1)), 2)
        o["up3d"] = round(float(cl[-1] / cl[-4] - 1) * 100, 1)
        o["closes"] = [round(float(v), 2) for v in cl[-30:]]
        contracts = pick_contracts(tk, side, px, c)
        o["play"] = contracts[0] if contracts else None; o["alts"] = contracts[1:3]
        opt = o["play"]["q"] if o["play"] else 0
        flags = []
        if side == "calls":
            o["rsiUp"] = o["rsi"] > o["rsi3"]; o["wasOversold"] = min(rs[-10:]) < 35
            o["low20"] = round(float(cl[-1] / cl[-20:].min() - 1) * 100, 1)
            fuel = clamp(o["sf"] or 0, 0, 50) / 50 * 25 + clamp(o["sr"] or 0, 0, 10) / 10 * 10
            bot = 0
            if 25 <= o["rsi"] <= 45: bot += 10
            elif o["rsi"] < 25 or o["rsi"] <= 52: bot += 4
            if o["rsiUp"]: bot += 10
            if o["wasOversold"]: bot += 5
            if o["low20"] <= 8 or (o["lo"] is not None and o["lo"] <= 15): bot += 5
            if o["vr"] >= 1.3: bot += 5
            s = fuel + bot + opt - (5 if o["up3d"] < -10 else 0)
            if o["up3d"] < -10: flags.append(f"still falling ({o['up3d']}% / 3d)")
            if o["rsiUp"]: flags.append("RSI turning up")
            if o["vr"] >= 1.3: flags.append(f"volume pickup {o['vr']}x")
            o.update(fuel=round(fuel, 1), bot=bot)
        else:
            o["rsiDown"] = o["rsi"] < o["rsi3"]
            o["high20"] = round(float(cl[-1] / hi[-20:].max() - 1) * 100, 1)                 # % below 20-day high (≤ 0)
            o["reversal"] = bool(cl[-1] < lo[-2])                                            # closed under yesterday's low
            o["nohigh3"] = bool(hi[-3:].max() < hi[-10:-3].max())                             # no new high in 3 sessions
            peak_v = float(vol[-10:].max()); o["volFade"] = bool(vol[-3:].mean() < 0.6 * peak_v)
            run = max(x.get("pw") or 0, x.get("p10") or 0)
            press = clamp(run, 0, 100) / 100 * 12 + clamp(x.get("s20") or 0, 0, 40) / 40 * 8
            if (x.get("sf") or 0) < 8: press += 3
            if x.get("runway") is not None and x["runway"] < 1: press += 6
            elif x.get("runway") is not None and x["runway"] < 2: press += 3
            if x.get("dil"): press += 6
            if x.get("ipo_days") is not None and 150 <= x["ipo_days"] <= 200: press += 4
            press = min(press, 35)
            top_ = 0
            if o["rsi"] >= 70: top_ += 8
            elif o["rsi"] >= 60: top_ += 4
            if o["rsiDown"]: top_ += 8
            if o["reversal"]: top_ += 7
            if o["nohigh3"]: top_ += 5
            if o["volFade"]: top_ += 4
            if o["high20"] <= -5: top_ += 3                                                   # already rolling over
            if x.get("gapfail"): top_ += 6                                                     # gap up that failed = trapped buyers
            top_ = min(top_, 35)
            s = press + top_ + opt - (5 if o["up3d"] > 10 else 0)                                # still ripping = don't step in front
            if o["up3d"] > 10: flags.append(f"still ripping (+{o['up3d']}% / 3d)")
            if o["rsiDown"]: flags.append("RSI rolling over")
            if o["reversal"]: flags.append("reversal day")
            if o["volFade"]: flags.append("volume fading")
            if x.get("gapfail"): flags.append(f"failed gap (+{x['gapfail']}%)")
            if x.get("dil"): flags.append("dilution: " + ", ".join(f"{f} {d}" for f, d in x["dil"][:2]))
            if x.get("runway") is not None and x["runway"] < 1: flags.append(f"cash runway {x['runway']}y")
            if x.get("ipo_days") is not None and 150 <= x["ipo_days"] <= 200: flags.append("lockup window")
            o.update(fuel=round(press, 1), bot=top_)
        if not o["play"]: flags.insert(0, f"no tradeable {'call' if side == 'calls' else 'put'} ≤ ${c['max_ask']:.2f}")
        m = datetime.now(ET).month; mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        if re.match(f"^({mn[m-1]}|{mn[m % 12]})", o.get("earn") or ""): flags.append(f"earnings {o['earn']}")
        o.update(flags=flags, opt=round(opt, 1), score=round(s if o["play"] else min(s, 40), 1))
    except Exception as e:
        log.warning("%s: %s", x["t"], e); o["err"] = str(e)[:120]; o["score"] = -1
    return o

def stage2(top, side):
    out = []
    for i, x in enumerate(top):
        out.append(deep(x, side)); time.sleep(0.4)
        if i % 10 == 9: log.info("deep %d/%d", i + 1, len(top))
    out.sort(key=lambda o: (0 if o.get("play") else 1, -o["score"]))   # names with a real contract rank first
    return out

def _clean(v):
    if isinstance(v, dict): return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_clean(x) for x in v]
    if isinstance(v, np.floating): return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, float) and v != v: return None
    return v

def assemble(side, mode, universe, pass1, top_in, deep_out, full_asof):
    keys = ("t","co","px","sf","sr","rsi","rsi3","lo","hi","low20","high20","up3d","vr","pq","pm","pw","p10","s20","av","mc","earn",
            "runway","dil","ipo_days","gapfail","fuel","bot","opt","score","flags","play","alts","closes","err")
    top = [{k: o.get(k) for k in keys} for o in deep_out[:16]]
    for o in top: o["lo52"] = o.pop("lo"); o["hi52"] = o.pop("hi")
    rest = [[o["t"], o.get("px"), o.get("sf"), o.get("rsi"), o.get("score"), 1 if o.get("play") else 0] for o in deep_out[16:]]
    return _clean(dict(side=side, asof=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                       full_asof=full_asof, mode=mode, universe=universe, pass1=pass1, deep=len(deep_out),
                       stage1=top_in, top=top, rest=rest, cfg=CFG[side]))

def run_side(side, mode, now_et):
    data = DATA_ROOT if side == "calls" else DATA_ROOT / "puts"
    data.mkdir(parents=True, exist_ok=True); (data / "history").mkdir(exist_ok=True)
    today = now_et.strftime("%Y-%m-%d"); prev = None
    if (data / "latest.json").exists():
        try: prev = json.load(open(data / "latest.json"))
        except Exception: pass
    if mode == "auto":
        mode = "refresh" if (prev and (prev.get("full_asof") or "")[:10] == today) else "full"
    if mode == "refresh" and not (prev and prev.get("stage1")):
        mode = "full"
    log.info("== %s · %s", side, mode)
    if mode == "full":
        universe, pass1, top = (stage1_calls if side == "calls" else stage1_puts)()
        full_asof = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    else:
        universe, pass1, top, full_asof = prev["universe"], prev["pass1"], prev["stage1"], prev.get("full_asof")
    out = assemble(side, mode, universe, pass1, top, stage2(top, side), full_asof)
    json.dump(out, open(data / "latest.json", "w"), separators=(",", ":"))
    if mode == "full":
        json.dump(out, open(data / "history" / f"{today}.json", "w"), separators=(",", ":"))
        json.dump(sorted(p.stem for p in (data / "history").glob("*.json") if p.stem != "index"), open(data / "history" / "index.json", "w"))
    log.info("%s %s done: %d deep, top: %s", side, mode, out["deep"], ", ".join(f"{o['t']} {o['score']}" for o in out["top"][:5]))

def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "auto").lower()
    side = (sys.argv[2] if len(sys.argv) > 2 else "both").lower()
    now_et = datetime.now(ET)
    if mode == "auto" and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        mode = "full"
    if mode == "auto":
        hhmm = now_et.hour * 100 + now_et.minute
        if now_et.weekday() >= 5 or not (825 <= hhmm <= 1610):
            log.info("auto: outside 08:25–16:10 ET weekday window — skipping"); return 0
    for s in (["calls", "puts"] if side == "both" else [side]):
        try: run_side(s, mode, now_et)
        except Exception as e: log.error("%s side failed: %s", s, e)
    return 0

if __name__ == "__main__":
    sys.exit(main())


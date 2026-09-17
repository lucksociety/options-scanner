#!/usr/bin/env python3
"""
Luck Society Option Scanner — three boards, one engine.

  calls     Heavily shorted, sub-$10 stocks that look bottomed → best OTM call ≤ $0.25, 2–6 weeks out.
  breakout  The same heavily shorted, sub-$10 universe, but already turned: higher highs, above the 20/50-day,
            coiling or pressing the 20-day high on rising volume → best OTM call ≤ $0.25.
  puts      Overextended $5–$30 stocks with a forced seller behind them (trapped momentum buyers, dilution,
            lockup expiry) that look like they're topping → best OTM put ≤ $0.35, 3–6 weeks out.

Usage:  python scanner/scan.py <mode> [side]
  mode   full | refresh | auto | daily      (daily: one full scan per trading day, exit 3 if today's exists;
                                             auto: full first run after 08:25 ET, refresh until 16:10 ET, else skip)
  side   calls | puts | breakout | both     (default both = all three)

Data: Yahoo Finance via yfinance (screener, batched history, key stats, option chains) + SEC EDGAR (filings)
      + iBorrowDesk (borrow fee / shares available, Interactive Brokers data) + our own daily IV log (data/iv/).
Output: data/latest.json (calls), data/<side>/latest.json for the others, plus data/<side>/history/YYYY-MM-DD.json after a full scan.
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

from market import regime

log = logging.getLogger("scan")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
SEC_UA = {"User-Agent": "LuckSocietyOptionScanner/1.0 (lucksociety@users.noreply.github.com)",   # SEC fair-access policy wants name + contact
          "Accept-Encoding": "gzip, deflate", "Accept": "application/json, text/plain, */*"}

SIDES = ("calls", "puts", "breakout")
MKT = None          # market regime, scored once per run (scanner/market.py)

CFG = {
    "calls": dict(price_min=1.0, price_max=10.0, si_min=15.0, avgvol_min=300_000, dv_min=1.0, deep_n=40,   # dv_min = $M traded a day
                  min_dte=14, max_dte=45, max_ask=0.25, min_oi=25, max_be=60.0),
    # same universe as calls (heavily shorted and cheap) but timed off momentum instead of a bottom
    "breakout": dict(price_min=1.0, price_max=10.0, si_min=15.0, avgvol_min=300_000, dv_min=1.0, deep_n=40,
                     min_dte=14, max_dte=45, max_ask=0.35, min_oi=25, max_be=60.0),
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

_cik = None          # None = not loaded yet; {} = SEC unreachable from this runner, stop trying
def _load_cik():
    """Ticker -> CIK map. SEC blocks some cloud IPs outright; if so we give up once instead of per-ticker."""
    global _cik
    if _cik is not None: return _cik
    for url, parse in (("https://www.sec.gov/files/company_tickers.json",
                        lambda r: {v["ticker"].upper(): v["cik_str"] for v in r.json().values()}),
                       ("https://www.sec.gov/include/ticker.txt",
                        lambda r: {a.upper(): int(b) for a, b in (ln.split("\t") for ln in r.text.strip().splitlines() if "\t" in ln)})):
        try:
            r = requests.get(url, headers=SEC_UA, timeout=20)
            if r.status_code == 200:
                _cik = parse(r); log.info("edgar: loaded %d CIKs", len(_cik)); return _cik
            log.warning("edgar: %s -> HTTP %s", url.rsplit("/", 1)[-1], r.status_code)
        except Exception as e:
            log.warning("edgar: %s -> %s", url.rsplit("/", 1)[-1], e)
        time.sleep(0.5)
    # Diagnose once: is data.sec.gov reachable even though www.sec.gov isn't? (CIK 1318605 = Tesla)
    try:
        probe = requests.get("https://data.sec.gov/submissions/CIK0001318605.json", headers=SEC_UA, timeout=20)
        log.warning("edgar: ticker map unavailable; data.sec.gov probe -> HTTP %s", probe.status_code)
    except Exception as e:
        log.warning("edgar: ticker map unavailable; data.sec.gov probe -> %s", e)
    _cik = {}
    return _cik

def edgar_recent(t):
    """SEC filings that matter: shelf/offering forms in the last 120 days (dilution) and 8-Ks in the last 5 days (news catalyst)."""
    cikmap = _load_cik()
    if not cikmap: return None, None            # SEC unreachable from this runner — skip silently
    cik = cikmap.get(t.upper())
    if not cik: return None, None
    try:
        r = requests.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", headers=SEC_UA, timeout=20); r.raise_for_status()
        f = r.json()["filings"]["recent"]; now = datetime.now(timezone.utc)
        c120 = (now - timedelta(days=120)).strftime("%Y-%m-%d"); c5 = (now - timedelta(days=5)).strftime("%Y-%m-%d")
        dil = [(f["form"][i], f["filingDate"][i]) for i in range(len(f["form"]))
               if re.match(r"^(S-3|S-1|F-3|F-1|424B5|424B4|424B7)", f["form"][i]) and f["filingDate"][i] >= c120]
        k8 = [f["filingDate"][i] for i in range(len(f["form"])) if f["form"][i].startswith("8-K") and f["filingDate"][i] >= c5]
        time.sleep(0.12)
        return dil[:3], (k8[0] if k8 else None)
    except Exception as e:
        log.warning("edgar %s: %s", t, e); return None, None

def edgar_dilution(t):
    return edgar_recent(t)[0]

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/140.0.0.0 Safari/537.36")
_borrow_note = [0]     # log the first failure reason once instead of 40 silent misses
def borrow_info(t):
    """iBorrowDesk (Interactive Brokers stock-loan data): borrow fee % and shares available. Free, no key.
    Needs the www. host AND a browser User-Agent — Cloudflare 403s unknown agents."""
    try:
        r = requests.get(f"https://www.iborrowdesk.com/api/ticker/{t}",
                         headers={"User-Agent": BROWSER_UA, "Accept": "application/json, text/plain, */*",
                                  "Referer": f"https://www.iborrowdesk.com/report/{t}"}, timeout=15)
        if r.status_code != 200:
            if not _borrow_note[0]: _borrow_note[0] = 1; log.warning("borrow: %s -> HTTP %s", t, r.status_code)
            return None
        j = r.json()
        fee, avail, asof = j.get("latest_fee"), j.get("latest_available"), str(j.get("updated") or "")[:16]
        if fee is None:
            rows = j.get("real_time") or j.get("daily") or []
            if not rows: return None
            last = rows[-1]; fee = last.get("fee"); avail = last.get("available"); asof = str(last.get("datetime") or last.get("date") or "")[:16]
        if fee is None: return None
        return dict(fee=round(float(fee), 1), avail=int(float(avail or 0)), asof=asof)
    except Exception as e:
        if not _borrow_note[0]: _borrow_note[0] = 1; log.warning("borrow: %s -> %s", t, e)
        return None

def share_growth(t):
    """Realized dilution straight from Yahoo: % change in shares outstanding over ~6 months.
    Replaces the SEC shelf-filing signal, which is unreachable from GitHub runners (sec.gov 403s their IPs).
    Measures dilution that actually happened rather than dilution a company is merely allowed to do."""
    try:
        end = datetime.now(timezone.utc); start = end - timedelta(days=200)
        tk = yf.Ticker(t)
        sp = tk.splits                                   # a split rewrites the share count: the signal would be nonsense
        if sp is not None and len(sp):
            idx = sp.index
            try: idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
            except Exception: pass
            if (idx >= pd.Timestamp(start.replace(tzinfo=None))).any(): return None
        ser = tk.get_shares_full(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if ser is None or len(ser) < 2: return None
        ser = ser.dropna()
        if len(ser) < 2: return None
        first, last = float(ser.iloc[0]), float(ser.iloc[-1])
        if first <= 0: return None
        return round((last / first - 1) * 100, 1)
    except Exception as e:
        log.debug("shares %s: %s", t, e); return None

def hv_percentile(cl, n=20):
    """Where today's 20-day realized vol sits in its 1-year range (0-100). Proxy for IV rank until our own IV log is long enough."""
    c = np.asarray(cl, dtype=float)
    if len(c) < n + 30: return None
    lr = np.diff(np.log(c)); hv = pd.Series(lr).rolling(n).std().dropna().to_numpy() * math.sqrt(252) * 100
    if len(hv) < 30: return None
    return int(round((hv < hv[-1]).mean() * 100)), round(float(hv[-1]), 1)

def _ts(v):
    try: return datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d") if v else None
    except Exception: return None

def key_stats(info, r):
    """Float, ownership and — the part that matters most — the two most recent EXCHANGE short-interest
    reports, so we can show the reported trend and the date it was measured instead of implying it is live."""
    r["flt"] = info.get("floatShares"); r["shout"] = info.get("sharesOutstanding")
    r["ins"] = round(float(info.get("heldPercentInsiders") or 0) * 100, 1)
    r["inst"] = round(float(info.get("heldPercentInstitutions") or 0) * 100, 1)
    r["si_date"] = _ts(info.get("dateShortInterest"))
    ss, ssp = info.get("sharesShort"), info.get("sharesShortPriorMonth")
    r["ss"], r["ssp"] = ss, ssp
    r["ssp_date"] = _ts(info.get("sharesShortPreviousMonthDate"))
    r["si_rep"] = round((ss / ssp - 1) * 100, 1) if (ss and ssp) else None   # change between the two reports
    r["exch"] = info.get("exchange")                                          # NMS/NGM/NCM = Nasdaq, NYQ = NYSE, ASE = NYSE American
    r["state"] = info.get("marketState")                                      # PRE / REGULAR / POST / CLOSED
    # Yahoo's shortPercentOfFloat is sometimes missing or zero while the raw counts are fine: derive it rather than drop the name.
    if not r.get("sf") and ss and r["flt"]:
        r["sf"] = round(ss / r["flt"] * 100, 2); r["sf_src"] = "derived"

_regsho = None
def regsho():
    """Reg SHO threshold lists (persistent failures to deliver). Each exchange publishes its own; SEC's copy
    blocks our runner. Returns {"nasdaq": set|None, "nyse": set|None} — None means that list could not be
    loaded, so names on that exchange read as UNKNOWN, never as clean."""
    global _regsho
    if _regsho is not None: return _regsho
    _regsho = {"nasdaq": None, "nyse": None}
    for back in (0, 1, 2, 3, 4):
        d = datetime.now(ET) - timedelta(days=back); ds = d.strftime("%Y%m%d")
        if _regsho["nasdaq"] is None:
            try:
                r = requests.get(f"https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{ds}.txt",
                                 headers={"User-Agent": BROWSER_UA}, timeout=15)
                if r.status_code == 200 and "|" in r.text:
                    _regsho["nasdaq"] = {ln.split("|")[0].strip().upper() for ln in r.text.splitlines()[1:] if "|" in ln}
            except Exception as e: log.debug("regsho nasdaq %s: %s", ds, e)
        if _regsho["nyse"] is None:
            try:
                r = requests.get("https://www.nyse.com/api/regulatory/threshold-securities/download",
                                 params={"selectedDate": d.strftime("%Y-%m-%d"), "market": "ALL"},
                                 headers={"User-Agent": BROWSER_UA}, timeout=15)
                if r.status_code == 200 and "|" in r.text:
                    _regsho["nyse"] = {ln.split("|")[0].strip().upper() for ln in r.text.splitlines()[1:] if "|" in ln}
            except Exception as e: log.debug("regsho nyse %s: %s", ds, e)
        if _regsho["nasdaq"] is not None and _regsho["nyse"] is not None: break
    log.info("reg sho: nasdaq %s · nyse %s", *(("%d names" % len(v)) if v is not None else "unavailable" for v in (_regsho["nasdaq"], _regsho["nyse"])))
    return _regsho

def on_threshold(t, exch):
    """True / False / None(unknown) for this ticker given its listing exchange."""
    th = regsho(); t = t.upper()
    lst = th["nasdaq"] if (exch or "").upper() in ("NMS", "NGM", "NCM", "NAS") else th["nyse"]
    if lst is None:
        other = th["nyse"] if lst is th["nasdaq"] else th["nasdaq"]
        return True if (other and t in other) else None
    return t in lst

def attention(t):
    """StockTwits messages in the last 24h for this symbol. Free and keyless; the public stream endpoint
    returns the 30 most recent messages with timestamps, so the count saturates at 30 — plenty for a
    sub-$10 name, where 30 posts in a day IS the signal. Called in full mode only (rate limits)."""
    try:
        r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{t}.json",
                         headers={"User-Agent": BROWSER_UA, "Accept": "application/json"}, timeout=12)
        if r.status_code != 200: return None
        msgs = r.json().get("messages") or []
        cut = datetime.now(timezone.utc) - timedelta(hours=24)
        n = 0
        for m in msgs:
            try:
                ts = datetime.strptime(m["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if ts >= cut: n += 1
            except Exception: pass
        return n
    except Exception as e:
        log.debug("attention %s: %s", t, e); return None

def zscore(kind, t, value, days=30):
    """Log today's value and return (z vs our own history, mean) once we have >= 7 samples. Rate of change
    of attention is what matters, never the level: 20->400 mentions beats 8000->8100."""
    try:
        if value is None: return None, None
        d = DATA_ROOT / kind; d.mkdir(parents=True, exist_ok=True); f = d / f"{t}.json"
        hist = json.load(open(f)) if f.exists() else {}
        hist[datetime.now(ET).strftime("%Y-%m-%d")] = int(value)
        hist = dict(sorted(hist.items())[-120:]); json.dump(hist, open(f, "w"), separators=(",", ":"))
        cutoff = (datetime.now(ET) - timedelta(days=days)).strftime("%Y-%m-%d"); today = datetime.now(ET).strftime("%Y-%m-%d")
        past = [v for k, v in sorted(hist.items()) if cutoff <= k < today]
        if len(past) < 7: return None, None
        mu = float(np.mean(past)); sd = float(np.std(past)) or 1.0
        return round((value - mu) / sd, 2), round(mu, 1)
    except Exception as e:
        log.debug("zscore %s %s: %s", kind, t, e); return None, None

def log_metric(kind, t, value, days=30):
    """Append today's value to data/<kind>/<T>.json and report the change vs the oldest sample in the window.
    The LEVEL says how crowded a short is; the TREND says whether shorts are actually being forced."""
    try:
        if value is None: return None
        d = DATA_ROOT / kind; d.mkdir(parents=True, exist_ok=True); f = d / f"{t}.json"
        hist = json.load(open(f)) if f.exists() else {}
        hist[datetime.now(ET).strftime("%Y-%m-%d")] = round(float(value), 3)
        hist = dict(sorted(hist.items())[-260:]); json.dump(hist, open(f, "w"), separators=(",", ":"))
        cutoff = (datetime.now(ET) - timedelta(days=days)).strftime("%Y-%m-%d")
        win = [v for k, v in sorted(hist.items()) if k >= cutoff]
        if len(win) < 2: return None
        return round(float(value) - win[0], 2)
    except Exception as e:
        log.debug("log_metric %s %s: %s", kind, t, e); return None

def float_pts(fl):
    """Squeeze mechanics: the same short interest is far more dangerous against a small tradable float."""
    if not fl: return 0, None
    m = fl / 1e6
    return (8 if m < 10 else 6 if m < 25 else 4 if m < 50 else 2 if m < 100 else 1 if m < 300 else 0), round(m, 1)

def gamma_setup(chain, px, adv=None):
    """Dealer-hedging fuel: calls stacked just above spot that market makers must hedge into if price gets there."""
    try:
        cs, ps = chain.calls, chain.puts
        coi = float(cs["openInterest"].fillna(0).sum()); poi = float(ps["openInterest"].fillna(0).sum())
        cvol = float(cs["volume"].fillna(0).sum()); pvol = float(ps["volume"].fillna(0).sum())
        band = cs[(cs["strike"] >= px) & (cs["strike"] <= px * 1.20)]
        ramp = float(band["openInterest"].fillna(0).sum()) * 100        # shares behind those calls
        return dict(coi=int(coi), poi=int(poi), cp=round(coi / poi, 2) if poi else None,
                    cvol=int(cvol), pvol=int(pvol), cpv=round(cvol / pvol, 2) if pvol else None,
                    ramp=int(ramp), ramp_adv=round(ramp / adv, 2) if adv else None)
    except Exception as e:
        log.debug("gamma: %s", e); return None

def gamma_pts(g):
    """0-8, matching the weight the playbook gives options positioning."""
    if not g: return 0, []
    pts, fl = 0, []
    ra = g.get("ramp_adv")
    if ra is not None:
        if ra >= 1.0: pts += 4; fl.append(f"call wall {ra}x daily volume")
        elif ra >= 0.4: pts += 2; fl.append(f"call wall {ra}x daily volume")
    cp = g.get("cp")
    if cp is not None:
        if cp >= 2.5: pts += 2; fl.append(f"calls {cp}x puts (OI)")
        elif cp >= 1.3: pts += 1
    cpv = g.get("cpv")
    if cpv is not None and cpv >= 2.5: pts += 2; fl.append(f"call volume {cpv}x puts today")
    return min(pts, 8), fl

IV_DIR = DATA_ROOT / "iv"
def iv_rank(t, iv_now):
    """Log today's ATM IV for this ticker and return its rank in our own history (needs >= 20 days), else None."""
    try:
        IV_DIR.mkdir(parents=True, exist_ok=True); f = IV_DIR / f"{t}.json"
        hist = json.load(open(f)) if f.exists() else {}
        if iv_now: hist[datetime.now(ET).strftime("%Y-%m-%d")] = iv_now
        hist = dict(sorted(hist.items())[-260:]); json.dump(hist, open(f, "w"), separators=(",", ":"))
        vals = list(hist.values())
        if len(vals) < 20 or not iv_now: return None
        lo, hi = min(vals), max(vals)
        return int(round((iv_now - lo) / (hi - lo) * 100)) if hi > lo else 50
    except Exception as e:
        log.debug("iv log %s: %s", t, e); return None

def premarket(info):
    """Yahoo pre-market quote, only while the pre-market session is actually on. Yahoo keeps the last
    pre-market print in `preMarketPrice` all day, so without the marketState check a 7am tick was
    scoring bottoming points and a "pre-market +x%" flag at 3pm."""
    if (info.get("marketState") or "").upper() != "PRE": return None
    p = info.get("preMarketPrice"); prev = info.get("regularMarketPreviousClose") or info.get("previousClose")
    if p and prev: return round(float(p), 2), round((float(p) / float(prev) - 1) * 100, 1)
    return None

# --------------------------------------------------------------------------- stage 1 · calls
def stage1_calls(kind="calls"):
    """One screen, two rankings: 'calls' wants the shorted names that have stopped falling,
    'breakout' wants the shorted names that have already turned and are pressing highs."""
    c = CFG[kind]
    q = EQ("and", [EQ("gt", ["short_percentage_of_float.value", c["si_min"]]),
                   EQ("btwn", ["intradayprice", c["price_min"], c["price_max"]]),
                   EQ("gt", ["avgdailyvol3m", c["avgvol_min"]]), EQ("eq", ["region", "us"])])
    quotes = screen(q, "short_percentage_of_float.value"); syms = [x["symbol"] for x in quotes]
    log.info("%s screener: %d names", kind, len(syms))
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
            rs = rsi_series(cl[-60:]); av = float(vol[-63:].mean()); hvp = hv_percentile(cl)
            av10 = float(vol[-10:].mean()); hi_ = h["High"].to_numpy(); lo_ = h["Low"].to_numpy()
            tr = np.maximum(hi_[-15:] - lo_[-15:], np.maximum(abs(hi_[-15:] - cl[-16:-1]), abs(lo_[-15:] - cl[-16:-1])))
            rows.append(dict(t=t, co=x.get("longName") or x.get("shortName") or t, mc=x.get("marketCap"), sf=0.0, sr=0.0,
                             pw=perf(5), pm=perf(21), pq=perf(63), s20=sma(20), s50=sma(50), s200=sma(200),
                             hi=round((p / float(cl[-252:].max()) - 1) * 100, 2), lo=round((p / float(cl[-252:].min()) - 1) * 100, 2),
                             h20=round((p / float(hi_[-20:].max()) - 1) * 100, 2), h50=round((p / float(hi_[-50:].max()) - 1) * 100, 2),
                             atr=round(float(tr.mean()) / p * 100, 2),                       # ATR(14) as % of price: how big a normal day is
                             dv=round(av * p / 1e6, 2),                                       # average daily dollar volume, $M
                             rv10=round(float(vol[-1]) / (av10 or 1), 2),
                             rsi=round(float(rs[-1]), 2), av=av, rv=round(float(vol[-1]) / (av or 1), 2), p=round(p, 2), earn="-",
                             hvp=hvp[0] if hvp else None, hv=hvp[1] if hvp else None, earn_ts=None))
        except Exception as e: log.warning("%s stage1 %s: %s", kind, t, e)
    for r in rows:
        try:
            info = yf.Ticker(r["t"]).info
            r["sf"] = round(float(info.get("shortPercentOfFloat") or 0) * 100, 2); r["sr"] = round(float(info.get("shortRatio") or 0), 2)
            r["earn"] = earn_label(info); r["earn_ts"] = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
            key_stats(info, r)
        except Exception as e: log.warning("info %s: %s", r["t"], e)
        time.sleep(0.15)
    keep = [x for x in rows if x["av"] >= c["avgvol_min"] and x["sf"] >= c["si_min"] and (x.get("dv") or 0) >= c.get("dv_min", 0)]
    for x in keep:
        x["ftd"] = on_threshold(x["t"], x.get("exch"))
        # Squeeze fuel is scored here too, not only after the cut: otherwise a 15M-float name at 30% short
        # loses its deep-scan slot to a 300M-float name at 35% and the float weight never gets a say.
        s = min(x["sf"], 60) / 60 * 30 + min(x["sr"] or 0, 10) / 10 * 8 + float_pts(x.get("flt"))[0]
        sir = x.get("si_rep")
        if sir is not None: s += 4 if sir >= 25 else (2 if sir >= 10 else 0)
        if x.get("ftd"): s += 3
        if kind == "calls":                                   # bottoming: beaten down, starting to turn
            if 25 <= x["rsi"] <= 45: s += 15
            elif 45 < x["rsi"] <= 55: s += 8
            elif x["rsi"] < 25: s += 5
            if x["lo"] <= 25: s += 15
            elif x["lo"] <= 50: s += 8
            if (x["pq"] or 0) < -20: s += 8
            if (x["pw"] or 0) > -3: s += 6
            if (x["s20"] or 0) > 0: s += 6
            if x["rv"] >= 1.2: s += 5
        else:                                                 # breakout: already moving, pressing highs
            if 55 <= x["rsi"] <= 72: s += 15
            elif 45 <= x["rsi"] < 55: s += 8
            elif x["rsi"] > 72: s += 3
            if (x["hi"] if x["hi"] is not None else -100) >= -15: s += 12
            elif (x["hi"] if x["hi"] is not None else -100) >= -30: s += 6
            if (x["s20"] or 0) > 0: s += 8
            if (x["s50"] or 0) > 0: s += 6
            if (x["pw"] or 0) > 3: s += 6
            if x["rv"] >= 1.5: s += 8
            elif x["rv"] >= 1.2: s += 4
        x["s1"] = round(s, 1)
    keep.sort(key=lambda x: -x["s1"])
    top = []
    for x in keep:
        if len(top) >= c["deep_n"]: break
        if has_options(x["t"]): top.append(x)
        time.sleep(0.1)
    for x in top:
        x["borrow"] = borrow_info(x["t"]); time.sleep(0.2)
        x["dil"], x["k8"] = edgar_recent(x["t"])
        x["shrg"] = share_growth(x["t"])
        x["att"] = attention(x["t"]); x["att_z"], x["att_mu"] = zscore("att", x["t"], x["att"]); time.sleep(0.3)
    log.info("%s stage1: universe=%d pass=%d deep=%d · borrow %d/%d, filings %d/%d", kind, len(rows), len(keep), len(top),
             sum(1 for x in top if x.get("borrow")), len(top), sum(1 for x in top if x.get("dil") is not None), len(top))
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
            rs = rsi_series(cl[-60:]); av = float(vol[-63:].mean()); hvp = hv_percentile(cl)
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
                             runway=None, dil=None, ipo_days=None, gapfail=gapfail,
                             hvp=hvp[0] if hvp else None, hv=hvp[1] if hvp else None, earn_ts=None))
        except Exception as e: log.warning("puts stage1 %s: %s", t, e)
    log.info("puts: %d names passed the run filter", len(rows))
    for r in rows:
        try:
            info = yf.Ticker(r["t"]).info
            r["sf"] = round(float(info.get("shortPercentOfFloat") or 0) * 100, 2); r["sr"] = round(float(info.get("shortRatio") or 0), 2)
            r["earn"] = earn_label(info); r["earn_ts"] = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
            key_stats(info, r)
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
    for x in top:
        x["dil"], x["k8"] = edgar_recent(x["t"])
        x["shrg"] = share_growth(x["t"])
        x["borrow"] = borrow_info(x["t"]); time.sleep(0.2)
        x["ftd"] = on_threshold(x["t"], x.get("exch"))
    log.info("puts stage1: universe=%d pass=%d deep=%d · borrow %d/%d, filings %d/%d, share-count %d/%d", len(syms), len(keep), len(top),
             sum(1 for x in top if x.get("borrow")), len(top), sum(1 for x in top if x.get("dil") is not None), len(top),
             sum(1 for x in top if x.get("shrg") is not None), len(top))
    return len(syms), len(keep), top

# --------------------------------------------------------------------------- stage 2 · deep scan (both sides)
def pick_contracts(tk, side, px, c, ivpen=0, adv=None):
    now = datetime.now(timezone.utc); lst = []; atm_iv = None; gam = None
    exps = []
    for e in tk.options:
        dte = (datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) - now).days + 1
        if c["min_dte"] <= dte <= c["max_dte"]: exps.append((e, dte))
    for e, dte in exps[:3]:
        chain = tk.option_chain(e)
        if gam is None: gam = gamma_setup(chain, px, adv)      # nearest expiry carries the most gamma
        table = chain.puts if side == "puts" else chain.calls
        if atm_iv is None and len(table):
            try:
                near = table.iloc[(table["strike"] - px).abs().argsort()[:1]]
                v = float(near["impliedVolatility"].iloc[0]); atm_iv = round(v * 100) if v == v and v > 0 else None
            except Exception: pass
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
            q = clamp(oi, 0, 2000) / 2000 * 10 + (1 - spread) * 8 + clamp(1 - (be - (15 if side == "calls" else 10)) / (45 if side == "calls" else 25), 0, 1) * 12 + (0 if dte >= 21 else -2) - ivpen
            lst.append(dict(exp=e, dte=dte, strike=k, bid=round(bid, 2), ask=round(ask, 2), last=round(last, 2), oi=oi,
                            vol=int(r.volume) if r.volume == r.volume else 0, iv=iv, be=round(be, 1), spread=round(spread * 100), q=round(q, 1)))
        time.sleep(0.25)
    lst.sort(key=lambda z: -z["q"])
    return lst, atm_iv, gam

# Measured on 2y of daily history for today's shorted $1-10 universe (211 names, 9,267 pattern days; survivorship-biased,
# entry = next day's open). Baseline for a random day on these names: 34% reach +20% within 20 sessions, 11% reach +50%.
TRIG_STATS = {"strong":   dict(hit20=52.5, hit50=24.3, best=21.7, mae=-10.2, close=-6.7, n=1080),
              "standard": dict(hit20=46.5, hit50=19.1, best=17.6, mae=-8.2,  close=-5.4, n=2372),
              "armed":    dict(hit20=38.3, hit50=12.3, best=14.4, mae=-5.9,  close=-4.0, n=9267),
              "baseline": dict(hit20=34.2, hit50=10.6, best=13.0, mae=-5.0,  close=-4.6, n=1513), "asof": "2026-09-17"}

def trigger(o, cl, hi, lo, op, vol, av, band, last_is_today=True):
    """Fuel says the gun is loaded; this says whether it is firing. Calibrated on the backtest above, which
    overturned three folklore rules: (1) volume is the signal — RVOL >= 3 or 3-day/20-day volume >= 2 lifts the
    +20% hit-rate from 38% to 48%, while 1.5x barely moves it; (2) relative strength only helps once it is LARGE
    (>= 20pp vs SPY over 5 days: 52%) — the +5pp/+8pp bars were noise, and 0-10pp was actually the WORST bucket;
    (3) closing in the top of the range does nothing (38.1 / 38.4 / 38.3 across thirds). A 15%+ gap raises both
    the hit-rate and the drawdown — flagged, not blocked. Bear vs bull bars were not supported either: fired
    setups hit +20% 46% of the time in bear tapes and 45% in bull; the regime stays in the score, not the gate.
    Every bucket has a NEGATIVE median close 20 sessions later: these are spike names. Sell into strength."""
    now = datetime.now(ET); t = {}
    mins = (now.hour - 9) * 60 + now.minute - 30
    frac = clamp(mins / 390, 0.12, 1.0) if (last_is_today and now.weekday() < 5) else 1.0   # a finished bar is a full day
    rvol = float(vol[-1]) / (av * frac) if av else None; t["rvol"] = round(rvol, 2) if rvol is not None else None
    t["vr20"] = round(float(vol[-3:].mean() / (vol[-23:-3].mean() or 1)), 2)
    bench = (MKT or {}).get("bench") or {}
    t["rs5_spy"] = round(float(cl[-1] / cl[-6] - 1) * 100 - (bench.get("spy5") or 0), 1) if len(cl) > 6 and bench.get("spy5") is not None else None
    rng = float(hi[-1] - lo[-1]); t["clpos"] = round(float(cl[-1] - lo[-1]) / rng, 2) if rng > 0 else None
    t["brk10"] = bool(len(hi) > 11 and cl[-1] > float(hi[-11:-1].max()))            # closing above the prior 10-day high
    t["gap"] = round(float(op[-1] / cl[-2] - 1) * 100, 1) if len(cl) > 1 and op[-1] else None
    fb = False
    if len(cl) > 26:
        prior_low = float(lo[-26:-6].min()); recent_low_i = int(np.argmin(lo[-6:])); recent_low = float(lo[-6:][recent_low_i])
        fb = recent_low < prior_low and cl[-1] > prior_low and cl[-1] > hi[-2] and recent_low_i < 5
    t["failbd"] = bool(fb)
    pattern = t["brk10"] or fb
    vol_strong = (rvol is not None and rvol >= 3.0) or t["vr20"] >= 2.0
    vol_ok = (rvol is not None and rvol >= 2.0) or t["vr20"] >= 2.0
    rs_big = t["rs5_spy"] is not None and t["rs5_spy"] >= 20
    if pattern and vol_strong and rs_big: state, tier = "fired", "strong"
    elif pattern and vol_ok: state, tier = "fired", "standard"
    elif pattern: state, tier = "armed", "armed"
    else: state, tier = "idle", None
    t["state"], t["tier"] = state, tier
    t["stats"] = TRIG_STATS.get(tier) if tier else None
    t["kind"] = "failed breakdown" if fb else ("10-day breakout" if t["brk10"] else None)
    t["entry"] = round(float(hi[-1]), 2); t["stop"] = round(float(lo[-1]), 2)           # signal-day high / low
    t["extended"] = bool(t["gap"] is not None and t["gap"] >= 15)                          # spikier both ways, not excluded
    t["why"] = []
    if pattern and not vol_ok: t["why"].append(f"volume RVOL {t['rvol']}x / 3d-vs-20d {t['vr20']}x (need RVOL 2 or 3d 2x)")
    if pattern and vol_ok and not (vol_strong and rs_big):
        if not vol_strong: t["why"].append(f"RVOL {t['rvol']}x — 3x would make it strong")
        if not rs_big: t["why"].append(f"RS vs SPY {t['rs5_spy']:+}pp — 20pp would make it strong" if t["rs5_spy"] is not None else "RS n/a")
    return t

def deep(x, side):
    c = CFG[side]; o = dict(x); o["err"] = ""
    try:
        tk = yf.Ticker(x["t"])
        h = tk.history(period="3mo", interval="1d", auto_adjust=False).dropna(subset=["Close"])
        if len(h) < 25: raise RuntimeError("not enough price history")
        cl = h["Close"].to_numpy(); hi = h["High"].to_numpy(); lo = h["Low"].to_numpy(); vol = h["Volume"].to_numpy(); op = h["Open"].to_numpy()
        px = round(float(cl[-1]), 2); o["px"] = px
        rs = rsi_series(cl)
        o["rsi"] = round(rs[-1], 1); o["rsi3"] = round(rs[-4], 1)
        o["vr"] = round(float(vol[-3:].mean() / (vol[-23:-3].mean() or 1)), 2)
        o["up3d"] = round(float(cl[-1] / cl[-4] - 1) * 100, 1)
        o["closes"] = [round(float(v), 2) for v in cl[-30:]]
        # live info: pre-market quote + earnings timestamp (cheap, one call)
        try: info = tk.info or {}
        except Exception: info = {}
        pm = premarket(info); o["pm_px"], o["pm_chg"] = (pm if pm else (None, None))
        earn_ts = info.get("earningsTimestampStart") or info.get("earningsTimestamp") or x.get("earn_ts")
        if info.get("earningsTimestampStart"): o["earn"] = earn_label(info)
        # IV rank: our own log once it has 20+ days, else realized-vol percentile as a proxy
        contracts, atm_iv, gam = pick_contracts(tk, side, px, c, adv=x.get("av"))
        o["gam"] = gam
        ivr = iv_rank(x["t"], atm_iv); o["ivr"] = ivr; o["ivr_src"] = "iv" if ivr is not None else "hv"
        if ivr is None: ivr = x.get("hvp")
        o["iv_rank"] = ivr; o["atm_iv"] = atm_iv
        ivpen = 4 if (ivr is not None and ivr >= 80) else (2 if (ivr is not None and ivr >= 60) else 0)
        if ivpen and contracts:
            for z in contracts: z["q"] = round(z["q"] - ivpen, 1)
            contracts.sort(key=lambda z: -z["q"])
        # A contract only counts as tradeable if its quality is actually positive. A zero bid with a 100%
        # spread and a breakeven halfway to the moon is not a play, and shouldn't pass the gate.
        o["play"] = contracts[0] if contracts and contracts[0]["q"] > 0 else None; o["alts"] = contracts[1:3]
        if o["play"] is None and contracts: o["dud"] = contracts[0]     # keep it visible on the card
        opt = o["play"]["q"] if o["play"] else 0
        flags = []
        # earnings inside the contract window = binary event, not a squeeze/topping trade
        o["earn_in"] = False
        if o["play"] and earn_ts:
            ed = datetime.fromtimestamp(earn_ts, timezone.utc).date(); ex = datetime.strptime(o["play"]["exp"], "%Y-%m-%d").date()
            o["earn_in"] = datetime.now(timezone.utc).date() <= ed <= ex
        b = x.get("borrow") or {}
        if side in ("calls", "breakout"):
            o["rsiUp"] = o["rsi"] > o["rsi3"]; o["wasOversold"] = min(rs[-10:]) < 35
            o["low20"] = round(float(cl[-1] / cl[-20:].min() - 1) * 100, 1)
            fpts, fm = float_pts(x.get("flt")); o["fltm"] = fm
            fuel = clamp(o["sf"] or 0, 0, 60) / 60 * 15 + fpts + clamp(o["sr"] or 0, 0, 10) / 10 * 6   # 0-29
            if b.get("fee") is not None:
                fuel += 4 if b["fee"] >= 50 else (3 if b["fee"] >= 20 else (1 if b["fee"] >= 5 else 0))
                fuel += 2 if b["avail"] < 100_000 else (1 if b["avail"] < 500_000 else 0)
            else:
                fuel *= 35 / 29            # no borrow feed: rescale rather than strand 6 pts for everyone
            # trend beats level: shorts piling in, or borrow getting expensive, is the actual tell.
            # Our own SI log only moves when the exchange report does, so it is NOT scored (that would count
            # the same event twice with si_rep below); it is logged for the card. Borrow fee/availability
            # are daily data, so their 30-day change is real information.
            o["si_chg"] = log_metric("si", x["t"], o.get("sf"))
            o["fee_chg"] = log_metric("fee", x["t"], b.get("fee"))
            o["avail_chg"] = log_metric("avail", x["t"], b.get("avail"))
            if (o["fee_chg"] or 0) >= 5: fuel += 2
            if b.get("avail") and o["avail_chg"] is not None and o["avail_chg"] <= -0.5 * (b["avail"] - o["avail_chg"]): fuel += 1   # shares to borrow halved
            sir = x.get("si_rep")                              # exchange-reported change between the last two settlement dates
            if sir is not None: fuel += 3 if sir >= 25 else (2 if sir >= 10 else 0)
            if x.get("ftd"): fuel += 2                         # Reg SHO threshold list = persistent failures to deliver
            if (x.get("inst") or 0) >= 40: fuel += 2           # institutions sit on it: the effective tradable float is smaller still
            fuel = min(fuel, 35)
            # Ignition inputs shared by both call boards
            bench = (MKT or {}).get("bench") or {}
            o["rs5"] = round(float(cl[-1] / cl[-6] - 1) * 100 - (bench.get("iwm5") or 0), 1) if len(cl) > 6 and bench.get("iwm5") is not None else None
            o["rs10"] = round(float(cl[-1] / cl[-11] - 1) * 100 - (bench.get("iwm10") or 0), 1) if len(cl) > 11 and bench.get("iwm10") is not None else None
            rng = float(hi[-1] - lo[-1]); o["clpos"] = round(float(cl[-1] - lo[-1]) / rng, 2) if rng > 0 else None     # where today closed in its range
            o["sma20_up"] = bool(len(cl) >= 25 and cl[-20:].mean() > cl[-25:-5].mean())
            att_z = x.get("att_z"); apts = 3 if (att_z or 0) >= 2 else (1 if (att_z or 0) >= 1 else 0)
            if apts: flags.append(f"attention {x.get('att')} posts/24h vs {x.get('att_mu')} avg")
            gpts, gflags = gamma_pts(o.get("gam")); flags += gflags
            if side == "calls":
                bot = 0
                if 25 <= o["rsi"] <= 45: bot += 10
                elif o["rsi"] < 25 or o["rsi"] <= 52: bot += 4
                if o["rsiUp"]: bot += 10
                if o["wasOversold"]: bot += 5
                if o["low20"] <= 8 or (o["lo"] is not None and o["lo"] <= 15): bot += 5
                if o["vr"] >= 1.3: bot += 5
                if o["pm_chg"] is not None and 2 <= o["pm_chg"] <= 15: bot += 3
                if (o["rs5"] or 0) > 0: bot += 2                                        # already outperforming small caps off the low
                bot = min(bot + gpts + apts, 35)
                setup = fuel + bot - (5 if o["up3d"] < -10 else 0)
                if o["up3d"] < -10: flags.append(f"still falling ({o['up3d']}% / 3d)")
                if o["rsiUp"]: flags.append("RSI turning up")
            else:
                # Breakout timing: the trend has already turned — structure, location and volume, not oversold bounce.
                o["hh"] = bool(hi[-10:].max() > hi[-20:-10].max() and lo[-10:].min() > lo[-20:-10].min())
                s20v = float(cl[-20:].mean()); s50v = float(cl[-50:].mean()) if len(cl) >= 50 else s20v
                o["above20"] = bool(cl[-1] > s20v); o["stack"] = bool(cl[-1] > s20v > s50v)
                o["high20"] = round(float(cl[-1] / hi[-20:].max() - 1) * 100, 1)                 # % below 20-day high (≤ 0)
                w1 = float(hi[-10:].max() - lo[-10:].min())
                w0 = float(hi[-30:-10].max() - lo[-30:-10].min()) if len(cl) >= 30 else 0.0
                o["coil"] = bool(w0 and w1 / w0 < 0.6)                                           # range compressing under resistance
                o["recl50"] = bool(len(cl) >= 55 and cl[-1] > s50v and float(min(cl[-5:])) <= float(cl[-55:-5].mean()) * 1.02)
                brk = 0
                if o["hh"]: brk += 6
                if o["stack"]: brk += 6
                elif o["above20"]: brk += 3
                if o["high20"] >= -3: brk += 8
                elif o["high20"] >= -8: brk += 4
                if o["coil"]: brk += 5
                if o["recl50"]: brk += 4
                if o["vr"] >= 2: brk += 8
                elif o["vr"] >= 1.3: brk += 5
                if 50 <= o["rsi"] <= 72: brk += 4
                elif o["rsi"] > 78: brk -= 3
                if o["pm_chg"] is not None and 1 <= o["pm_chg"] <= 12: brk += 3
                if (o["rs5"] or 0) > 5: brk += 3                                         # relative strength vs IWM, not just up
                if (o["rs10"] or 0) > 8: brk += 3
                if o["clpos"] is not None and o["clpos"] >= 0.8: brk += 2                # closed in the top fifth of its range
                if o["sma20_up"]: brk += 2
                bot = min(max(brk, 0) + gpts + apts, 35)
                setup = fuel + bot - (6 if o["up3d"] > 25 else 0)          # chasing something already vertical is how you buy the top
                if o["up3d"] > 25: flags.append(f"already vertical (+{o['up3d']}% / 3d)")
                if o["hh"]: flags.append("higher highs and lows")
                if o["high20"] >= -3: flags.append("at 20-day high")
                elif o["high20"] >= -8: flags.append(f"{abs(o['high20'])}% under 20-day high")
                if o["coil"]: flags.append("coiling under resistance")
                if o["recl50"]: flags.append("reclaimed 50-day")
                if o["stack"]: flags.append("above 20 & 50-day")
                if (o["rs5"] or 0) > 5: flags.append(f"+{o['rs5']}pp vs IWM / 5d")
                if o["clpos"] is not None and o["clpos"] >= 0.8: flags.append("closed near the high")
            if o["vr"] >= 1.3: flags.append(f"volume pickup {o['vr']}x")
            if b.get("fee") is not None and b["fee"] >= 20: flags.append(f"borrow fee {b['fee']}%")
            if b.get("avail") is not None and b["avail"] < 100_000: flags.append(f"{b['avail']:,} shares to borrow")
            if o.get("fltm") is not None and o["fltm"] < 25: flags.append(f"{o['fltm']}M float")
            if (o.get("si_chg") or 0) >= 2: flags.append(f"short interest +{o['si_chg']}pp")
            if (o.get("fee_chg") or 0) >= 5: flags.append(f"borrow fee +{o['fee_chg']}pp")
            if (x.get("si_rep") or 0) >= 10: flags.append(f"shorts added {x['si_rep']}% since {x.get('ssp_date') or 'last report'}")
            if x.get("ftd"): flags.append("Reg SHO threshold list")
            # Dilution is the squeeze killer: a company that sells stock into every rally caps the move. Graded, not a nudge.
            g = x.get("shrg") or 0
            if g >= 50: setup -= 12; flags.append(f"heavy dilution: shares +{g}% in 6mo")
            elif g >= 25: setup -= 8; flags.append(f"dilution: shares +{g}% in 6mo")
            elif g >= 10: setup -= 4; flags.append(f"shares +{g}% in 6mo")
            if o.get("avail_chg") is not None and b.get("avail") and o["avail_chg"] < 0 and -o["avail_chg"] >= 0.5 * (b["avail"] - o["avail_chg"]):
                flags.append("shares to borrow halved in 30d")
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
            elif (x.get("shrg") or 0) >= 10: press += 6          # share count actually ballooned
            elif (x.get("shrg") or 0) >= 5: press += 3
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
            if o["pm_chg"] is not None and o["pm_chg"] <= -2: top_ += 3
            top_ = min(top_, 35)
            setup = press + top_ - (5 if o["up3d"] > 10 else 0)                                # still ripping = don't step in front
            if o["up3d"] > 10: flags.append(f"still ripping (+{o['up3d']}% / 3d)")
            if o["rsiDown"]: flags.append("RSI rolling over")
            if o["reversal"]: flags.append("reversal day")
            if o["volFade"]: flags.append("volume fading")
            if x.get("gapfail"): flags.append(f"failed gap (+{x['gapfail']}%)")
            if x.get("dil"): flags.append("dilution: " + ", ".join(f"{f} {d}" for f, d in x["dil"][:2]))
            elif (x.get("shrg") or 0) >= 5: flags.append(f"shares +{x['shrg']}% in 6mo")
            if x.get("runway") is not None and x["runway"] < 1: flags.append(f"cash runway {x['runway']}y")
            if x.get("ipo_days") is not None and 150 <= x["ipo_days"] <= 200: flags.append("lockup window")
            if x.get("ftd"): press -= 4; flags.append("on the Reg SHO threshold list — squeeze risk against you")
            o["si_chg"] = log_metric("si", x["t"], o.get("sf")); o["fee_chg"] = log_metric("fee", x["t"], b.get("fee"))
            gp = gamma_pts(o.get("gam"))[0]
            if gp >= 4: press -= 3; flags.append("heavy call positioning against you")   # gamma cuts the other way on puts
            o["fltm"] = float_pts(x.get("flt"))[1]
            o.update(fuel=round(press, 1), bot=top_)
        if o["pm_chg"] is not None and abs(o["pm_chg"]) >= 2: flags.append(f"pre-market {o['pm_chg']:+}%")
        if o.get("earn_in"): flags.append(f"earnings inside window ({o.get('earn')}) — binary")
        elif re.match(r"^[A-Z][a-z]{2} \d", o.get("earn") or ""):
            m = datetime.now(ET).month; mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            if re.match(f"^({mn[m-1]}|{mn[m % 12]})", o["earn"]): flags.append(f"earnings {o['earn']}")
        if x.get("k8"): flags.append(f"8-K filed {x['k8'][5:]}")
        if ivr is not None and ivr >= 60: flags.append(f"IV rank {ivr} (rich)")
        if not o["play"]:
            d0 = o.get("dud")
            kind_ = "put" if side == "puts" else "call"
            flags.insert(0, (f"only a junk {kind_} (bid {d0['bid']:.2f}, "
                             f"{d0['spread']}% spread, needs {round(d0['be'])}%)") if d0
                            else f"no tradeable {kind_} ≤ ${c['max_ask']:.2f}")
        if side in ("calls", "breakout"):
            try: last_today = h.index[-1].date() == datetime.now(ET).date()
            except Exception: last_today = True
            o["trig"] = trigger(o, cl, hi, lo, op, vol, x.get("av"), (MKT or {}).get("band"), last_today)
            tg = o["trig"]
            if tg["state"] == "fired": flags.insert(0, f"TRIGGERED ({tg['tier']}) — {tg['kind']}")
            elif tg["kind"]: flags.append(f"{tg['kind']} without volume ({'; '.join(tg['why'][:1])})")
            if tg.get("extended"): flags.append(f"gapped +{tg['gap']}% — spikier both ways (median dip −11%)")
        # Ranking = the stock setup only (fuel/pressure + bottoming/topping), rescaled 0-70 -> 0-100.
        # Contract quality does NOT lift the score: it gates (no tradeable contract caps at 40 and sorts last)
        # and breaks ties between equal setups. A great option can no longer carry a weak setup.
        s = clamp(setup, 0, 70) / 70 * 100
        # The tape gets 15% of the score: the same setup is worth less when shorts are being paid.
        mk = (MKT or {}).get("score")
        o["mkt"] = mk
        if mk is not None: s = 0.85 * s + 0.15 * mk
        if o.get("trig") and o["trig"]["state"] == "fired" and s < 60:
            # the backtest did not test the setup score, so the bar is a sanity floor, not a calibrated cut
            o["trig"]["state"] = "armed"; o["trig"]["why"].append(f"score {round(s)} under the 60 floor")
            ti = next((i for i, f in enumerate(flags) if f.startswith("TRIGGERED")), None)
            if ti is not None: flags[ti] = f"{o['trig']['kind']} on volume, but setup score {round(s)} is under the 60 floor"
        o.update(flags=flags, setup=round(clamp(setup, 0, 70), 1), opt=round(opt, 1),
                 score=round(s if o["play"] else min(s, 40), 1),
                 pot=round(o["fuel"] / 35 * 100), imm=round(o["bot"] / 35 * 100))   # how explosive vs. is it starting now
    except Exception as e:
        log.warning("%s: %s", x["t"], e); o["err"] = str(e)[:120]; o["score"] = -1
    return o

def stage2(top, side):
    out = []
    for i, x in enumerate(top):
        out.append(deep(x, side)); time.sleep(0.4)
        if i % 10 == 9: log.info("deep %d/%d", i + 1, len(top))
    out.sort(key=lambda o: (0 if o.get("play") else 1, 0 if (o.get("trig") or {}).get("state") == "fired" else 1, -o["score"], -(o.get("opt") or 0)))   # firing first, then setup; contract breaks ties
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
            "runway","dil","ipo_days","gapfail","borrow","k8","shrg","flt","fltm","ins","si_date","si_chg","fee_chg","avail_chg","gam","mkt","exch","sf_src","att","att_z","att_mu","rs5","rs10","clpos","sma20_up","pot","imm","trig","inst","ss","ssp","ssp_date","si_rep","ftd","atr","dv","rv10","h20","h50","hh","stack","coil","recl50","pm_px","pm_chg","iv_rank","ivr_src","atm_iv","earn_in",
            "fuel","bot","opt","setup","score","flags","play","dud","alts","closes","err")
    top = [{k: o.get(k) for k in keys} for o in deep_out[:16]]
    for o in top: o["lo52"] = o.pop("lo"); o["hi52"] = o.pop("hi")
    rest = [[o["t"], o.get("px"), o.get("sf"), o.get("rsi"), o.get("score"), 1 if o.get("play") else 0] for o in deep_out[16:]]
    return _clean(dict(side=side, asof=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                       full_asof=full_asof, mode=mode, universe=universe, pass1=pass1, deep=len(deep_out), market=MKT,
                       stage1=top_in, top=top, rest=rest, cfg=CFG[side]))

def run_side(side, mode, now_et):
    data = DATA_ROOT if side == "calls" else DATA_ROOT / side
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
        universe, pass1, top = (stage1_puts() if side == "puts" else stage1_calls(side))
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
    global MKT
    try:
        MKT = regime()
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        json.dump(MKT, open(DATA_ROOT / "market.json", "w"), separators=(",", ":"))
    except Exception as e:
        log.error("market regime failed: %s", e); MKT = None
    if mode == "daily":
        # one full scan per trading day: if today's already exists, exit 3 so a fallback trigger does nothing
        if now_et.weekday() >= 5: log.info("daily: weekend — skipping"); return 3
        try: prev = json.load(open(DATA_ROOT / "latest.json"))
        except Exception: prev = None
        if prev and (prev.get("full_asof") or "")[:10] == now_et.strftime("%Y-%m-%d"):
            log.info("daily: today's full scan already published — skipping"); return 3
        mode = "full"
    if mode == "auto":
        hhmm = now_et.hour * 100 + now_et.minute
        if now_et.weekday() >= 5 or not (825 <= hhmm <= 1610):
            log.info("auto: outside 08:25–16:10 ET weekday window — skipping"); return 3   # 3 = the workflow loop stops here
    for s in (SIDES if side in ("both", "all") else [side]):
        try: run_side(s, mode, now_et)
        except Exception as e: log.error("%s side failed: %s", s, e)
    return 0

if __name__ == "__main__":
    sys.exit(main())

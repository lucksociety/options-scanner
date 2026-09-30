#!/usr/bin/env python3
"""
Squeeze engine v2 — the live side of squeeze_model.py.

Every morning: screen the whole $1-10 band (no short-interest gate: the audit showed the SI >= 15% gate excluded
81% of the names that actually squeezed), keep the optionable ones, compute the model's features for every name
from the same daily bars, score each with the fitted bucket table, calibrate to a real probability, and rank.
The top names go to the deep scan (contracts, trigger, card). Short interest, float and days-to-cover are FEATURES
here, cached in data/keystats.json and refreshed on a rolling basis (Yahoo's info call is slow).

Outputs per name: p50 (calibrated P(+50% within 20 sessions)), p30, why (the buckets that moved it most, with
their measured hit-rates), plus the raw feature row the card shows.
"""
import json, logging, math, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as EQ

import squeeze_model as SM

log = logging.getLogger("engine")
ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data"
OPT_CACHE = DATA / "optionable.json"; KS_CACHE = DATA / "keystats.json"; MODEL = DATA / "backtest" / "squeeze_model.json"
FEATS = list(SM.EDGES) + SM.FLAGS
LABEL = {"rvol": "volume vs 63-day", "vr3": "3-day volume vs 20-day", "ret1": "1-day move", "ret5": "5-day move", "ret20": "20-day move",
         "rs5": "5-day RS vs SPY", "h20": "vs 20-day high", "lo20": "above 20-day low", "lo52": "above 52-week low", "rsi": "RSI-14",
         "clpos": "close in day's range", "gap": "opening gap", "s20": "vs 20-day avg", "s50": "vs 50-day avg", "atr": "ATR % of price",
         "dv": "$ volume/day (M)", "px": "price", "sf": "short float %", "sr": "days to cover", "flt": "float (M)",
         "brk10": "10-day-high breakout", "brk20": "20-day-high breakout", "above20": "above 20-day", "stack": "20 over 50",
         "rvol_x_flt": "volume × float", "rvol_x_sf": "volume × short float", "brk_x_rvol": "breakout × volume"}


def load_model():
    try: return json.load(open(MODEL))
    except Exception as e:
        log.error("squeeze model missing (%s) — run the backtest workflow", e); return None


def _load(path):
    try: return json.load(open(path))
    except Exception: return {}


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); json.dump(obj, open(path, "w"), separators=(",", ":"))


def universe(price_min=1.0, price_max=10.0, avgvol_min=300_000):
    q = EQ("and", [EQ("btwn", ["intradayprice", price_min, price_max]), EQ("gt", ["avgdailyvol3m", avgvol_min]), EQ("eq", ["region", "us"])])
    out, offset = {}, 0
    while True:
        res = yf.screen(q, offset=offset, size=250, sortField="avgdailyvol3m", sortAsc=False)
        quotes = res.get("quotes", []) if res else []
        for x in quotes:
            if x.get("symbol"): out[x["symbol"]] = x
        if len(quotes) < 250 or offset > 4000: break
        offset += 250; time.sleep(0.4)
    return out


def optionable(syms, max_checks=400, max_age_days=21):
    """Which names have listed options. Cached; `syms` is taken in priority order (best model score first), so on a
    cold start the names that matter get checked first: unknown/stale ones in that order, up to max_checks per run."""
    cache = _load(OPT_CACHE); today = datetime.now(timezone.utc).date()
    def age(t):
        a = cache.get(t, {}).get("asof")
        return 999 if not a else (today - datetime.strptime(a, "%Y-%m-%d").date()).days
    todo = [t for t in syms if age(t) > max_age_days][:max_checks]
    for t in todo:
        try: ok = bool(yf.Ticker(t).options)
        except Exception: ok = False
        cache[t] = dict(opt=ok, asof=str(today)); time.sleep(0.08)
    _save(OPT_CACHE, cache)
    known = {t: cache[t]["opt"] for t in syms if t in cache}
    log.info("optionable: checked %d, known %d/%d, optionable %d", len(todo), len(known), len(syms), sum(known.values()))
    return known


def keystats(syms, priority, max_calls=350, max_age_days=4):
    """Short float / days-to-cover / float / ownership / earnings / exchange for each name, cached. Refreshes the
    stalest first but always includes the `priority` names (this run's top candidates) if they are stale."""
    cache = _load(KS_CACHE); today = datetime.now(timezone.utc).date()
    def age(t):
        a = cache.get(t, {}).get("asof")
        return 999 if not a else (today - datetime.strptime(a, "%Y-%m-%d").date()).days
    pri = [t for t in priority if age(t) > 1][:max_calls]
    rest = sorted([t for t in syms if t not in pri and age(t) > max_age_days], key=lambda t: -age(t))[:max(0, max_calls - len(pri))]
    for t in pri + rest:
        try:
            info = yf.Ticker(t).info or {}
            fl = info.get("floatShares"); ss = info.get("sharesShort"); ssp = info.get("sharesShortPriorMonth")
            sf = float(info.get("shortPercentOfFloat") or 0) * 100
            if not sf and ss and fl: sf = ss / fl * 100
            ts = lambda v: datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d") if v else None
            cache[t] = dict(asof=str(today), sf=round(sf, 2), sr=round(float(info.get("shortRatio") or 0), 2),
                            flt=fl, shout=info.get("sharesOutstanding"), ss=ss, ssp=ssp,
                            si_date=ts(info.get("dateShortInterest")), ssp_date=ts(info.get("sharesShortPreviousMonthDate")),
                            ins=round(float(info.get("heldPercentInsiders") or 0) * 100, 1), inst=round(float(info.get("heldPercentInstitutions") or 0) * 100, 1),
                            exch=info.get("exchange"), earn_ts=info.get("earningsTimestampStart") or info.get("earningsTimestamp"),
                            co=info.get("longName") or info.get("shortName"), mc=info.get("marketCap"))
        except Exception as e:
            log.debug("keystats %s: %s", t, e); cache.setdefault(t, {})["asof"] = str(today)
        time.sleep(0.12)
    _save(KS_CACHE, cache)
    log.info("keystats: refreshed %d (%d priority), cached %d/%d", len(pri) + len(rest), len(pri), sum(1 for t in syms if t in cache), len(syms))
    return {t: cache.get(t, {}) for t in syms}


def score_rows(rows, model):
    """rows: DataFrame with the model's feature columns. Returns (p50, p30, z50, z30) as Series."""
    p50, z50 = SM.score(rows, model["t50"]); p30, z30 = SM.score(rows, model["t30"])
    c50 = SM.apply_calib(z50, model["calib50"]) if model.get("calib50") else p50
    c30 = SM.apply_calib(z30, model["calib30"]) if model.get("calib30") else p30
    return pd.Series(c50, index=rows.index), pd.Series(c30, index=rows.index), z50, z30


def explain(row, model, k_pos=4, k_neg=2):
    """The buckets that moved this name most, with the measured hit-rate of each bucket."""
    T = model["t50"]; base = T["base"]; items = []
    for f in FEATS:
        b = SM.bucket(f, row.get(f)); v = T["feats"].get(f, {}).get(b)
        if v: items.append((v["w"], f, b, v["p"], v["n"]))
    for name in SM.INTER:
        key = SM.inter_key(name, row); v = T["inter"].get(name, {}).get(key)
        if v: items.append((v["w"], name, key, v["p"], v["n"]))
    items.sort(key=lambda x: -x[0])
    fmt = lambda w, f, b, p, n: dict(f=f, label=LABEL.get(f, f), bucket=b, w=round(w, 2), p=round(p * 100, 1), n=n)
    pos = [fmt(*x) for x in items if x[0] > 0.15][:k_pos]; neg = [fmt(*x) for x in items[::-1] if x[0] < -0.15][:k_neg]
    return dict(base=round(base * 100, 1), pos=pos, neg=neg)


def stage1(cfg, model, deep_n=40):
    """The v2 stage 1: broad universe -> optionable -> features -> model -> ranked candidates.
    Returns (universe_size, n_scored, top list of dicts in scan.py's stage-1 shape, all_scored DataFrame)."""
    U = universe(cfg["price_min"], cfg["price_max"], cfg["avgvol_min"]); syms = sorted(U)
    log.info("v2 universe: %d names", len(syms))
    if not syms: raise RuntimeError("screener returned nothing")
    hist = yf.download(syms + ["SPY"], period="1y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    spy = hist["SPY"]["Close"].dropna()
    last = []
    frames = {}
    for t in syms:
        try:
            h = hist[t].dropna(subset=["Close"])
            if len(h) < 70: continue
            F = SM.features(h, spy.reindex(h.index).ffill())
            r = F.iloc[-1].to_dict(); r["t"] = t; r["date"] = str(h.index[-1].date()); r["av"] = float(h["Volume"].tail(63).mean())
            r["closes"] = [round(float(v), 2) for v in h["Close"].tail(30)]
            last.append(r); frames[t] = h
        except Exception as e: log.debug("features %s: %s", t, e)
    D = pd.DataFrame(last).set_index("t")
    D = D[D["dv"] >= cfg.get("dv_min", 0)]
    # pass 1 with cached key stats (NaN for unknown names), to know who deserves a fresh info call
    ks = _load(KS_CACHE)
    for c in ("sf", "sr", "flt"): D[c] = [ks.get(t, {}).get(c) for t in D.index]
    D["flt"] = D["flt"].astype(float) / 1e6
    p50, p30, _, _ = score_rows(D[FEATS], model); D["p50"] = p50
    # optionability is checked in model order, so the first run already knows the names that matter
    opt = optionable(list(D.sort_values("p50", ascending=False).index))
    D = D[[bool(opt.get(t)) for t in D.index]]
    if D.empty: raise RuntimeError("no optionable names known yet")
    pri = list(D.sort_values("p50", ascending=False).index[:200])
    ks = keystats(list(D.index), pri)
    for c in ("sf", "sr", "flt"): D[c] = [ks.get(t, {}).get(c) for t in D.index]
    D["flt"] = D["flt"].astype(float) / 1e6
    p50, p30, z50, z30 = score_rows(D[FEATS], model); D["p50"] = p50; D["p30"] = p30; D["z50"] = z50
    D = D.sort_values("p50", ascending=False)
    top = []
    for t, r in D.head(deep_n).iterrows():
        k = ks.get(t, {}); q = U.get(t, {})
        top.append(dict(t=t, co=k.get("co") or q.get("longName") or q.get("shortName") or t, mc=k.get("mc") or q.get("marketCap"),
                        p=float(r["px"]), av=float(r["av"]), dv=round(float(r["dv"]), 2), closes=r["closes"],
                        sf=float(k.get("sf") or 0), sr=float(k.get("sr") or 0), flt=k.get("flt"), shout=k.get("shout"), ss=k.get("ss"), ssp=k.get("ssp"),
                        si_date=k.get("si_date"), ssp_date=k.get("ssp_date"), ins=k.get("ins"), inst=k.get("inst"), exch=k.get("exch"),
                        earn_ts=k.get("earn_ts"), earn="-", si_rep=(round((k["ss"] / k["ssp"] - 1) * 100, 1) if (k.get("ss") and k.get("ssp")) else None),
                        p50=round(float(r["p50"]) * 100, 1), p30=round(float(r["p30"]) * 100, 1), z50=round(float(r["z50"]), 3),
                        feat={f: (None if (isinstance(r[f], float) and r[f] != r[f]) else round(float(r[f]), 2)) for f in FEATS},
                        why=explain(r, model), s1=round(float(r["p50"]) * 100, 1),
                        pw=round(float(r["ret5"]), 2), pm=round(float(r["ret20"]), 2), rsi=round(float(r["rsi"]), 2) if r["rsi"] == r["rsi"] else None,
                        s20=round(float(r["s20"]), 2), s50=round(float(r["s50"]), 2), h20=round(float(r["h20"]), 2), lo=round(float(r["lo52"]), 2) if r["lo52"] == r["lo52"] else None,
                        hi=None, atr=round(float(r["atr"]), 2), rv=round(float(r["rvol"]), 2), rv10=None, hvp=None, hv=None))
    rest = [[t, round(float(r["px"]), 2), round(float(r["sf"]) if r["sf"] == r["sf"] else 0, 1), round(float(r["rsi"]), 0) if r["rsi"] == r["rsi"] else None,
             round(float(r["p50"]) * 100, 1), 0] for t, r in D.iloc[deep_n:deep_n + 60].iterrows()]
    return len(U), int(len(D)), top, rest

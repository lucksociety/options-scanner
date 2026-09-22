#!/usr/bin/env python3
"""
Market regime for the Luck Society Option Scanner.

A heavily shorted stock needs a tape that rewards risk before shorts feel any pressure. This scores the
broad environment 0-100 from free Yahoo data and writes data/market.json, which the dashboard shows at the
top of the page and the scanner blends into every board score at 15% weight.

The regime we want: market recently corrected -> VIX spiked -> indexes reverse higher -> VIX falls ->
small caps start leading -> speculative groups rip -> volume expands. The regime we don't: SPY and IWM
down together, VIX and yields rising, breadth deteriorating -- shorts are being paid, so they don't cover.

Components (100 total):
  trend 20 · small-cap leadership 20 · VIX 15 · speculative appetite 15 · rebound 10 · breadth 10 · yields 10
"""
import json, logging, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yfinance as yf

import outlook as _outlook

log = logging.getLogger("market")

ROOT = Path(__file__).resolve().parent.parent
TICKERS = ["SPY", "QQQ", "IWM", "RSP", "ARKK", "XBI", "SPHB", "^VIX", "^TNX", "BTC-USD"]


def _series(hist, t):
    try:
        h = hist[t].dropna(subset=["Close"]) if len(TICKERS) > 1 else hist.dropna(subset=["Close"])
        return h["Close"].to_numpy(dtype=float), h["Volume"].to_numpy(dtype=float)
    except Exception:
        return np.array([]), np.array([])


def _chg(c, n):
    return float(c[-1] / c[-1 - n] - 1) * 100 if len(c) > n else None


def _sma(c, n):
    return float(c[-n:].mean()) if len(c) >= n else None


def regime():
    """Score the tape 0-100. Every component reports what it saw, so a low score is explainable."""
    hist = yf.download(TICKERS, period="6mo", interval="1d", group_by="ticker",
                       auto_adjust=False, threads=True, progress=False)
    px = {t: _series(hist, t) for t in TICKERS}
    parts, notes, missing = {}, [], []

    def close(t):
        c, _ = px.get(t, (np.array([]), np.array([])))
        return c if len(c) else None

    spy, qqq, iwm, rsp = close("SPY"), close("QQQ"), close("IWM"), close("RSP")
    vix, tnx = close("^VIX"), close("^TNX")

    # 1. TREND (20) — are the indexes above their own short-term averages and going up
    if spy is not None and qqq is not None:
        s = 0
        for c in (spy, qqq):
            if _sma(c, 20) and c[-1] > _sma(c, 20): s += 4
            if _sma(c, 50) and c[-1] > _sma(c, 50): s += 3
            if (_chg(c, 5) or 0) > 0: s += 3
        parts["trend"] = min(s, 20)
        notes.append(f"SPY {_chg(spy,5):+.1f}% / 5d, QQQ {_chg(qqq,5):+.1f}% / 5d")
    else:
        missing.append("trend")

    # 2. SMALL-CAP LEADERSHIP (20) — the single biggest tell for speculative squeezes
    if iwm is not None and spy is not None:
        r5 = (_chg(iwm, 5) or 0) - (_chg(spy, 5) or 0)
        r20 = (_chg(iwm, 20) or 0) - (_chg(spy, 20) or 0)
        s = 0
        s += 11 if r5 >= 1.5 else (8 if r5 >= 0.5 else (5 if r5 >= 0 else (2 if r5 >= -1 else 0)))
        s += 9 if r20 >= 2 else (6 if r20 >= 0.5 else (3 if r20 >= -1 else 0))
        parts["smallcap"] = min(s, 20)
        notes.append(f"IWM vs SPY {r5:+.1f}pp / 5d, {r20:+.1f}pp / 20d")
    else:
        missing.append("smallcap")

    # 3. VIX (15) — elevated but falling is the sweet spot; dead calm and panic both score poorly
    if vix is not None:
        v = float(vix[-1]); v5 = _chg(vix, 5) or 0
        lvl = 9 if 18 <= v <= 25 else (7 if 15 <= v < 18 else (5 if 25 < v <= 30 else (3 if v < 15 else 0)))
        dirn = 6 if v5 <= -10 else (4 if v5 < 0 else (2 if v5 < 5 else 0))
        parts["vix"] = lvl + dirn
        notes.append(f"VIX {v:.1f} ({v5:+.1f}% / 5d)")
    else:
        missing.append("vix")

    # 4. SPECULATIVE APPETITE (15) — unprofitable tech, biotech, high beta, crypto
    spec, have = 0.0, 0
    for t, w in (("ARKK", 4), ("XBI", 4), ("SPHB", 4), ("BTC-USD", 3)):
        c = close(t)
        if c is None or spy is None: continue
        have += w
        rel = (_chg(c, 10) or 0) - (_chg(spy, 10) or 0)
        spec += w if rel >= 2 else (w * 0.66 if rel >= 0 else (w * 0.33 if rel >= -2 else 0))
    if have:
        parts["spec"] = round(spec / have * 15, 1)
        notes.append("speculative groups " + ("leading" if spec / have > 0.6 else "lagging") + " SPY over 10d")
    else:
        missing.append("spec")

    # 5. REBOUND AFTER A CORRECTION (10) — shorts pressed into the hole, then the tape turned
    if spy is not None and len(spy) > 60:
        peak = float(spy[-60:].max()); trough = float(spy[-20:].min()); now = float(spy[-1])
        drawdown = (trough / peak - 1) * 100
        off_low = (now / trough - 1) * 100
        s = 0
        if drawdown <= -5 and off_low >= 2: s = 10 if drawdown <= -8 else 7
        elif drawdown <= -3 and off_low >= 1.5: s = 4
        parts["rebound"] = s
        if s: notes.append(f"recovering from a {abs(drawdown):.0f}% pullback (+{off_low:.1f}% off the low)")
    else:
        missing.append("rebound")

    # 6. BREADTH (10) — equal weight vs cap weight: is the average stock participating
    if rsp is not None and spy is not None:
        b5 = (_chg(rsp, 5) or 0) - (_chg(spy, 5) or 0)
        b20 = (_chg(rsp, 20) or 0) - (_chg(spy, 20) or 0)
        parts["breadth"] = min((5 if b5 >= 0.3 else (3 if b5 >= -0.3 else 0)) +
                               (5 if b20 >= 0.5 else (3 if b20 >= -0.5 else 0)), 10)
        notes.append(f"breadth (RSP vs SPY) {b5:+.1f}pp / 5d")
    else:
        missing.append("breadth")

    # 7. YIELDS (10) — falling yields ease conditions for exactly the high-duration names that carry shorts
    if tnx is not None:
        y = float(tnx[-1]) / 10 if float(tnx[-1]) > 20 else float(tnx[-1])   # ^TNX quotes 10x in some feeds
        y20 = _chg(tnx, 20) or 0
        parts["yields"] = 10 if y20 <= -3 else (7 if y20 < 0 else (4 if y20 < 3 else 0))
        notes.append(f"10-year {y:.2f}% ({y20:+.1f}% / 20d)")
    else:
        missing.append("yields")

    # Missing components are uncertainty, not free points: rescale to what we could actually measure.
    WEIGHTS = dict(trend=20, smallcap=20, vix=15, spec=15, rebound=10, breadth=10, yields=10)
    measured = sum(WEIGHTS[k] for k in parts)
    raw = sum(parts.values())
    score = round(raw / measured * 100, 1) if measured else None

    band = "unknown" if score is None else ("green" if score >= 65 else ("amber" if score >= 40 else "red"))
    verdict = {"green": "Risk-on: heavily shorted names get help from the tape.",
               "amber": "Mixed: isolated squeezes possible, nothing broad.",
               "red": "Risk-off: shorts are being paid, so they have little reason to cover.",
               "unknown": "Market data unavailable."}[band]
    out = dict(asof=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
               score=score, band=band, verdict=verdict, parts=parts, weights=WEIGHTS,
               bench={**{k: (round(_chg(iwm, n), 2) if iwm is not None and _chg(iwm, n) is not None else None)
                         for k, n in (("iwm5", 5), ("iwm10", 10), ("iwm20", 20))},
                      **{k: (round(_chg(spy, n), 2) if spy is not None and _chg(spy, n) is not None else None)
                         for k, n in (("spy5", 5), ("spy10", 10))}},
               measured=measured, missing=missing, notes=notes)
    log.info("market regime %s (%s) · %s", score, band, "; ".join(notes))
    # 3-4 week directional outlook (calls vs puts) — its own download, a year of history for the 50-day math
    try:
        h2 = yf.download(_outlook.TICKERS, period="1y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
        try: table = json.load(open(ROOT / "data" / "backtest" / "outlook_model.json"))
        except Exception: table = None
        out["outlook"] = _outlook.outlook_today(h2, table)
        try:
            bt = json.load(open(ROOT / "data" / "backtest" / "outlook.json"))
            o = out["outlook"]; b = o["band"]
            # held-out (last ~30% of history) odds for today's band are the honest ones; fall back to the full-sample odds
            o["measured"] = (bt.get("xbands_test", {}).get(b) or bt.get("xbands_all", {}).get(b)) if o.get("calibrated") else bt.get("bands", {}).get(b)
            o["measured_scope"] = "held-out" if (o.get("calibrated") and bt.get("xbands_test", {}).get(b)) else "all"
            o["measured_all"] = bt.get("test_all") if o["measured_scope"] == "held-out" else bt.get("all")
            o["measured_asof"] = bt.get("asof"); o["measured_span"] = [bt.get("first"), bt.get("last")]
        except Exception: pass
        log.info("outlook %s (%s) · %s", out["outlook"]["bias"], out["outlook"]["label"], "; ".join(out["outlook"]["notes"]))
    except Exception as e:
        log.error("outlook failed: %s", e); out["outlook"] = None
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = regime()
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "data" / "market.json", "w"), separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

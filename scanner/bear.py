#!/usr/bin/env python3
"""
Bear-put engine for the Luck Society Option Scanner (the Puts board).

The question is not "which stock is going down" but "where is the options market underpricing the probability
and size of a decline over the next 3-4 weeks, and which contract captures it with the least wasted theta".
So the stock is ranked first (structure + relative weakness + catalyst), then every candidate put is repriced
under a modelled distribution of where the stock could be at the planned exit (10 sessions out), and the
contract is chosen on probability of profit, expected return and payoff asymmetry -- not on "looks cheap".

Everything in here is pure math on numbers scan.py already has (daily bars, chain rows, ATM IV, earnings date).
No network. Model assumptions are deliberately simple and printed on the card, so they can be argued with:

  paths     lognormal, sigma = 50/50 blend of 20-day realized vol and the put's own implied vol
  drift     bearish tilt = -(0..0.6) x sigma over the horizon, sized by how bearish the structure and relative
            weakness actually are (a 15/15 chart with -10pp relative weakness gets the full tilt; a neutral one gets none)
  tails     8% of paths draw from a 1.8x-wider distribution (gaps happen), total variance held at sigma^2
  earnings  if the report lands inside the horizon, the variance IV carries ABOVE realized is treated as the event
            jump and added to every path; the exit IV is then crushed back toward realized (the "priced-in" test)
  exit IV   rises when the stock falls (skew), falls when it rallies -- half the move, capped at +/-15%
  repricing Black-Scholes at exit with the remaining days; intrinsic if the horizon reaches expiry
"""
import math
import numpy as np

R_F = 0.04            # risk-free, annual
HORIZON = 10          # planned exit, sessions (hold 7-15 days per the playbook; 10 is the middle)
N_PATHS = 6000
SEED = 11


# ---------------------------------------------------------------- Black-Scholes (vectorised, no scipy)
def _ncdf(x):
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    pdf = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    c = 1.0 - pdf * poly
    return np.where(x >= 0, c, 1.0 - c)


def put_price(S, K, T, iv, r=R_F):
    """European put, T in years, iv annual. Works on arrays of S (and iv)."""
    S = np.asarray(S, dtype=float); iv = np.asarray(iv, dtype=float)
    if T <= 1e-6: return np.maximum(K - S, 0.0)
    sd = np.maximum(iv, 0.01) * math.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * np.maximum(iv, 0.01) ** 2) * T) / sd
    d2 = d1 - sd
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def put_delta(S, K, T, iv, r=R_F):
    if T <= 1e-6 or iv <= 0: return -1.0 if S < K else 0.0
    sd = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / sd
    return float(_ncdf(d1)) - 1.0


# ---------------------------------------------------------------- the distribution
def simulate(px, sigma, tilt, horizon=HORIZON, earn_jump_sd=0.0, n=N_PATHS, seed=SEED):
    """Terminal prices at the exit horizon. sigma annual (decimal); tilt in [0, 0.6] = bearish drift in sigmas."""
    rng = np.random.default_rng(seed)
    T = horizon / 252.0
    sd = sigma * math.sqrt(T)
    z = rng.standard_normal(n)
    wide = rng.random(n) < 0.08                                   # fat tails: 8% of paths are 1.8x as volatile...
    z = np.where(wide, z * 1.8, z) / math.sqrt(0.92 + 0.08 * 1.8 ** 2)   # ...with the total variance held at sigma^2, so the
                                                                    # tails are a shape, not free volatility that makes every option look cheap
    mu = -tilt * sd - 0.5 * sd * sd
    lr = mu + sd * z
    if earn_jump_sd > 0:
        lr = lr + earn_jump_sd * rng.standard_normal(n)           # the event: symmetric jump, size from the IV premium
    return px * np.exp(lr)


def exit_iv(iv_now, paths, px, crush=1.0):
    """Skew-aware exit IV per path: down moves lift IV, rallies deflate it. crush < 1 after an earnings report."""
    move = paths / px - 1.0
    adj = np.clip(-0.5 * move, -0.15, 0.15)
    return np.maximum(iv_now * (1.0 + adj) * crush, 0.08)


def evaluate(paths, px, K, ask, dte, iv, horizon=HORIZON, crush=1.0):
    """Reprice one put on every path at the exit; return the numbers the ranking uses."""
    T_left = max(dte - horizon * 365.0 / 252.0, 0) / 365.0        # dte is calendar days, the horizon is sessions
    ivx = exit_iv(iv, paths, px, crush)
    val = put_price(paths, K, T_left, ivx)
    val = np.maximum(val, np.maximum(K - paths, 0.0))              # never below intrinsic
    ret = val / ask - 1.0
    win = ret > 0
    pop = float(win.mean())
    ev = float(ret.mean())
    avg_win = float(ret[win].mean()) if win.any() else 0.0
    avg_loss = float(-ret[~win].mean()) if (~win).any() else 1e-9
    return dict(pop=round(pop * 100), ev=round(ev * 100), asym=round(avg_win / max(avg_loss, 1e-9), 2),
                up90=round(float(np.percentile(ret, 90)) * 100), p2x=round(float((ret >= 1.0).mean()) * 100),
                p50=round(float((ret >= 0.5).mean()) * 100), med=round(float(np.median(ret)) * 100))


def scenario_grid(px, K, ask, dte, iv, moves=(-15, -10, -5, 0, 5), days=(3, 5, 10)):
    """What the contract is worth if the stock is X% away after N sessions (IV moved by the skew rule). Returns % P/L."""
    out = []
    for m in moves:
        S = px * (1 + m / 100.0)
        row = []
        for d in days:
            T_left = max(dte - d * 365.0 / 252.0, 0) / 365.0
            v = float(put_price(np.array([S]), K, T_left, exit_iv(iv, np.array([S]), px))[0])
            v = max(v, max(K - S, 0.0))
            row.append(round((v / ask - 1) * 100))
        out.append(row)
    return dict(moves=list(moves), days=list(days), pl=out)


# ---------------------------------------------------------------- stock-side structure
def structure(cl, hi, lo, op, vol):
    """Bearish technical structure, 0-15, plus the flags that explain it. Arrays are daily bars, oldest first."""
    n = len(cl); pts = 0; fl = []; d = {}
    s20 = float(cl[-20:].mean()); s50 = float(cl[-50:].mean()) if n >= 50 else s20
    d["below20"] = bool(cl[-1] < s20); d["below50"] = bool(cl[-1] < s50); d["stackdn"] = bool(s20 < s50)
    if d["below20"] and d["below50"] and d["stackdn"]: pts += 4; fl.append("below the 20 & 50-day, 20 under 50")
    elif d["below20"] and d["below50"]: pts += 3; fl.append("below the 20 & 50-day")
    elif d["below20"]: pts += 1
    # lower highs and lower lows, last 10 sessions vs the 10 before
    d["lhll"] = bool(n >= 20 and hi[-10:].max() < hi[-20:-10].max() and lo[-10:].min() < lo[-20:-10].min())
    if d["lhll"]: pts += 3; fl.append("lower highs and lower lows")
    # break of the prior 20-day low inside the last 5 sessions
    d["brk20"] = False
    if n >= 26:
        for i in range(n - 5, n):
            if cl[i] < float(lo[i - 20:i].min()): d["brk20"] = True
    # failed retest: support broke (a low from 10-30 sessions ago, taken out in the last 10), price bounced back
    # up to it (within 3%) and is closing under it again -- the playbook's favourite pattern
    d["failed_retest"] = False; d["support"] = None
    if n >= 32:
        sup = float(lo[-30:-10].min()); d["support"] = round(sup, 2)
        broke = bool((cl[-10:-2] < sup).any())
        bounced = bool(hi[-5:].max() >= sup * 0.97)
        d["failed_retest"] = broke and bounced and cl[-1] < sup and cl[-1] < hi[-1] * 0.995
    if d["failed_retest"]: pts += 4; fl.append(f"failed retest of ${d['support']} support")
    elif d["brk20"]: pts += 3; fl.append("broke the 20-day low")
    # volume: heavier on down days than up days (20 sessions)
    ch = np.diff(cl[-21:]); v = vol[-20:]
    dn = v[ch < 0].mean() if (ch < 0).any() else 0.0; up = v[ch > 0].mean() if (ch > 0).any() else 0.0
    d["dv_ratio"] = round(float(dn / up), 2) if up else None
    if d["dv_ratio"] is not None and d["dv_ratio"] >= 1.5: pts += 2; fl.append(f"down-day volume {d['dv_ratio']}x up-day")
    elif d["dv_ratio"] is not None and d["dv_ratio"] >= 1.2: pts += 1
    # weak closes: average position in the day's range over the last 5 sessions
    rng_ = hi[-5:] - lo[-5:]
    pos = np.where(rng_ > 0, (cl[-5:] - lo[-5:]) / np.where(rng_ > 0, rng_, 1), 0.5)
    d["clpos5"] = round(float(pos.mean()), 2)
    if d["clpos5"] <= 0.35: pts += 1; fl.append("closing near the lows")
    # gap down in the last 6 sessions that has not been recovered
    d["gapdn"] = None
    for i in range(max(1, n - 6), n):
        g = (op[i] / cl[i - 1] - 1) * 100
        if g <= -5 and cl[-1] < cl[i - 1]: d["gapdn"] = round(float(g), 1)
    if d["gapdn"] is not None: pts += 1; fl.append(f"gap down ({d['gapdn']}%) not recovered")
    # momentum: 20-day rate of change turning down while price sits under the 20-day
    d["roc20"] = round(float(cl[-1] / cl[-21] - 1) * 100, 1) if n > 21 else None
    return min(pts, 15), d, fl


def relative_weakness(cl, bench):
    """RS vs SPY and QQQ over 1/3/5/10/20 sessions (pp). 0-10 points; the doc wants weakness on several windows."""
    out = {}
    for n_, key in ((1, "1"), (3, "3"), (5, "5"), (10, "10"), (20, "20")):
        r = float(cl[-1] / cl[-1 - n_] - 1) * 100 if len(cl) > n_ else None
        out["rs" + key + "_spy"] = round(r - bench["spy"][key], 1) if (r is not None and bench["spy"].get(key) is not None) else None
        out["rs" + key + "_qqq"] = round(r - bench["qqq"][key], 1) if (r is not None and bench["qqq"].get(key) is not None) else None
    pts = 0; fl = []
    w5, w10, w20 = out.get("rs5_spy"), out.get("rs10_spy"), out.get("rs20_spy")
    if w10 is not None:
        pts += 4 if w10 <= -8 else (3 if w10 <= -4 else (1 if w10 < 0 else 0))
    if w20 is not None:
        pts += 3 if w20 <= -12 else (2 if w20 <= -6 else (1 if w20 < 0 else 0))
    if w5 is not None:
        pts += 3 if w5 <= -5 else (2 if w5 <= -2 else (1 if w5 < 0 else 0))
    neg = sum(1 for k in ("rs3_spy", "rs5_spy", "rs10_spy", "rs20_spy") if out.get(k) is not None and out[k] < 0)
    out["neg_windows"] = neg
    if w10 is not None and w10 <= -4: fl.append(f"{w10:+}pp vs SPY / 10d")
    if neg >= 4: fl.append("weaker than SPY on every window")
    return min(pts, 10), out, fl


def tilt_from(tech_pts, rw_pts):
    """How hard to lean the distribution. 0 = no view, 0.6 = the full bearish drift."""
    return round(min(0.6, 0.35 * tech_pts / 15 + 0.25 * rw_pts / 10), 3)

#!/usr/bin/env python3
"""
Luck Society Morning Brief — the 8:30 am "which sector, which stocks" routine, automated.

  python scanner/morning.py brief    pre-market: futures + sector ETFs + holdings → Claude picks → Discord
  python scanner/morning.py score    after the close: how did this morning's picks actually do → Discord + running hit rate

What it does at 8:30 ET (all free Yahoo data, two batched downloads):
  1. Index / commodity / rate futures and VIX, overnight change.
  2. Pre-market change and pre-market volume for the 11 sector ETFs + a few themes (semis, biotech, banks, gold, builders, ARK).
  3. Pre-market change + volume + 5-day return for ~12 liquid names inside every sector.
  4. Sends that whole table to Claude (secret ANTHROPIC_API_KEY) and asks for exactly one bullish sector, one bearish sector,
     five names expected to lead each, and one line of why. Without the key it posts the pure ranking (ETF pre-market move,
     names sorted by pre-market move with volume) so the morning never goes silent.
  5. Posts to Discord (secret DISCORD_WEBHOOK) and saves data/morning/<date>.json for the scorecard.

Exit 3 = nothing to do (weekend, already posted today, outside the pre-market window) so a fallback cron trigger is harmless.
This is a momentum snapshot at 8:30 — pre-market prints are thin and reverse often. The scorecard exists so the hit rate,
not the confidence of the prose, decides whether it is worth reading.
"""
import json, os, sys, logging, time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, requests, yfinance as yf

log = logging.getLogger("morning"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data" / "morning"
ET = ZoneInfo("America/New_York")
MODEL = os.environ.get("MORNING_MODEL", "claude-sonnet-5-5")

FUTURES = {"ES=F": "S&P 500", "NQ=F": "Nasdaq 100", "YM=F": "Dow", "RTY=F": "Russell 2000", "CL=F": "Crude oil", "GC=F": "Gold",
           "SI=F": "Silver", "HG=F": "Copper", "NG=F": "Nat gas", "ZN=F": "10-yr note", "DX-Y.NYB": "Dollar", "^VIX": "VIX"}

# sector ETF -> (name, liquid holdings). Edit freely; nothing else depends on the exact list.
SECTORS = {
    "XLK":  ("Technology",     "AAPL MSFT NVDA AVGO ORCL CRM AMD ADBE CSCO ACN INTC QCOM TXN PLTR MU AMAT"),
    "XLC":  ("Communication",  "META GOOGL NFLX DIS TMUS VZ T CMCSA EA TTWO WBD RBLX SPOT"),
    "XLY":  ("Consumer disc.", "AMZN TSLA HD MCD NKE LOW SBUX BKNG TJX CMG LULU ABNB GM F"),
    "XLP":  ("Staples",        "PG COST WMT KO PEP PM MDLZ MO CL KHC TGT KR DG"),
    "XLF":  ("Financials",     "JPM BRK-B V MA BAC WFC GS MS C SCHW AXP BLK COIN"),
    "XLV":  ("Health care",    "LLY UNH JNJ ABBV MRK PFE TMO AMGN ISRG GILD CVS MRNA HIMS"),
    "XLI":  ("Industrials",    "GE CAT RTX UNP HON BA DE LMT UPS ETN GD FDX UBER"),
    "XLE":  ("Energy",         "XOM CVX COP SLB EOG MPC PSX OXY VLO HAL DVN FANG"),
    "XLB":  ("Materials",      "LIN SHW FCX NEM APD ECL NUE DOW CTVA VMC ALB CLF"),
    "XLU":  ("Utilities",      "NEE SO DUK CEG SRE AEP VST D EXC PCG XEL"),
    "XLRE": ("Real estate",    "PLD AMT EQIX WELL SPG PSA O DLR CCI VICI"),
    "SMH":  ("Semiconductors", "NVDA AMD AVGO TSM MU AMAT LRCX KLAC ARM MRVL INTC QCOM SMCI"),
    "XBI":  ("Biotech",        "MRNA VRTX REGN ALNY BNTX EXAS SRPT NTLA CRSP IONS INSM"),
    "KRE":  ("Regional banks", "KEY TFC RF HBAN FITB CFG ZION WAL CMA MTB"),
    "GDX":  ("Gold miners",    "NEM GOLD AEM FNV WPM KGC AU HMY PAAS"),
    "XHB":  ("Homebuilders",   "DHI LEN PHM NVR TOL KBH HD LOW BLDR"),
    "ARKK": ("Speculative growth", "TSLA COIN ROKU PLTR HOOD SHOP CRSP RBLX SOFI DKNG"),
}


# ------------------------------------------------------------------------------------------------ data
def _pct(a, b):
    return None if a is None or b is None or not b or np.isnan(a) or np.isnan(b) else round((a / b - 1) * 100, 2)


def snapshot(now_et):
    """One row per symbol: prev close, pre-market last, pre-market %, pre-market volume vs 20-day average, 5d %."""
    stocks = sorted({s for _, h in SECTORS.values() for s in h.split()})
    syms = list(FUTURES) + list(SECTORS) + stocks
    daily = yf.download(syms, period="40d", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    intra = yf.download(syms, period="1d", interval="1m", group_by="ticker", prepost=True, auto_adjust=False, threads=True, progress=False)
    today = now_et.date()
    out = {}
    for s in syms:
        try:
            d = daily[s].dropna(subset=["Close"])
            if len(d) and d.index[-1].date() == today: d = d.iloc[:-1]          # a partial bar for today would poison "prev close"
            if not len(d): continue
            prev = float(d["Close"].iloc[-1]); c5 = float(d["Close"].iloc[-6]) if len(d) > 6 else None
            avgv = float(d["Volume"].tail(20).mean()) if "Volume" in d else 0.0
            row = dict(prev=round(prev, 2), d5=_pct(prev, c5), pm=None, pmc=None, pmv=0, rv=None)
            try:
                m = intra[s].dropna(subset=["Close"])
                m = m[m.index.tz_convert(ET).date == today] if len(m) else m
                if len(m):
                    last = float(m["Close"].iloc[-1]); vol = float(m["Volume"].sum())
                    row.update(pm=round(last, 2), pmc=_pct(last, prev), pmv=int(vol), rv=round(vol / avgv * 100, 1) if avgv else None)
            except Exception: pass
            out[s] = row
        except Exception as e:
            log.debug("%s: %s", s, e)
    log.info("snapshot: %d/%d symbols, %d with a pre-market print", len(out), len(syms), sum(1 for r in out.values() if r["pm"]))
    return out


def sector_table(snap):
    rows = []
    for etf, (name, hold) in SECTORS.items():
        r = snap.get(etf, {}); hs = [snap[h] for h in hold.split() if h in snap and snap[h].get("pmc") is not None]
        rows.append(dict(etf=etf, name=name, pmc=r.get("pmc"), d5=r.get("d5"), rv=r.get("rv"), n=len(hs),
                         breadth=round(float(np.mean([h["pmc"] for h in hs])), 2) if hs else None,
                         up=sum(1 for h in hs if h["pmc"] > 0)))
    return rows


def names_in(snap, etf, side, k=5):
    hs = [(h, snap[h]) for h in SECTORS[etf][1].split() if h in snap and snap[h].get("pmc") is not None]
    hs = [(h, r) for h, r in hs if r["pmv"] > 0] or hs
    hs.sort(key=lambda x: x[1]["pmc"], reverse=(side == "bull"))
    return [dict(t=h, pmc=r["pmc"], pm=r["pm"], rv=r["rv"], d5=r["d5"]) for h, r in hs[:k]]


# ------------------------------------------------------------------------------------------------ Claude
PROMPT = """You are a pre-market desk analyst. It is {when} ET, one hour before the US open. Below is a JSON snapshot: overnight
change in index/commodity/rate futures and VIX; for each sector ETF its pre-market % change (pmc), pre-market volume as % of
its 20-day average daily volume (rv), 5-day return (d5), and breadth (mean pre-market change of its liquid holdings, and how
many are up); and for every holding its pre-market price, pmc, rv and d5.

Task, exactly as a discretionary trader would answer it:
1. Which ONE sector is most likely to be the strongest performer today, and which ONE the weakest? Use the futures context
   (rates, dollar, crude, gold, VIX) plus the pre-market tape, not just the biggest pre-market print.
2. Inside the bullish sector, the 5 stocks most likely to move up the most today; inside the bearish sector, the 5 most likely
   to move down the most. Prefer names with real pre-market volume; a 1% move on no volume is noise.
3. One line of reasoning per pick, one short paragraph for each sector call, and one honest line on what would invalidate it.

Respond with ONLY a JSON object, no markdown fences, no preamble:
{{"bull_sector": "XLK", "bull_why": "...", "bear_sector": "XLE", "bear_why": "...",
  "bull_picks": [{{"t": "NVDA", "why": "..."}}, ...5 items], "bear_picks": [{{"t": "XOM", "why": "..."}}, ...5 items],
  "market_read": "one or two sentences on the overall tape", "invalidation": "..."}}
Tickers must come from the snapshot. Keep every "why" under 25 words.

SNAPSHOT:
{data}"""


def ask_claude(payload, when):
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key: log.info("ANTHROPIC_API_KEY not set — posting the quant ranking only"); return None
    body = dict(model=MODEL, max_tokens=1500, messages=[dict(role="user", content=PROMPT.format(when=when, data=json.dumps(payload, separators=(",", ":"))))])
    for attempt in range(3):
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=120, json=body,
                              headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            if r.status_code >= 300: log.error("claude %s: %s", r.status_code, r.text[:300]); time.sleep(5); continue
            text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
            text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            out = json.loads(text)
            if out.get("bull_sector") in SECTORS and out.get("bear_sector") in SECTORS: return out
            log.error("claude returned unknown sectors: %s", text[:200])
        except Exception as e:
            log.error("claude attempt %d: %s", attempt + 1, e); time.sleep(5)
    return None


# ------------------------------------------------------------------------------------------------ Discord
def post(embeds, content=None):
    hook = os.environ.get("DISCORD_WEBHOOK", "").strip()
    if not hook: log.warning("DISCORD_WEBHOOK not set — printing instead"); print(json.dumps(embeds, indent=1)); return True
    payload = dict(username="Luck Society Morning Brief", embeds=embeds)
    if content: payload["content"] = content
    r = requests.post(hook, json=payload, timeout=20)
    if r.status_code >= 300: log.error("discord %s: %s", r.status_code, r.text[:200]); return False
    return True


def fmt(v, suf="%", plus=True):
    if v is None: return "—"
    return f"{v:+.2f}{suf}" if plus else f"{v:.2f}{suf}"


def picks_lines(picks, snap):
    out = []
    for p in picks:
        r = snap.get(p["t"], {})
        line = f"**{p['t']}** {fmt(r.get('pmc'))} pre · ${r.get('pm') or r.get('prev') or 0:.2f}"
        if r.get("rv"): line += f" · vol {r['rv']:.0f}% of avg"
        if p.get("why"): line += f"\n  ↳ {p['why']}"
        out.append(line)
    return "\n".join(out) or "no pre-market prints yet"


def brief_embeds(now_et, snap, sectors, result, source):
    fut = " · ".join(f"{n} {fmt(snap[s]['pmc'])}" for s, n in FUTURES.items() if s in snap and snap[s].get("pmc") is not None)
    ranked = sorted([x for x in sectors if x["pmc"] is not None], key=lambda x: x["pmc"], reverse=True)
    board = "\n".join(f"`{x['etf']:<5}` {fmt(x['pmc'])}  breadth {fmt(x['breadth'])} ({x['up']}/{x['n']} up)" for x in ranked)
    b, s = result["bull_sector"], result["bear_sector"]
    color = 0x2FB35A if (snap.get("ES=F", {}).get("pmc") or 0) >= 0 else 0xE8604F
    head = dict(title=f"Morning Brief — {now_et.strftime('%a %b %d')} · {now_et.strftime('%I:%M %p').lstrip('0')} ET", color=color,
                description=(result.get("market_read") or "") + f"\n\n**Futures**\n{fut}",
                fields=[dict(name="Sector pre-market board", value=board[:1024] or "—", inline=False)],
                footer=dict(text=f"picks by {source} · pre-market prints are thin, size accordingly"))
    bull = dict(title=f"🟢 Bullish sector: {SECTORS[b][0]} ({b}) {fmt(snap.get(b, {}).get('pmc'))}", color=0x2FB35A,
                description=(result.get("bull_why") or "").strip(),
                fields=[dict(name="Expected leaders", value=picks_lines(result["bull_picks"], snap)[:1024], inline=False)])
    bear = dict(title=f"🔴 Bearish sector: {SECTORS[s][0]} ({s}) {fmt(snap.get(s, {}).get('pmc'))}", color=0xE8604F,
                description=(result.get("bear_why") or "").strip(),
                fields=[dict(name="Expected laggards", value=picks_lines(result["bear_picks"], snap)[:1024], inline=False)])
    if result.get("invalidation"): bear["fields"].append(dict(name="What breaks it", value=result["invalidation"][:1024], inline=False))
    return [head, bull, bear]


# ------------------------------------------------------------------------------------------------ modes
def brief():
    now = datetime.now(ET); today = now.strftime("%Y-%m-%d")
    if now.weekday() >= 5: log.info("weekend"); return 3
    if (DATA / f"{today}.json").exists(): log.info("already posted today"); return 3
    hhmm = now.hour * 100 + now.minute
    if not (700 <= hhmm <= 925) and os.environ.get("MORNING_FORCE") != "1": log.info("outside 07:00–09:25 ET; set MORNING_FORCE=1 to run anyway"); return 3
    snap = snapshot(now); sectors = sector_table(snap)
    if sum(1 for x in sectors if x["pmc"] is not None) < 6: log.error("too little pre-market data — try again later"); return 1
    payload = dict(futures={n: snap[s] for s, n in FUTURES.items() if s in snap},
                   sectors={x["etf"]: dict(name=x["name"], pmc=x["pmc"], rv=x["rv"], d5=x["d5"], breadth=x["breadth"], up=f"{x['up']}/{x['n']}") for x in sectors},
                   holdings={etf: {h: {k: snap[h][k] for k in ("pm", "pmc", "rv", "d5")} for h in hold.split() if h in snap}
                             for etf, (_, hold) in SECTORS.items()})
    result = ask_claude(payload, now.strftime("%I:%M %p").lstrip("0")); source = MODEL
    if not result:
        ranked = sorted([x for x in sectors if x["pmc"] is not None], key=lambda x: x["pmc"])
        b, s = ranked[-1]["etf"], ranked[0]["etf"]; source = "pre-market ranking (no ANTHROPIC_API_KEY)"
        result = dict(bull_sector=b, bear_sector=s, bull_picks=names_in(snap, b, "bull"), bear_picks=names_in(snap, s, "bear"),
                      bull_why=f"Strongest sector ETF pre-market; breadth {ranked[-1]['up']}/{ranked[-1]['n']} up.",
                      bear_why=f"Weakest sector ETF pre-market; breadth {ranked[0]['up']}/{ranked[0]['n']} up.", market_read="", invalidation="")
    # keep only tickers we actually have, so the scorecard can price them
    for k in ("bull_picks", "bear_picks"): result[k] = [p for p in result[k] if p.get("t") in snap][:5]
    embeds = brief_embeds(now, snap, sectors, result, source)
    if not post(embeds): return 1
    DATA.mkdir(parents=True, exist_ok=True)
    json.dump(dict(date=today, asof=now.isoformat(timespec="seconds"), source=source, picks=result, sectors=sectors,
                   snap={t: snap[t] for t in [result["bull_sector"], result["bear_sector"]] + [p["t"] for p in result["bull_picks"] + result["bear_picks"]] if t in snap}),
              open(DATA / f"{today}.json", "w"), separators=(",", ":"))
    log.info("posted: bull %s bear %s", result["bull_sector"], result["bear_sector"])
    return 0


def score():
    """After the close: open→close and prev-close→close for every pick and both sector ETFs; running hit rate."""
    now = datetime.now(ET); today = now.strftime("%Y-%m-%d"); f = DATA / f"{today}.json"
    if not f.exists(): log.info("no brief for today"); return 3
    d = json.load(open(f))
    if d.get("scored"): log.info("already scored"); return 3
    P = d["picks"]; syms = [P["bull_sector"], P["bear_sector"]] + [p["t"] for p in P["bull_picks"] + P["bear_picks"]]
    h = yf.download(syms, period="5d", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    def day(s):
        try:
            x = h[s].dropna(subset=["Close"]); x = x[x.index.date == now.date()]
            if not len(x): return None
            r = x.iloc[-1]; prev = d["snap"].get(s, {}).get("prev")
            return dict(oc=_pct(float(r["Close"]), float(r["Open"])), cc=_pct(float(r["Close"]), prev) if prev else None)
        except Exception: return None
    res = {s: day(s) for s in syms}
    bs, ss = res.get(P["bull_sector"]), res.get(P["bear_sector"])
    sector_win = bool(bs and ss and bs["cc"] is not None and ss["cc"] is not None and bs["cc"] > ss["cc"])
    bull_hits = [p["t"] for p in P["bull_picks"] if (res.get(p["t"]) or {}).get("cc") is not None and res[p["t"]]["cc"] > 0]
    bear_hits = [p["t"] for p in P["bear_picks"] if (res.get(p["t"]) or {}).get("cc") is not None and res[p["t"]]["cc"] < 0]
    d.update(scored=True, result=res, sector_win=sector_win, bull_hits=len(bull_hits), bear_hits=len(bear_hits),
             n_bull=len(P["bull_picks"]), n_bear=len(P["bear_picks"]))
    json.dump(d, open(f, "w"), separators=(",", ":"))
    # running tally
    days = [json.load(open(p)) for p in sorted(DATA.glob("20*.json"))]; days = [x for x in days if x.get("scored")]
    n = len(days); sw = sum(1 for x in days if x["sector_win"])
    ph = sum(x["bull_hits"] + x["bear_hits"] for x in days); pn = sum(x["n_bull"] + x["n_bear"] for x in days)
    json.dump(dict(days=n, sector_win_rate=round(sw / n * 100, 1) if n else None, pick_hit_rate=round(ph / pn * 100, 1) if pn else None,
                   asof=today), open(DATA / "scorecard.json", "w"))
    def lines(picks):
        return "\n".join(f"**{p['t']}** {fmt((res.get(p['t']) or {}).get('cc'))} close / {fmt((res.get(p['t']) or {}).get('oc'))} from open" for p in picks) or "—"
    e = dict(title=f"Scorecard — {now.strftime('%a %b %d')}", color=0x2FB35A if sector_win else 0xE8604F,
             description=(f"Bull **{P['bull_sector']}** {fmt(bs['cc'] if bs else None)} vs bear **{P['bear_sector']}** {fmt(ss['cc'] if ss else None)} → "
                          f"{'✅ sector call right' if sector_win else '❌ sector call wrong'}\n"
                          f"Picks: {len(bull_hits)}/{len(P['bull_picks'])} bulls closed up · {len(bear_hits)}/{len(P['bear_picks'])} bears closed down"),
             fields=[dict(name="Bull picks", value=lines(P["bull_picks"])[:1024], inline=True), dict(name="Bear picks", value=lines(P["bear_picks"])[:1024], inline=True)],
             footer=dict(text=f"running: {n} days · sector call right {round(sw / n * 100) if n else 0}% · picks right {round(ph / pn * 100) if pn else 0}% (coin flip = 50%)"))
    return 0 if post([e]) else 1


if __name__ == "__main__":
    mode = (sys.argv[1] if len(sys.argv) > 1 else "brief").lower()
    sys.exit(score() if mode == "score" else brief())

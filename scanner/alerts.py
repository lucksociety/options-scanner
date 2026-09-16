#!/usr/bin/env python3
"""
Luck Society Option Scanner — Discord alerts.

Runs right after scan.py. Reads data/latest.json and data/puts/latest.json, compares with what was already sent today
(data/alerts/sent.json) and posts a Discord embed for anything new:

  calls   score >= 70, or a name entering the top 3 for the first time today
  puts    score >= 60, or a name entering the top 3, or a "reversal day" flag with score >= 50

Needs the DISCORD_WEBHOOK secret (Discord channel -> Edit channel -> Integrations -> Webhooks -> New webhook -> Copy URL).
Silently does nothing when the secret is missing, so the scan never fails because of alerts.
Usage: python scanner/alerts.py
"""
import json, os, sys, logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

log = logging.getLogger("alerts"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data"; SENT = DATA / "alerts" / "sent.json"
ET = ZoneInfo("America/New_York")
PAGE = "https://lucksociety.github.io/options-scanner/"
SIDES = ("calls", "puts", "breakout")
THRESH = {"calls": 70, "puts": 60, "breakout": 70}
COLOR = {"calls": 0x2FB35A, "puts": 0xE8604F, "breakout": 0x3B82F6}
LABEL = {"calls": "Calls", "puts": "Puts", "breakout": "Breakout"}
MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def code(t, p, side):
    d = datetime.strptime(p["exp"], "%Y-%m-%d"); k = p["strike"]
    return f"{t} {MON[d.month-1]} {d.day} ${k:g} {'P' if side == 'puts' else 'C'}"

def embed(side, rank, o, why):
    p = o["play"]
    fields = [dict(name="Contract", value=f"**{code(o['t'], p, side)}**  {p['bid']:.2f} × {p['ask']:.2f} (last {p['last']:.2f})", inline=False),
              dict(name="Score", value=f"**{round(o['score'])}** · {('pressure' if side=='puts' else 'fuel')} {round(o['fuel'])} · {({'calls':'bottom','puts':'topping'}.get(side,'breakout'))} {round(o['bot'])} · contract {round(o['opt'])}", inline=True),
              dict(name="Needs", value=f"{'−' if side=='puts' else '+'}{round(p['be'])}% to B/E · {p['dte']}d · OI {p['oi']:,}", inline=True)]
    stats = (f"Short float {o.get('sf') or 0:.1f}% · DTC {o.get('sr') or 0:.1f} · RSI {round(o.get('rsi') or 0)}" if side != "puts"
             else f"Run +{round(o.get('pw') or 0)}% / +{round(o.get('p10') or 0)}% · RSI {round(o.get('rsi') or 0)} · SI {o.get('sf') or 0:.1f}%")
    fields.append(dict(name="Stock", value=f"${o['px']:.2f} · {stats}", inline=False))
    if o.get("flags"): fields.append(dict(name="Flags", value=" · ".join(o["flags"][:6]), inline=False))
    return dict(title=f"#{rank} {o['t']} — {o['co']}"[:256], url=f"https://finviz.com/quote.ashx?t={o['t']}",
                description=why, color=COLOR[side], fields=fields,
                footer=dict(text=f"Luck Society Option Scanner · {LABEL[side]} · {datetime.now(ET).strftime('%b %d %I:%M %p ET')}"))

def main():
    hook = os.environ.get("DISCORD_WEBHOOK", "").strip()
    if not hook: log.info("DISCORD_WEBHOOK not set — skipping alerts"); return 0
    SENT.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    try: sent = json.load(open(SENT))
    except Exception: sent = {}
    if sent.get("date") != today: sent = {"date": today, "keys": []}
    keys = set(sent["keys"]); embeds = []
    for side in SIDES:
        f = DATA / ("latest.json" if side == "calls" else f"{side}/latest.json")
        if not f.exists(): continue
        try: d = json.load(open(f))
        except Exception as e: log.warning("%s: %s", f, e); continue
        for i, o in enumerate(d.get("top", [])[:10], 1):
            if not o.get("play"): continue
            why = []
            if o["score"] >= THRESH[side]: why.append(f"Score {round(o['score'])} ≥ {THRESH[side]}")
            if i <= 3: why.append(f"Top {i} on the {side} board")
            if side == "puts" and o["score"] >= 50 and any("reversal" in fl for fl in o.get("flags", [])): why.append("Reversal day")
            if not why: continue
            k = f"{side}:{o['t']}:{'hi' if o['score'] >= THRESH[side] else 'top3'}"
            if k in keys: continue
            keys.add(k); embeds.append(embed(side, i, o, " · ".join(why)))
    if not embeds: log.info("nothing new to alert"); return 0
    for i in range(0, len(embeds), 5):    # Discord allows 10 embeds per message; keep it readable
        payload = dict(username="Luck Society Scanner", content=f"**New setups** — {PAGE}" if i == 0 else None, embeds=embeds[i:i+5])
        r = requests.post(hook, json={k: v for k, v in payload.items() if v is not None}, timeout=20)
        if r.status_code >= 300: log.error("discord %s: %s", r.status_code, r.text[:200]); return 0
    sent["keys"] = sorted(keys); json.dump(sent, open(SENT, "w"), separators=(",", ":"))
    log.info("sent %d alerts", len(embeds))
    return 0

if __name__ == "__main__":
    sys.exit(main())

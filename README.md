# Luck Society Option Scanner

Two boards, one engine, zero servers. Runs on GitHub Actions, publishes to GitHub Pages.

- **Calls** — heavily shorted (≥ 15% short float), $1–$10, optionable stocks that look bottomed → best OTM call ≤ $0.25, 2–6 weeks out.
- **Puts** — $3–$50 stocks that ran too far, too fast (or gapped up and failed) with a forced seller behind them (cash burn,
  shelf filings, lockup expiry) → best OTM put ≤ $0.35, 3–6 weeks out.

Live page: https://lucksociety.github.io/options-scanner/

## What runs when

| workflow | schedule | what it does |
|---|---|---|
| **Luck Society Option Scanner** (`scan.yml`) | every 5 min, 08:25–16:10 ET weekdays | first run of the day = full screen + deep scan; later runs re-quote and rescore. Posts Discord alerts, commits `data/`. |
| **Luck Society Track Record** (`track.yml`) | 5:40 pm ET weekdays | re-checks every pick from the last 45 days: stock +5/+10/+20 sessions, option mark, hit/bust. Writes `data/track/`. |

Both can be started by hand from the **Actions** tab (or the page's **Scan now** button).

## Data sources (all free)

- **Yahoo Finance** via `yfinance` — screener, price history, short float / days-to-cover, earnings dates, pre-market quotes, option chains.
- **iBorrowDesk** (Interactive Brokers stock-loan data) — borrow fee and shares available, near real time.
- **SEC EDGAR** — shelf / offering filings in the last 120 days (dilution), 8-Ks in the last 5 days (news catalyst).
- **Our own IV log** (`data/iv/`) — ATM implied vol per name per day; IV rank switches from a realized-vol proxy to real IV rank once a name has 20 days logged.

## Discord alerts

1. In Discord: channel → **Edit channel → Integrations → Webhooks → New webhook → Copy webhook URL**.
2. On GitHub: **Settings → Secrets and variables → Actions → New repository secret**, name `DISCORD_WEBHOOK`, paste the URL.

That's it. Each scan posts anything new: calls scoring 70+, puts scoring 60+, any name entering the top 3, and puts with a reversal
day and score 50+. Each alert goes out once per day per name. No secret = no alerts, scans still run.

## Scoring (0–100)

**Calls** = squeeze fuel (short float up to 20, days-to-cover up to 7, borrow fee up to 5, borrow availability up to 3; capped at 35)
+ bottoming (RSI 25–45, RSI turning up, recent oversold, near 20-day / 52-week low, volume pickup, pre-market gap up; 0–35)
+ contract (open interest, spread, breakeven; 0–30).

**Puts** = downside pressure (size of the run, extension over the 20-day average, low short interest, cash runway, dilution filings,
lockup window; 0–35) + topping (RSI ≥ 70/60, RSI rolling over, reversal day, failed gap, no new high, volume fade, off the 20-day high,
pre-market gap down; 0–35) + contract (0–30).

Both sides: IV rank ≥ 60 costs the contract 2 pts, ≥ 80 costs 4; earnings inside the contract window costs 8 and is flagged red;
names with no tradeable contract are capped at 40 and rank below anything with one.

## Track record

The dashboard's **Track record** panel shows hit-rate by score bucket. A **hit** = the contract is (or expired) worth 2× the entry ask,
or the stock reached its breakeven within 20 sessions. A **bust** = expired worthless, or down 70%+ with under 5 days left.
Everything else is open and not counted yet. Under ~30 decided picks per bucket it's noise — give it a month.

The **history** dropdown on the page loads any previous morning's board and shows what happened next on each card.
**Took it** logs a paper trade (in your browser only); the tracker fills in the mark and P/L each evening.

## Tuning

Edit `CFG` at the top of `scanner/scan.py` (price band, short-interest cutoffs, run triggers, DTE window, max ask, min OI, max breakeven).
Weights live in `deep()`; alert thresholds in `scanner/alerts.py`; hit/bust definitions in `scanner/track.py`.

## Files

```
index.html                     the dashboard (static; polls data/*.json every 60 s)
assets/                        banner, coin, favicon
scanner/scan.py                the scanner (both sides)
scanner/alerts.py              Discord alerts (runs after each scan)
scanner/track.py               outcome tracker (runs after the close)
.github/workflows/scan.yml     scan schedule
.github/workflows/track.yml    tracker schedule
data/latest.json               calls board        data/puts/latest.json      puts board
data/history/                  one JSON per full scan (calls)   data/puts/history/   (puts)
data/track/summary.json        hit-rates          data/track/<side>/<date>.json   per-pick outcomes
data/iv/<TICKER>.json          daily ATM IV log   data/alerts/sent.json      alert dedupe
```

## Scan now from the page (optional)

Create a fine-grained token at *GitHub → Settings → Developer settings → Personal access tokens → Fine-grained*: repository access =
this repo only, permission **Actions: Read and write**. On the dashboard click ⚙, enter `lucksociety/options-scanner` and the token.
It's stored only in your browser and only ever sent to api.github.com.

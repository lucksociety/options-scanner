# Luck Society Option Scanner

Finds heavily shorted, sub-$10, optionable stocks that look like they've bottomed, and picks the best call
contract under $0.25 expiring 2–6 weeks out. Runs on GitHub Actions, publishes to GitHub Pages, no server needed.

- **Full scan** every trading day at the first run after 08:25 ET: Finviz screener → filter → deep scan of the
  top 40 (price history, RSI, option chains via Yahoo) → score.
- **Quote refresh** every 5 minutes until 16:10 ET: re-quotes the same names and rescores.
- **Page** checks for new data every 60 seconds, shows the data's age, and has a **Scan now** button.

## Setup (about 5 minutes)

1. **Create the repo.** On GitHub, create a new **public** repository (e.g. `squeeze-scanner`). Don't add a README.
2. **Push these files** (from the unzipped folder):
   ```bash
   git init && git add -A && git commit -m "Luck Society Option Scanner"
   git branch -M main
   git remote add origin https://github.com/YOURNAME/squeeze-scanner.git
   git push -u origin main
   ```
3. **Turn on Pages from Actions.** Repo → *Settings* → *Pages* → under *Build and deployment*, set **Source = GitHub Actions**.
4. **Allow the workflow to write.** *Settings* → *Actions* → *General* → *Workflow permissions* → **Read and write permissions** → Save.
5. **Run it once.** *Actions* tab → *Luck Society Option Scanner* → *Run workflow* → mode `full` → *Run workflow*.
   First run takes 3–5 minutes. When it's green, your dashboard is at
   `https://YOURNAME.github.io/squeeze-scanner/`.
6. **(Optional) Scan now from the page.** Create a fine-grained token at *GitHub → Settings → Developer settings →
   Personal access tokens → Fine-grained*: repository access = this repo only, permission **Actions: Read and write**.
   On the dashboard click ⚙, paste `YOURNAME/squeeze-scanner` and the token. It's stored only in your browser.

The scheduled runs start automatically once the workflow file is on `main`. GitHub's cron can lag 5–15 minutes
during busy periods; the "data … ago" indicator on the page tells you how fresh what you're looking at is.

## Tuning

Edit `CFG` at the top of `scanner/scan.py`:

| key | default | meaning |
|---|---|---|
| `price_min` / `price_max` | 1 / 10 | stock price band |
| `si_min` | 15 | minimum short float % (the Finviz URL filter is `sh_short_o15`; change both if you go higher) |
| `avgvol_min` | 300,000 | minimum average daily volume |
| `deep_n` | 40 | how many stage-1 names get the full option-chain scan |
| `min_dte` / `max_dte` | 14 / 45 | expiration window in days |
| `max_ask` | 0.25 | max ask for the call |
| `min_oi` | 25 | minimum open interest |
| `max_be` | 60 | max % move needed to break even at expiry |

Scoring weights live in `deep()` (fuel 0–35, bottoming 0–35, contract 0–30) and are documented on the page itself.

## Files

```
scanner/scan.py            the scanner (modes: full | refresh | auto)
.github/workflows/scan.yml schedule + Scan-now dispatch + Pages deploy
site/index.html            the dashboard (static; polls data/latest.json every 60 s)
site/data/latest.json      newest results (deployed by the workflow, seeded with the Sep 13 2026 scan)
site/data/history/         one JSON per full scan, committed daily by the workflow
```

## Notes and caveats

- Short interest comes from Finviz (FINRA settlement data, published twice a month) — it can be up to two weeks stale.
- Finviz may occasionally rate-limit or block cloud IPs. If a run fails in `stage1`, re-run it; if it persists, the
  workflow log will show a 403 and you'll want to add a proxy or switch the screener source.
- Yahoo quotes outside market hours often show bid 0.00; the score falls back to the last trade.
- This is a screening tool, not advice. Check the chain, news, and pending earnings before trading.

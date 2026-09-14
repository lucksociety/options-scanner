# Luck Society Option Scanner

Finds heavily shorted, sub-$10, optionable stocks that look like they've bottomed, and picks the best call
contract under $0.25 expiring 2–6 weeks out. Runs on GitHub Actions, publishes to GitHub Pages, no server needed.

- **Full scan** every trading day at the first run after 08:25 ET: Finviz screener → filter → deep scan of the
  top 40 (price history, RSI, option chains via Yahoo) → score.
- **Quote refresh** every 5 minutes until 16:10 ET: re-quotes the same names and rescores.
- **Page** checks for new data every 60 seconds, shows the data's age, and has a **Scan now** button.

## Setup

1. **Create a public repo** on GitHub. Name it whatever you like — `lucksociety.github.io` makes the dashboard live at
   `https://lucksociety.github.io/`; any other name (say `option-scanner`) puts it at
   `https://lucksociety.github.io/option-scanner/`. It **must be public** — GitHub Pages doesn't publish private repos on a free account.
2. **Upload these files to the root of the repo** so that `index.html`, `assets/`, `data/`, `scanner/` and `.github/` sit at the top level
   (if you drag-and-drop on github.com, drag the *contents* of the unzipped folder, not the folder itself — and make sure the
   hidden `.github` folder came along; if it didn't, create `.github/workflows/scan.yml` by hand with the file's contents).
3. **Settings → Pages → Build and deployment → Source: "GitHub Actions"** (if it says "Deploy from a branch", change it).
4. **Settings → Actions → General → Workflow permissions → "Read and write permissions"** → Save.
5. **Actions tab → Luck Society Option Scanner → Run workflow → mode `full`.** Wait for the green check (3–5 min), then open the URL from step 1.

The page ships with a seeded scan so it shows something the moment the site is live; scheduled runs replace it during market hours.

## If the page 404s

- The repo is private, or the site was published from a folder other than the one holding `index.html`. Check step 1–2.
- Pages source is "Deploy from a branch" pointed at `/docs` — switch it to GitHub Actions (or branch `main`, folder `/ (root)`, which also works since results are committed to `data/`).
- Nothing deployed yet — run the workflow once (step 5).

## Scan now from the page (optional)

Create a fine-grained token at *GitHub → Settings → Developer settings → Personal access tokens → Fine-grained*:
repository access = this repo only, permission **Actions: Read and write**. On the dashboard click ⚙, enter `lucksociety/REPO-NAME`
and the token. It's stored only in your browser and only ever sent to api.github.com.

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
index.html                 the dashboard (static; polls data/latest.json every 60 s)
assets/                    Luck Society banner, coin, favicon
data/latest.json           newest results (committed by the workflow after every run)
data/history/              one JSON per full scan
scanner/scan.py            the scanner (modes: full | refresh | auto)
.github/workflows/scan.yml schedule + Scan-now dispatch + Pages deploy
```

## Notes and caveats

- Short interest comes from Finviz (FINRA settlement data, published twice a month) — it can be up to two weeks stale.
- Finviz may occasionally rate-limit or block cloud IPs. If a run fails in `stage1`, re-run it; if it persists, the
  workflow log will show a 403 and you'll want to add a proxy or switch the screener source.
- Yahoo quotes outside market hours often show bid 0.00; the score falls back to the last trade.
- This is a screening tool, not advice. Check the chain, news, and pending earnings before trading.

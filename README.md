# Hummingbot Dashboard – Local Backtesting + AI Coach

This repo contains a ready-to-run Hummingbot Dashboard setup with:
- CSV and Binance data for local backtesting
- SOL scalping backtest page with plots and metrics
- Groq AI recommendations that suggest parameter tweaks
- Persistent settings (dates, parameters, API key) across restarts

## Quick start

1) Prerequisites
- Docker Desktop running
- macOS (tested), 8GB+ RAM recommended

2) Start the stack
```bash
cd /Users/njajodia/deploy
docker compose up -d
```
Open http://127.0.0.1:8501

3) Data
- Use Data → Local Data & Date Range to upload CSVs or set a folder
- Or fetch candles from Data → Download Candles
- Cached files live in `/home/dashboard/data` (host mount recommended: `./data:/home/dashboard/data`)

4) SOL backtest
- Pinned → 03 SOL Scalping Backtest
- Source: CSV, paste path (e.g., `/home/dashboard/data/sol_usdt_15m.csv`)
- Adjust parameters → Run Backtest
- See equity curve, metrics, and trades

5) AI coach (Groq)
- In SOL page sidebar, expand “AI Analysis (Groq)”
- Paste key (`gsk_...`) or set env `GROQ_API_KEY`
- Run backtest → click “Get AI Recommendations” (or use the persistent button)
- “Apply AI Suggestions” will autofill the sliders

## Project layout
- `docker-compose.yml` – services; dashboard builds `Dockerfile.dashboard-mpl`
- `Dockerfile.dashboard-mpl` – baked matplotlib for native plots
- `pages/` – mounted Streamlit pages used by the container
- `pages/custom/sol_scalping/` – SOL backtester and config
- `pages/_persist` – mirrored cache for preferences/CSV
- `data/` – put your CSVs here (bind mount to `/home/dashboard/data`)

## Environment
- Parameters, date range, and Groq key are saved in `/home/dashboard/data/user_prefs.json`
- To make data persistent, add the volume under `dashboard → volumes`:
```yaml
- ./data:/home/dashboard/data
```

## Common commands
```bash
# start/stop
docker compose up -d
docker compose down

# rebuild only dashboard image
docker compose build dashboard && docker compose up -d dashboard

# logs
docker logs -f dashboard
```

## Git hygiene
This directory is ready for `git init`:
- `.gitignore` excludes virtualenvs, caches and .env
- Do not commit API keys; if you saved the Groq key, it lives in `data/user_prefs.json`

## Troubleshooting
- App port: http://127.0.0.1:8501
- Blank AI section: run a backtest first, then click the AI button
- Missing CSV: verify path exists inside container (`/home/dashboard/data/...`)

## License
MIT (if you plan to publish). Update this file if different.

import json
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="SOL Scalping Backtest", page_icon="📈", layout="wide")
st.title("📈 SOL Scalping Backtest")

st.markdown("Run your EMA(9/21) + RSI(14) scalping backtest on SOL/USDT 15m data.")

# Persistent paths and preferences
DATA_ROOT = Path("/home/dashboard/data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
PERSIST_ROOT = Path("/home/dashboard/frontend/pages/_persist")  # host-mounted
PERSIST_ROOT.mkdir(parents=True, exist_ok=True)
PREFS_FILE = DATA_ROOT / "user_prefs.json"
PREFS_FILE_MIRROR = PERSIST_ROOT / "user_prefs.json"
DEFAULT_CSV = DATA_ROOT / "sol_usdt_15m.csv"
DEFAULT_CSV_MIRROR = PERSIST_ROOT / "sol_usdt_15m.csv"

def load_prefs() -> dict:
    try:
        if PREFS_FILE.exists():
            return json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_prefs(prefs: dict) -> None:
    try:
        PREFS_FILE.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except Exception:
        pass
    # Mirror to host-mounted persist folder
    try:
        PREFS_FILE_MIRROR.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except Exception:
        pass

def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    url = "https://api.binance.com/api/v3/klines"
    out = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": min(end_ms, cursor + 1000 * 15 * 60 * 1000),
            "limit": 1000,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        kl = r.json()
        if not kl:
            break
        out.extend(kl)
        cursor = kl[-1][0] + 1
    return out

def normalize_binance_klines(raw: list) -> pd.DataFrame:
    rows = [
        (
            pd.to_datetime(r[0], unit="ms"),
            float(r[1]),
            float(r[2]),
            float(r[3]),
            float(r[4]),
            float(r[5]),
        )
        for r in raw
    ]
    return (
        pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

# Import custom backtester
try:
    from frontend.pages.custom.sol_scalping.backtest_sol_scalping import SolScalpingBacktester
    from frontend.pages.custom.sol_scalping.sol_scalping_config import get_config, validate_config
except Exception as e:
    st.error(f"Failed to import backtester: {e}")
    st.stop()

prefs = load_prefs()
saved_start = prefs.get("data", {}).get("start_date")
saved_end = prefs.get("data", {}).get("end_date")
saved_params = prefs.get("sol_scalping_params", {})
groq_prefs = prefs.get("groq", {})
default_start = pd.to_datetime(saved_start).date() if saved_start else pd.Timestamp("2024-07-01").date()
default_end = pd.to_datetime(saved_end).date() if saved_end else pd.Timestamp("2024-12-31").date()

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Select source", ["API (Binance)", "CSV"], index=0)
    if source == "API (Binance)":
        start_date = st.date_input("Start date", value=default_start, key="api_start")
        end_date = st.date_input("End date", value=default_end, key="api_end")
        cache_to_file = st.toggle("Cache to CSV", value=True, help=f"Save fetched klines to {DEFAULT_CSV}")
        fetch_btn = st.button("Fetch from Binance", type="primary")
        # Prefer persistent mirror if it already exists
        default_path = str(DEFAULT_CSV if DEFAULT_CSV.exists() or not DEFAULT_CSV_MIRROR.exists() else DEFAULT_CSV_MIRROR)
        data_file = st.text_input("Cached CSV path", value=default_path)
    else:
        start_date = st.date_input("Start date", value=default_start, key="csv_start")
        end_date = st.date_input("End date", value=default_end, key="csv_end")
        default_path = str(DEFAULT_CSV if DEFAULT_CSV.exists() or not DEFAULT_CSV_MIRROR.exists() else DEFAULT_CSV_MIRROR)
        data_file = st.text_input("CSV file path", value=default_path, help="Columns: timestamp, open, high, low, close, volume")

    st.markdown("### Strategy Parameters")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        ema_fast = st.number_input("EMA Fast", min_value=5, max_value=50, value=int(saved_params.get("ema_fast", 12)))
    with colB:
        ema_slow = st.number_input("EMA Slow", min_value=10, max_value=200, value=int(saved_params.get("ema_slow", 26)))
    with colC:
        rsi_high = st.number_input("RSI Long Threshold", min_value=30, max_value=90, value=int(saved_params.get("rsi_high", 55)))
    with colD:
        rsi_low = st.number_input("RSI Short Threshold", min_value=10, max_value=70, value=int(saved_params.get("rsi_low", 45)))
    colE, colF, colG = st.columns(3)
    with colE:
        vol_thresh = st.number_input("Volume Ratio Threshold", min_value=1.0, max_value=5.0, value=float(saved_params.get("vol_thresh", 1.5)), step=0.1)
    with colF:
        tp_pct = st.number_input("Take Profit %", min_value=0.1, max_value=10.0, value=float(saved_params.get("tp_pct", 3.0)), step=0.1) / 100.0
    with colG:
        sl_pct = st.number_input("Stop Loss %", min_value=0.1, max_value=10.0, value=float(saved_params.get("sl_pct", 1.0)), step=0.1) / 100.0
    max_trades = st.number_input("Max trades/day", min_value=1, max_value=50, value=int(saved_params.get("max_trades", 3)))

    # Save parameters button
    if st.button("Save Parameters"):
        prefs["sol_scalping_params"] = {
            "ema_fast": int(ema_fast),
            "ema_slow": int(ema_slow),
            "rsi_high": int(rsi_high),
            "rsi_low": int(rsi_low),
            "vol_thresh": float(vol_thresh),
            "tp_pct": float(tp_pct * 100.0),  # store as percent for readability
            "sl_pct": float(sl_pct * 100.0),
            "max_trades": int(max_trades),
        }
        save_prefs(prefs)
        st.success("Parameters saved and will persist across restarts.")

    st.divider()
    st.header("Run")
    run_btn = st.button("Run Backtest", type="primary")

# Persist latest date range
prefs.setdefault("data", {})
prefs["data"]["start_date"] = str(start_date)
prefs["data"]["end_date"] = str(end_date)
save_prefs(prefs)

# Sidebar: AI analysis configuration
with st.sidebar.expander("🤖 AI Analysis (Groq)", expanded=False):
    groq_api_key_input = st.text_input(
        "GROQ API Key",
        value=groq_prefs.get("api_key", os.getenv("GROQ_API_KEY", "")),
        type="password",
        help="Starts with gsk_..."
    )
    groq_model = st.selectbox(
        "Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )
    save_key = st.checkbox("Save key to preferences", value=bool(groq_prefs.get("api_key")))
    if save_key and groq_api_key_input:
        prefs.setdefault("groq", {})["api_key"] = groq_api_key_input
        save_prefs(prefs)

if source == "API (Binance)" and 'fetch_btn' in locals() and fetch_btn:
    try:
        start_ms = int(pd.Timestamp(start_date).tz_localize('UTC').timestamp() * 1000)
        end_ms = int(pd.Timestamp(end_date).tz_localize('UTC').timestamp() * 1000)
        raw = fetch_binance_klines('SOLUSDT', '15m', start_ms, end_ms)
        df = normalize_binance_klines(raw)
        if df.empty:
            st.warning("No data returned from Binance.")
        else:
            if cache_to_file:
                Path(data_file).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(data_file, index=False)
                # Also mirror to host-mounted persistent file
                try:
                    DEFAULT_CSV_MIRROR.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(DEFAULT_CSV_MIRROR, index=False)
                except Exception:
                    pass
            st.success(f"Fetched {len(df)} rows{' and cached' if cache_to_file else ''}.")
            # Keep in memory for this session
            st.session_state["sol_api_df"] = df
    except Exception as e:
        st.exception(e)

if run_btn:
    try:
        config = get_config()
        if not validate_config(config):
            st.error("Invalid configuration.")
            st.stop()

        # Inject UI overrides into strategy config
        config["strategy"].update({
            "ema_fast": int(ema_fast),
            "ema_slow": int(ema_slow),
            "rsi_neutral_high": float(rsi_high),
            "rsi_neutral_low": float(rsi_low),
            "volume_threshold": float(vol_thresh),
            "take_profit_pct": float(tp_pct),
            "stop_loss_pct": float(sl_pct),
            "max_trades_per_day": int(max_trades),
        })

        # Persist latest parameters on run
        prefs["sol_scalping_params"] = {
            "ema_fast": int(ema_fast),
            "ema_slow": int(ema_slow),
            "rsi_high": int(rsi_high),
            "rsi_low": int(rsi_low),
            "vol_thresh": float(vol_thresh),
            "tp_pct": float(tp_pct * 100.0),
            "sl_pct": float(sl_pct * 100.0),
            "max_trades": int(max_trades),
        }
        save_prefs(prefs)

        backtester = SolScalpingBacktester(config)
        if source == "API (Binance)" and st.session_state.get("sol_api_df") is not None:
            df = st.session_state["sol_api_df"]
        else:
            # Try selected path, fallback to persistent mirror
            try_paths = [Path(data_file), DEFAULT_CSV, DEFAULT_CSV_MIRROR]
            load_path = None
            for p in try_paths:
                if p and Path(p).exists():
                    load_path = str(p)
                    break
            if not load_path:
                st.error("CSV file not found. Set a valid CSV path or fetch from Binance.")
                st.stop()
            df = backtester.load_data(load_path)
        results = backtester.run_backtest(df)

        st.subheader("Results")
        if "error" in results:
            st.error(results["error"])
            st.stop()

        cols = st.columns(5)
        cols[0].metric("Total Trades", results.get("total_trades", 0))
        cols[1].metric("Win Rate", f"{results.get('win_rate', 0)*100:.1f}%")
        cols[2].metric("Profit Factor", f"{results.get('profit_factor', 0):.2f}")
        cols[3].metric("Max Drawdown", f"{results.get('max_drawdown', 0)*100:.1f}%")
        # Profit consistency signal: equity slope sign
        try:
            eq = pd.DataFrame(backtester.equity_curve)
            slope = 0.0
            verdict = "Neutral"
            if not eq.empty:
                eq["t"] = (pd.to_datetime(eq["timestamp"]) - pd.to_datetime(eq["timestamp"]).min()).dt.total_seconds()
                x = eq["t"].values
                y = eq["equity"].values
                if len(x) >= 2:
                    m = ((x * y).mean() - x.mean() * y.mean()) / (x.var() if x.var() else 1)
                    slope = float(m)
                    verdict = "Consistent Uptrend" if m > 0 else ("Downtrend Risk" if m < 0 else "Flat")
            cols[4].metric("Equity Slope", f"{slope:.2f}", help="> 0 suggests consistent profits; < 0 indicates downtrend risk")
            st.info(f"Profit consistency: {verdict}")
        except Exception:
            cols[4].metric("Equity Slope", "n/a")

        st.write("\n")
        st.subheader("Profit Consistency")
        # Plot with matplotlib if available, else plotly fallback
        plotted = False
        try:
            from frontend.pages.custom.sol_scalping.backtest_sol_scalping import plt as _plt
            if _plt is not None:
                fig = backtester.plot_results(df)
                if fig is not None:
                    import streamlit as _st
                    _st.pyplot(fig, use_container_width=True)
                    plotted = True
        except Exception:
            pass
        if not plotted:
            try:
                import plotly.express as px
                eq = pd.DataFrame(backtester.equity_curve)
                if not eq.empty:
                    fig = px.line(eq, x="timestamp", y="equity", title="Equity Curve (Fallback Plot)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No equity data available for plotting.")
            except Exception as pe:
                st.warning(f"Plotting issue: {pe}")

        # Show recent trades
        st.write("\n")
        st.subheader("Trades (last 200)")
        try:
            trades_df = pd.DataFrame(backtester.trades)
            if not trades_df.empty:
                st.dataframe(trades_df.tail(200), use_container_width=True)
            else:
                st.info("No trades executed.")
        except Exception as te:
            st.warning(f"Trades display issue: {te}")

        # Store last run context for AI
        st.session_state["sol_last_metrics"] = results
        st.session_state["sol_last_params"] = {
            "ema_fast": int(ema_fast),
            "ema_slow": int(ema_slow),
            "rsi_high": int(rsi_high),
            "rsi_low": int(rsi_low),
            "vol_thresh": float(vol_thresh),
            "tp_pct": float(tp_pct * 100.0),
            "sl_pct": float(sl_pct * 100.0),
            "max_trades": int(max_trades),
        }
        st.session_state["sol_last_time"] = {
            "start": str(start_date),
            "end": str(end_date),
            "csv": data_file,
        }

        st.markdown("---")
        st.subheader("🤖 AI Recommendations")
        if st.button("Get AI Recommendations", type="secondary"):
            api_key = groq_api_key_input or os.getenv("GROQ_API_KEY", "")
            if not api_key:
                st.error("Please provide a GROQ API key in the sidebar.")
            else:
                try:
                    # Build concise context
                    summary = {
                        "metrics": {
                            "total_trades": results.get("total_trades"),
                            "win_rate": round(results.get("win_rate", 0), 4),
                            "profit_factor": round(results.get("profit_factor", 0), 4),
                            "max_drawdown": round(results.get("max_drawdown", 0), 4),
                        },
                        "parameters": st.session_state.get("sol_last_params", {}),
                        "data": st.session_state.get("sol_last_time", {}),
                    }
                    system = (
                        "You are a quantitative trading coach. Given backtest metrics and parameters, "
                        "propose 3-5 concrete parameter changes (with new values) to improve profit factor and risk, "
                        "and explain briefly why. Keep under 180 words. Format as numbered list."
                    )
                    payload = {
                        "model": groq_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(summary, indent=2)},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 600,
                    }
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"]
                        # Persist AI output so it survives reruns
                        st.session_state["sol_ai_content"] = content
                        st.session_state["sol_ai_params"] = summary
                        st.markdown(content)
                    else:
                        st.error(f"Groq API error: {resp.status_code} {resp.text}")
                except Exception as e:
                    st.exception(e)

    except Exception as e:
        st.exception(e)

# Persistent AI section (visible after any successful run)
st.markdown("---")
st.subheader("🤖 AI Recommendations")
if "sol_last_metrics" in st.session_state:
    api_key_pref = prefs.get("groq", {}).get("api_key", os.getenv("GROQ_API_KEY", ""))
    with st.expander("Configure (optional)", expanded=False):
        groq_api_key_input2 = st.text_input("GROQ API Key", value=api_key_pref, type="password")
        groq_model2 = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], index=0, key="groq_model_persist")
        if st.checkbox("Save key", value=bool(api_key_pref), key="save_groq_key_persist") and groq_api_key_input2:
            prefs.setdefault("groq", {})["api_key"] = groq_api_key_input2
            save_prefs(prefs)
    if st.button("Get AI Recommendations (persistent)"):
        api_key = groq_api_key_input2 or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            st.error("Please provide a GROQ API key.")
        else:
            try:
                summary = {
                    "metrics": {
                        "total_trades": st.session_state["sol_last_metrics"].get("total_trades"),
                        "win_rate": round(st.session_state["sol_last_metrics"].get("win_rate", 0), 4),
                        "profit_factor": round(st.session_state["sol_last_metrics"].get("profit_factor", 0), 4),
                        "max_drawdown": round(st.session_state["sol_last_metrics"].get("max_drawdown", 0), 4),
                    },
                    "parameters": st.session_state.get("sol_last_params", {}),
                    "data": st.session_state.get("sol_last_time", {}),
                }
                system = (
                    "You are a quantitative trading coach. Given backtest metrics and parameters, "
                    "propose 3-5 concrete parameter changes (with new values) to improve profit factor and risk, "
                    "and explain briefly why. Keep under 180 words. Format as numbered list."
                )
                payload = {
                    "model": groq_model2,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(summary, indent=2)},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 600,
                }
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=30,
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    st.session_state["sol_ai_content"] = content
                    st.session_state["sol_ai_params"] = summary
                    st.markdown(content)
                else:
                    st.error(f"Groq API error: {resp.status_code} {resp.text}")
            except Exception as e:
                st.exception(e)
else:
    st.info("Run a backtest first to enable AI recommendations.")

# Show last AI output (pinned) and controls
if "sol_ai_content" in st.session_state:
    st.markdown("---")
    st.subheader("AI Recommendations (last result)")
    _ai_content_cached = st.session_state.get("sol_ai_content", "")
    st.markdown(_ai_content_cached)  # persisted content
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Clear AI Output"):
            st.session_state.pop("sol_ai_content", None)
            st.session_state.pop("sol_ai_params", None)
            st.success("Cleared.")
    with c2:
        st.download_button(
            label="Download AI Suggestions",
            data=_ai_content_cached,
            file_name="ai_recommendations.md",
            mime="text/markdown",
        )
    with c3:
        # Try to parse common fields from AI output and apply
        if st.button("Apply AI Suggestions"):
            content = st.session_state.get("sol_ai_content", "")
            # crude parsing: look for numbers after field names
            import re
            def find_num(key, cast=int):
                m = re.search(rf"{key}[^0-9]*([0-9]+(?:\.[0-9]+)?)", content, re.IGNORECASE)
                return cast(m.group(1)) if m else None
            new_vals = {}
            v = find_num("ema_fast", int);
            if v is not None: new_vals["ema_fast"] = v
            v = find_num("ema_slow", int);
            if v is not None: new_vals["ema_slow"] = v
            v = find_num("rsi_high", int);
            if v is not None: new_vals["rsi_high"] = v
            v = find_num("rsi_low", int);
            if v is not None: new_vals["rsi_low"] = v
            v = find_num("vol_thresh", float);
            if v is not None: new_vals["vol_thresh"] = v
            v = find_num("tp_pct", float);
            if v is not None: new_vals["tp_pct"] = v
            v = find_num("sl_pct", float);
            if v is not None: new_vals["sl_pct"] = v

            # Update prefs so controls reload with these values
            sp = prefs.get("sol_scalping_params", {})
            sp.update({
                "ema_fast": int(new_vals.get("ema_fast", sp.get("ema_fast", 12))),
                "ema_slow": int(new_vals.get("ema_slow", sp.get("ema_slow", 26))),
                "rsi_high": int(new_vals.get("rsi_high", sp.get("rsi_high", 55))),
                "rsi_low": int(new_vals.get("rsi_low", sp.get("rsi_low", 45))),
                "vol_thresh": float(new_vals.get("vol_thresh", sp.get("vol_thresh", 1.5))),
                "tp_pct": float(new_vals.get("tp_pct", sp.get("tp_pct", 3.0))),
                "sl_pct": float(new_vals.get("sl_pct", sp.get("sl_pct", 1.0))),
            })
            prefs["sol_scalping_params"] = sp
            save_prefs(prefs)
            st.success("Applied AI suggestions to controls. Rerun backtest to evaluate.")

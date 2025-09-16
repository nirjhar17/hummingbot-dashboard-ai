import streamlit as st
from pathlib import Path


def main_page():
    return [st.Page("frontend/pages/landing.py", title="Hummingbot Dashboard", icon="📊", url_path="landing")]


def _pinned_pages():
    """Dynamically load top-level custom pages from the mounted frontend/pages directory."""
    pinned = []
    try:
        base = Path("frontend/pages")
        # Exclude core files that are already referenced elsewhere
        exclude = {"__init__.py", "permissions.py", "landing.py", "binance_data_upload.py", "local_data_range.py"}
        if base.exists() and base.is_dir():
            for py_file in sorted(base.glob("*.py")):
                if py_file.name in exclude:
                    continue
                title = py_file.stem.replace("_", " ").title()
                url_path = py_file.stem
                pinned.append(
                    st.Page(str(py_file.as_posix()), title=title, icon="📌", url_path=url_path)
                )
    except Exception:
        pass
    return pinned


def public_pages():
    result = {
        "Config Generator": [
            st.Page("frontend/pages/config/grid_strike/app.py", title="Grid Strike", icon="🎳", url_path="grid_strike"),
            st.Page("frontend/pages/config/pmm_simple/app.py", title="PMM Simple", icon="👨‍🏫", url_path="pmm_simple"),
            st.Page("frontend/pages/config/pmm_dynamic/app.py", title="PMM Dynamic", icon="👩‍🏫", url_path="pmm_dynamic"),
            st.Page("frontend/pages/config/dman_maker_v2/app.py", title="D-Man Maker V2", icon="🤖", url_path="dman_maker_v2"),
            st.Page("frontend/pages/config/bollinger_v1/app.py", title="Bollinger V1", icon="📈", url_path="bollinger_v1"),
            st.Page("frontend/pages/config/macd_bb_v1/app.py", title="MACD_BB V1", icon="📊", url_path="macd_bb_v1"),
            st.Page("frontend/pages/config/supertrend_v1/app.py", title="SuperTrend V1", icon="👨‍🔬", url_path="supertrend_v1"),
            st.Page("frontend/pages/config/xemm_controller/app.py", title="XEMM Controller", icon="⚡️", url_path="xemm_controller"),
        ],
        "Data": [
            st.Page("frontend/pages/data/download_candles/app.py", title="Download Candles", icon="💹", url_path="download_candles"),
            st.Page("frontend/pages/binance_data_upload.py", title="Binance Data Upload", icon="🗂️", url_path="binance_data_upload_v1"),
            st.Page("frontend/pages/local_data_range.py", title="Local Data & Date Range", icon="🗂️", url_path="local_data_range"),
        ],
        "Community Pages": [
            st.Page("frontend/pages/data/tvl_vs_mcap/app.py", title="TVL vs Market Cap", icon="🦉", url_path="tvl_vs_mcap"),
        ]
    }

    pinned = _pinned_pages()
    if pinned:
        result["Pinned"] = pinned

    return result


def private_pages():
    return {
        "Bot Orchestration": [
            st.Page("frontend/pages/orchestration/instances/app.py", title="Instances", icon="🦅", url_path="instances"),
            st.Page("frontend/pages/orchestration/launch_bot_v2/app.py", title="Deploy V2", icon="🚀", url_path="launch_bot_v2"),
            st.Page("frontend/pages/orchestration/credentials/app.py", title="Credentials", icon="🔑", url_path="credentials"),
            st.Page("frontend/pages/orchestration/portfolio/app.py", title="Portfolio", icon="💰", url_path="portfolio"),
            st.Page("frontend/pages/orchestration/trading/app.py", title="Trading", icon="🪄", url_path="trading"),
            st.Page("frontend/pages/orchestration/archived_bots/app.py", title="Archived Bots", icon="🗃️", url_path="archived_bots"),
        ]
    }

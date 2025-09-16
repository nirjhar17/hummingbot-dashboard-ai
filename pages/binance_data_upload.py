import json
import os
from pathlib import Path

import streamlit as st


CONFIG_FILE = Path("/home/dashboard/data/source_paths.json")
DEFAULT_DATA_DIR = Path("/home/dashboard/data/binance")
UPLOADS_DIR = Path("/home/dashboard/data/uploads/binance")


def load_config():
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_config(cfg: dict):
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"Failed to save config: {e}")
        return False


def ensure_dirs():
    try:
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def main():
    st.title("Binance Data Upload")
    st.caption(
        "Configure your Binance data directory or upload CSV files. Settings are saved in /home/dashboard/data and persist across restarts."
    )

    ensure_dirs()

    cfg = load_config()
    current_dir = cfg.get("exchanges", {}).get("binance", {}).get("data_dir", str(DEFAULT_DATA_DIR))

    col1, col2 = st.columns([1, 2])
    with col1:
        st.selectbox("Exchange", options=["binance"], index=0, disabled=True)
    with col2:
        data_dir = st.text_input(
            "Data directory path (inside container)",
            value=current_dir,
            help="Use a directory accessible to the dashboard container. Default: /home/dashboard/data/binance",
            placeholder=str(DEFAULT_DATA_DIR),
        )

    # Save settings
    if st.button("Save Settings", type="primary"):
        cfg.setdefault("exchanges", {}).setdefault("binance", {})["data_dir"] = data_dir or str(DEFAULT_DATA_DIR)
        if save_config(cfg):
            st.success("Settings saved.")

    # Validate path and preview files
    if data_dir:
        path = Path(data_dir)
        if path.is_dir():
            files = sorted([p for p in path.glob("**/*") if p.is_file()])
            st.info(f"Directory exists: {path}")
            if files:
                st.write("Sample files:")
                st.write([str(p.relative_to(path)) for p in files[:10]])
            else:
                st.warning("No files found in the specified directory.")
        else:
            st.warning("Specified directory does not exist yet.")

    st.divider()
    st.subheader("Upload CSV Files")
    uploads = st.file_uploader(
        "Select one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Files will be stored under /home/dashboard/data/uploads/binance",
    )

    if uploads:
        saved = 0
        for up in uploads:
            try:
                target = UPLOADS_DIR / up.name
                with open(target, "wb") as f:
                    f.write(up.getbuffer())
                saved += 1
            except Exception as e:
                st.error(f"Failed to save {up.name}: {e}")
        st.success(f"Saved {saved} file(s) to {UPLOADS_DIR}")

    st.caption(
        "Tip: If your data is on the host machine, mount that folder into the container as /home/dashboard/data to make it accessible here."
    )


if __name__ == "__main__":
    main()



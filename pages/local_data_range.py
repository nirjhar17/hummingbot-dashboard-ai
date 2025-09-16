import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


PREFS_FILE = Path("/home/dashboard/data/user_prefs.json")
DEFAULT_DIR = Path("/home/dashboard/data")
UPLOAD_DIR = Path("/home/dashboard/data/uploads/local")


def load_prefs() -> dict:
    try:
        if PREFS_FILE.exists():
            return json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_prefs(prefs: dict) -> bool:
    try:
        PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PREFS_FILE.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"Failed to save preferences: {e}")
        return False


def ensure_dirs():
    try:
        DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def list_csv_files(path: Path) -> list[Path]:
    try:
        return sorted([p for p in path.glob("**/*.csv") if p.is_file()])
    except Exception:
        return []


def main():
    st.title("Local Data & Date Range")
    st.caption("Choose a local data folder or upload CSVs, and set a date range. Settings persist across restarts.")

    ensure_dirs()
    prefs = load_prefs()

    current_dir = prefs.get("data", {}).get("source_dir", str(DEFAULT_DIR))
    start_str = prefs.get("data", {}).get("start_date")
    end_str = prefs.get("data", {}).get("end_date")

    today = date.today()
    default_start = pd.to_datetime(start_str).date() if start_str else today.replace(day=1)
    default_end = pd.to_datetime(end_str).date() if end_str else today

    col1, col2 = st.columns([2, 1])
    with col1:
        source_dir = st.text_input(
            "Data source directory (inside container)",
            value=current_dir,
            help="Mount your host folder to /home/dashboard/data for persistence.",
            placeholder=str(DEFAULT_DIR),
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Save", type="primary"):
            prefs.setdefault("data", {})["source_dir"] = source_dir or str(DEFAULT_DIR)
            prefs["data"]["start_date"] = str(default_start)
            prefs["data"]["end_date"] = str(default_end)
            if save_prefs(prefs):
                st.success("Preferences saved.")

    st.subheader("Date Range")
    start_date, end_date = st.date_input(
        "Select date range",
        value=(default_start, default_end),
        help="Used by other pages to filter the time window.",
    )

    # Update in-memory and persisted prefs when changed
    prefs.setdefault("data", {})
    prefs["data"]["start_date"] = str(start_date)
    prefs["data"]["end_date"] = str(end_date)
    save_prefs(prefs)

    st.divider()
    st.subheader("Upload CSVs (optional)")
    uploads = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help=f"Files are stored in {UPLOAD_DIR}",
    )

    if uploads:
        saved = 0
        for up in uploads:
            try:
                target = UPLOAD_DIR / up.name
                with open(target, "wb") as f:
                    f.write(up.getbuffer())
                saved += 1
            except Exception as e:
                st.error(f"Failed to save {up.name}: {e}")
        st.success(f"Saved {saved} file(s) to {UPLOAD_DIR}")

    # Preview files
    st.divider()
    st.subheader("Folder Preview")
    if source_dir:
        p = Path(source_dir)
        if p.is_dir():
            files = list_csv_files(p)
            if files:
                st.write(f"Found {len(files)} CSV file(s). Showing up to 10:")
                st.write([str(f.relative_to(p)) for f in files[:10]])
            else:
                st.warning("No CSV files found in the selected directory.")
        else:
            st.warning("Directory does not exist. Mount or create it, or use uploads above.")


if __name__ == "__main__":
    main()



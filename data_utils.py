import streamlit as st
import pandas as pd
import time
import os
from io import BytesIO

try:
    import dropbox  # type: ignore
except Exception:  # pragma: no cover - dropbox may not be installed during tests
    dropbox = None

def _read_csv_from_dropbox(dropbox_path: str, token: str) -> pd.DataFrame:
    """Download CSV file from Dropbox and return as DataFrame."""
    if dropbox is None:
        raise RuntimeError("Dropbox SDK is not available")

    dbx = dropbox.Dropbox(token)
    _, res = dbx.files_download(dropbox_path)
    return pd.read_csv(BytesIO(res.content))


@st.cache_data
def load_data():
    """Load net export data either from Dropbox or from local storage."""
    data_path = os.path.join("data", "net_export.csv")
    dropbox_token = os.environ.get("DROPBOX_TOKEN")
    dropbox_path = os.environ.get("NET_EXPORT_DROPBOX_PATH", "/net_export.csv")

    start_time = time.time()
    if dropbox_token:
        try:
            df = _read_csv_from_dropbox(dropbox_path, dropbox_token)
            load_time = time.time() - start_time
            return df, load_time
        except Exception as e:  # pragma: no cover - network operations
            st.warning(f"Не удалось загрузить данные из Dropbox: {e}. Использую локальный файл.")

    df = pd.read_csv(data_path)
    load_time = time.time() - start_time
    return df, load_time


@st.cache_data
def load_gdp_data():
    """Load GDP data either from Dropbox or local storage."""
    data_path = os.path.join("data", "gdp.csv")
    dropbox_token = os.environ.get("DROPBOX_TOKEN")
    dropbox_path = os.environ.get("GDP_DROPBOX_PATH", "/gdp.csv")

    start_time = time.time()
    if dropbox_token:
        try:
            df = _read_csv_from_dropbox(dropbox_path, dropbox_token)
            load_time = time.time() - start_time
            return df, load_time
        except Exception as e:  # pragma: no cover - network operations
            st.warning(f"Не удалось загрузить данные из Dropbox: {e}. Использую локальный файл.")

    df = pd.read_csv(data_path)
    load_time = time.time() - start_time
    return df, load_time

import streamlit as st
import pandas as pd
import dropbox
import time
import io

@st.cache_data
def load_data_from_dropbox(file_path, app_key, app_secret, refresh_token):
    """
    Downloads a single CSV file from Dropbox and loads it into a pandas DataFrame.
    Args:
        file_path: Path to the file in Dropbox
        app_key: Dropbox app key
        app_secret: Dropbox app secret
        refresh_token: Dropbox refresh token
    Returns: DataFrame, load_time
    """
    try:
        start_time = time.time()
        dbx = dropbox.Dropbox(
            app_key=app_key,
            app_secret=app_secret,
            oauth2_refresh_token=refresh_token
        )
        # Download the specified file
        _, res = dbx.files_download(path=file_path)
        df = pd.read_csv(io.BytesIO(res.content))
        
        load_time = time.time() - start_time
        st.success(f"Successfully loaded {file_path} from Dropbox in {load_time:.2f}s")
        
        return df, load_time
        
    except Exception as e:
        st.error(f"Error loading {file_path} from Dropbox: {e}")
        raise e

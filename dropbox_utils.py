import streamlit as st
import pandas as pd
import dropbox
import time
import io

@st.cache_data
def load_data_from_dropbox(file_path):
    """
    Downloads a single CSV file from Dropbox and loads it into a pandas DataFrame.
    Args:
        file_path: Path to the file in Dropbox
    Returns: DataFrame, load_time
    """
    try:
        token = st.secrets["DROPBOX_ACCESS_TOKEN"]
        
        start_time = time.time()
        dbx = dropbox.Dropbox(token)
        
        # Download the specified file
        _, res = dbx.files_download(path=file_path)
        df = pd.read_csv(io.BytesIO(res.content))
        
        load_time = time.time() - start_time
        st.success(f"Successfully loaded {file_path} from Dropbox in {load_time:.2f}s")
        
        return df, load_time
        
    except Exception as e:
        st.error(f"Error loading {file_path} from Dropbox: {e}")
        raise e

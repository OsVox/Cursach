import streamlit as st
import pandas as pd
import time

@st.cache_data
def load_data():
    """
    Загрузка и подготовка данных о чистом экспорте.
    Данная функция кэшируется, поэтому данные загружаются только один раз,
    независимо от того, на какой странице приложения она вызывается.
    """
    data_path = "/Users/andrejvorsin/Desktop/Курсовая/Coding part/data/net_export.csv"
    start_time = time.time()
    df = pd.read_csv(data_path)
    load_time = time.time() - start_time
    return df, load_time 
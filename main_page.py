import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
import plotly.express as px

# Импортируем функцию загрузки данных из общего модуля
from dropbox_utils import load_data_from_dropbox

# Настройка страницы
st.set_page_config(
    page_title="Анализ сравнительных преимуществ",
    page_icon="📊",
    layout="wide"
)

# Заголовок приложения
st.title("🌐 Анализ сравнительных преимуществ стран")
st.write("Этот инструмент позволяет анализировать данные о чистом экспорте и выявлять сравнительные преимущества стран.")

# Загрузка данных (используем общую функцию кэширования)
with st.spinner("Загрузка данных..."):
    df, load_time = load_data_from_dropbox(
        st.secrets["NET_EXPORT_DATA_PATH"],
        app_key=st.secrets["APP_KEY"],
        app_secret=st.secrets["APP_SECRET"],
        refresh_token=st.secrets["REFRESH_TOKEN"]
    )

    df['year'] = 2022

    st.success(f"Данные успешно загружены за {load_time:.2f} секунд!")

# Показ базовой информации о данных
st.subheader("Обзор данных")
st.write(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

# Основные метрики
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Всего стран", df['country'].nunique())
with col2:
    st.metric("Всего продуктов", df['hs_product_description'].nunique())
with col3:
    st.metric("Всего записей", df.shape[0])

# Фильтры
st.subheader("Настройки фильтрации")

# Слайдеры для выбора топ стран и продуктов
with st.form("filter_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        top_countries = st.slider(
            "Топ стран по объему торговли", 
            min_value=5, 
            max_value=min(200, df['country'].nunique()), 
            value=50,
            help="Выберите количество стран для анализа"
        )
    
    with col2:
        top_products = st.slider(
            "Топ продуктов по объему торговли", 
            min_value=5, 
            max_value=min(5000, df['hs_product_description'].nunique()), 
            value=100,
            help="Выберите количество продуктов для анализа"
        )

    # Добавляем слайдеры для выбора диапазона лет
    st.write("Выберите диапазон лет:")
    # Предполагаем, что в df есть столбец 'year'
    min_year = int(df['year'].min() - 1)
    max_year = int(df['year'].max() + 1)
    selected_years = st.slider(
        "Годы",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        help="Выберите начальный и конечный год для анализа"
    )
    
    # Добавляем кнопку отправки формы
    submit_button = st.form_submit_button("Применить фильтры")

# Применение фильтров и отображение результатов
if submit_button or 'filtered_data' not in st.session_state:
    with st.spinner("Применение фильтров..."):
        # Определение топ стран по объему торговли
        top_countries_list = df.groupby('country')['value_1000_usd'].sum().sort_values(ascending=False).head(top_countries).index.tolist()
        
        # Определение топ продуктов по объему торговли
        top_products_list = df.groupby('hs_product_description')['value_1000_usd'].sum().sort_values(ascending=False).head(top_products).index.tolist()
        
        # Фильтрация данных
        filtered_data = df[
            (df['country'].isin(top_countries_list)) & 
            (df['hs_product_description'].isin(top_products_list)) &
            (df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1]) # Фильтр по годам
        ]
        
        st.session_state['filtered_data'] = filtered_data
        st.session_state['top_countries_list'] = top_countries_list
        st.session_state['top_products_list'] = top_products_list

# Отображение отфильтрованных данных
if 'filtered_data' in st.session_state:
    filtered_data = st.session_state['filtered_data']
    
    st.subheader("Отфильтрованные данные")
    st.write(f"Выбрано {len(st.session_state['top_countries_list'])} стран и {len(st.session_state['top_products_list'])} продуктов")
    st.write(f"Размер отфильтрованных данных: {filtered_data.shape[0]} строк")
    
    # Показать первые несколько строк отфильтрованных данных с форматированием
    display_data = filtered_data.copy()
    display_data['value_1000_usd'] = display_data['value_1000_usd'].apply(lambda x: f"${x/1000000:.0f} bln")
    display_data.rename(columns={'value_1000_usd': 'Объем торговли (млрд USD)'}, inplace=True)
    st.dataframe(display_data.head(20))

    # Анализ стран
    st.subheader("Статистика по странам")
    country_stats = filtered_data.groupby('country')['value_1000_usd'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
    country_stats.columns = ['Общий объем торговли (тыс. USD)', 'Средний объем торговли (тыс. USD)', 'Количество товаров']
    st.dataframe(country_stats)
    
    # Визуализация стран на карте мира с использованием plotly
    st.subheader("Распределение объема торговли по странам на карте мира (учитываются только продукты в которых страна является чистым экспортером)")
    
    # Подготовка данных для карты
    country_trade_data = country_stats.reset_index()
    
    # Создание карты с помощью plotly
    fig = px.choropleth(
        country_trade_data,
        locations='country',  # название столбца с названиями стран
        locationmode='country names',  # режим определения местоположения
        color='Общий объем торговли (тыс. USD)',  # данные для цветовой шкалы
        hover_name='country',  # данные для всплывающей подсказки
        color_continuous_scale='Viridis',  # цветовая схема
        title='Объем торговли по странам',
        labels={'Общий объем торговли (тыс. USD)': 'Объем торговли (тыс. USD)'}
    )
    
    # Настройка внешнего вида карты
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=600,
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    # Отображение карты
    st.plotly_chart(fig, use_container_width=True)
    
    # Кнопка для перехода к обучению модели
    st.subheader("Обучение модели сравнительных преимуществ")
    st.write("Для обучения модели на отфильтрованных данных перейдите на страницу обучения модели:")
    if st.button("Перейти к обучению модели", type="primary"):
        st.switch_page("pages/learn_model.py")

# Инструкции по использованию
with st.expander("Инструкции по использованию"):
    st.markdown("""
    ### Как использовать это приложение
    1. Используйте слайдеры для выбора количества стран и продуктов для анализа
    2. Нажмите кнопку "Применить фильтры" для обновления результатов
    3. Изучите статистику и графики на основе отфильтрованных данных
    4. Чтобы обучить модель сравнительных преимуществ, перейдите на соответствующую страницу
    
    Данные загружаются из файла `data/net_export.csv` и кэшируются для ускорения работы.
    """)

# Информация о проекте
st.sidebar.header("О проекте")
st.sidebar.info("""
    Этот инструмент создан для анализа сравнительных преимуществ стран
    на основе данных о чистом экспорте.
""")

# Навигация между страницами
st.sidebar.header("Навигация")
st.sidebar.page_link("main_page.py", label="Главная страница", icon="🏠")
st.sidebar.page_link("pages/learn_model.py", label="Обучение модели", icon="🧠") 
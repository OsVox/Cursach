import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
import umap
import plotly.express as px


from data_utils import load_data

from learning_a_model import ComparativeAdvantageModel

# Настройка страницы
st.set_page_config(
    page_title="Обучение модели сравнительных преимуществ",
    page_icon="🧠",
    layout="wide"
)

# Заголовок страницы
st.title("🧠 Обучение модели сравнительных преимуществ")
st.write("На этой странице вы можете обучить модель сравнительных преимуществ на основе отфильтрованных данных.")

# Проверяем, есть ли в session_state отфильтрованные данные
if 'filtered_data' not in st.session_state:
    # Если данных нет, загружаем и фильтруем данные
    with st.spinner("Загрузка данных..."):
        df, load_time = load_data()  # Используем общую функцию кэширования
        st.success(f"Данные успешно загружены за {load_time:.2f} секунд!")
    
    st.warning("Вы перешли на страницу обучения модели без предварительной фильтрации данных. Используйте полный набор данных или вернитесь на главную страницу для фильтрации.")
    
    # Используем полный набор данных, но ограничиваем его для производительности
    top_countries = 50
    top_products = 100
    
    # Определение топ стран по объему торговли
    top_countries_list = df.groupby('country')['value_1000_usd'].sum().sort_values(ascending=False).head(top_countries).index.tolist()
    
    # Определение топ продуктов по объему торговли
    top_products_list = df.groupby('hs_product_description')['value_1000_usd'].sum().sort_values(ascending=False).head(top_products).index.tolist()
    
    # Фильтрация данных
    filtered_data = df[
        (df['country'].isin(top_countries_list)) & 
        (df['hs_product_description'].isin(top_products_list))
    ].copy()
    
    st.session_state['filtered_data'] = filtered_data
    st.session_state['top_countries_list'] = top_countries_list
    st.session_state['top_products_list'] = top_products_list
else:
    filtered_data = st.session_state['filtered_data']
    st.success(f"Используются данные, отфильтрованные на главной странице: {len(st.session_state['top_countries_list'])} стран и {len(st.session_state['top_products_list'])} продуктов")

# Показываем основную информацию об отфильтрованных данных
st.subheader("Информация о данных для обучения")
st.write(f"Размер отфильтрованных данных: {filtered_data.shape[0]} строк")

# Показать первые несколько строк отфильтрованных данных с форматированием
with st.expander("Предварительный просмотр данных"):
    display_data = filtered_data.copy()
    display_data['value_1000_usd'] = display_data['value_1000_usd'].apply(lambda x: f"${x/1000000:.0f} bln")
    display_data.rename(columns={'value_1000_usd': 'Объем торговли (млрд USD)'}, inplace=True)
    st.dataframe(display_data.head(20))

# Секция для обучения модели
st.subheader("Настройки модели")

col1, col2 = st.columns(2)

with col1:
    # Выбор количества измерений
    n_factors = st.slider(
        "Количество измерений в модели", 
        min_value=2, 
        max_value=10, 
        value=2,
        help="Большее количество измерений может улучшить точность модели, но усложнит интерпретацию"
    )

with col2:
    # Выбор порога для бинаризации
    threshold = st.slider(
        "Порог индекса специализации", 
        min_value=1.0, 
        max_value=5.0, 
        value=1.0, 
        step=0.1,
        help="Значение индекса специализации, выше которого считается, что страна имеет сравнительное преимущество"
    )

# Дополнительные параметры обучения
with st.expander("Дополнительные параметры обучения"):
    use_weights = st.checkbox(
        "Использовать веса на основе объема торговли", 
        value=False,
        help="Если отмечено, модель будет учитывать объем торговли при обучении"
    )

if st.button("Обучить модель", type="primary"):
    with st.spinner("Подготовка данных для обучения модели..."):
        # Формируем бинарную матрицу сравнительных преимуществ на основе индекса специализации
        # Рассчитываем индекс специализации
        filtered_data['total_export_of_country'] = filtered_data.groupby('country')['value_1000_usd'].transform('sum')
        filtered_data['total_export_of_product'] = filtered_data.groupby('hs_product_description')['value_1000_usd'].transform('sum')
        
        filtered_data['share_in_countries_export'] = filtered_data['value_1000_usd'] / filtered_data['total_export_of_country']
        filtered_data['share_in_world_export'] = filtered_data['total_export_of_product'] / filtered_data['value_1000_usd'].sum()
        
        filtered_data['specialization_index'] = filtered_data['share_in_countries_export'] / filtered_data['share_in_world_export']
        
        # Создаем сводную таблицу с индексом специализации
        # Получаем список продуктов в порядке их появления в top_products_list
        
        # Создаем сводную таблицу
        pivot_data = filtered_data.pivot_table(
            index='country', 
            columns='hs_product_description', 
            values='specialization_index', 
            aggfunc='sum'
        ).fillna(0)
        
        # Переупорядочиваем столбцы в соответствии с порядком в product_order
        pivot_data = pivot_data[[col for col in st.session_state['top_products_list'] if col in pivot_data.columns]]

        # Преобразуем в бинарную матрицу (L) с учетом порога
        L = (pivot_data > threshold).astype(int)
        Y = L.to_numpy()
        
        # Создаем сводную таблицу с индексом специализации
        weights = None
        if use_weights:
            trade_weights = filtered_data.pivot_table(
                index='country', 
                columns='hs_product_description', 
                values='value_1000_usd', 
                aggfunc='sum'
            ).fillna(0)
            
            total_world_trade = trade_weights.sum().sum()
            weights = (trade_weights / total_world_trade * 100).to_numpy()
        
    with st.spinner(f"Обучение модели с {n_factors} измерениями..."):
        try:
            # Запускаем обучение модели с отображением прогресса
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Создаем и обучаем модель
            model = ComparativeAdvantageModel(n_factors=n_factors)
            
            # Запускаем обучение модели
            result = model.fit(Y, weights=weights, method='L-BFGS-B')
            
            st.session_state['model'] = model
            st.session_state['model_results'] = result
            st.session_state['country_names'] = L.index.tolist()
            st.session_state['product_names'] = L.columns.tolist()
            st.session_state['model_threshold'] = threshold
            st.session_state['n_factors'] = min(3, model.n_factors)

            country_reducer = umap.UMAP(n_components=st.session_state['n_factors'], random_state=42)
            country_embedding = country_reducer.fit_transform(model.country_vectors)
            
            # Создаем UMAP-проекцию для продуктов
            product_reducer = umap.UMAP(n_components=st.session_state['n_factors'], random_state=42)
            product_embedding = product_reducer.fit_transform(model.product_vectors)
            
            st.session_state['country_embedding'] = country_embedding
            st.session_state['product_embedding'] = product_embedding
            
            st.success("Модель успешно обучена!")
        except Exception as e:
            st.error(f"Ошибка при обучении модели: {e}")

# Визуализация результатов модели (если модель уже обучена)
if 'model' in st.session_state:
    st.header("Результаты модели")
    
    model = st.session_state['model']
    country_names = st.session_state['country_names']
    product_names = st.session_state['product_names']
    
    st.subheader("Интерпретация факторов")

    col1, col2 = st.columns(2)
    with col1:
        include_countries = st.checkbox("Включить страны", value=True)
    with col2:
        include_products = st.checkbox("Включить продукты", value=True)
        
    if not include_countries and not include_products:
        st.warning("Необходимо выбрать хотя бы один параметр для отображения")
        include_countries = True  # Активируем страны по умолчанию, если ничего не выбрано
    
    use_umap = False
    if not (include_countries and include_products):
        use_umap = st.checkbox("Использовать UMAP для визуализации", value=False, disabled=(include_countries and include_products))

    color_by_gdp = False
    if include_countries and not include_products:
        color_by_gdp = st.checkbox("Окрасить страны по ВВП", value=False)

    if use_umap:
        country_reducer = umap.UMAP(n_components=st.session_state['n_factors'], random_state=42)
        country_embedding = country_reducer.fit_transform(model.country_vectors)
        
        product_reducer = umap.UMAP(n_components=st.session_state['n_factors'], random_state=42)
        product_embedding = product_reducer.fit_transform(model.product_vectors)
    else:
        country_embedding = model.country_vectors
        product_embedding = model.product_vectors

    data_frames_3d = []
    if include_countries:
        # Для совместной визуализации используем обычные векторы стран
        country_df_3d = pd.DataFrame({
            'x': country_embedding[:, 0],
            'y': country_embedding[:, 1],
            'z': country_embedding[:, 2] if model.n_factors >= 3 else 0,
            'name': country_names,
            'type': 'Страна'
        })
        
        # Добавляем данные о ВВП, если выбрана соответствующая опция
        if color_by_gdp:
            gdp_data = pd.read_csv('data/gdp.csv')
            
            # Приводим названия стран к одному формату для корректного слияния
            gdp_data['country'] = gdp_data['country'].str.strip()
            country_df_3d['country'] = country_df_3d['name'].str.strip()
            
            # Объединяем данные
            country_df_3d = country_df_3d.merge(
                gdp_data[['country', 'gdp_ppp']], 
                left_on='country', 
                right_on='country', 
                how='left'
            )
    
        data_frames_3d.append(country_df_3d)
    if include_products:
        product_df_3d = pd.DataFrame({
            'x': product_embedding[:min(100, len(product_names)), 0],
            'y': product_embedding[:min(100, len(product_names)), 1],
            'z': product_embedding[:min(100, len(product_names)), 2] if model.n_factors >= 3 else 0,
            'name': product_names[:min(100, len(product_names))],
            'type': 'Продукт'
        })
        data_frames_3d.append(product_df_3d)

    all_data_3d = pd.concat(data_frames_3d, ignore_index=True)

    fig = px.scatter_3d(
        all_data_3d,
        x='x',
        y='y',
        z='z',
        color='gdp_ppp' if color_by_gdp else 'type',
        hover_name='name',
        title='Векторы стран и продуктов в 3D пространстве факторов',
        labels={'x': 'Компонент 1', 'y': 'Компонент 2', 'z': 'Компонент 3'},
        width=1000,
        height=800
    )

    # Отображаем график
    st.plotly_chart(fig, use_container_width=True)

    # Показываем таблицы с эмбеддингами стран и продуктов
    st.subheader("Эмбеддинги стран")
    country_table = pd.DataFrame(
        model.country_vectors,
        columns=[f"factor_{i+1}" for i in range(model.n_factors)]
    )
    country_table.insert(0, "Страна", country_names)
    st.dataframe(country_table)

    st.subheader("Эмбеддинги продуктов")
    product_table = pd.DataFrame(
        model.product_vectors,
        columns=[f"factor_{i+1}" for i in range(model.n_factors)]
    )
    product_table.insert(0, "Продукт", product_names)
    st.dataframe(product_table)

    # Экспорт результатов
    if st.button("Экспортировать результаты"):
        # Создаем DataFrame для экспорта
        export_df = pd.DataFrame(index=country_names)
        export_df['country_intercept'] = model.country_intercepts
        
        for i in range(model.n_factors):
            export_df[f'country_factor_{i+1}'] = model.country_vectors[:, i]
        
        # Сохраняем во временный CSV
        csv = export_df.to_csv().encode('utf-8')
        st.download_button(
            label="Скачать результаты (CSV)",
            data=csv,
            file_name="country_factors.csv",
            mime="text/csv",
        )

# Информация о модели
st.sidebar.header("О модели")
st.sidebar.info("""
    **Модель сравнительных преимуществ** основана на вероятностном подходе к выявлению скрытых факторов, 
    которые влияют на сравнительные преимущества стран в различных секторах экономики.
    
    Модель позволяет:
    - Выявить скрытые факторы, влияющие на торговую специализацию стран
    - Определить кластеры стран с похожими экономическими структурами
    - Прогнозировать потенциальные новые сферы экспортной специализации
""")

# Навигация
st.sidebar.header("Навигация")
st.sidebar.page_link("main_page.py", label="Главная страница", icon="🏠")
st.sidebar.page_link("pages/learn_model.py", label="Обучение модели", icon="🧠") 

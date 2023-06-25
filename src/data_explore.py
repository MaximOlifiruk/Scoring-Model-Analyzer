import streamlit as st
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from utils import create_pipelines
import pandas as pd


def display_tables(tables):
    data_selected = st.sidebar.selectbox('Таблицы:', list(tables.keys()))

    if data_selected in tables:
        data = tables[data_selected]
        st.write("Выбрана таблица", data_selected, "из папки import")

    return data, st.sidebar


def display_data_info(data):
    st.sidebar.header('Посмотреть на данные: ')
    choice = st.sidebar.selectbox('Menu:', ["Ничего не выбрано", "Размер таблицы", "Начало таблицы", "Конец таблицы", "Описание таблицы", "Информация таблицы", "Колонки таблицы"])

    options = {
        "Размер таблицы": data.shape,
        "Начало таблицы": data.head(),
        "Конец таблицы": data.tail(),
        "Описание таблицы": data.describe(),
        "Информация таблицы": data.info(),
        "Колонки таблицы": data.columns
    }

    if choice in options:
        st.write(choice + ":")
        st.write(options[choice])


def display_column_values(data):
    selected_column = st.sidebar.selectbox("Выберите столбец", ["Ничего не выбрано"] + data.columns.tolist())

    if selected_column != "Ничего не выбрано":
        st.write("Значения", selected_column, ":")
        st.write(data[selected_column].value_counts())


def display_histogram(data):
    numeric_columns = data.select_dtypes(include='number').columns.tolist()
    selected_column_hist = st.sidebar.selectbox("Выберите столбец для гистограммы", ["Ничего не выбрано"] + numeric_columns)
    if selected_column_hist != "Ничего не выбрано":
        fig, ax = plt.subplots()
        ax.hist(data[selected_column_hist], bins=50)
        ax.set_xlabel(selected_column_hist)
        ax.set_ylabel("Count")
        st.pyplot(fig)


def display_pie_chart(data):
    selected_column_pie = st.sidebar.selectbox("Выберите столбец для круговой диаграммы", ["Ничего не выбрано"] + data.columns.tolist())
    if selected_column_pie != "Ничего не выбрано":
        counts = data[selected_column_pie].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
        ax.set_aspect('equal')
        ax.set_title(selected_column_pie)
        st.pyplot(fig)


def display_visualization(data):
    st.sidebar.header('Визуализация данных: ')
    display_histogram(data)
    display_pie_chart(data)


def display_correlations(data):
    st.sidebar.header('Корреляция данных: ')
    menu_corr = ["Ничего не выбрано", "Таблица корелляции", "Графики корелляции"]
    choice_corr = st.sidebar.selectbox("Выберите в какой форме показать корреляцию", menu_corr)

    options_corr = {
        "Таблица корелляции": data.corr(),
        "Графики корелляции": scatter_matrix(data, figsize=(12, 8))
    }

    if choice_corr in options_corr:
        st.write(choice_corr + ":")
    
        if choice_corr == "Графики корреляции":
            scatter_matrix(data, figsize=(12, 8))
            st.pyplot()
        else:
            st.write(options_corr[choice_corr])


def display_preparation(numeric_columns, label_columns, ordinal_columns, onehot_columns, data, tables):
    st.sidebar.header('Подготовить данные: ')
    data_selected = st.sidebar.selectbox('Выберите таблицу для подготовки:', ["Ничего не выбрано"] + list(tables.keys()))
    full_pipeline, label_data = create_pipelines(numeric_columns, label_columns, ordinal_columns, onehot_columns, data)

    if data_selected != "Ничего не выбрано":
        data = tables[data_selected]
        data_prepared = full_pipeline.fit_transform(data)
        final_columns = numeric_columns + ordinal_columns + ['Property_area_' + column for column in data['Property_Area'].value_counts().index]

        st.write(f"Таблица " + data_selected + " преобразована")
        data_prepared = pd.DataFrame(data_prepared, columns=final_columns)
        data_combined = pd.concat([data_prepared, label_data], axis=1)
        data_combined = data_combined.dropna()
        tables["data_prepared"] = data_combined
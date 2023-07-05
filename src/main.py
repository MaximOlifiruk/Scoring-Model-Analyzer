import streamlit as st
import pandas as pd
from data_explore import display_tables, display_data_info, display_column_values, display_visualization, display_correlations, display_preparation
from utils import create_columns
from models import model_options, get_selected_model, calculate_scores
from sklearn.ensemble import VotingClassifier


@st.cache_data
def load_tables():
    return {
        "data.csv": pd.read_csv("import/data.csv", usecols=lambda column: column != 'Loan_ID'),
    }
tables = load_tables()


st.title('Скоринговая модель')
st.markdown("""
В данном веб-приложении: можно:
1. Просматривать данные, визуализировать их.
2. Стандитизировать данные.
3. Применять к данным модели машинного обучения.
""")


data, sidebar_data = display_tables(tables)
display_data_info(data)
display_column_values(data)
display_visualization(data)
#display_correlations(data)
numeric_columns, label_columns, ordinal_columns, onehot_columns = create_columns(data)
display_preparation(numeric_columns, label_columns, ordinal_columns, onehot_columns, data, tables)


if "data_prepared" in tables.keys():
    data_prepared = tables["data_prepared"]

    st.write("Преобразованная таблица")
    st.write(data_prepared)

    st.write("Eё колонки: ")
    st.write(data_prepared.columns.tolist())

    y = data_prepared.iloc[:, -1]
    X_full = data_prepared.drop(data_prepared.columns[-1], axis=1)
    selected_columns = st.multiselect("Выберите столбцы для X", X_full.columns.tolist())
    X = X_full[selected_columns]

    if len(selected_columns) != 0:
        st.write("Выбранные столбцы для X:")
        st.write(X)

        model_options_keys = model_options.keys()
        selected_model = st.selectbox('Выберите модель', model_options_keys)

        model = get_selected_model(selected_model)

        mean_score, std_score = calculate_scores(model, X, y)
        st.write("Среднее:", mean_score)
        st.write("Стандартное отклонение:", std_score)

        st.write("Выберите модели с которыми хотите сделать ансамбль")
        selected_models = st.multiselect("Выберите модели", model_options_keys)

        if len(selected_models) != 0:
            classifiers = [model_options[elem] for elem in selected_models]
            estimators = [(selected_model, classifier) for selected_model, classifier in zip(selected_models, classifiers)]

            ensemble = VotingClassifier(estimators=estimators, voting='hard')
            mean_score, std_score = calculate_scores(ensemble, X, y)
            st.write("Среднее:", mean_score)
            st.write("Стандартное отклонение:", std_score)



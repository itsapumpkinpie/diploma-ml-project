import streamlit as st
import pandas as pd
import numpy as np
import json
from models.data_flat import data
import keras
from tensorflow.keras.losses import MeanSquaredError as mse


loaded_model = keras.models.load_model("models/flat_model.h5",
                                       custom_objects={'mse': mse})
geo_data = pd.read_csv('geolocation_data.csv')

with open('models/config.json') as file:
    districts = json.load(file)

with open('models/scaling_params.json', 'r') as f:
    scaling_params = json.load(f)

#  Основная часть страницы
st.header("Предсказание стоимости квартир на рынке первичного жилья")
st.markdown("""
    <h4>Тепловая карта стоимости жилья за квадратный метр
""", unsafe_allow_html=True)
st.map(data=geo_data,
       latitude='geo_lat',
       longitude='geo_lon',
       color='color', zoom=9)
trine_data = pd.read_csv('krasnodar_flats.csv',  index_col=None)
st.header("Данные, на которых обучалась модель")
st.write(trine_data[:4])
st.markdown("""
    <h4>
""", unsafe_allow_html=True)

# Поля
st.sidebar.header("Предсказать стоимость жилья")
flat_area = st.sidebar.number_input("Общая площадь квартиры", min_value=0.0, max_value=None)
flat_rooms = st.sidebar.selectbox("Количество комнат", (1, 2, 3, 4, 5, 6))
flat_storey = st.sidebar.number_input("Количество этажей в доме", min_value=0, max_value=25)
flat_floor = st.sidebar.number_input("Этаж", min_value=0, max_value=25)
flat_district = st.sidebar.selectbox("Район", (districts["district"]))

# Подставление данных пользователя в датафрейм
data["total_meters"] = flat_area
data["floor"] = flat_floor
data["floors_count"] = flat_storey
data["rooms_count"] = flat_rooms
data[f"district_{flat_district}"] = 1.0

# Предобработка данных
data_frame = pd.DataFrame(data, index=[0])
data_frame = data_frame.to_numpy()
mean = np.array(scaling_params['mean'])
std = np.array(scaling_params['std'])
data_frame -= mean
data_frame /= std

button = st.sidebar.button("Предсказать стоимость квартиры", type="primary")
if button:
    st.sidebar.write(f"Результат: {loaded_model.predict(data_frame)[0][0]} рублей")


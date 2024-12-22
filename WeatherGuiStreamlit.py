import streamlit as st
from PIL import Image
import numpy as np
import joblib
from datetime import datetime

model = joblib.load("random_forest_model.pkl")

direction_mapping = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90,
    'ESE': 112.5, 'SE': 135, 'SSE': 157.5, 'S': 180,
    'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
    'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}
unique_directions = list(direction_mapping.keys())

def encode_wind_direction(value):
    angle = direction_mapping[value]
    return np.sin(np.radians(angle)), np.cos(np.radians(angle))

def encode_rain_today(value):
    return 1 if value == "Yes" else 0

def predict_weather(inputs):
    try:
        prediction = model.predict([inputs])[0]
        return "Yes, it will rain tomorrow." if prediction == 1 else "No, it won't rain tomorrow."
    except Exception as e:
        return f"Error: {e}"

st.title("Weather Prediction App")

bg_image = Image.open(r"C:\Users\DELL\Desktop\machine learning\fcds ml\fcds ml final project\rain.jpg")
st.image(bg_image, use_container_width=True)

with st.form("weather_form"):
    st.subheader("Enter Weather Details")

    input_date = st.text_input("Date (YYYY-MM-DD)")
    rain_today = st.selectbox("Rain Today (Yes/No):", ["Yes", "No"])

    wind_gust_dir = st.selectbox("Wind Gust Direction:", unique_directions)
    wind_dir9am = st.selectbox("Wind Direction 9AM:", unique_directions)
    wind_dir3pm = st.selectbox("Wind Direction 3PM:", unique_directions)

    fields = [
        "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
        "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
        "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"
    ]
    numerical_inputs = {}
    for field in fields:
        numerical_inputs[field] = st.number_input(field, value=0.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            date_obj = datetime.strptime(input_date, "%Y-%m-%d")
            year, month, day = date_obj.year, date_obj.month, date_obj.day
            weekday = date_obj.weekday()
            duration = (date_obj - datetime(2008, 12, 1)).days

            rain_today_encoded = encode_rain_today(rain_today)
            wind_gust_sin, wind_gust_cos = encode_wind_direction(wind_gust_dir)
            wind_dir9am_sin, wind_dir9am_cos = encode_wind_direction(wind_dir9am)
            wind_dir3pm_sin, wind_dir3pm_cos = encode_wind_direction(wind_dir3pm)

            features = [
                rain_today_encoded, year, month, day, weekday, duration,
                wind_gust_sin, wind_gust_cos, wind_dir9am_sin, wind_dir9am_cos,
                wind_dir3pm_sin, wind_dir3pm_cos,
                *numerical_inputs.values()
            ]

            if len(features) != 28:
                st.error(f"Expected 28 features, but got {len(features)}.")
            else:
                result = predict_weather(features)
                st.success(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")

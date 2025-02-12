import os
import streamlit as st
from PIL import Image
import numpy as np
import joblib
from datetime import datetime

# Get the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_path = os.path.join(BASE_DIR, "random_forest_model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("❌ Model file not found! Please make sure 'random_forest_model.pkl' is in the project folder.")

# Load the background image
image_path = os.path.join(BASE_DIR, "rain.jpg")
if os.path.exists(image_path):
    bg_image = Image.open(image_path)
else:
    st.error("❌ Background image file not found! Please make sure 'rain.jpg' is in the project folder.")

# Wind direction encoding
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
        return "🌧️ Yes, it will rain tomorrow." if prediction == 1 else "☀️ No, it won't rain tomorrow."
    except Exception as e:
        return f"Error: {e}"

st.set_page_config(page_title="Weather Prediction App", page_icon="🌦️", layout="wide")

# Display background image
if os.path.exists(image_path):
    st.image(bg_image, use_container_width=True)

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        color: #333;
        font-family: Arial, sans-serif;
    }
    .main-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #007BFF;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌦️ Weather Prediction App")
st.markdown("**Enter the weather details below to predict if it will rain tomorrow.**")

with st.form("weather_form"):
    st.subheader("Enter Weather Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        input_date = st.text_input("Date (YYYY-MM-DD)", "2024-12-31")
        rain_today = st.selectbox("Rain Today (Yes/No):", ["Yes", "No"])
    
    with col2:
        wind_gust_dir = st.selectbox("Wind Gust Direction:", unique_directions)
        wind_dir9am = st.selectbox("Wind Direction 9AM:", unique_directions)
    
    with col3:
        wind_dir3pm = st.selectbox("Wind Direction 3PM:", unique_directions)

    st.write("---")
    st.subheader("Weather Data")

    fields = [
        "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
        "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
        "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"
    ]
    col1, col2 = st.columns(2)
    numerical_inputs = {}

    # Temperature fields
    temp_fields = ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]
    non_temp_fields = [field for field in fields if field not in temp_fields]

    for field in temp_fields:
        with col1 if field in ["MinTemp", "Temp9am"] else col2:
            numerical_inputs[field] = st.slider(
                field,
                min_value=-10.0,
                max_value=50.0,
                value=20.0,
                step=0.5
            )

    for i, field in enumerate(non_temp_fields):
        with col1 if i % 2 == 0 else col2:
            numerical_inputs[field] = st.number_input(field, value=0.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            # Parse date and extract features
            date_obj = datetime.strptime(input_date, "%Y-%m-%d")
            year, month, day = date_obj.year, date_obj.month, date_obj.day
            weekday = date_obj.weekday()
            duration = (date_obj - datetime(2008, 12, 1)).days

            # Encode categorical inputs
            rain_today_encoded = encode_rain_today(rain_today)
            wind_gust_sin, wind_gust_cos = encode_wind_direction(wind_gust_dir)
            wind_dir9am_sin, wind_dir9am_cos = encode_wind_direction(wind_dir9am)
            wind_dir3pm_sin, wind_dir3pm_cos = encode_wind_direction(wind_dir3pm)

            # Construct feature list
            features = [
                rain_today_encoded, year, month, day, weekday, duration,
                wind_gust_sin, wind_gust_cos, wind_dir9am_sin, wind_dir9am_cos,
                wind_dir3pm_sin, wind_dir3pm_cos,
                *numerical_inputs.values()
            ]

            # Validate feature count
            if len(features) != 28:
                st.error(f"Expected 28 features, but got {len(features)}.")
            else:
                result = predict_weather(features)
                st.success(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# GitHub link
st.markdown(
    """
    <a href="https://github.com/seifkhalled/Weather-prediction-project" target="_blank" style="
        text-decoration: none;
        color: #ffffff;
        background-color: #007BFF;
        padding: 10px 15px;
        border-radius: 5px;
        font-size: 16px;
        display: inline-block;
    ">
        View My GitHub Repository
    </a>
    """,
    unsafe_allow_html=True
)

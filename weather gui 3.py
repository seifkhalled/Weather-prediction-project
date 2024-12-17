import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
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

def predict_weather():
    try:
        rain_today = dropdown_rain_today.get()  
        rain_today_encoded = encode_rain_today(rain_today) 
        
        input_date = entry_date.get()
        date_obj = datetime.strptime(input_date, "%Y-%m-%d")
        year, month, day = date_obj.year, date_obj.month, date_obj.day
        weekday = date_obj.weekday()
        duration = (date_obj - datetime(2008, 12, 1)).days
        
        wind_gust_dir = dropdown_gust_dir.get()
        wind_dir9am = dropdown_dir9am.get()
        wind_dir3pm = dropdown_dir3pm.get()
        wind_speed_9am = float(entry_wind_speed_9am.get())
        
        wind_gust_sin, wind_gust_cos = encode_wind_direction(wind_gust_dir)
        wind_dir9am_sin, wind_dir9am_cos = encode_wind_direction(wind_dir9am)
        wind_dir3pm_sin, wind_dir3pm_cos = encode_wind_direction(wind_dir3pm)

        numerical_inputs = [
            float(entry_min_temp.get()), float(entry_max_temp.get()),
            float(entry_rainfall.get()), float(entry_evaporation.get()),
            float(entry_sunshine.get()), float(entry_wind_gust_speed.get()),
            wind_speed_9am, float(entry_wind_speed_3pm.get()),
            float(entry_humidity_9am.get()), float(entry_humidity_3pm.get()),
            float(entry_pressure_9am.get()), float(entry_pressure_3pm.get()),
            float(entry_cloud_9am.get()), float(entry_cloud_3pm.get()),
            float(entry_temp_9am.get()), float(entry_temp_3pm.get())
        ]

        features = [
            rain_today_encoded, year, month, day, weekday, duration, wind_gust_sin, wind_gust_cos,
            wind_dir9am_sin, wind_dir9am_cos, wind_dir3pm_sin, wind_dir3pm_cos,
            *numerical_inputs
        ]
        
        if len(features) != 28:
            raise ValueError(f"Expected 28 features, but got {len(features)}.")

        prediction = model.predict([features])[0]
        result = "Yes, it will rain tomorrow." if prediction == 1 else "No, it won't rain tomorrow."
        
        # Display the prediction result in a message box
        messagebox.showinfo("Prediction Result", result)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("Weather Prediction App")
root.geometry("1000x700")  

bg_image = Image.open(r"C:\Users\DELL\Desktop\machine learning\fcds ml\fcds ml final project\rain.jpg")  # Use the raw string to handle backslashes
bg_image = bg_image.resize((1000, 1100), Image.Resampling.LANCZOS)  
bg_image_tk = ImageTk.PhotoImage(bg_image)

canvas = tk.Canvas(root, width=975, height=800)
canvas.create_image(0, 0, anchor="nw", image=bg_image_tk) 
canvas.grid(row=0, column=0, sticky="nsew")

scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

canvas.configure(yscrollcommand=scrollbar.set)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all")) 
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
scrollbar.grid(row=0, column=1, sticky="ns")

title_frame = tk.Frame(scrollable_frame, bg="lightblue")  
title_frame.grid(row=0, column=0, pady=(20, 10))  

title_label = tk.Label(title_frame, text="Weather Prediction", font=("Helvetica", 30, "bold"), fg="white", bg="lightblue")
title_label.grid(row=0, column=0)

frame = tk.Frame(scrollable_frame, bg="#ffffff", padx=20, pady=20)
frame.grid(row=1, column=0)

frame.grid_rowconfigure(0, minsize=10)
frame.grid_rowconfigure(1, minsize=10)

label_style = {"font": ("Arial", 12), "bg": "#ffffff", "fg": "#333"}
entry_style = {"font": ("Arial", 12), "width": 30}

tk.Label(frame, text="Date (YYYY-MM-DD):", **label_style).grid(row=0, column=0, padx=10, pady=10)
entry_date = tk.Entry(frame, **entry_style)
entry_date.grid(row=0, column=1, padx=10, pady=10)

tk.Label(frame, text="Rain Today (Yes/No):", **label_style).grid(row=1, column=0, padx=10, pady=10)
dropdown_rain_today = ttk.Combobox(frame, values=["Yes", "No"], state="readonly", **entry_style)
dropdown_rain_today.grid(row=1, column=1, padx=10, pady=10)

tk.Label(frame, text="Wind Gust Direction:", **label_style).grid(row=2, column=0, padx=10, pady=10)
dropdown_gust_dir = ttk.Combobox(frame, values=unique_directions, state="readonly", **entry_style)
dropdown_gust_dir.grid(row=2, column=1, padx=10, pady=10)

tk.Label(frame, text="Wind Direction 9AM:", **label_style).grid(row=3, column=0, padx=10, pady=10)
dropdown_dir9am = ttk.Combobox(frame, values=unique_directions, state="readonly", **entry_style)
dropdown_dir9am.grid(row=3, column=1, padx=10, pady=10)

tk.Label(frame, text="Wind Direction 3PM:", **label_style).grid(row=4, column=0, padx=10, pady=10)
dropdown_dir3pm = ttk.Combobox(frame, values=unique_directions, state="readonly", **entry_style)
dropdown_dir3pm.grid(row=4, column=1, padx=10, pady=10)

fields = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
    "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"
]
entries = {}
for i, field in enumerate(fields, start=5):
    tk.Label(frame, text=f"{field}:", **label_style).grid(row=i, column=0, padx=10, pady=10)
    entries[field] = tk.Entry(frame, **entry_style)
    entries[field].grid(row=i, column=1, padx=10, pady=10)

entry_min_temp = entries["MinTemp"]
entry_max_temp = entries["MaxTemp"]
entry_rainfall = entries["Rainfall"]
entry_evaporation = entries["Evaporation"]
entry_sunshine = entries["Sunshine"]
entry_wind_gust_speed = entries["WindGustSpeed"]
entry_wind_speed_9am = entries["WindSpeed9am"]
entry_wind_speed_3pm = entries["WindSpeed3pm"]
entry_humidity_9am = entries["Humidity9am"]
entry_humidity_3pm = entries["Humidity3pm"]
entry_pressure_9am = entries["Pressure9am"]
entry_pressure_3pm = entries["Pressure3pm"]
entry_cloud_9am = entries["Cloud9am"]
entry_cloud_3pm = entries["Cloud3pm"]
entry_temp_9am = entries["Temp9am"]
entry_temp_3pm = entries["Temp3pm"]

btn_predict = tk.Button(frame, text="Predict Rain Tomorrow", font=("Arial", 14), bg="cyan", fg="black", command=predict_weather)
btn_predict.grid(row=21, column=0, columnspan=2, pady=20)

lbl_result = tk.Label(scrollable_frame, font=("Arial", 16), bg="lightblue")
lbl_result.grid(row=2, column=0, pady=20)

root.mainloop()

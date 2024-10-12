import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__, template_folder='templates')

with open('arko_combined_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

rf_model = model_data["RandomForest"]
gb_model = model_data["GradientBoosting"]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    # Capture form data and convert to float
    data = [float(x) for x in request.form.values()]

    # Ensure all features are captured and not empty
    if len(data) < 13:  # Check if all required features are present
        return render_template('index.html', prediction_text="Error: All fields are required.")

    # Prepare features for model prediction
    final_features = [np.array(data)]

    rf_prediction = rf_model.predict(final_features)
    gb_prediction = gb_model.predict(final_features)
    combined_prediction = (rf_prediction + gb_prediction) / 2

    # Return the prediction result to the webpage
    return render_template('index.html', prediction_text=f'Predicted Rent: ₹{np.expm1(combined_prediction[0]):,.2f}')


if __name__ == '__main__':
    app.run(debug=True)

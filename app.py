from app import app as application
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('depression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        gender = int(request.form['gender'])
        suicidal_thoughts = int(request.form['suicidal_thoughts'])
        family_history = int(request.form['family_history'])
        sleep_duration = int(request.form['sleep_duration'])
        new_degree = int(request.form['new_degree'])
        academic_pressure = int(request.form['academic_pressure'])
        stress_level = int(request.form['stress_level'])  # Added missing feature

        # Handle one-hot encoding for dietary habits (ensuring 3 features)
        dietary_habits = request.form['dietary_habits']
        dietary_features = {'Dietary_Non-Vegetarian': 0, 'Dietary_Vegetarian': 0, 'Dietary_Vegan': 0}
        if dietary_habits in dietary_features:
            dietary_features[f'Dietary_{dietary_habits}'] = 1  # Set selected diet to 1

        # Create feature array in the exact order used during training
        features = np.array([[gender, suicidal_thoughts, family_history, sleep_duration, new_degree, academic_pressure,
                              stress_level,  # NEW FEATURE ADDED
                              dietary_features['Dietary_Non-Vegetarian'],
                              dietary_features['Dietary_Vegetarian'],
                              dietary_features['Dietary_Vegan']]])

        # Ensure the number of features matches model training
        if features.shape[1] != 10:
            return render_template('index.html', prediction=f"Error: Expected 10 features, got {features.shape[1]}")

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        result_text = f"You are {'prone to depression' if prediction == 1 else 'not prone to depression'}. Probability: {probability:.2f}%"

        return render_template('index.html', prediction=result_text)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

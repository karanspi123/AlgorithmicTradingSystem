from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
regression_model = joblib.load("../models/regression_model.pkl")
classification_model = joblib.load("../models/classification_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_values = data['features']

    # Convert input values to DataFrame
    columns = ['Open', 'High', 'Low', 'Volume', 'SMA9', 'SMA21', 'SMA220', 'EMA_Cross']
    input_df = pd.DataFrame([input_values], columns=columns)

    # Predictions
    predicted_price = regression_model.predict(input_df)[0]
    probability = classification_model.predict_proba(input_df)[0][1]
    confidence = probability

    # Return response
    return jsonify({
        "input_values": input_values,
        "predicted_price": predicted_price,
        "probability": probability,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(port=5000)
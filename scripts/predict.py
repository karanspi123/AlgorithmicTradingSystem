import pandas as pd
import joblib

def batch_predict(input_file, output_file):
    # Load models
    regressor = joblib.load("/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/models/regression_model.pkl")
    classifier = joblib.load("/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/models/classification_model.pkl")

    # Load input data
    data = pd.read_csv(input_file)
    X = data[['Open', 'High', 'Low', 'Volume', 'SMA9', 'SMA21', 'SMA220', 'EMA_Cross']]

    # Predict
    data['Predicted_Close'] = regressor.predict(X).round(2)
    data['Probability'] = classifier.predict_proba(X)[:, 1]
    data['Confidence'] = data['Probability']  # Same as probability for now

    # Save results
    data.to_csv(output_file, index=False, float_format="%.2f")
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    batch_predict("/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/processed_nq_data.csv", "/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/predicted_results.csv")
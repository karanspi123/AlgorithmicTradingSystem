import os
import pandas as pd
import joblib

def create_table_with_predictions(input_file, output_file, regression_model_file, classification_model_file):
    # Load data
    data = pd.read_csv(input_file)

    # Load models
    regressor = joblib.load(regression_model_file)
    classifier = joblib.load(classification_model_file)

    # Select features for predictions
    features = ['Open', 'High', 'Low', 'Volume', 'SMA9', 'SMA21', 'SMA220', 'EMA_Cross']
    X = data[features]

    # Make predictions
    data['Predicted_Close'] = regressor.predict(X).round(2)  # Predict closing price
    data['Probability'] = classifier.predict_proba(X)[:, 1].round(2)  # Predict probability of reaching the target price

    # Add Target_Reached column for binary classification (optional)
    data['Target_Reached'] = (data['High'] >= data['Close']).astype(int)  # Example logic: Check if High >= Close

    # Save results to a new CSV
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Return DataFrame for display or further use
    return data


if __name__ == "__main__":
    # Get the absolute path to the directory containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define absolute paths relative to the script's location
    input_file = os.path.join(base_dir, "../data/processed_nq_data.csv")
    output_file = os.path.join(base_dir, "../data/final_predictions.csv")
    regression_model_file = os.path.join(base_dir, "../models/regression_model.pkl")
    classification_model_file = os.path.join(base_dir, "../models/classification_model.pkl")

    # Generate table with predictions
    result = create_table_with_predictions(input_file, output_file, regression_model_file, classification_model_file)

    # Display a preview of the results
    print(result.head())
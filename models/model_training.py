import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(input_file, output_model):
    # Load preprocessed data
    data = pd.read_csv(input_file)

    # Define features and target
    X = data[['Open', 'High', 'Low', 'Volume', 'SMA9', 'SMA21', 'SMA220', 'EMA_Cross']]
    y = data['Signal']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

    # Save the trained model
    joblib.dump(model, output_model)
    print(f"Model saved to {output_model}")

if __name__ == "__main__":
    train_model("../data/processed_nq_data.csv", "../models/rf_model.pkl")
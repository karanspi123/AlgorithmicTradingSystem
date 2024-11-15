import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_classification_model(input_file, model_output_path):
    # Load preprocessed data
    data = pd.read_csv(input_file)

    # Define features and target for classification
    X = data[['Open', 'High', 'Low', 'Volume', 'SMA9', 'SMA21', 'SMA220', 'EMA_Cross']]
    y = data['Target_Reached']  # Target variable: 1 if target price is reached, else 0

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the classification model
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Evaluate model performance (optional)
    accuracy = classifier.score(X_test, y_test)
    print(f"Classification Model Accuracy: {accuracy}")

    # Save the trained classification model
    joblib.dump(classifier, model_output_path)
    print(f"Classification model saved to {model_output_path}")

if __name__ == "__main__":
    train_classification_model("/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/processed_nq_data.csv", "/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/models/classification_model.pkl")
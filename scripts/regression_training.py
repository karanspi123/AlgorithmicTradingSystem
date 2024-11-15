import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_regression_model(input_file, model_output_path):
    # Load preprocessed data
    data = pd.read_csv(input_file)

    # Define features and target for regression
    X = data[['Open', 'High', 'Low', 'Volume', 'SMA9', 'SMA21', 'SMA220', 'EMA_Cross']]
    y = data['Close']  # Target variable is the closing price

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the regression model
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)

    # Evaluate model performance (optional)
    score = regressor.score(X_test, y_test)
    print(f"Regression Model R^2 Score: {score}")

    # Save the trained regression model
    joblib.dump(regressor, model_output_path)
    print(f"Regression model saved to {model_output_path}")

if __name__ == "__main__":
    train_regression_model("/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/processed_nq_data.csv", "/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/models/regression_model.pkl")
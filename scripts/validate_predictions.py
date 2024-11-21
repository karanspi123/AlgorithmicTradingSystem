import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

def validate_predictions(predictions_file):
    print("Loading predictions...")
    # Load predicted results
    data = pd.read_csv(predictions_file)
    print(f"Data loaded. Number of rows: {len(data)}")

    # --- Regression Model Validation ---
    print("Validating Regression Model...")
    # Calculate regression metrics
    mse = mean_squared_error(data['Close'], data['Predicted_Close'])
    mae = mean_absolute_error(data['Close'], data['Predicted_Close'])

    # Calculate variance of residuals
    residuals = data['Close'] - data['Predicted_Close']
    variance = residuals.var()

    print(f"Regression Model Evaluation:")
    print(f"- Mean Squared Error (MSE): {mse:.4f}")
    print(f"- Mean Absolute Error (MAE): {mae:.4f}")
    print(f"- Variance of Residuals (Least Squares): {variance:.4f}")

    # Plot actual vs predicted closing prices
    print("Plotting Actual vs Predicted Closing Prices...")
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Actual Close', color='blue')
    plt.plot(data['Predicted_Close'], label='Predicted Close', color='orange')
    plt.legend()
    plt.title("Actual vs Predicted Closing Prices")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Price")
    plt.grid()
    plt.show()

    # --- Classification Model Validation ---
    print("Validating Classification Model...")
    # Evaluate classification performance
    y_true = data['Target_Reached']  # Actual target reached status
    y_pred = (data['Probability'] > 0.5).astype(int)  # Predicted status (threshold = 0.5)

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nClassification Model Evaluation:")
    print(f"- Accuracy: {accuracy:.2f}")

    # Plot probabilities
    print("Plotting Predicted Probabilities...")
    plt.figure(figsize=(10, 6))
    plt.plot(data['Probability'], label='Predicted Probability', color='green')
    plt.title("Predicted Probability of Reaching Target")
    plt.xlabel("Time Steps")
    plt.ylabel("Probability")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    validate_predictions("/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/predicted_results.csv")
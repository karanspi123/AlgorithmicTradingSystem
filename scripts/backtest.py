import pandas as pd

def backtest_strategy(input_file, output_file):
    # Load predicted results
    data = pd.read_csv(input_file)

    # Initialize trading signals
    # Example strategy: go long (buy) if Probability > 0.7 and Predicted_Close > Current_Close
    data['Signal'] = ((data['Probability'] > 0.7) & (data['Predicted_Close'] > data['Close'])).astype(int)

    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()  # Percentage change in Close price

    # Calculate strategy returns based on the Signal (if Signal is 1, apply return; if 0, return is 0)
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']  # Shift signal to avoid look-ahead bias

    # Calculate cumulative returns
    data['Cumulative_Returns'] = (1 + data['Strategy_Return']).cumprod()  # Compound returns

    # Save backtest results
    data.to_csv(output_file, index=False)
    print(f"Backtest results saved to {output_file}")

if __name__ == "__main__":
    backtest_strategy(
        "/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/predicted_results.csv",
        "/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/backtest_results.csv"
    )
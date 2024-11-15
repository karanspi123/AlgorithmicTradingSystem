import pandas as pd
import matplotlib.pyplot as plt

def visualize_data(data_file):
    # Load data
    data = pd.read_csv(data_file)

    # Plot Close price and indicators
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['SMA9'], label='SMA9')
    plt.plot(data['SMA21'], label='SMA21')
    plt.legend()
    plt.title("Price and Indicators")
    plt.show()

    # Plot strategy performance
    plt.figure(figsize=(10, 6))
    plt.plot(data['Cumulative_Returns'], label='Cumulative Returns')
    plt.legend()
    plt.title("Strategy Performance")
    plt.show()

if __name__ == "__main__":
    visualize_data("../data/backtest_results.csv")
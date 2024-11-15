import pandas as pd
import matplotlib.pyplot as plt

def visualize_backtest(data_file):
    # Load backtest results
    data = pd.read_csv(data_file)

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(data['Cumulative_Returns'], label='Cumulative Strategy Returns', color='green')
    plt.title("Backtest: Strategy Cumulative Returns")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    visualize_backtest("/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/backtest_results.csv")
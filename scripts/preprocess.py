import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(input_file, output_file):
    # Load data and strip whitespace from column headers
    data = pd.read_csv(input_file)
    data.columns = data.columns.str.strip()  # Remove leading and trailing spaces

    # Print the column names to verify
    print("Cleaned Columns in the dataset:", data.columns)

    # Define the columns for numerical scaling
    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA9', 'SMA21', 'SMA220']

    # Verify that the required columns are present
    missing_cols = [col for col in numerical_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in dataset: {missing_cols}")

    # Normalize numerical features
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Add additional features if needed
    data['EMA_Cross'] = data['SMA9'] - data['SMA21']
    data['Target_Reached'] = (data['Close'] >= data['High']).astype(int)

    # Save processed data with limited decimal places
    data.to_csv(output_file, index=False, float_format="%.2f")
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data(
        "/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/nq_data.csv",
        "/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/processed_nq_data.csv"
    )
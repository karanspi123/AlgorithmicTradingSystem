import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file):
    # Define the columns for numerical scaling
    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA9', 'SMA21', 'SMA220']

    # Initialize the scaler
    scaler = StandardScaler()

    # Process data in chunks
    chunk_size = 100000  # Adjust chunk size based on available memory
    chunks = pd.read_csv(input_file, chunksize=chunk_size)

    processed_chunks = []
    for chunk in chunks:
        chunk.columns = chunk.columns.str.strip()  # Remove leading and trailing spaces

        # Verify that the required columns are present
        missing_cols = [col for col in numerical_cols if col not in chunk.columns]
        if missing_cols:
            raise KeyError(f"Missing columns in dataset: {missing_cols}")

        # Normalize numerical features
        chunk[numerical_cols] = scaler.fit_transform(chunk[numerical_cols])

        # Add additional features
        chunk['EMA_Cross'] = chunk['SMA9'] - chunk['SMA21']
        chunk['Target_Reached'] = (chunk['Close'] >= chunk['High']).astype(int)

        processed_chunks.append(chunk)

    # Concatenate all processed chunks
    processed_data = pd.concat(processed_chunks)

    # Save processed data with limited decimal places
    processed_data.to_csv(output_file, index=False, float_format="%.2f")
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data(
        "/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/nq_data.csv",
        "/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/processed_nq_data.csv"
    )
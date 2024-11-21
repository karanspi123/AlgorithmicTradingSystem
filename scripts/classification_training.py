import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

def train_classification_model(input_file, model_output_path):
    # Load preprocessed data
    data = pd.read_csv(input_file)

    # Define features and target for classification
    features = ['Open', 'High', 'Low', 'Volume', 'SMA9', 'SMA21', 'SMA220', 'EMA_Cross']
    X = data[features]
    y = data['Target_Reached']  # Target variable: 1 if target price is reached, else 0

    # Split data into training and testing sets using stratified sampling
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in strat_split.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the classification model with hyperparameter tuning
    classifier = RandomForestClassifier(n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_classifier = grid_search.best_estimator_

    # Evaluate model performance
    accuracy = best_classifier.score(X_test, y_test)
    print(f"Classification Model Accuracy: {accuracy}")

    # Save the trained classification model
    joblib.dump(best_classifier, model_output_path)
    print(f"Classification model saved to {model_output_path}")

if __name__ == "__main__":
    train_classification_model("/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/data/processed_nq_data.csv", "/Users/karankanekar/Trading-Algorithms/PythonAlgorithmicTrading/models/classification_model.pkl")
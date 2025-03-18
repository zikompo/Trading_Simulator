import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def read_csv(path):
    """Read CSV file from the given path."""
    return pd.read_csv(path)

def get_data():
    # Load the data
    df = read_csv('../data/data.csv')
    # Rename and drop the date column if it's not needed for prediction
    df.columns = ['date'] + list(df.columns[1:])
    df = df.drop(columns=['date']) 

    # Assume the last column is the target and the rest are features
    features = df.columns[:-1]
    target = df.columns[-1]

    X = df[features]
    y = df[target]

    # For time series, it's important not to shuffle the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def main():
    # Load and split the data
    X_train, X_test, y_train, y_test = get_data()

    # Check if the data is valid
    if X_train is None or X_test is None or y_train is None or y_test is None:
        print("No data available for training.")
        return

    # Apply feature scaling since KNN is sensitive to the scale of the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up hyperparameter grid for tuning
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }

    # Use TimeSeriesSplit for cross-validation to maintain the time order
    tscv = TimeSeriesSplit(n_splits=5)

    # Set up the GridSearchCV with KNeighborsRegressor
    knn_reg = KNeighborsRegressor()
    grid_search = GridSearchCV(knn_reg, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)

    print("Best parameters found:", grid_search.best_params_)

    # Use the best estimator from the grid search
    model = grid_search.best_estimator_

    # Make predictions on the scaled test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    import pickle  
    with open(f'../backend/models/knn_model.pkl', "wb") as f:  
        pickle.dump(model, f)  # Save  

if __name__ == '__main__':
    main()
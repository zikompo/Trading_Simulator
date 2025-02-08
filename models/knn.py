from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
import pandas as pd

def read_csv(path):
    """Read CSV file from the given path."""
    return pd.read_csv(path)

def get_data():
    """Read data from CSV, split into features and target, then into training and test sets."""
    df = read_csv('data/data.csv')
    columns = df.columns
    features = columns[:-1]
    X = df[features]
    y = df['target']
    # Using shuffle=False for time series data; remove if not needed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def main():
    # Load the data
    X_train, X_test, y_train, y_test = get_data()
    
    # Define hyperparameters to tune
    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]  # Only relevant if metric='minkowski'
    }
    
    # Set up TimeSeriesSplit for sequential data
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize the KNN classifier
    knn = KNeighborsClassifier()
    
    # Set up grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=tscv,  # Use TimeSeriesSplit; change to KFold if data is not sequential
        scoring='accuracy'
    )
    
    # Fit grid search on the training data
    grid_search.fit(X_train, y_train)
    
    # Output the best parameters and the corresponding score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

if __name__ == '__main__':
    main()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Assuming X_train and y_train are available'
# X_train = ...
# y_train = ...

# Example hyperparameters to tune
param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]  # Only relevant if metric='minkowski'
}

# TimeSeriesSplit if working with sequential data
tscv = TimeSeriesSplit(n_splits=5)

knn = KNeighborsClassifier()

grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=tscv,  # or use KFold for non-time-series data
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
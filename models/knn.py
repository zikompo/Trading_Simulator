import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import os

class SymbolTimeSeriesSplitter:
    def __init__(self, data_path):
        """
        Initialize the time series splitter
        
        Args:
            data_path (str): Path to the CSV file
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self):
        """
        Load and preprocess the data
        
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        try:
            # Read the CSV file
            self.df = pd.read_csv(self.data_path)
            
            # Convert Date to datetime
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            
            # Sort by Symbol and Date
            self.df.sort_values(['Symbol', 'Date'], inplace=True)
            
            # Log initial data info
            logging.info(f"Total data shape: {self.df.shape}")
            logging.info(f"Unique symbols: {self.df['Symbol'].nunique()}")
            
            return self.df
        
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def split_symbol_data(self, symbol_data):
        """
        Split a single symbol's data into train, validation, and test sets
        
        Args:
            symbol_data (pd.DataFrame): Data for a single symbol
        
        Returns:
            tuple: Train, validation, and test dataframes
        """
        # Total rows for the symbol
        total_rows = len(symbol_data)
        
        # Calculate split points
        train_end = int(total_rows * 2/3)  # First 2 years
        val_end = int(total_rows * 5/6)    # Next 6 months
        
        # Split the data
        train_data = symbol_data.iloc[:train_end]
        val_data = symbol_data.iloc[train_end:val_end]
        test_data = symbol_data.iloc[val_end:]
        
        return train_data, val_data, test_data

    def prepare_features_target(self, data, target_col='Close'):
        """
        Prepare features and target
        
        Args:
            data (pd.DataFrame): Input dataframe
            target_col (str): Target column name
        
        Returns:
            tuple: Features and target
        """
        # Drop non-numeric columns
        feature_cols = ['High', 'Low', 'Open', 'Volume']
        
        X = data[feature_cols]
        y = data[target_col]
        
        return X, y

    def collect_symbol_splits(self):
        """
        Collect train, validation, and test sets across all symbols
        
        Returns:
            dict: Collected datasets
        """
        # Prepare collection lists
        X_train_all, y_train_all = [], []
        X_val_all, y_val_all = [], []
        X_test_all, y_test_all = [], []
        
        # Group by symbol
        for symbol, symbol_data in self.df.groupby('Symbol'):
            # Reset index for proper splitting
            symbol_data = symbol_data.reset_index(drop=True)
            
            # Split the symbol's data
            train_data, val_data, test_data = self.split_symbol_data(symbol_data)
            
            # Prepare features and targets
            X_train, y_train = self.prepare_features_target(train_data)
            X_val, y_val = self.prepare_features_target(val_data)
            X_test, y_test = self.prepare_features_target(test_data)
            
            # Collect
            X_train_all.append(X_train)
            y_train_all.append(y_train)
            X_val_all.append(X_val)
            y_val_all.append(y_val)
            X_test_all.append(X_test)
            y_test_all.append(y_test)
        
        # Concatenate across symbols
        X_train_concat = pd.concat(X_train_all)
        y_train_concat = pd.concat(y_train_all)
        X_val_concat = pd.concat(X_val_all)
        y_val_concat = pd.concat(y_val_all)
        X_test_concat = pd.concat(X_test_all)
        y_test_concat = pd.concat(y_test_all)
        
        return {
            'X_train': X_train_concat,
            'y_train': y_train_concat,
            'X_val': X_val_concat,
            'y_val': y_val_concat,
            'X_test': X_test_concat,
            'y_test': y_test_concat
        }

    def explore_data_distribution(self, splits):
        """
        Visualize data distributions and characteristics
        
        Args:
            splits (dict): Train, validation, test splits
        """
        plt.figure(figsize=(15, 10))
        
        # Feature distributions
        plt.subplot(2, 2, 1)
        splits['X_train'].boxplot()
        plt.title('Feature Distributions')
        plt.xticks(rotation=45)
        
        # Target distribution
        plt.subplot(2, 2, 2)
        splits['y_train'].hist()
        plt.title('Target Distribution')
        
        # Correlation heatmap
        plt.subplot(2, 2, 3)
        corr_matrix = splits['X_train'].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Feature Correlation')
        
        # Scatter plot of features vs target
        plt.subplot(2, 2, 4)
        plt.scatter(splits['X_train']['Volume'], splits['y_train'])
        plt.title('Volume vs Target')
        plt.xlabel('Volume')
        plt.ylabel('Close Price')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png')
        plt.close()

    def hyperparameter_tuning(self, splits):
        """
        Perform hyperparameter tuning for KNN
        
        Args:
            splits (dict): Train, validation, test splits
        
        Returns:
            dict: Best hyperparameters and their performance
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(splits['X_train'])
        X_val_scaled = self.scaler.transform(splits['X_val'])
        X_test_scaled = self.scaler.transform(splits['X_test'])
        
        # Hyperparameter grid
        n_neighbors_range = range(3, 100, 2)
        weights_options = ['uniform', 'distance']
        
        best_params = {}
        best_performance = float('inf')
        
        # Grid search
        for n_neighbors in n_neighbors_range:
            for weights in weights_options:
                knn = KNeighborsRegressor(
                    n_neighbors=n_neighbors, 
                    weights=weights
                )
                
                # Train on training data
                knn.fit(X_train_scaled, splits['y_train'])
                
                # Predict on validation
                val_pred = knn.predict(X_val_scaled)
                
                # Calculate MSE
                val_mse = mean_squared_error(splits['y_val'], val_pred)
                
                # Update best parameters
                if val_mse < best_performance:
                    best_performance = val_mse
                    best_params = {
                        'n_neighbors': n_neighbors,
                        'weights': weights
                    }
        
        # Train best model
        best_knn = KNeighborsRegressor(
            n_neighbors=best_params['n_neighbors'],
            weights=best_params['weights']
        )
        best_knn.fit(X_train_scaled, splits['y_train'])
        
        # Evaluate on validation and test
        val_pred = best_knn.predict(X_val_scaled)
        test_pred = best_knn.predict(X_test_scaled)
        
        results = {
            'best_params': best_params,
            'Validation': {
                'MSE': mean_squared_error(splits['y_val'], val_pred),
                'MAE': mean_absolute_error(splits['y_val'], val_pred),
                'R2': r2_score(splits['y_val'], val_pred)
            },
            'Test': {
                'MSE': mean_squared_error(splits['y_test'], test_pred),
                'MAE': mean_absolute_error(splits['y_test'], test_pred),
                'R2': r2_score(splits['y_test'], test_pred)
            }
        }

        import pickle

        # Save the model
        with open('backend/models/knn_model.pkl', "wb") as f:
            pickle.dump(best_knn, f)
        with open('backend/models/scaler.pkl', "wb") as f:
            pickle.dump(self.scaler, f)
        
        # Log results
        logging.info("Best Hyperparameters:")
        logging.info(f"Number of Neighbors: {best_params['n_neighbors']}")
        logging.info(f"Weights: {best_params['weights']}")
        logging.info("\nValidation Metrics:")
        for metric, value in results['Validation'].items():
            logging.info(f"{metric}: {value}")
        logging.info("\nTest Metrics:")
        for metric, value in results['Test'].items():
            logging.info(f"{metric}: {value}")
        
        return results

    def run_analysis(self):
        """
        Run complete analysis pipeline
        """
        try:
            # Load data
            self.load_and_preprocess_data()
            
            # Split data across symbols
            splits = self.collect_symbol_splits()
            
            # Explore data characteristics
            self.explore_data_distribution(splits)
            
            # Perform hyperparameter tuning
            results = self.hyperparameter_tuning(splits)
            
            return splits, results
        
        except Exception as e:
            logging.error(f"Error in analysis: {e}")
            raise

def main():
    # Path to your CSV file
    data_path = '/Users/devshah/Documents/WorkSpace/University/year 3/CSC392/Trading_Simulator/data/updated_data.csv'
    
    # Initialize and run analyzer
    analyzer = SymbolTimeSeriesSplitter(data_path)
    splits, results = analyzer.run_analysis()

if __name__ == '__main__':
    main()
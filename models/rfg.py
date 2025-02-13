from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def read_csv(path):
    return pd.read_csv(path)

def get_data():
    df = read_csv('data/data.csv')
    df.columns = ['date'] + list(df.columns[1:])  
    df = df.drop(columns=['date']) 

    features = df.columns[:-1]  
    target = df.columns[-1]    

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = get_data()
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
  
if __name__ == "__main__":
    main()

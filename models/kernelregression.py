import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import gc
import os

from preprocessing import prepare_data_for_training

def train_kernel_regression(file_path, sequence_length=60, 
                           steps_per_epoch=100, epochs=10, 
                           batch_size=32, model_save_path="kernel_model.pkl"):
    """
    Train a kernel regression model with memory efficiency in mind.
    
    Args:
        file_path: Path to the CSV data file
        sequence_length: Number of days in each sequence
        steps_per_epoch: Number of batches to process per epoch
        epochs: Number of training epochs
        batch_size: Number of samples per batch
        model_save_path: Where to save the trained model
    
    Returns:
        Trained model and validation metrics
    """
    print("Preparing data...")
    train_gen, X_val, y_val, scaler = prepare_data_for_training(
        file_path, sequence_length
    )
    
    print("Initializing model...")
    
    if len(X_val) > 1000:
        X_val_sample = X_val[:1000]
        y_val_sample = y_val[:1000]
    else:
        X_val_sample = X_val
        y_val_sample = y_val
    
    model = KernelRidge(alpha=1.0, kernel='linear')
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        epoch_losses = []
        for step in range(steps_per_epoch):
            X_batch, y_batch = next(train_gen)
            
            model.fit(X_batch, y_batch)
            
            y_pred = model.predict(X_batch)
            loss = mean_squared_error(y_batch, y_pred)
            epoch_losses.append(loss)
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{steps_per_epoch} - Loss: {loss:.6f}")
                
            gc.collect()
            
        val_chunk_size = 200
        val_predictions = []
        
        for i in range(0, len(X_val), val_chunk_size):
            X_val_chunk = X_val[i:i+val_chunk_size]
            val_predictions.extend(model.predict(X_val_chunk))
            
        val_mse = mean_squared_error(y_val, val_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        
        print(f"Epoch {epoch+1} completed - Avg Loss: {np.mean(epoch_losses):.6f}, Val MSE: {val_mse:.6f}, Val R²: {val_r2:.4f}")
        
        
    print("Training completed. Final evaluation:")
    val_predictions = []
    for i in range(0, len(X_val), val_chunk_size):
        X_val_chunk = X_val[i:i+val_chunk_size]
        val_predictions.extend(model.predict(X_val_chunk))
        
    val_mse = mean_squared_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    print(f"Final validation MSE: {val_mse:.6f}, R²: {val_r2:.4f}")
    
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model, (val_mse, val_r2)

if __name__ == "__main__":
    model_dir = os.path.join("backend", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_dir, "kernel_model.pkl")
    
    model, metrics = train_kernel_regression(
        "data/updated_data.csv",
        sequence_length=30,  
        steps_per_epoch=50,
        epochs=5,
        batch_size=16,  
        model_save_path=model_save_path 
    )
    
    print(f"Training completed with MSE: {metrics[0]:.6f}, R²: {metrics[1]:.4f}")
    print(f"Model saved to {model_save_path}")
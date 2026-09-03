import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNorm, Dropout, Bidirectional, LSTM, Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. DATA LOADING & WINDOWING FUNCTION
# -------------------------------------------------------------
def load_dataset_pairs(base_path, window_size=20, stride=1):
    X_list, y_list = [], []
    
    # Locate all synchronized S- and V- CSV pairs
    s_files = glob.glob(os.path.join(base_path, "**/S-*.csv"), recursive=True)
    print(f"Found {len(s_files)} smartphone CSV files.")
    
    for idx, s_file in enumerate(s_files, 1):
        v_file = s_file.replace("S-", "V-")
        if not os.path.exists(v_file):
            continue
            
        try:
            df_s = pd.read_csv(s_file, encoding='latin1')
            df_v = pd.read_csv(v_file, encoding='latin1')
            
            min_len = min(len(df_s), len(df_v))
            if min_len < window_size + 10:
                continue
                
            df_s = df_s.iloc[:min_len]
            df_v = df_v.iloc[:min_len]
            
            # Extract features from S- (Smartphone)
            ax = df_s['ACCELEROMETER X (m/s²)'].values
            ay = df_s['ACCELEROMETER Y (m/s²)'].values
            az = df_s['ACCELEROMETER Z (m/s²)'].values
            gx = df_s['GRAVITY X (m/s²)'].values
            gy = df_s['GRAVITY Y (m/s²)'].values
            gz = df_s['GRAVITY Z (m/s²)'].values
            
            # Gravity-free linear acceleration
            lax, lay, laz = ax - gx, ay - gy, az - gz
            acc_mag = np.sqrt(lax**2 + lay**2 + laz**2)
            
            gy_yaw = df_s['GYROSCOPE Yaw (rad/s)'].values
            gy_pitch = df_s['GYROSCOPE Pitch (rad/s)'].values
            gy_roll = df_s['GYROSCOPE Roll (rad/s)'].values
            gyro_mag = np.sqrt(gy_yaw**2 + gy_pitch**2 + gy_roll**2)
            
            yaw = df_s['ORIENTATION (Yaw) (°)'].values / 360.0
            pitch = df_s['ORIENTATION (Pitch) (°)'].values / 180.0
            
            features = np.column_stack([lax, lay, laz, acc_mag, gy_yaw, gy_pitch, gy_roll, gyro_mag, yaw, pitch])
            
            # Target speed from V- (Vehicle speed in m/s)
            target_speed = df_v['Velocity (km/hr)'].values / 3.6
            
            # Create sliding windows
            for i in range(0, min_len - window_size, stride):
                X_list.append(features[i : i + window_size])
                y_list.append(target_speed[i + window_size - 1])
                
        except Exception as e:
            print(f"Skipping {s_file}: {e}")
            
    return np.array(X_list), np.array(y_list)

# -------------------------------------------------------------
# 2. MODEL BUILDING FUNCTION
# -------------------------------------------------------------
def build_velocity_model(input_shape=(20, 10)):
    model = Sequential([
        Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# -------------------------------------------------------------
# 3. MAIN TRAINING & INFERENCE PIPELINE
# -------------------------------------------------------------
if __name__ == "__main__":
    dataset_path = "../IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset"
    
    print("Loading and windowing IO-VNBD dataset...")
    X, y = load_dataset_pairs(dataset_path, window_size=20, stride=2)
    print(f"Dataset Loaded! Samples: {X.shape[0]}, Window Shape: {X.shape[1:]}")
    
    # Train / Val Split
    split_idx = int(0.85 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    print("Building 1D-CNN + BiLSTM Model...")
    model = build_velocity_model(input_shape=(20, 10))
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=128,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    
    # Save model for edge export (TFLite / ONNX)
    os.makedirs("./trained_models", exist_ok=True)
    model.save("./trained_models/velocity_bilstm_model.h5")
    print("Model saved to ./trained_models/velocity_bilstm_model.h5")
    
    # Evaluate & Plot Speed Predictions
    y_pred = model.predict(X_val)
    plt.figure(figsize=(12, 5))
    plt.plot(y_val[:500], label='True Vehicle Speed (m/s)', color='blue')
    plt.plot(y_pred[:500], label='Predicted IMU Speed (m/s)', color='red', linestyle='--')
    plt.title("Vehicle Forward Velocity Estimation from Smartphone IMU")
    plt.xlabel("Time Step (100ms)")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid(True)
    plt.savefig("./trained_models/velocity_prediction_plot.png")
    print("Prediction plot saved to ./trained_models/velocity_prediction_plot.png")

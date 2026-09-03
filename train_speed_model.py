import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Bidirectional, LSTM, Dense
import matplotlib.pyplot as plt

def find_column(df, keyword):
    """Robust helper to find a column matching a keyword regardless of whitespace or special characters."""
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    raise KeyError(f"Keyword '{keyword}' not found in columns: {list(df.columns)}")

# -------------------------------------------------------------
# 1. DATA LOADING & WINDOWING FUNCTION
# -------------------------------------------------------------
def load_dataset_pairs(base_path, window_size=20, stride=1):
    X_list, y_list = [], []
    
    # Check current directory and parent directory if dataset folder exists
    possible_paths = [
        base_path,
        os.path.join(".", "IO-VNBD", "Synchronised V abd S datasets", "Categorised IOVNB Dataset"),
        os.path.join("..", "IO-VNBD", "Synchronised V abd S datasets", "Categorised IOVNB Dataset"),
        os.path.join(".", "Synchronised V abd S datasets", "Categorised IOVNB Dataset")
    ]
    
    selected_path = None
    for p in possible_paths:
        if os.path.exists(p):
            selected_path = p
            break
            
    if selected_path is None:
        selected_path = base_path

    print(f"Searching dataset in: {os.path.abspath(selected_path)}")
    
    # Locate all synchronized S- and V- CSV pairs
    s_files = glob.glob(os.path.join(selected_path, "**/S-*.csv"), recursive=True)
    print(f"Found {len(s_files)} smartphone CSV files.")
    
    for idx, s_file in enumerate(s_files, 1):
        v_file = s_file.replace("S-", "V-")
        if not os.path.exists(v_file):
            continue
            
        try:
            df_s = pd.read_csv(s_file, encoding='latin1')
            df_v = pd.read_csv(v_file, encoding='latin1')
            
            # Clean column names by stripping leading/trailing whitespace
            df_s.columns = [c.strip() for c in df_s.columns]
            df_v.columns = [c.strip() for c in df_v.columns]
            
            min_len = min(len(df_s), len(df_v))
            if min_len < window_size + 10:
                continue
                
            df_s = df_s.iloc[:min_len]
            df_v = df_v.iloc[:min_len]
            
            # Extract features from S- (Smartphone) using robust column search
            col_ax = find_column(df_s, 'ACCELEROMETER X')
            col_ay = find_column(df_s, 'ACCELEROMETER Y')
            col_az = find_column(df_s, 'ACCELEROMETER Z')
            col_gx = find_column(df_s, 'GRAVITY X')
            col_gy = find_column(df_s, 'GRAVITY Y')
            col_gz = find_column(df_s, 'GRAVITY Z')
            
            col_gy_yaw = find_column(df_s, 'GYROSCOPE Yaw')
            col_gy_pitch = find_column(df_s, 'GYROSCOPE Pitch')
            col_gy_roll = find_column(df_s, 'GYROSCOPE Roll')
            
            col_yaw = find_column(df_s, 'ORIENTATION (Yaw)')
            col_pitch = find_column(df_s, 'ORIENTATION (Pitch)')
            
            ax = df_s[col_ax].values
            ay = df_s[col_ay].values
            az = df_s[col_az].values
            gx = df_s[col_gx].values
            gy = df_s[col_gy].values
            gz = df_s[col_gz].values
            
            # Gravity-free linear acceleration
            lax, lay, laz = ax - gx, ay - gy, az - gz
            acc_mag = np.sqrt(lax**2 + lay**2 + laz**2)
            
            gy_yaw = df_s[col_gy_yaw].values
            gy_pitch = df_s[col_gy_pitch].values
            gy_roll = df_s[col_gy_roll].values
            gyro_mag = np.sqrt(gy_yaw**2 + gy_pitch**2 + gy_roll**2)
            
            yaw = df_s[col_yaw].values / 360.0
            pitch = df_s[col_pitch].values / 180.0
            
            features = np.column_stack([lax, lay, laz, acc_mag, gy_yaw, gy_pitch, gy_roll, gyro_mag, yaw, pitch])
            
            # Target speed from V- (Vehicle speed in m/s)
            col_vel = find_column(df_v, 'Velocity')
            target_speed = df_v[col_vel].values / 3.6
            
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
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
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
    dataset_path = "./IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset"
    
    print("Loading and windowing IO-VNBD dataset...")
    X, y = load_dataset_pairs(dataset_path, window_size=20, stride=2)
    print(f"Dataset Loaded! Samples: {X.shape[0]}, Window Shape: {X.shape[1:]}")
    
    if len(X) == 0:
        print("\nERROR: No dataset files were loaded! Please ensure your dataset directory exists.")
        exit(1)
        
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

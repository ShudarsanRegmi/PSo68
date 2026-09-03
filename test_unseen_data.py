import os
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

# Try importing Plotly for interactive web-based charts
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt

def find_column(df, keyword):
    """Helper to find a column matching a keyword regardless of whitespace/encoding."""
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    raise KeyError(f"Keyword '{keyword}' not found in columns: {list(df.columns)}")

def predict_on_unseen_file(s_csv_path, v_csv_path=None, model_path="./trained_models/velocity_bilstm_model.h5", window_size=20):
    # 1. Load trained model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading trained model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()}, compile=False)
        
    print("Model loaded successfully!")
    
    # 2. Resolve Smartphone CSV Path
    possible_s_paths = [
        s_csv_path,
        os.path.join(".", s_csv_path),
        os.path.join("..", s_csv_path),
        os.path.join(".", "IO-VNBD", "Synchronised V abd S datasets", "Categorised IOVNB Dataset", "M (Driver B)", "S-M.csv"),
        os.path.join("..", "IO-VNBD", "Synchronised V abd S datasets", "Categorised IOVNB Dataset", "M (Driver B)", "S-M.csv")
    ]
    
    actual_s_path = None
    for p in possible_s_paths:
        if p and os.path.exists(p):
            actual_s_path = p
            break
            
    if actual_s_path is None:
        print(f"Error: Could not find smartphone CSV file at {s_csv_path}")
        return

    print(f"Loading unseen smartphone CSV: {actual_s_path}")
    df_s = pd.read_csv(actual_s_path, encoding='latin1')
    df_s.columns = [c.strip() for c in df_s.columns]
    
    # Extract 10 IMU feature channels
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
    
    lax, lay, laz = ax - gx, ay - gy, az - gz
    acc_mag = np.sqrt(lax**2 + lay**2 + laz**2)
    
    gy_yaw = df_s[col_gy_yaw].values
    gy_pitch = df_s[col_gy_pitch].values
    gy_roll = df_s[col_gy_roll].values
    gyro_mag = np.sqrt(gy_yaw**2 + gy_pitch**2 + gy_roll**2)
    
    yaw = df_s[col_yaw].values / 360.0
    pitch = df_s[col_pitch].values / 180.0
    
    features = np.column_stack([lax, lay, laz, acc_mag, gy_yaw, gy_pitch, gy_roll, gyro_mag, yaw, pitch])
    
    # Create sliding windows
    X_unseen = []
    for i in range(len(features) - window_size):
        X_unseen.append(features[i : i + window_size])
    X_unseen = np.array(X_unseen)
    
    print(f"Constructed {len(X_unseen)} inference windows.")
    
    # 3. Predict Velocity
    y_pred = model.predict(X_unseen).flatten() # Speed in m/s
    y_pred_kmh = y_pred * 3.6 # Convert to km/h
    
    # 4. Optional: Load Vehicle Ground Truth if available for comparison
    y_true_kmh = None
    actual_v_path = None
    if v_csv_path:
        possible_v_paths = [
            v_csv_path,
            os.path.join(".", v_csv_path),
            os.path.join("..", v_csv_path),
            os.path.join(".", "IO-VNBD", "Synchronised V abd S datasets", "Categorised IOVNB Dataset", "M (Driver B)", "V-M.csv"),
            os.path.join("..", "IO-VNBD", "Synchronised V abd S datasets", "Categorised IOVNB Dataset", "M (Driver B)", "V-M.csv")
        ]
        for p in possible_v_paths:
            if p and os.path.exists(p):
                actual_v_path = p
                break

    if actual_v_path and os.path.exists(actual_v_path):
        print(f"Loading vehicle ground truth CSV: {actual_v_path}")
        df_v = pd.read_csv(actual_v_path, encoding='latin1')
        df_v.columns = [c.strip() for c in df_v.columns]
        col_vel = find_column(df_v, 'Velocity')
        y_true_kmh = df_v[col_vel].values[window_size:]
        min_len = min(len(y_pred_kmh), len(y_true_kmh))
        y_pred_kmh = y_pred_kmh[:min_len]
        y_true_kmh = y_true_kmh[:min_len]
    
    time_seconds = np.arange(len(y_pred_kmh)) * 0.1 # 10 Hz = 0.1s per step
    os.makedirs("./trained_models", exist_ok=True)

    # 5. Generate Interactive Plotly HTML Chart
    if HAS_PLOTLY:
        print("Generating interactive Plotly graph...")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_seconds, 
            y=y_pred_kmh,
            mode='lines',
            name='AI Predicted Speed (km/h)',
            line=dict(color='red', width=1.5)
        ))

        if y_true_kmh is not None:
            fig.add_trace(go.Scatter(
                x=time_seconds, 
                y=y_true_kmh,
                mode='lines',
                name='True Vehicle Ground Speed (km/h)',
                line=dict(color='blue', width=1.5, dash='dash')
            ))

        fig.update_layout(
            title=f"Interactive Velocity Inference: {os.path.basename(actual_s_path)}",
            xaxis_title="Elapsed Time (Seconds)",
            yaxis_title="Vehicle Speed (km/h)",
            hovermode="x unified",
            xaxis=dict(
                rangeslider=dict(visible=True),  # Interactive range slider at bottom
                type="linear"
            ),
            template="plotly_white"
        )

        out_html = "./trained_models/interactive_velocity_inference.html"
        fig.write_html(out_html)
        print(f"Interactive HTML graph saved to: {os.path.abspath(out_html)}")
        print("Open this file in any browser to pan, zoom, and inspect data interactively!")
        
        # Try launching default browser automatically
        try:
            import webbrowser
            webbrowser.open(os.path.abspath(out_html))
        except Exception:
            pass

    else:
        # Fallback Matplotlib plot with pan/zoom enabled
        print("Plotly not installed. Install with 'pip install plotly' for web charts.")
        print("Rendering Matplotlib chart with pan/zoom window...")
        plt.figure(figsize=(14, 6))
        plt.plot(time_seconds, y_pred_kmh, label='AI Predicted Speed (km/h)', color='red', linewidth=1.5)
        if y_true_kmh is not None:
            plt.plot(time_seconds, y_true_kmh, label='True Vehicle Speed (km/h)', color='blue', alpha=0.7, linestyle='--')
        plt.title(f"Velocity Inference: {os.path.basename(actual_s_path)}")
        plt.xlabel("Elapsed Time (Seconds)")
        plt.ylabel("Vehicle Speed (km/h)")
        plt.legend()
        plt.grid(True)
        plt.savefig("./trained_models/unseen_test_inference.png")
        plt.show()

if __name__ == "__main__":
    default_s_file = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"
    default_v_file = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv"
    
    if len(sys.argv) > 1:
        s_file = sys.argv[1]
        v_file = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        s_file = default_s_file
        v_file = default_v_file
        
    predict_on_unseen_file(s_file, v_file)

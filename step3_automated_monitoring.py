import os
from datetime import datetime
import pandas as pd
import joblib

# 1. Working directory and file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

model_path = os.path.join(base_dir, 'telecom_rf_model.pkl')
log_file = os.path.join(base_dir, 'network_health_log.csv')
html_report = os.path.join(base_dir, 'dashboard.html')

def run_network_diagnostic():
    # 2. Ingest trained classifier
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run step2_fault_prediction_model.py first.")

    print(f"[INFO] Ingesting trained Random Forest model from '{model_path}'...")
    model = joblib.load(model_path)
    
    # 3. Simulate real-time streaming telemetry sample
    telemetry_sample = pd.DataFrame({
        'Connected_Users': [450],
        'SNR_dB': [8.5], 
        'Latency_ms': [65.0]
    })
    
    # 4. Execute inference
    prediction = model.predict(telemetry_sample)[0]
    status = "Critical" if prediction == 1 else "Normal"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 5. Persist audit log record
    log_entry = pd.DataFrame([[current_time, 450, 8.5, 65.0, status]], 
                             columns=['Timestamp', 'Users', 'SNR', 'Latency', 'Status'])
    
    if not os.path.isfile(log_file):
        log_entry.to_csv(log_file, index=False)
    else:
        log_entry.to_csv(log_file, mode='a', header=False, index=False)
        
    # 6. Render NOC (Network Operations Center) HTML Dashboard
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Network Operations Center - Health Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1e1e2f;
            color: #ffffff;
            padding: 40px;
            margin: 0;
        }}
        .container {{
            max-width: 800px;
            margin: auto;
            background-color: #27293d;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }}
        h2 {{
            color: #e1e1e1;
            border-bottom: 2px solid #3b3b53;
            padding-bottom: 15px;
            margin-top: 0;
        }}
        .metric-box {{
            background-color: #32325d;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 1.1em;
            display: flex;
            justify-content: space-between;
        }}
        .metric-label {{
            color: #a4a6b3;
            font-weight: bold;
        }}
        .status-critical {{
            color: #ff5d5d;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .status-normal {{
            color: #00f2c3;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .footer {{
            margin-top: 30px;
            font-size: 0.85em;
            color: #8b8b99;
            text-align: right;
            border-top: 1px solid #3b3b53;
            padding-top: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Cellular Network Operations Health Monitor</h2>
        
        <div class="metric-box">
            <span class="metric-label">Last Telemetry Timestamp:</span>
            <span>{current_time}</span>
        </div>
        
        <div class="metric-box">
            <span class="metric-label">Network Status Assessment:</span> 
            <span class="{'status-critical' if status == 'Critical' else 'status-normal'}">{status}</span>
        </div>
        
        <div class="metric-box">
            <span class="metric-label">Signal Quality (SNR):</span>
            <span>8.5 dB</span>
        </div>

        <div class="metric-box">
            <span class="metric-label">Connected Subscribers:</span>
            <span>450</span>
        </div>

        <div class="metric-box">
            <span class="metric-label">Radio Latency:</span>
            <span>65.0 ms</span>
        </div>
        
        <div class="footer">Automated Telemetry Fault Classification Pipeline</div>
    </div>
</body>
</html>
"""
    
    with open(html_report, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"[{current_time}] Diagnostic execution completed. Network Status: {status}")
    print(f"[INFO] Audit log updated: '{log_file}'")
    print(f"[INFO] NOC HTML dashboard rendered: '{html_report}'")

if __name__ == "__main__":
    run_network_diagnostic()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Working directory configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Set random seed for reproducibility
np.random.seed(42)

# 2. Network simulation parameters
num_records = 1000
cell_ids = ['TOWER_001', 'TOWER_002', 'TOWER_003', 'TOWER_004', 'TOWER_005']

# 3. Generate synthetic cellular telemetry
timestamps = pd.date_range(start='2026-08-01', periods=num_records, freq='h')
towers = np.random.choice(cell_ids, size=num_records)
users = np.random.randint(50, 500, size=num_records)

# Signal-to-Noise Ratio (SNR in dB): Normal distribution (mean 15 dB, std 5 dB)
snr = np.random.normal(loc=15.0, scale=5.0, size=num_records)
snr = np.round(snr, 2)

# Drop Call Rate (%): Base rate between 0.1% and 3.0%
# Physics-based correlation: degrade drop rate when SNR < 10.0 dB
drop_rate = np.random.uniform(0.1, 3.0, size=num_records)
drop_rate = np.where(snr < 10.0, drop_rate * 2, drop_rate)
drop_rate = np.round(drop_rate, 2)

# Latency in milliseconds (20ms to 150ms)
latency = np.random.randint(20, 150, size=num_records)

# 4. Assemble Telemetry DataFrame
df = pd.DataFrame({
    'Timestamp': timestamps,
    'Cell_ID': towers,
    'Connected_Users': users,
    'SNR_dB': snr,
    'Drop_Call_Rate_Pct': drop_rate,
    'Latency_ms': latency
})

print("--- Initial Telemetry Records ---")
print(df.head())

print("\n--- Network Global Statistics ---")
print(df.describe())

# 5. Cell Tower Aggregation and KPI Performance Evaluation
tower_stats = df.groupby('Cell_ID').agg({
    'Connected_Users': 'mean',
    'SNR_dB': 'mean',
    'Drop_Call_Rate_Pct': 'max', 
    'Latency_ms': 'mean'
}).reset_index()

tower_stats.rename(columns={
    'Connected_Users': 'Avg_Users',
    'SNR_dB': 'Avg_SNR',
    'Drop_Call_Rate_Pct': 'Max_Drop_Rate',
    'Latency_ms': 'Avg_Latency'
}, inplace=True)

worst_towers = tower_stats.sort_values(by='Avg_SNR', ascending=True)
print("\n--- Tower Performance Summary (Worst SNR First) ---")
print(worst_towers)

# 6. Telemetry Visualization
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Maximum Drop Call Rate per Tower
sns.barplot(
    data=tower_stats, 
    x='Cell_ID', 
    y='Max_Drop_Rate', 
    ax=axes[0], 
    hue='Cell_ID',
    palette="viridis",
    legend=False
)
axes[0].set_title("Maximum Drop Call Rate per Tower", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Cell Tower ID")
axes[0].set_ylabel("Max Drop Rate (%)")
axes[0].tick_params(axis='x', rotation=45)

# Subplot 2: Physical Signal Correlation (SNR vs. Drop Rate)
sns.scatterplot(
    data=df, 
    x='SNR_dB', 
    y='Drop_Call_Rate_Pct', 
    alpha=0.6, 
    color="darkred", 
    ax=axes[1]
)
axes[1].axvline(
    x=10.0, 
    color='black', 
    linestyle='--', 
    label='Critical SNR Threshold (10 dB)'
)
axes[1].set_title("Impact of Signal Quality on Drop Calls", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Signal-to-Noise Ratio (dB)")
axes[1].set_ylabel("Drop Call Rate (%)")
axes[1].legend()

plt.tight_layout()
plt.savefig("network_kpi_visualization.png")
plt.close()

# 7. Anomaly Filtering and Field Maintenance Reporting
critical_events = df[(df['Drop_Call_Rate_Pct'] > 3.0) | (df['SNR_dB'] < 10.0)]
critical_events = critical_events.sort_values(by='Drop_Call_Rate_Pct', ascending=False)

report_file = "critical_tower_report.csv"
critical_events.to_csv(report_file, index=False)

print("\n--- Field Maintenance Report ---")
print(f"Total critical anomalies detected: {len(critical_events)} of {num_records} records")
print(f"Report exported to: {report_file}")
print("\nTop 5 Critical Events Requiring Urgent Remediation:")
print(critical_events.head())

# Export processed dataset for machine learning pipeline
dataset_file = "telecom_dataset.csv"
df.to_csv(dataset_file, index=False)
print(f"\n[SUCCESS] Baseline dataset exported to '{dataset_file}'.")

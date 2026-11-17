import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

np.random.seed(42)
N_SAMPLES = 6000

print("Generating cluster workload telemetry dataset...")

# Simulate realistic node states:
# 1. Base load (steady background services)
# 2. Trend/Momentum (whether traffic is ramping up or cooling down)
cpu_5m_avg = np.random.uniform(5.0, 75.0, N_SAMPLES)
trend = np.random.normal(0.0, 4.5, N_SAMPLES)  # Positive = spiking, Negative = cooling

cpu_1m_ago = np.clip(cpu_5m_avg + trend * 0.6 + np.random.normal(0, 1.5, N_SAMPLES), 1.0, 98.0)
cpu_now = np.clip(cpu_1m_ago + trend * 0.8 + np.random.normal(0, 1.5, N_SAMPLES), 1.0, 99.0)

# Engineered feature: rate of change (slope)
cpu_slope = cpu_now - cpu_1m_ago

ram_avail_mb = np.clip(8192 - (cpu_now * 55) + np.random.normal(0, 400, N_SAMPLES), 256, 8000)
pod_count = np.clip(np.round((cpu_now / 6.0) + np.random.randint(1, 5, N_SAMPLES)), 1, 30)

# Target variable: Future CPU utilization 5 minutes ahead (t + 5m)
# Influenced by current CPU, momentum (slope), pod density, and non-linear congestion above 70%
congestion_penalty = np.where(cpu_now > 65.0, (cpu_now - 65.0) * 0.35, 0.0)
future_cpu_5m = np.clip(
    cpu_now + (cpu_slope * 1.4) + (pod_count * 0.25) + congestion_penalty + np.random.normal(0, 2.2, N_SAMPLES),
    1.0,
    100.0,
)

df = pd.DataFrame({
    "cpu_now": cpu_now,
    "cpu_1m_ago": cpu_1m_ago,
    "cpu_5m_avg": cpu_5m_avg,
    "cpu_slope": cpu_slope,
    "ram_avail_mb": ram_avail_mb,
    "pod_count": pod_count,
    "future_cpu_5m": future_cpu_5m,
})

# Save dataset for inspection / reproducibility
df.to_csv("D:/kubesopt/cluster_telemetry.csv", index=False)

features = ["cpu_now", "cpu_1m_ago", "cpu_5m_avg", "cpu_slope", "ram_avail_mb", "pod_count"]
X = df[features]
y = df["future_cpu_5m"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost Regressor...")
model = XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.85,
    random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\n--- Model Evaluation Results ---")
print(f"RMSE (Prediction Error): {rmse:.2f}% CPU")
print(f"R2 Score (Accuracy):     {r2:.4f}")

joblib.dump(model, "D:/kubesopt/kubesopt_model.joblib")
print("Saved trained model to D:/kubesopt/kubesopt_model.joblib")

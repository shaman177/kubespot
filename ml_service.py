import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="KubeSopt ML Load Prediction Service")
model = joblib.load("D:/kubesopt/kubesopt_model.joblib")


class NodeMetrics(BaseModel):
    node_name: str
    cpu_now: float
    cpu_1m_ago: float
    cpu_5m_avg: float
    ram_avail_mb: float
    pod_count: int


@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost-v1"}


@app.post("/predict")
def predict_node_score(metrics: NodeMetrics):
    cpu_slope = metrics.cpu_now - metrics.cpu_1m_ago

    features = np.array([[
        metrics.cpu_now,
        metrics.cpu_1m_ago,
        metrics.cpu_5m_avg,
        cpu_slope,
        metrics.ram_avail_mb,
        metrics.pod_count,
    ]])

    predicted_cpu_5m = float(model.predict(features)[0])
    predicted_cpu_5m = max(0.0, min(100.0, predicted_cpu_5m))

    # Calculate final KubeSopt suitability score (0 to ~180 scale, higher = better node)
    # Heavily penalize nodes predicted to spike in the next 5 minutes
    suitability_score = (100.0 - predicted_cpu_5m) + (metrics.ram_avail_mb / 120.0) - (metrics.pod_count * 0.5)

    return {
        "node_name": metrics.node_name,
        "cpu_now": round(metrics.cpu_now, 2),
        "cpu_slope": round(cpu_slope, 2),
        "predicted_cpu_5m": round(predicted_cpu_5m, 2),
        "suitability_score": round(suitability_score, 2),
    }

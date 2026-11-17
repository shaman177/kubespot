import json
from datetime import datetime, timezone
import requests
from kubernetes import client, config, watch

SCHEDULER_NAME = "kubesopt-scheduler"
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
ML_SERVICE_URL = "http://localhost:8000/predict"
DECISION_LOG_FILE = "D:/kubesopt/scheduling_logs.jsonl"

config.load_kube_config()
v1 = client.CoreV1Api()


def get_schedulable_nodes():
    """Filter out control-plane/tainted nodes and return healthy worker nodes."""
    workers = []
    for node in v1.list_node().items:
        taints = node.spec.taints or []
        if any(t.effect == "NoSchedule" for t in taints):
            continue

        is_ready = any(
            cond.type == "Ready" and cond.status == "True"
            for cond in (node.status.conditions or [])
        )
        if not is_ready:
            continue

        internal_ip = next(
            (addr.address for addr in node.status.addresses if addr.type == "InternalIP"),
            None,
        )
        if internal_ip:
            workers.append({"name": node.metadata.name, "ip": internal_ip})
    return workers


def get_node_pod_counts():
    """Count active pods currently assigned to each node."""
    counts = {}
    for pod in v1.list_pod_for_all_namespaces().items:
        node_name = pod.spec.node_name
        if node_name and pod.status.phase in ("Running", "Pending"):
            counts[node_name] = counts.get(node_name, 0) + 1
    return counts


def query_prometheus(promql):
    """Query Prometheus HTTP API and return {instance: float_value}."""
    try:
        resp = requests.get(PROMETHEUS_URL, params={"query": promql}, timeout=3)
        results = resp.json().get("data", {}).get("result", [])
        return {item["metric"]["instance"]: float(item["value"][1]) for item in results}
    except Exception as e:
        print(f"[WARN] Prometheus query failed: {e}")
        return {}


def pick_best_node_with_ml(workers, pod_name, namespace):
    """Gather time-series metrics from Prometheus and score nodes via the XGBoost ML API."""
    cpu_now_map = query_prometheus('100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)')
    cpu_1m_ago_map = query_prometheus('100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m] offset 1m)) * 100)')
    cpu_5m_avg_map = query_prometheus('100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)')
    ram_map = query_prometheus('node_memory_MemAvailable_bytes / 1024 / 1024')
    pod_counts = get_node_pod_counts()

    best_node = None
    best_score = -9999.0
    evaluated_nodes = []

    print("\n--- KubeSopt ML Predictive Telemetry & Scoring ---")
    for w in workers:
        instance = f"{w['ip']}:9100"
        cpu_now = cpu_now_map.get(instance, 15.0)
        cpu_1m_ago = cpu_1m_ago_map.get(instance, cpu_now)
        cpu_5m_avg = cpu_5m_avg_map.get(instance, cpu_now)
        ram_avail_mb = ram_map.get(instance, 2048.0)
        pods_on_node = pod_counts.get(w["name"], 0)

        payload = {
            "node_name": w["name"],
            "cpu_now": cpu_now,
            "cpu_1m_ago": cpu_1m_ago,
            "cpu_5m_avg": cpu_5m_avg,
            "ram_avail_mb": ram_avail_mb,
            "pod_count": pods_on_node,
        }

        try:
            ml_resp = requests.post(ML_SERVICE_URL, json=payload, timeout=3).json()
            pred_cpu = ml_resp["predicted_cpu_5m"]
            score = ml_resp["suitability_score"]
            slope = ml_resp["cpu_slope"]
        except Exception as e:
            print(f"[WARN] ML Service unreachable ({e}), falling back to heuristic.")
            pred_cpu = cpu_now
            slope = cpu_now - cpu_1m_ago
            score = (100.0 - cpu_now) + (ram_avail_mb / 120.0)

        print(
            f"Node: {w['name']} | CPU Now: {cpu_now:.2f}% (Slope: {slope:+.2f}%) | "
            f"Pred CPU (t+5m): {pred_cpu:.2f}% | Avail RAM: {ram_avail_mb:.0f} MB | "
            f"Pods: {pods_on_node} | ML Score: {score:.2f}"
        )

        evaluated_nodes.append({
            "node_name": w["name"],
            "ip": w["ip"],
            "cpu_now": round(cpu_now, 2),
            "cpu_slope": round(slope, 2),
            "predicted_cpu_5m": round(pred_cpu, 2),
            "ram_avail_mb": round(ram_avail_mb, 0),
            "pod_count": pods_on_node,
            "suitability_score": round(score, 2),
        })

        if score > best_score:
            best_score = score
            best_node = w["name"]

    # Save structured decision log for Phase 4 RAG Observability Assistant
    if best_node:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pod": f"{namespace}/{pod_name}",
            "selected_node": best_node,
            "winning_score": round(best_score, 2),
            "candidates": evaluated_nodes,
        }
        with open(DECISION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    return best_node


def bind_pod(pod_name, namespace, node_name):
    """Bind the pending pod to the chosen worker node via Kubernetes API."""
    target = client.V1ObjectReference(kind="Node", api_version="v1", name=node_name)
    meta = client.V1ObjectMeta(name=pod_name)
    body = client.V1Binding(target=target, metadata=meta)

    v1.create_namespaced_binding(namespace=namespace, body=body, _preload_content=False)
    print(f"[SCHEDULED] Bound pod '{pod_name}' -> '{node_name}' (Decision logged)\n")


def main():
    print(f"Starting {SCHEDULER_NAME} (ML-Enabled)... Watching for Pending pods.")
    watcher = watch.Watch()
    for event in watcher.stream(v1.list_pod_for_all_namespaces):
        pod = event["object"]
        if (
            pod.status.phase == "Pending"
            and pod.spec.scheduler_name == SCHEDULER_NAME
            and pod.spec.node_name is None
        ):
            pod_name = pod.metadata.name
            namespace = pod.metadata.namespace
            print(f"\n[EVENT] Intercepted unscheduled pod: {namespace}/{pod_name}")

            workers = get_schedulable_nodes()
            if not workers:
                print("[ERROR] No healthy worker nodes available!")
                continue

            chosen_node = pick_best_node_with_ml(workers, pod_name, namespace)
            if chosen_node:
                try:
                    bind_pod(pod_name, namespace, chosen_node)
                except Exception as e:
                    print(f"[ERROR] Failed to bind {pod_name}: {e}")


if __name__ == "__main__":
    main()

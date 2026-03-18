from __future__ import annotations

import json
from typing import Any, Dict, Optional


def parse_json_dict(raw: str) -> Dict[str, str]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v is not None}
        if isinstance(data, list):
            parsed: Dict[str, str] = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = item.get("Name") or item.get("name")
                value = item.get("Value") or item.get("value")
                if name and value is not None:
                    parsed[str(name)] = str(value)
            if parsed:
                return parsed
    except Exception:
        return {}
    return {}


def extract_metrics(ingress: Optional[Dict[str, Any]], proc: Optional[Dict[str, Any]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    def _num(val: Any) -> Optional[float]:
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _nested_num(data: Optional[Dict[str, Any]], *path: str) -> Optional[float]:
        cur: Any = data
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return _num(cur)

    if isinstance(ingress, dict) and "sse_connections" in ingress:
        sse = ingress.get("sse_connections") or {}
        total = _num(sse.get("global_total_connections") or sse.get("total_connections"))
        max_conn = _num(sse.get("global_max_connections") or sse.get("max_connections"))
        sessions = _num(sse.get("global_sessions") or sse.get("sessions"))
        if total is not None:
            metrics["ingress.sse.total_connections"] = total
        if max_conn is not None:
            metrics["ingress.sse.max_connections"] = max_conn
        if sessions is not None:
            metrics["ingress.sse.sessions"] = sessions
        if total is not None and max_conn and max_conn > 0:
            metrics["ingress.sse.saturation_percent"] = (total / max_conn) * 100.0
            metrics["ingress.sse.saturation_ratio"] = total / max_conn

    if isinstance(proc, dict):
        queue_total = _num((proc.get("enhanced_queue_stats") or {}).get("total_queue"))
        queue_util = _num(proc.get("queue_utilization"))
        pressure_ratio = _num((proc.get("capacity_info") or {}).get("pressure_ratio"))
        wait_times = (proc.get("queue_analytics") or {}).get("wait_times") or {}
        queue_wait_p95_ms = _nested_num(proc, "components", "proc", "latency", "queue_wait_ms", "p95")
        exec_p95_ms = _nested_num(proc, "components", "proc", "latency", "exec_ms", "p95")
        queue_depth_1m = _nested_num(proc, "components", "proc", "queue", "windows", "depth", "1m")
        queue_pressure_1m = _nested_num(proc, "components", "proc", "queue", "windows", "pressure_ratio", "1m")

        if queue_total is not None:
            metrics["proc.queue.total"] = queue_total
        if queue_util is not None:
            metrics["proc.queue.utilization_percent"] = queue_util
            metrics["proc.queue.utilization_ratio"] = queue_util / 100.0
        if pressure_ratio is not None:
            metrics["proc.pressure_ratio_percent"] = pressure_ratio * 100.0
            metrics["proc.queue.pressure_ratio"] = pressure_ratio
        if queue_wait_p95_ms is not None:
            metrics["proc.queue.wait.p95_ms"] = queue_wait_p95_ms
        if exec_p95_ms is not None:
            metrics["proc.exec.p95_ms"] = exec_p95_ms
        if queue_depth_1m is not None:
            metrics["proc.queue.depth.1m"] = queue_depth_1m
        if queue_pressure_1m is not None:
            metrics["proc.queue.pressure_ratio.1m"] = queue_pressure_1m

        for role in ("registered", "paid", "privileged", "anonymous"):
            w = _num(wait_times.get(role))
            if w is not None:
                metrics[f"proc.queue.avg_wait_seconds.{role}"] = w

    if isinstance(proc, dict) and "throttling_stats" in proc:
        ts = proc.get("throttling_stats") or {}
    elif isinstance(ingress, dict) and "throttling_stats" in ingress:
        ts = ingress.get("throttling_stats") or {}
    else:
        ts = None

    if ts:
        rate_429 = _num(ts.get("rate_limit_429"))
        back_503 = _num(ts.get("backpressure_503"))
        if rate_429 is not None:
            metrics["throttling.rate_limit_429"] = rate_429
        if back_503 is not None:
            metrics["throttling.backpressure_503"] = back_503

    return metrics

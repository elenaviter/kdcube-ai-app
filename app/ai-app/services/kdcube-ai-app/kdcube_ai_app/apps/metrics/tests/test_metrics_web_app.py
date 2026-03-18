from kdcube_ai_app.apps.metrics.export_metrics import extract_metrics, parse_json_dict


def test_parse_json_dict_accepts_ecs_dimensions_array():
    raw = '[{"Name":"Environment","Value":"staging"},{"Name":"Tenant","Value":"demo"}]'

    assert parse_json_dict(raw) == {
        "Environment": "staging",
        "Tenant": "demo",
    }


def test_extract_metrics_exports_proc_autoscaling_signals():
    system_data = {
        "sse_connections": {
            "global_total_connections": 35,
            "global_max_connections": 100,
            "global_sessions": 12,
        },
        "enhanced_queue_stats": {
            "total_queue": 14,
        },
        "queue_utilization": 70.0,
        "capacity_info": {
            "pressure_ratio": 0.82,
        },
        "queue_analytics": {
            "wait_times": {
                "registered": 1.4,
                "privileged": 0.2,
            },
        },
        "components": {
            "proc": {
                "queue": {
                    "windows": {
                        "depth": {"1m": 14},
                        "pressure_ratio": {"1m": 0.81},
                    }
                },
                "latency": {
                    "queue_wait_ms": {"p95": 5400},
                    "exec_ms": {"p95": 1600},
                },
            }
        },
    }

    metrics = extract_metrics(system_data, system_data)

    assert metrics["ingress.sse.saturation_percent"] == 35.0
    assert metrics["ingress.sse.saturation_ratio"] == 0.35
    assert metrics["proc.queue.total"] == 14.0
    assert metrics["proc.queue.utilization_percent"] == 70.0
    assert metrics["proc.queue.utilization_ratio"] == 0.7
    assert metrics["proc.pressure_ratio_percent"] == 82.0
    assert metrics["proc.queue.pressure_ratio"] == 0.82
    assert metrics["proc.queue.wait.p95_ms"] == 5400.0
    assert metrics["proc.exec.p95_ms"] == 1600.0
    assert metrics["proc.queue.depth.1m"] == 14.0
    assert metrics["proc.queue.pressure_ratio.1m"] == 0.81
    assert metrics["proc.queue.avg_wait_seconds.registered"] == 1.4
    assert metrics["proc.queue.avg_wait_seconds.privileged"] == 0.2

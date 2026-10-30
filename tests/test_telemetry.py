from telemetry import summarize


def test_summary_metrics():
    records = [
        {"event": "object_result", "success": True, "attempts": 1, "predicted_label": "circle", "true_label": "circle"},
        {"event": "object_result", "success": False, "attempts": 3, "failure": "empty_grasp", "predicted_label": "square", "true_label": "triangle"},
    ]
    summary = summarize(records)
    assert summary["success_rate"] == 0.5
    assert summary["classification_accuracy"] == 0.5
    assert summary["mean_attempts"] == 2.0
    assert summary["failure_distribution"] == {"empty_grasp": 1}

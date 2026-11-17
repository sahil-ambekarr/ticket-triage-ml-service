import json


def test_training_smoke(small_model):
    m = small_model["metrics"]
    assert small_model["model_path"].exists()
    assert json.loads(small_model["metrics_path"].read_text())["served_model"] == "logreg"
    assert (small_model["reports_dir"] / "confusion_category.png").exists()
    assert (small_model["reports_dir"] / "confusion_priority.png").exists()
    for target, chance in (("category", 1 / 7), ("priority", 1 / 4)):
        result = m["targets"][target]
        assert set(result["cv"]) == {"logreg", "linear_svc", "complement_nb"}
        # Even on 600 rows the model must beat chance by a wide margin.
        assert result["test"]["logreg"]["accuracy"] > chance * 2
        assert set(result["served"]["per_class"]) == set(result["served"]["labels"])


def test_split_is_stratified_and_seeded(small_model):
    d = small_model["metrics"]["dataset"]
    assert d["rows_used"] == 600 and d["test_rows"] == 120
    assert small_model["metrics"]["split"]["random_state"] == 42


def test_routing_summary_present(small_model):
    routing = small_model["metrics"]["routing_at_default_thresholds"]
    assert routing["category_threshold"] == 0.6 and routing["priority_threshold"] == 0.4
    assert 0 <= routing["flagged_share"] <= 1

"""Input, threshold, artifact and HTTP regression tests with real inference."""

import io
from pathlib import Path

import pytest

from ethereum_fraud.schema import FEATURES, validate_records
from ethereum_fraud.service import PredictionService
from ethereum_fraud.web import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="service", scope="module")
def fixture_service():
    """Load the reviewed artifact once."""
    return PredictionService(ROOT / "models")


@pytest.fixture(name="client")
def fixture_client():
    """Create isolated HTTP test client."""
    return create_app({"TESTING": True, "MODEL_DIR": ROOT / "models"}).test_client()


@pytest.mark.parametrize("records", [[], {}, None, [None], [{}], [{}] * 501])
def test_invalid_structure(records):
    """Require bounded batches with the complete feature contract."""
    with pytest.raises(ValueError):
        validate_records(records)


@pytest.mark.parametrize("value", ["bad", float("nan"), float("inf"), True, 1e30])
def test_invalid_numbers(service, value):
    """Reject nonnumeric, non-finite and out-of-range values."""
    record = dict(service.sample)
    record[next(iter(FEATURES))] = value
    with pytest.raises(ValueError):
        validate_records([record])


@pytest.mark.parametrize(
    "threshold", [True, -1, 0, 1, 2, "bad", float("nan"), float("inf")]
)
def test_invalid_threshold(service, threshold):
    """Thresholds are bounded numbers, not boolean flags."""
    with pytest.raises(ValueError):
        service.predict([service.sample], threshold)


def test_threshold_monotonicity(service):
    """Lower thresholds cannot reduce the number of positive predictions."""
    low = service.predict([service.sample], 0.01)[0]
    high = service.predict([service.sample], 0.99)[0]
    assert int(low["category"]) >= int(high["category"])
    assert low["scores"] == high["scores"]


def test_batch_and_column_order(service):
    """Batch scoring, ignored IDs and reordered inputs agree."""
    record = dict(reversed(list(service.sample.items())))
    record["private_identifier"] = "ignored"
    result = service.predict([service.sample])[0]
    assert service.predict([record] * 3) == [result] * 3
    assert abs(sum(result["scores"].values()) - 1) < 0.00001


def test_missing_values(service):
    """Missing observations use fitted imputation; empty rows fail."""
    record = dict(service.sample)
    record[next(iter(FEATURES))] = None
    assert service.predict([record])
    with pytest.raises(ValueError):
        validate_records([{key: None for key in FEATURES}])


def test_http_and_csv(client, service):
    """HTML, JSON and CSV scoring agree with the shared prediction service."""
    assert client.get("/").status_code == 200
    response = client.post("/api/predict", json={"records": [service.sample]})
    assert response.status_code == 200
    assert response.json["predictions"] == service.predict([service.sample])
    response = client.post(
        "/api/predict", json={"records": [service.sample], "threshold": 0.25}
    )
    assert response.json["predictions"] == service.predict([service.sample], 0.25)
    response = client.post("/predict", data=dict(service.sample, threshold=0.5))
    assert response.status_code == 200 and b"MODEL RESULT" in response.data
    sample = client.get("/sample.csv")
    response = client.post(
        "/predict-csv", data={"file": (io.BytesIO(sample.data), "sample.csv")}
    )
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]


def test_http_errors_and_limits(client):
    """Input failures remain safe client errors with bounded uploads."""
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.post("/predict-csv").status_code == 400
    assert client.get("/missing").status_code == 404
    assert (
        client.post(
            "/api/predict",
            data=b"x" * (1024 * 1024 + 1),
            content_type="application/json",
        ).status_code
        == 413
    )
    sample = client.get("/sample.csv").data.decode().splitlines()
    content = "\n".join([sample[0]] + [sample[1]] * 501).encode()
    assert (
        client.post(
            "/predict-csv", data={"file": (io.BytesIO(content), "large.csv")}
        ).status_code
        == 400
    )


def test_report_and_headers(client):
    """Published metrics match totals and defensive headers are present."""
    report = client.get("/api/metrics").json
    assert sum(map(sum, report["test"]["confusion_matrix"])) == report["test"]["rows"]
    assert 0 <= report["test"]["average_precision"] <= 1
    assert report["classes"] == ["0", "1"]
    ready = client.get("/health/ready")
    assert ready.json["status"] == "ready"
    assert ready.headers["X-Content-Type-Options"] == "nosniff"


def test_model_integrity(tmp_path):
    """Verify hashes before loading serialized models."""
    (tmp_path / "metrics.json").write_bytes((ROOT / "models/metrics.json").read_bytes())
    (tmp_path / "pipeline.joblib").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        PredictionService(tmp_path)

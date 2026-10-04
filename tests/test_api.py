import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fastapi.testclient import TestClient

from medintel.api import create_app


VALID_INPUT = {
    "age_years": 58,
    "sex": "female",
    "race_ethnicity": "non_hispanic_asian",
    "income_poverty_ratio": 2.5,
    "systolic_bp": 132,
    "diastolic_bp": 82,
    "bmi": 29.4,
    "waist_cm": 96,
    "total_cholesterol_mg_dl": 205,
    "hdl_cholesterol_mg_dl": 48,
    "hba1c_percent": 5.9,
    "ever_smoked": False,
}


class FixedModel:
    def predict_proba(self, frame):
        assert list(frame.columns)[0] == "RIDAGEYR"
        assert frame.iloc[0]["RIAGENDR"] == "2"
        return np.array([[0.76, 0.24]])


class ApiContractTests(unittest.TestCase):
    def test_health_metadata_and_prediction_contract(self):
        client = TestClient(create_app(model=FixedModel()))
        self.assertEqual(client.get("/healthz").json()["status"], "ok")
        metadata = client.get("/v1/model").json()
        self.assertEqual(metadata["intended_use"], "research and education")
        response = client.post("/v1/predictions", json=VALID_INPUT)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertAlmostEqual(body["probability"], 0.24)
        self.assertTrue(body["above_exploratory_threshold"])
        self.assertIn("not future cardiovascular risk", body["interpretation"])

    def test_invalid_and_extra_fields_are_rejected(self):
        client = TestClient(create_app(model=FixedModel()))
        invalid = {**VALID_INPUT, "age_years": 12, "diagnosis": "yes"}
        response = client.post("/v1/predictions", json=invalid)
        self.assertEqual(response.status_code, 422)

    def test_inconsistent_clinical_measurements_are_rejected(self):
        client = TestClient(create_app(model=FixedModel()))
        invalid = {**VALID_INPUT, "systolic_bp": 70, "diastolic_bp": 90}
        self.assertEqual(client.post("/v1/predictions", json=invalid).status_code, 422)
        invalid = {
            **VALID_INPUT,
            "total_cholesterol_mg_dl": 100,
            "hdl_cholesterol_mg_dl": 120,
        }
        self.assertEqual(client.post("/v1/predictions", json=invalid).status_code, 422)

    def test_missing_model_returns_service_unavailable(self):
        with TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.joblib"
            client = TestClient(create_app(model_path=missing))
            self.assertEqual(client.get("/healthz").json()["status"], "degraded")
            response = client.post("/v1/predictions", json=VALID_INPUT)
            self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()

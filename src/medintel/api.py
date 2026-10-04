"""Versioned research API for the locally built MedIntel model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from medintel import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_logistic.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "reports" / "metrics" / "v0.5-interpretation.json"
RACE_CODES = {
    "mexican_american": "1",
    "other_hispanic": "2",
    "non_hispanic_white": "3",
    "non_hispanic_black": "4",
    "non_hispanic_asian": "6",
    "other_or_multiracial": "7",
}


class ClinicalFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_years: int = Field(ge=20, le=80)
    sex: Literal["male", "female"]
    race_ethnicity: Literal[
        "mexican_american", "other_hispanic", "non_hispanic_white",
        "non_hispanic_black", "non_hispanic_asian", "other_or_multiracial",
    ]
    income_poverty_ratio: float | None = Field(default=None, ge=0, le=5)
    systolic_bp: float | None = Field(default=None, ge=70, le=250)
    diastolic_bp: float | None = Field(default=None, ge=30, le=150)
    bmi: float | None = Field(default=None, ge=10, le=80)
    waist_cm: float | None = Field(default=None, ge=40, le=200)
    total_cholesterol_mg_dl: float | None = Field(default=None, ge=50, le=500)
    hdl_cholesterol_mg_dl: float | None = Field(default=None, ge=5, le=200)
    hba1c_percent: float | None = Field(default=None, ge=2, le=20)
    ever_smoked: bool | None = None

    @model_validator(mode="after")
    def validate_clinical_relationships(self):
        if (
            self.systolic_bp is not None
            and self.diastolic_bp is not None
            and self.diastolic_bp >= self.systolic_bp
        ):
            raise ValueError("diastolic_bp must be lower than systolic_bp")
        if (
            self.total_cholesterol_mg_dl is not None
            and self.hdl_cholesterol_mg_dl is not None
            and self.hdl_cholesterol_mg_dl > self.total_cholesterol_mg_dl
        ):
            raise ValueError("HDL cholesterol cannot exceed total cholesterol")
        return self

    def as_model_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "RIDAGEYR": self.age_years,
            "INDFMPIR": self.income_poverty_ratio,
            "mean_systolic_bp": self.systolic_bp,
            "mean_diastolic_bp": self.diastolic_bp,
            "BMXBMI": self.bmi,
            "BMXWAIST": self.waist_cm,
            "LBXTC": self.total_cholesterol_mg_dl,
            "LBDHDD": self.hdl_cholesterol_mg_dl,
            "LBXGH": self.hba1c_percent,
            "RIAGENDR": "1" if self.sex == "male" else "2",
            "RIDRETH3": RACE_CODES[self.race_ethnicity],
            "ever_smoked": None if self.ever_smoked is None else ("yes" if self.ever_smoked else "no"),
        }]).replace({None: np.nan})


class PredictionResponse(BaseModel):
    model_version: str
    probability: float
    exploratory_threshold: float
    above_exploratory_threshold: bool
    interpretation: str
    warning: str


class ModelRuntime:
    def __init__(self, model=None, model_path: Path | None = None, metadata_path: Path | None = None):
        self.model_path = model_path or Path(
            os.environ.get("MEDINTEL_MODEL_PATH", DEFAULT_MODEL_PATH)
        )
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.threshold = float(self.metadata["threshold_selection"]["threshold"])
        self.model = model
        self.load_error: str | None = None
        if self.model is None:
            if self.model_path.exists():
                try:
                    self.model = joblib.load(self.model_path)
                except Exception as error:  # pragma: no cover - defensive startup path
                    self.load_error = type(error).__name__
            else:
                self.load_error = "model_artifact_missing"

    @property
    def available(self) -> bool:
        return self.model is not None


def create_app(*, model=None, model_path: Path | None = None) -> FastAPI:
    runtime = ModelRuntime(model=model, model_path=model_path)
    application = FastAPI(
        title="MedIntel AI Research API",
        version=__version__,
        description=(
            "Educational API for a cross-sectional model of self-reported cardiovascular "
            "disease history. It is not a diagnostic or clinical decision tool."
        ),
    )

    @application.get("/healthz")
    def health() -> dict[str, object]:
        return {
            "status": "ok" if runtime.available else "degraded",
            "model_loaded": runtime.available,
            "version": __version__,
        }

    @application.get("/v1/model")
    def model_metadata() -> dict[str, object]:
        metrics = runtime.metadata["frozen_test"]
        return {
            "version": __version__,
            "target": "prevalent self-reported cardiovascular disease history",
            "calibration": runtime.metadata["calibration_selection"]["selected"],
            "exploratory_threshold": runtime.threshold,
            "test_roc_auc": metrics["roc_auc"],
            "test_average_precision": metrics["average_precision"],
            "intended_use": "research and education",
            "prohibited_use": "diagnosis, treatment, screening, or individual medical advice",
        }

    @application.post("/v1/predictions", response_model=PredictionResponse)
    def predict(features: ClinicalFeatures) -> PredictionResponse:
        if not runtime.available:
            raise HTTPException(
                status_code=503,
                detail="Model artifact unavailable. Rebuild the local V0.5 model before serving predictions.",
            )
        probability = float(runtime.model.predict_proba(features.as_model_frame())[0, 1])
        return PredictionResponse(
            model_version=__version__,
            probability=probability,
            exploratory_threshold=runtime.threshold,
            above_exploratory_threshold=probability >= runtime.threshold,
            interpretation=(
                "Estimated association with prevalent self-reported history in the NHANES "
                "research cohort; this is not future cardiovascular risk."
            ),
            warning="For research and education only. Do not use for clinical decisions.",
        )

    return application


app = create_app()


def run() -> None:
    uvicorn.run("medintel.api:app", host="127.0.0.1", port=8000, reload=False)

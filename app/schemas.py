"""Request/response schemas for the prediction API."""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, le=20, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, le=20, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, le=20, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, le=20, description="Petal width in cm")

    def as_feature_vector(self) -> list[float]:
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class PredictResponse(BaseModel):
    prediction: str
    class_id: int
    probabilities: dict[str, float]
    model_name: str
    model_version: str
    trace_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


class ModelInfo(BaseModel):
    name: str
    version: str
    algorithm: str
    trained_at: str | None
    training_accuracy: float | None
    test_accuracy: float | None
    features: list[str]
    classes: list[str]

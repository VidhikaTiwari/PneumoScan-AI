from pydantic import BaseModel
from typing import Dict, Optional

class PredictionResponse(BaseModel):
    #building template for answer
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    warning: Optional[str] = None
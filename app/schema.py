from pydantic import BaseModel, Field
from typing import Optional, List

class HouseFeatures(BaseModel):
    OverallQual: int = Field(...,ge=1,le=10,description="Overall material and finish quality (1-10)")
    GrLivArea: float = Field(...,ge=0,description="Above grade (ground) living area square feet")
    GarageCars: int = Field(...,ge=0,le=5,description="Size of garage in car capacity")
    TotalBsmtSF: float = Field(...,ge=0,description="Total square feet of basement area")
    FullBath: int = Field(...,ge=0,description="Full bathrooms above grade")
    YearBuilt: int   = Field(..., ge=1800, le=2025, description="Year house was built")
    YearRemodAdd: int   = Field(..., ge=1800, le=2025, description="Year of remodel")
    LotArea: float = Field(..., gt=0, description="Lot size in sqft")
    Neighborhood: str   = Field(..., description="Neighborhood name")
    ExterQual: int   = Field(..., ge=0, le=5, description="Exterior quality 0-5")

class PredictionResponse(BaseModel):
    predicted_price: float 
    predicted_price_low: float
    predicted_price_high: float
    model_rmse: float
    

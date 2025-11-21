"""
Database Schemas for the Stock AI app

Each Pydantic model corresponds to a MongoDB collection (lowercased class name).
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class Stock(BaseModel):
    """
    Stocks collection schema
    Collection: "stock"
    """
    symbol: str = Field(..., description="Ticker symbol (e.g., RELIANCE.NS)")
    name: Optional[str] = Field(None, description="Company name")
    sector: Optional[str] = Field(None, description="Sector/industry")
    exchange: str = Field("NSE", description="Exchange name")

class Prediction(BaseModel):
    """
    Predictions collection schema
    Collection: "prediction"
    """
    symbol: str = Field(..., description="Ticker symbol")
    score: float = Field(..., ge=-100, le=100, description="Composite score (-100 to 100)")
    recommendation: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    reasons: List[str] = Field(default_factory=list, description="Key drivers for the score")
    run_id: Optional[str] = Field(None, description="Identifier for a prediction run")
    created_at: Optional[datetime] = None

class Feedback(BaseModel):
    """
    User feedback on predictions
    Collection: "feedback"
    """
    symbol: str
    run_id: str
    helpful: bool
    comment: Optional[str] = None

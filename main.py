import os
from typing import List, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from datetime import datetime, timezone

from database import db, create_document, get_documents
from schemas import Prediction as PredictionSchema, Stock as StockSchema, Feedback as FeedbackSchema

app = FastAPI(title="Stock AI India", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"


class PredictionOut(BaseModel):
    symbol: str
    name: Optional[str] = None
    exchange: str = "NSE"
    score: float
    recommendation: str
    reasons: List[str]
    run_id: str
    created_at: datetime


@app.get("/")
def read_root():
    return {"message": "Stock AI (India) backend ready"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            collections = db.list_collection_names()
            response["collections"] = collections
            response["connection_status"] = "Connected"
        else:
            response["database"] = "❌ Not Available"
    except Exception as e:
        response["database"] = f"⚠️ Error: {str(e)[:80]}"
    return response


@app.get("/api/stocks/search")
def search_stocks(q: str = Query(..., min_length=1)):
    params = {"q": q, "quotesCount": 10, "newsCount": 0, "region": "IN", "lang": "en-IN"}
    r = requests.get(YAHOO_SEARCH_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    results = []
    for itm in data.get("quotes", [])[:10]:
        # Prefer NSE listings
        exch = itm.get("exchange") or itm.get("exchDisp") or ""
        symbol = itm.get("symbol") or ""
        shortname = itm.get("shortname") or itm.get("longname")
        if not symbol:
            continue
        results.append({
            "symbol": symbol,
            "name": shortname,
            "exchange": exch
        })
    return {"results": results}


def _yahoo_quote(symbols: List[str]):
    params = {"symbols": ",".join(symbols)}
    r = requests.get(YAHOO_QUOTE_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("quoteResponse", {}).get("result", [])


def _score_stock(q: dict) -> (float, List[str], str):
    reasons = []
    # Factors
    chg_pct = q.get("regularMarketChangePercent") or 0.0
    pe = q.get("trailingPE") or None
    mcap = q.get("marketCap") or None
    fifty_two_wk_change = q.get("fiftyTwoWeekChange") or 0.0

    score = 0.0

    # Momentum factor
    score += max(min(chg_pct, 5), -5) * 1.5  # cap intraday effect
    if chg_pct is not None:
        reasons.append(f"Intraday momentum {chg_pct:.2f}%")

    # 52-week trend
    if fifty_two_wk_change is not None:
        score += (fifty_two_wk_change * 100) * 0.3
        reasons.append(f"52W performance {fifty_two_wk_change*100:.1f}%")

    # Valuation factor (lower PE better up to a threshold)
    if pe:
        if pe < 12:
            score += 10
            reasons.append("Attractive valuation (PE < 12)")
        elif pe < 20:
            score += 4
            reasons.append("Reasonable valuation (PE < 20)")
        elif pe > 40:
            score -= 6
            reasons.append("Rich valuation (PE > 40)")

    # Size/quality proxy
    if mcap:
        # Reward mid-large caps modestly; penalize extremes less
        if mcap >= 1_000_000_000_000:  # >= 1T INR approx when converted; Yahoo is in local
            score += 4
            reasons.append("Large, liquid company")
        elif mcap >= 100_000_000_000:
            score += 2
            reasons.append("Mid/Large cap stability")

    # Normalize bounds
    score = max(min(score, 100.0), -100.0)

    # Map to recommendation
    if score >= 20:
        rec = "Strong Buy" if score >= 30 else "Buy"
    elif score <= -10:
        rec = "Strong Sell" if score <= -20 else "Sell"
    else:
        rec = "Hold"

    return score, reasons, rec


@app.get("/api/predict", response_model=List[PredictionOut])
def predict(symbols: str = Query(..., description="Comma separated symbols, use .NS for NSE (e.g., RELIANCE.NS,TCS.NS)")):
    raw_symbols = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    if not raw_symbols:
        return []

    quotes = _yahoo_quote(raw_symbols)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d%H%M%S")

    outputs: List[PredictionOut] = []
    for q in quotes:
        score, reasons, rec = _score_stock(q)
        out = PredictionOut(
            symbol=q.get("symbol"),
            name=q.get("longName") or q.get("shortName"),
            exchange=q.get("fullExchangeName") or q.get("exchange") or "NSE",
            score=round(score, 2),
            recommendation=rec,
            reasons=reasons,
            run_id=run_id,
            created_at=now,
        )
        # Persist
        try:
            create_document("prediction", {
                "symbol": out.symbol,
                "score": out.score,
                "recommendation": out.recommendation,
                "reasons": out.reasons,
                "run_id": out.run_id,
                "created_at": out.created_at,
                "name": out.name,
                "exchange": out.exchange,
            })
        except Exception:
            # Database not mandatory for functionality
            pass
        outputs.append(out)

    return outputs


@app.get("/api/predictions/latest")
def latest_predictions(limit: int = 20):
    try:
        docs = get_documents("prediction", {}, limit=limit)
        # Convert ObjectId and datetime to serializable
        def _s(d):
            d = {**d}
            if "_id" in d:
                d["id"] = str(d.pop("_id"))
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = d["created_at"].isoformat()
            return d
        docs = [_s(d) for d in sorted(docs, key=lambda x: x.get("created_at"), reverse=True)[:limit]]
        return {"items": docs}
    except Exception:
        return {"items": []}


class FeedbackIn(BaseModel):
    symbol: str
    run_id: str
    helpful: bool
    comment: Optional[str] = None


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackIn):
    try:
        create_document("feedback", payload.model_dump())
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

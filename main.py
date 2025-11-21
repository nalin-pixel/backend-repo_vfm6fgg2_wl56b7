import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from datetime import datetime, timezone

from database import db, create_document, get_documents
from schemas import Prediction as PredictionSchema, Stock as StockSchema, Feedback as FeedbackSchema

app = FastAPI(title="Stock AI India", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}"


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
    reasons: List[str] = []
    # Base fields
    chg_pct = q.get("regularMarketChangePercent") or 0.0
    pe = q.get("trailingPE") or None
    mcap = q.get("marketCap") or None
    fifty_two_wk_change = q.get("fiftyTwoWeekChange") or 0.0

    # Enrichment fields
    price = q.get("regularMarketPrice") or None
    sma50 = q.get("fiftyDayAverage") or None
    sma200 = q.get("twoHundredDayAverage") or None
    volume = q.get("regularMarketVolume") or None
    avg_vol = q.get("averageDailyVolume3Month") or None
    beta = q.get("beta") or None

    score = 0.0

    # Momentum factor (intraday)
    score += max(min(chg_pct, 5), -5) * 1.5
    reasons.append(f"Intraday momentum {chg_pct:.2f}%")

    # 52-week trend
    score += (fifty_two_wk_change * 100) * 0.3
    reasons.append(f"52W performance {fifty_two_wk_change*100:.1f}%")

    # Valuation factor
    if pe is not None:
        if pe < 12:
            score += 10
            reasons.append("Attractive valuation (PE < 12)")
        elif pe < 20:
            score += 4
            reasons.append("Reasonable valuation (PE < 20)")
        elif pe > 60:
            score -= 10
            reasons.append("Expensive valuation (PE > 60)")
        elif pe > 40:
            score -= 6
            reasons.append("Rich valuation (PE > 40)")

    # Size/liquidity proxy
    if mcap:
        if mcap >= 1_000_000_000_000:
            score += 4
            reasons.append("Large, liquid company")
        elif mcap >= 100_000_000_000:
            score += 2
            reasons.append("Mid/Large cap stability")
        elif mcap < 10_000_000_000:
            score -= 3
            reasons.append("Small cap risk")

    # Trend strength via SMA relationships
    if price and sma50:
        pct = (price - sma50) / sma50 * 100
        score += max(min(pct, 20), -20) * 0.15
        reasons.append(f"Price vs 50D SMA: {pct:.1f}%")
    if price and sma200:
        pct = (price - sma200) / sma200 * 100
        score += max(min(pct, 40), -40) * 0.08
        reasons.append(f"Price vs 200D SMA: {pct:.1f}%")

    # Volume surge
    if volume and avg_vol and avg_vol > 0:
        vol_ratio = volume / avg_vol
        if vol_ratio > 1.5:
            boost = min((vol_ratio - 1) * 4, 8)
            score += boost
            reasons.append("Volume surge")
        elif vol_ratio < 0.5:
            score -= 2
            reasons.append("Low volume")

    # Volatility filter using beta
    if beta is not None:
        if beta > 2.0:
            score -= 3
            reasons.append("High volatility (beta > 2)")
        elif beta < 0.6:
            score += 2
            reasons.append("Defensive profile (beta < 0.6)")

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
            pass
        outputs.append(out)

    return outputs


@app.get("/api/predictions/latest")
def latest_predictions(limit: int = 20):
    try:
        docs = get_documents("prediction", {}, limit=limit)
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


@app.get("/api/leaderboard")
def leaderboard(limit: int = 20):
    """Return top symbols from the most recent prediction run_id."""
    try:
        # Fetch recent predictions and group by run_id
        docs = get_documents("prediction", {}, limit=1000)
        if not docs:
            return {"items": []}
        # Find latest run_id by created_at
        latest_run = max(docs, key=lambda d: d.get("created_at") or datetime.min).get("run_id")
        latest = [d for d in docs if d.get("run_id") == latest_run]
        # Deduplicate per symbol (keep the highest score if multiple)
        by_symbol: Dict[str, Dict[str, Any]] = {}
        for d in latest:
            sym = d.get("symbol")
            if sym not in by_symbol or (d.get("score", -1e9) > by_symbol[sym].get("score", -1e9)):
                by_symbol[sym] = d
        top = sorted(by_symbol.values(), key=lambda x: x.get("score", 0), reverse=True)[:limit]
        # Serialize
        items = []
        for d in top:
            items.append({
                "symbol": d.get("symbol"),
                "name": d.get("name"),
                "score": d.get("score"),
                "recommendation": d.get("recommendation"),
                "created_at": (d.get("created_at").isoformat() if isinstance(d.get("created_at"), datetime) else d.get("created_at")),
            })
        return {"items": items, "run_id": latest_run}
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


# ---------- Backtesting / Portfolio Simulation ----------

def _yahoo_chart_range(symbol: str, days: int):
    # Map days to Yahoo range strings
    if days <= 30:
        range_str = "1mo"
        interval = "1d"
    elif days <= 90:
        range_str = "3mo"
        interval = "1d"
    elif days <= 180:
        range_str = "6mo"
        interval = "1d"
    elif days <= 365:
        range_str = "1y"
        interval = "1d"
    else:
        range_str = "2y"
        interval = "1d"
    url = YAHOO_CHART_URL.format(symbol)
    params = {"range": range_str, "interval": interval}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        return []
    ind = result[0]
    ts = ind.get("timestamp", [])
    closes = ind.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    prices = []
    for t, c in zip(ts, closes):
        if c is None:
            continue
        prices.append({"ts": datetime.fromtimestamp(t, tz=timezone.utc), "close": c})
    return prices


def _series_return(prices: List[Dict[str, Any]]):
    if not prices:
        return 0.0
    start = prices[0]["close"]
    end = prices[-1]["close"]
    if not start:
        return 0.0
    return (end / start - 1) * 100.0


@app.get("/api/backtest")
def backtest(symbols: str = Query(...), days: int = Query(90, ge=5, le=730), benchmark: str = Query("^NSEI")):
    """
    Simple equal-weight portfolio backtest over the last N days compared to benchmark (default NIFTY 50 index).
    Returns cumulative % returns for portfolio and benchmark.
    """
    syms = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    if not syms:
        return {"portfolio_return": 0.0, "benchmark_return": 0.0, "symbols": []}

    portfolio_returns = []
    for s in syms:
        try:
            series = _yahoo_chart_range(s, days)
            r = _series_return(series)
            portfolio_returns.append(r)
        except Exception:
            continue

    bench_r = 0.0
    try:
        bench_series = _yahoo_chart_range(benchmark, days)
        bench_r = _series_return(bench_series)
    except Exception:
        pass

    port_r = sum(portfolio_returns) / len(portfolio_returns) if portfolio_returns else 0.0
    return {
        "symbols": syms,
        "days": days,
        "portfolio_return": round(port_r, 2),
        "benchmark": benchmark,
        "benchmark_return": round(bench_r, 2),
        "count": len(portfolio_returns)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

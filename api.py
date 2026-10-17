"""FastAPI service. Run: uvicorn api:app --port 8000   (docs at /docs)"""
from typing import Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import AlertService, INPUTS

app = FastAPI(title="Air-quality anomaly alert service", version="1.0")
svc = AlertService()

class Record(BaseModel):
    timestamp: str                                   # e.g. "2005-03-01 18:00:00"
    values: Dict[str, Optional[float]]               # sensor name -> reading; null or -200 = missing
    persist: bool = False                            # True: keep record in the history buffer

@app.get("/health")
def health(): return {"status": "ok", "inputs": INPUTS, "history_end": str(svc.history.index.max())}

@app.post("/score")
def score(r: Record):
    return svc.score_record(r.timestamp, {k: (None if v is None else v) for k, v in r.values.items()}, persist=r.persist)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional
from services.ml_service import process_vitals

from websocket.manager import manager

from database.queries import (
    insert_sensor_data,
    insert_prediction
)

from services.data_service import (
    fetch_latest_vitals,
    fetch_day_timeline,
    fetch_week_timeline,
    fetch_month_timeline
)

app = FastAPI()


# -----------------------------
# CORS (allow mobile app access)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Pydantic model for validation
# -----------------------------
class SensorData(BaseModel):
    patient_id: int
    hr: float
    temp: float
    rr: float
    spo2: float
    hrv: Optional[float] = None
    rrv: Optional[float] = None
    movement: float


# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/")
def root():
    return {"status": "SepsisGuard backend running"}


# -----------------------------
# Sensor data ingestion
# -----------------------------
@app.post("/sensor-data")
async def receive_sensor_data(data: SensorData):

    timestamp = datetime.now(timezone.utc)

    insert_sensor_data(
        data.patient_id,
        data.hr,
        data.temp,
        data.rr,
        data.spo2,
        data.hrv,
        data.rrv,
        data.movement,
        timestamp
    )

    # -----------------------------
    # Run ML model
    # -----------------------------
    ml_result = process_vitals(data)

    # -----------------------------
    # Store prediction ONLY if monitoring phase
    # -----------------------------
    if ml_result["phase"] == "MONITORING":
        insert_prediction(
            data.patient_id,
            ml_result["status"] in ["HIGH_RISK", "CRITICAL"],
            ml_result["score"],
            [ml_result["score"]],
            [timestamp]
        )
    

    # Broadcast to dashboard clients
    await manager.broadcast({
        "patient_id": data.patient_id,
        "hr": data.hr,
        "rr": data.rr,
        "spo2": data.spo2,
        "temp": data.temp,
        "hrv": data.hrv or 0.0,
        "rrv": data.rrv or 0.0,
        "movement": data.movement,
        "timestamp": str(timestamp),
        "ml": ml_result
    })

    return {
        "message": "data processed",
        "ml": ml_result
    }


# -----------------------------
# WebSocket endpoint
# -----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# -----------------------------
# Latest vitals endpoint
# -----------------------------
@app.get("/latest-vitals/{patient_id}")
def latest_vitals(patient_id: int):

    data = fetch_latest_vitals(patient_id)

    if not data:
        return {"error": "No data found"}

    return {
        "hr": data[0],
        "rr": data[1],
        "spo2": data[2],
        "temp": data[3],
        "hrv": data[4],
        "rrv": data[5],
        "timestamp": data[6]
    }


# -----------------------------
# Timeline analytics endpoints
# -----------------------------
@app.get("/timeline/day/{patient_id}")
def day_timeline(patient_id: int):

    data = fetch_day_timeline(patient_id)
    return {"data": data}


@app.get("/timeline/week/{patient_id}")
def week_timeline(patient_id: int):

    data = fetch_week_timeline(patient_id)
    return {"data": data}


@app.get("/timeline/month/{patient_id}")
def month_timeline(patient_id: int):

    data = fetch_month_timeline(patient_id)
    return {"data": data}
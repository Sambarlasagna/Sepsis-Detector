from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import uvicorn
from db import engine  # Your NeonDB connection setup
from models import AlertMessage  # Pydantic model for alerts
# --------------------------
# Lifespan for Startup/Shutdown
# --------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        print("Connected to NeonDB")
    except Exception as e:
        print("Failed to connect to NeonDB:", e)

    yield  # app runs here

    # Shutdown logic
    await engine.dispose()
    print("Database connection closed")


# --------------------------
# Initialize App
# --------------------------
app = FastAPI(
    title="Sepsis Detector API",
    description="Backend API for Sepsis Detection using AI/ML with Real-Time Alerts",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# Root & Prediction Endpoints
# --------------------------
@app.get("/")
def read_root():
    return {"message": "Sepsis Detector API is running "}

@app.post("/predict")
def predict(data: dict):
    return {"input": data, "prediction": "sepsis_risk_placeholder"}

# --------------------------
# WebSocket Connection Manager
# --------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, patient_id: str, websocket: WebSocket):
        await websocket.accept()
        if patient_id not in self.active_connections:
            self.active_connections[patient_id] = []
        self.active_connections[patient_id].append(websocket)

    def disconnect(self, patient_id: str, websocket: WebSocket):
        if patient_id in self.active_connections:
            self.active_connections[patient_id].remove(websocket)
            if not self.active_connections[patient_id]:
                del self.active_connections[patient_id]

    async def broadcast(self, patient_id: str, message: List[int]):
        if patient_id in self.active_connections:
            alert_payload = AlertMessage(alerts=message)  # Pydantic model
            for connection in self.active_connections[patient_id]:
                await connection.send_json(alert_payload.model_dump())


# Global manager & storage
manager = ConnectionManager()
alerts_storage: Dict[str, List[int]] = {}


# --------------------------
# WebSocket Endpoint for Alerts
# --------------------------
@app.websocket("/ws/alerts/{patient_id}")
async def websocket_alerts(websocket: WebSocket, patient_id: str):
    await manager.connect(patient_id, websocket)

    # Send existing alerts immediately
    if patient_id in alerts_storage and alerts_storage[patient_id]:
        await websocket.send_json({"alerts": alerts_storage[patient_id]})

    try:
        while True:
            await asyncio.sleep(1)  #keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(patient_id, websocket)

# --------------------------
# POST Endpoint to Trigger Alerts
# --------------------------
@app.post("/test-alerts/{patient_id}")
async def add_test_alerts(patient_id: str, request: AlertMessage):
    if patient_id not in alerts_storage:
        alerts_storage[patient_id] = []

    alerts_storage[patient_id].extend(request.hours_until_sepsis)
    await manager.broadcast(patient_id, alerts_storage[patient_id])
    return {"patient_id": patient_id, "alerts": alerts_storage[patient_id]}


# --------------------------
# Endpoint to Clear Alerts
# --------------------------
@app.delete("/clear-alerts/{patient_id}")
async def clear_alerts(patient_id: str):
    alerts_storage[patient_id] = []
    await manager.broadcast(patient_id, [])
    return {"patient_id": patient_id, "alerts": []}


# --------------------------
# Run Server
# --------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

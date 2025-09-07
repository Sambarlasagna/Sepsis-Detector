# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, List
import asyncio
import uvicorn
import time

import torch
import torch.nn as nn
import numpy as np

from db import engine  # NeonDB connection
from models import AlertMessage, PatientSequence  # shared Pydantic models

# ============================================================
# 🔹 Rich Logging Setup
# ============================================================
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("sepsis-detector")


# ============================================================
# 🔹 Lifespan for DB Startup/Shutdown
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        logger.info("✅ Connected to NeonDB")
    except Exception as e:
        logger.error(f"❌ Failed to connect to NeonDB: {e}")

    yield
    await engine.dispose()
    logger.info("Database connection closed")


# ============================================================
# 🔹 Initialize App
# ============================================================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://neondb_owner:npg_Q1RNakf5ZUpw@ep-weathered-field-a1da99tk-pooler.ap-southeast-1.aws.neon.tech/neondb"
)
engine: AsyncEngine = create_async_engine(DATABASE_URL, echo=False, future=True)

# ============================================================
# 🔹 GRU Model for Time-to-Sepsis Prediction
# ============================================================

FEATURES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3',
    'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
    'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]

SEQ_LENGTH = 6
INPUT_DIM = len(FEATURES)
MODEL_PATH = "../model/model_1_gru_time_to_sepsis.pth"

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out).squeeze(1)


# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

logger.info(f"Model loaded from {MODEL_PATH} on {device}")


@app.post("/predict-ml/")
async def predict_time_to_sepsis(input_data: PatientSequence):
    input_seq = input_data.features
    logger.info(f"Received ML prediction request with {len(input_seq)} timesteps")

    if len(input_seq) != SEQ_LENGTH:
        logger.warning(f"Invalid sequence length: expected {SEQ_LENGTH}, got {len(input_seq)}")
        return {"error": f"Expected {SEQ_LENGTH} timesteps, got {len(input_seq)}"}

    try:
        input_array = np.array([[step[feat] for feat in FEATURES] for step in input_seq], dtype=np.float32)
    except KeyError as e:
        logger.error(f"Missing feature in input: {e.args[0]}")
        return {"error": f"Missing feature in input: {e.args[0]}"}

    input_tensor = torch.tensor(input_array).unsqueeze(0).to(device)  # (1, 6, INPUT_DIM)
    with torch.no_grad():
        pred = model(input_tensor).item()

    logger.info(f"Prediction result: {pred}")
    return {"predicted_time_to_sepsis": pred}


# ============================================================
# 🔹 WebSocket Connection Manager for Real-Time Alerts
# ============================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, patient_id: str, websocket: WebSocket):
        await websocket.accept()
        if patient_id not in self.active_connections:
            self.active_connections[patient_id] = []
        self.active_connections[patient_id].append(websocket)
        logger.info(f"WebSocket connected for patient_id={patient_id}")

    def disconnect(self, patient_id: str, websocket: WebSocket):
        if patient_id in self.active_connections:
            self.active_connections[patient_id].remove(websocket)
            if not self.active_connections[patient_id]:
                del self.active_connections[patient_id]
        logger.info(f"WebSocket disconnected for patient_id={patient_id}")

    async def broadcast(self, patient_id: str, message: List[int]):
        if patient_id in self.active_connections:
            alert_payload = AlertMessage(hours_until_sepsis=message)
            for connection in self.active_connections[patient_id]:
                await connection.send_json(alert_payload.model_dump())
            logger.info(f"Broadcasted alerts to {patient_id}: {message}")


manager = ConnectionManager()
alerts_storage: Dict[str, List[int]] = {}


@app.websocket("/ws/alerts/{patient_id}")
async def websocket_alerts(websocket: WebSocket, patient_id: str):
    await manager.connect(patient_id, websocket)

    if patient_id in alerts_storage and alerts_storage[patient_id]:
        await websocket.send_json({"alerts": alerts_storage[patient_id]})
        logger.info(f"Sent stored alerts for {patient_id}")

    try:
        while True:
            await asyncio.sleep(1)  # keep alive
    except WebSocketDisconnect:
        manager.disconnect(patient_id, websocket)


@app.post("/test-alerts/{patient_id}")
async def add_test_alerts(patient_id: str, request: AlertMessage):
    if patient_id not in alerts_storage:
        alerts_storage[patient_id] = []

    alerts_storage[patient_id].extend(request.hours_until_sepsis)
    await manager.broadcast(patient_id, alerts_storage[patient_id])
    logger.info(f"Added test alerts for {patient_id}: {request.hours_until_sepsis}")
    return {"patient_id": patient_id, "alerts": alerts_storage[patient_id]}


@app.delete("/clear-alerts/{patient_id}")
async def clear_alerts(patient_id: str):
    alerts_storage[patient_id] = []
    await manager.broadcast(patient_id, [])
    logger.info(f"Cleared alerts for {patient_id}")
    return {"patient_id": patient_id, "alerts": []}


# ============================================================
# 🔹 Run Server
# ============================================================
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_config=None)

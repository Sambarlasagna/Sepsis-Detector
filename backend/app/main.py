# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, List
import asyncio
import uvicorn
import time
import pandas as pd
import os
import torch
import torch.nn as nn
import numpy as np
import logging
from rich.logging import RichHandler
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from models import AlertMessage, PatientSequence  # shared Pydantic models

# ============================================================
# 🔹 Logging
# ============================================================
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("sepsis-detector")

# ============================================================
# 🔹 Database (asyncpg for NeonDB)
# ============================================================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://neondb_owner:npg_Q1RNakf5ZUpw@ep-weathered-field-a1da99tk-pooler.ap-southeast-1.aws.neon.tech/neondb"
)
engine: AsyncEngine = create_async_engine(DATABASE_URL, echo=False, future=True)

# ============================================================
# 🔹 GRU Model Setup
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
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
logger.info(f"Model loaded from {MODEL_PATH} on {device}")

# ============================================================
# 🔹 WebSocket Manager
# ============================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, patient_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(patient_id, []).append(websocket)
        logger.info(f"WebSocket connected for {patient_id}")

    def disconnect(self, patient_id: str, websocket: WebSocket):
        if patient_id in self.active_connections:
            self.active_connections[patient_id].remove(websocket)
            if not self.active_connections[patient_id]:
                del self.active_connections[patient_id]
        logger.info(f"WebSocket disconnected for {patient_id}")

    async def broadcast(self, patient_id: str, message: List[int]):
        if patient_id in self.active_connections:
            payload = AlertMessage(hours_until_sepsis=message)
            for ws in list(self.active_connections[patient_id]):
                try:
                    await ws.send_json(payload.model_dump())
                except WebSocketDisconnect:
                    self.disconnect(patient_id, ws)
            logger.info(f"Broadcasted alerts to {patient_id}: {message}")

manager = ConnectionManager()
alerts_storage: Dict[str, List[int]] = {}

# ============================================================
# 🔹 CSV Simulation
# ============================================================
async def simulate_patient_alerts(patient_id: str, csv_path: str):
    try:
        reader = list(pd.read_csv(csv_path).to_dict(orient="records"))
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        return

    for start in range(len(reader) - SEQ_LENGTH + 1):
        window = reader[start:start + SEQ_LENGTH]
        try:
            arr = np.array([[float(step[feat]) for feat in FEATURES] for step in window], dtype=np.float32)
            tensor = torch.tensor(arr).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tensor).item()
                pred_int = int(pred)
        except Exception as e:
            logger.error(f"Prediction failed at rows {start}-{start+SEQ_LENGTH-1}: {e}")
            continue

        # Store cumulative alerts for reference
        alerts_storage.setdefault(patient_id, []).append(pred_int)

        # Broadcast only the latest alert
        await manager.broadcast(patient_id, [pred_int])
        logger.info(f"Simulated alert: {pred:.1f} hours (rows {start}-{start+SEQ_LENGTH-1})")
        await asyncio.sleep(3)

# ============================================================
# 🔹 Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect DB
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        logger.info("✅ Connected to NeonDB")
    except Exception as e:
        logger.error(f"❌ Failed to connect to NeonDB: {e}")

    # BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # go up to project root
    # csv_path = os.path.join(BASE_DIR, "frontend", "public", "patient1.csv")
    # asyncio.create_task(simulate_patient_alerts("patient1", csv_path))
    logger.info("🔹 Started patient1 simulation")

    yield

    await engine.dispose()
    logger.info("Database connection closed")

# ============================================================
# 🔹 FastAPI App
# ============================================================
app = FastAPI(title="Sepsis Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🔹 WebSocket Endpoint
# ============================================================
@app.websocket("/ws/alerts/{patient_id}")
async def websocket_alerts(websocket: WebSocket, patient_id: str):
    await manager.connect(patient_id, websocket)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(BASE_DIR, "frontend", "public", f"{patient_id}.csv")
    
    # Start simulation if not already started
    if patient_id not in alerts_storage:
        alerts_storage[patient_id] = []
        if os.path.exists(csv_path):
            asyncio.create_task(simulate_patient_alerts(patient_id, csv_path))
            logger.info(f"🔹 Started simulation for {patient_id}")
        else:
            logger.error(f"CSV file not found for {patient_id}: {csv_path}")

    # Do NOT send the entire past array here — let simulation push each new alert
    try:
        while True:
            await asyncio.sleep(1)  # keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(patient_id, websocket)

# ============================================================
# 🔹 Root & Predict
# ============================================================
@app.get("/")
def read_root():
    return {"message": "Sepsis Detector API is running"}

@app.post("/predict-ml/")
async def predict_ml(input_data: PatientSequence):
    input_seq = input_data.features
    if len(input_seq) != SEQ_LENGTH:
        return {"error": f"Expected {SEQ_LENGTH} timesteps, got {len(input_seq)}"}
    arr = np.array([[step[feat] for feat in FEATURES] for step in input_seq], dtype=np.float32)
    tensor = torch.tensor(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).item()
    return {"predicted_time_to_sepsis": pred}

# ============================================================
# 🔹 Run Server
# ============================================================
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_config=None)
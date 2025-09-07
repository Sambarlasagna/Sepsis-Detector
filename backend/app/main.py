from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, List
import asyncio
import uvicorn

# --- DB ---
from db import engine  # NeonDB connection setup
from models import AlertMessage, PatientSequence  # Pydantic models

# --- ML ---
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ------------------ GRU Model Definition ------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out).squeeze(1)


# ------------------ Load Model ------------------
def load_model(model_path, input_dim):
    model = GRUModel(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


# ------------------ Data Preprocessing ------------------
SEQ_LENGTH = 6
scaler = StandardScaler()
FEATURE_COLUMNS = ["feature1", "feature2", "feature3"]

def preprocess_input(data_list):
    df = pd.DataFrame(data_list)
    df = df[FEATURE_COLUMNS]
    df = df.ffill().bfill().fillna(df.median())
    df_scaled = scaler.transform(df)
    return df_scaled


# -------------------------- Lifespan --------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        print("Connected to NeonDB")
    except Exception as e:
        print("Failed to connect to NeonDB:", e)

    yield

    await engine.dispose()
    print("Database connection closed")


# -------------------------- FastAPI Init --------------------------
app = FastAPI(
    title="Sepsis Detector API",
    description="Backend API for Sepsis Detection using AI/ML with Real-Time Alerts",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------- Root --------------------------
@app.get("/")
def read_root():
    return {"message": "Sepsis Detector API is running 🚀"}


# -------------------------- ML Prediction --------------------------
MODEL_PATH = "gru_time_to_sepsis.pth"
model = load_model(MODEL_PATH, input_dim=len(FEATURE_COLUMNS))
scaler.fit(np.zeros((10, len(FEATURE_COLUMNS))))  # dummy fit

@app.post("/predict-ml/")
async def predict_time_to_sepsis(input_data: PatientSequence):
    seq_array = preprocess_input(input_data.features)
    input_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    prediction_clipped = max(0.0, min(6.0, prediction))
    return {"predicted_time_to_sepsis": prediction_clipped}


# -------------------------- WebSocket Connection Manager --------------------------
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
            alert_payload = AlertMessage(hours_until_sepsis=message)
            for connection in self.active_connections[patient_id]:
                await connection.send_json(alert_payload.model_dump())


# Global manager & storage
manager = ConnectionManager()
alerts_storage: Dict[str, List[int]] = {}


# -------------------------- WebSocket Endpoint --------------------------
@app.websocket("/ws/alerts/{patient_id}")
async def websocket_alerts(websocket: WebSocket, patient_id: str):
    await manager.connect(patient_id, websocket)

    if patient_id in alerts_storage and alerts_storage[patient_id]:
        await websocket.send_json({"alerts": alerts_storage[patient_id]})

    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(patient_id, websocket)


# -------------------------- POST Endpoint to Trigger Alerts --------------------------
@app.post("/test-alerts/{patient_id}")
async def add_test_alerts(patient_id: str, request: AlertMessage):
    if patient_id not in alerts_storage:
        alerts_storage[patient_id] = []

    alerts_storage[patient_id].extend(request.hours_until_sepsis)
    await manager.broadcast(patient_id, alerts_storage[patient_id])
    return {"patient_id": patient_id, "alerts": alerts_storage[patient_id]}


# -------------------------- Clear Alerts --------------------------
@app.delete("/clear-alerts/{patient_id}")
async def clear_alerts(patient_id: str):
    alerts_storage[patient_id] = []
    await manager.broadcast(patient_id, [])
    return {"patient_id": patient_id, "alerts": []}


# -------------------------- Run --------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

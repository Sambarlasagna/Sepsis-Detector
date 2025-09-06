from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, List
import asyncio

app = FastAPI()

# -------------------------------
# Connection Manager (Per Patient)
# -------------------------------
class ConnectionManager:
    def __init__(self):
        # Each patient will have its own list of WebSocket connections
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
                del self.active_connections[patient_id]  # cleanup if empty

    async def broadcast(self, patient_id: str, message: List[int]):
        """Send alerts to all connected clients of a specific patient."""
        if patient_id in self.active_connections:
            for connection in self.active_connections[patient_id]:
                await connection.send_json({"alerts": message})


# Create a single connection manager
manager = ConnectionManager()

# Store alerts separately for each patient
alerts_storage: Dict[str, List[int]] = {}

# --------------------------
# WebSocket Endpoint (Per Patient)
# --------------------------
@app.websocket("/ws/alerts/{patient_id}")
async def websocket_alerts(websocket: WebSocket, patient_id: str):
    await manager.connect(patient_id, websocket)

    # Send existing alerts immediately upon connection
    if patient_id in alerts_storage and alerts_storage[patient_id]:
        await websocket.send_json({"alerts": alerts_storage[patient_id]})

    try:
        while True:
            # Keep the connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(patient_id, websocket)


# --------------------------
# Pydantic Model for POST
# --------------------------
class AlertRequest(BaseModel):
    hours_until_sepsis: List[int]


# --------------------------
# POST Endpoint to Trigger Alerts
# --------------------------
@app.post("/test-alerts/{patient_id}")
async def add_test_alerts(patient_id: str, request: AlertRequest):
    # Initialize patient's alerts list if missing
    if patient_id not in alerts_storage:
        alerts_storage[patient_id] = []

    # Add new alerts for this patient
    alerts_storage[patient_id].extend(request.hours_until_sepsis)

    # Broadcast only to this patient's clients
    await manager.broadcast(patient_id, alerts_storage[patient_id])

    return {"patient_id": patient_id, "alerts": alerts_storage[patient_id]}


# --------------------------
# Endpoint to Clear Alerts (Optional)
# --------------------------
@app.delete("/clear-alerts/{patient_id}")
async def clear_alerts(patient_id: str):
    """Clear all alerts for a patient."""
    alerts_storage[patient_id] = []
    await manager.broadcast(patient_id, [])
    return {"patient_id": patient_id, "alerts": []}
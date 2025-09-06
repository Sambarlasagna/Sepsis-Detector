from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import asyncio 

app = FastAPI()

# Enable CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manager to track connected WebSocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: List[int]):
        for connection in self.active_connections:
            await connection.send_json({"alerts": message})

manager = ConnectionManager()
alerts_placeholder: List[int] = []

# WebSocket endpoint
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send current alerts immediately upon connection
        if alerts_placeholder:
            await websocket.send_json({"alerts": alerts_placeholder})

        while True:
            # Keep connection open
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Pydantic model for POST request
class AlertRequest(BaseModel):
    hours_until_sepsis: List[int]

# POST endpoint to manually trigger alerts
@app.post("/test-alerts")
async def add_test_alerts(request: AlertRequest):
    alerts_placeholder.extend(request.hours_until_sepsis)
    await manager.broadcast(alerts_placeholder)
    return {"alerts": alerts_placeholder}

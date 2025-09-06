from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from db import engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        print("Connected to NeonDB")
    except Exception as e:
        print("Failed to connect to NeonDB:", e)

    yield  #app runs here

    # Shutdown logic
    await engine.dispose()
    print("Database connection closed")

app = FastAPI(
    title="Sepsis Detector API",
    description="Backend API for Sepsis Detection using AI/ML",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Sepsis Detector API is running."}

@app.post("/predict")
def predict(data: dict):
    return {"input": data,"prediction":"sepsis_risk_placeholder"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Alert(BaseModel):
    timestamp: str
    location: str
    objects_involved: list
    severity_score: float
    frame_path: str

# Initialize video processor
processor = VideoProcessor('accident_model.h5')

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    # Save uploaded video temporarily
    temp_path = f"temp_{datetime.now().timestamp()}.mp4"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Process video
    processor.start_processing(temp_path)
    
    # Clean up
    os.remove(temp_path)
    
    return {"message": "Video processing started"}

@app.get("/alerts/")
async def get_alerts(start_time: str = None, end_time: str = None):
    # Query database for alerts in time range
    query = {}
    if start_time and end_time:
        query = {"timestamp": {"$gte": start_time, "$lte": end_time}}
    
    alerts = db.alerts.find(query)
    return [Alert(**alert) for alert in alerts]

@app.websocket("/ws-alerts")
async def websocket_alerts(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Listen for new alerts (could use Redis pub/sub)
        new_alert = await get_new_alert_from_queue()
        await websocket.send_json(new_alert)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
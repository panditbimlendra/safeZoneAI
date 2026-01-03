from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from detector import CrashDetector
import os
import uuid

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = CrashDetector()

@app.post("/detect-crash")
async def detect_crash(video: UploadFile = File(...)):
    # Save uploaded file
    file_path = f"uploads/{uuid.uuid4()}.mp4"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await video.read())
    
    # Process video
    result = detector.process_video(file_path)
    
    # Cleanup
    os.remove(file_path)
    
    return result
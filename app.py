import os
import shutil
import json
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # Import StaticFiles
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.mp4 import MP4
from mutagen.oggvorbis import OggVorbis

# Import the analytics functions from main.py
import main as analytics

# Initialize the FastAPI app
app = FastAPI(
    title="Call Center Analytics API",
    description="An API to analyze call center audio recordings using Gemini.",
    version="1.0.0",
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FIX: Mount a directory to serve the audio files ---
# This creates an endpoint like /audio/filename.mp3
app.mount("/audio", StaticFiles(directory=analytics.AUDIO_DIR), name="audio")


# Load configuration at startup
config = analytics.load_config()

def get_audio_duration(filepath):
    """Calculates the duration of an audio file in minutes."""
    try:
        if filepath.lower().endswith('.mp3'):
            audio = MP3(filepath)
        elif filepath.lower().endswith('.wav'):
            audio = WAVE(filepath)
        elif filepath.lower().endswith('.m4a'):
            audio = MP4(filepath)
        elif filepath.lower().endswith('.ogg'):
            audio = OggVorbis(filepath)
        else:
            return 0
        return round(audio.info.length / 60, 2)
    except Exception:
        return 0

@app.get("/list_audio", summary="List all available audio files")
def list_audio():
    """
    Scans the designated audio directory and returns a list of all
    supported audio files found, including their duration.
    """
    supported_formats = ['.mp3', '.wav', '.m4a', '.ogg']
    audio_files = []
    try:
        for filename in os.listdir(analytics.AUDIO_DIR):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                file_path = os.path.abspath(os.path.join(analytics.AUDIO_DIR, filename))
                duration = get_audio_duration(file_path)
                audio_files.append({
                    "filename": filename,
                    "path": file_path,
                    "duration_mins": duration
                })
        return {"audio_files": audio_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_audio", summary="Analyze an audio file")
async def analyze_audio(
    audio_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    if not audio_id and not file:
        raise HTTPException(status_code=400, detail="You must provide either an 'audio_id' or upload a 'file'.")

    if file:
        audio_path = os.path.join(analytics.AUDIO_DIR, file.filename)
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    else:
        audio_path = os.path.join(analytics.AUDIO_DIR, audio_id)
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail=f"Audio file with id '{audio_id}' not found.")

    analysis_result = analytics.analyze_call(audio_path, config)

    if "error" in analysis_result:
        raise HTTPException(status_code=500, detail=analysis_result["error"])

    return JSONResponse(content=analysis_result)


@app.get("/export_data", summary="Export analysis results as JSON")
def export_data(analysis_id: str):
    analysis_path = os.path.join(analytics.ANALYSIS_DIR, analysis_id)
    if not os.path.exists(analysis_path):
        raise HTTPException(status_code=404, detail=f"Analysis with id '{analysis_id}' not found.")

    with open(analysis_path, 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

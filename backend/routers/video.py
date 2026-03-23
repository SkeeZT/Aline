from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from loguru import logger
import shutil
import os
import uuid

from core.config import Config, settings
from engine.video_processor import VideoProcessor

router = APIRouter()

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(settings.upload_dir, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"filename": filename, "path": file_path}
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_analysis(video_path: str, config: dict):
    try:
        # Extract video_id from path
        filename = os.path.basename(video_path)
        video_id = os.path.splitext(filename)[0]

        processor = VideoProcessor(
            analysis_id=video_id,
            config=config,
            exercise_type="squat",
            video_path=video_path,
            use_webcam=False,
            video_id=video_id
        )
        processor.process()
    except Exception as e:
        logger.error(f"Analysis error: {e}")

@router.post("/analyze")
async def analyze_video(video_path: str, background_tasks: BackgroundTasks):
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
        
    try:
        config = Config.load_config(settings.config_path)
        # Disable visualization for API requests to prevent server-side window popup
        if "video" not in config:
            config["video"] = {}
        config["video"]["show_visualize"] = False
        
        # Run analysis in background
        background_tasks.add_task(run_analysis, video_path, config)
        
        return {"message": "Analysis started", "video_path": video_path}
        
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{filename}")
async def check_status(filename: str):
    """Check the status of video analysis."""
    try:
        # Extract ID from filename (remove extension)
        video_id = os.path.splitext(filename)[0]
        
        # Check output directory for processed video
        # Pattern: {video_id}_squat_{timestamp}_final.mp4
        output_dir = os.path.join(settings.output_dir, "videos")
        
        if not os.path.exists(output_dir):
            return {"status": "processing"}
            
        # List files in output directory
        files = os.listdir(output_dir)

        # Find matching files
        # 1. Try standard pattern: {video_id}_squat_*_final.mp4 (for uploads)
        matching_files = [f for f in files if f.startswith(f"{video_id}_squat") and f.endswith("_final.mp4")]
        
        # 2. If no match, try exact match or webcam pattern: {video_id} (if video_id is already the full base name)
        if not matching_files:
             matching_files = [f for f in files if f.startswith(video_id) and f.endswith("_final.mp4")]

        if matching_files:
            # Sort by modification time (newest first)
            matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
            latest_file = matching_files[0]
            
            # Extract timestamp from filename
            # Pattern: {video_id}_squat_{YYYYMMDD}_{HHMMSS}_final.mp4
            # We need: {YYYYMMDD}_{HHMMSS}
            parts = latest_file.replace('_final.mp4', '').split('_')
            # parts = [uuid, 'squat', YYYYMMDD, HHMMSS]
            if len(parts) >= 4:
                timestamp = f"{parts[-2]}_{parts[-1]}"
            else:
                # Fallback
                timestamp = parts[-1]
            
            return {
                "status": "completed",
                "video_url": f"http://localhost:8000/static/videos/{latest_file}",
                "metadata_url": f"http://localhost:8000/static/velocity_calculations/metadata_{timestamp}.json",
                "details_url": f"http://localhost:8000/static/squat_analysis_details_{timestamp}.json",
                "vbt_url": f"http://localhost:8000/static/velocity_calculations/squat_vbt_analysis_{timestamp}.json"
            }
            
        return {"status": "processing"}
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/monitor/{video_id}")
async def monitor_video(websocket: WebSocket, video_id: str):
    from core.monitor import monitor
    await monitor.connect(video_id, websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        monitor.disconnect(video_id, websocket)
    except Exception as e:
        logger.error(f"Monitor error: {e}")
        monitor.disconnect(video_id, websocket)

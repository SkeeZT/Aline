import pytest
from fastapi.testclient import TestClient
from api.app import app
import os
import cv2
import numpy as np
import base64

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Trainer API is running"}

def test_upload_video():
    # Create a dummy video file
    filename = "test_video.mp4"
    with open(filename, "wb") as f:
        f.write(b"dummy content")
    
    try:
        with open(filename, "rb") as f:
            response = client.post(
                "/api/video/upload",
                files={"file": (filename, f, "video/mp4")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "path" in data
        assert os.path.exists(data["path"])
        
        # Cleanup uploaded file
        if os.path.exists(data["path"]):
            os.remove(data["path"])
            
    finally:
        if os.path.exists(filename):
            os.remove(filename)

# Note: WebSocket testing with TestClient is limited. 
# We'll rely on manual verification for the full flow, 
# but we can test the connection logic if we mock the config/processor.

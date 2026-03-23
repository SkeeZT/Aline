import asyncio
from typing import Dict, Set
from fastapi import WebSocket
from loguru import logger

class AnalysisMonitor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnalysisMonitor, cls).__new__(cls)
            cls._instance.active_connections: Dict[str, Set[WebSocket]] = {}
        return cls._instance

    async def connect(self, video_id: str, websocket: WebSocket):
        await websocket.accept()
        if video_id not in self.active_connections:
            self.active_connections[video_id] = set()
        self.active_connections[video_id].add(websocket)
        logger.info(f"Client connected to monitor for video {video_id}")

    def disconnect(self, video_id: str, websocket: WebSocket):
        if video_id in self.active_connections:
            self.active_connections[video_id].remove(websocket)
            if not self.active_connections[video_id]:
                del self.active_connections[video_id]
        logger.info(f"Client disconnected from monitor for video {video_id}")

    async def broadcast_frame(self, video_id: str, frame_data: str, metadata: dict = None):
        if video_id in self.active_connections:
            message = {
                "frame": frame_data,
                "metadata": metadata or {}
            }
            
            # Create a list of connections to remove (if they fail)
            to_remove = []
            
            for connection in self.active_connections[video_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send frame to client: {e}")
                    to_remove.append(connection)
            
            # Clean up failed connections
            for connection in to_remove:
                self.disconnect(video_id, connection)

monitor = AnalysisMonitor()

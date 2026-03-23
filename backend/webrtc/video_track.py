"""WebRTC video track for streaming analysis frames."""

import asyncio
from aiortc import VideoStreamTrack
from av import VideoFrame
import numpy as np
from loguru import logger
from streaming.latest_frame_bus import latest_frame_bus


class AnalysisVideoTrack(VideoStreamTrack):
    """Video track that streams frames from an analysis session."""
    
    kind = "video"
    
    def __init__(self, analysis_id: str, fps: int = 30):
        """
        Initialize the video track.
        
        Args:
            analysis_id: The analysis ID to stream frames for
            fps: Target frame rate (default 30)
        """
        super().__init__()
        self.analysis_id = analysis_id
        self.frame_interval = 1 / max(1, fps)
        self._last_frame_rgb = None
        
    async def recv(self) -> VideoFrame:
        """Receive the next video frame from the frame bus."""
        try:
            # Pull the freshest frame; wait briefly to avoid long stalls
            try:
                frame_task = asyncio.create_task(latest_frame_bus.get(self.analysis_id))
                frame_bgr, meta = await asyncio.wait_for(frame_task, timeout=self.frame_interval * 2)
            except asyncio.TimeoutError:
                # On timeout, reuse last frame or emit black frame to keep stream alive
                if self._last_frame_rgb is None:
                    frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    frame_rgb = self._last_frame_rgb
                vf = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                pts, time_base = await self.next_timestamp()
                vf.pts = pts
                vf.time_base = time_base
                return vf
            
            # Convert BGR (OpenCV) -> RGB for PyAV
            frame_rgb = frame_bgr[:, :, ::-1]
            
            # Create VideoFrame from numpy array
            vf = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            
            # VideoStreamTrack's next_timestamp() manages pts/time_base automatically
            pts, time_base = await self.next_timestamp()
            vf.pts = pts
            vf.time_base = time_base
            
            # Cache last frame
            self._last_frame_rgb = frame_rgb
            return vf
        except Exception as exc:
            logger.error(f"Error in AnalysisVideoTrack.recv for {self.analysis_id}: {exc}", exc_info=True)
            raise


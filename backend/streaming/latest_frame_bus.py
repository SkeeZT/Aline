import asyncio
from collections import defaultdict
from typing import Dict, Optional, Tuple
import numpy as np


class LatestFrameBus:
    """Frame bus that publishes only the latest frame per analysis_id for low latency streaming."""
    
    def __init__(self):
        self._qs: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue(maxsize=1))
        self._closed = set()

    def publish(self, analysis_id: str, frame_bgr: np.ndarray, meta: dict):
        """Publish a frame, dropping stale frames to keep latency low."""
        if analysis_id in self._closed:
            return
        q = self._qs[analysis_id]
        # Drop stale to keep latency low
        while not q.empty():
            try:
                q.get_nowait()
            except:
                break
        q.put_nowait((frame_bgr, meta))

    async def get(self, analysis_id: str) -> Tuple[np.ndarray, dict]:
        """Get the latest frame for an analysis_id, blocking until one arrives."""
        return await self._qs[analysis_id].get()

    def close(self, analysis_id: str):
        """Mark an analysis_id as closed and clean up its queue."""
        self._closed.add(analysis_id)
        if analysis_id in self._qs:
            q = self._qs[analysis_id]
            # Drain the queue
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break


# Global instance
latest_frame_bus = LatestFrameBus()








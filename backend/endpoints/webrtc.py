"""WebRTC signaling endpoints for establishing peer connections."""

import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from webrtc.video_track import AnalysisVideoTrack
from streaming.latest_frame_bus import latest_frame_bus
from services.video_service import video_service
from loguru import logger


router = APIRouter(prefix="/api/v1/webrtc", tags=["WebRTC"])


class OfferBody(BaseModel):
    """Request body for WebRTC offer."""
    analysis_id: str
    sdp: str
    type: str  # "offer"


# Load ICE servers from environment
_ICE_SERVERS_JSON = os.getenv("ICE_SERVERS_JSON", '["stun:stun.l.google.com:19302"]')
try:
    parsed = json.loads(_ICE_SERVERS_JSON)
    # Normalize ICE servers to RTCIceServer objects
    # Input can be: ["stun:..."] or [{"urls": ["stun:..."], "username": "...", "credential": "..."}]
    if isinstance(parsed, list) and len(parsed) > 0:
        ice_servers = []
        for item in parsed:
            if isinstance(item, str):
                # Single URL string
                ice_servers.append(RTCIceServer(urls=item))
            elif isinstance(item, dict):
                # Dictionary with urls and optional username/credential
                ice_servers.append(RTCIceServer(**item))
            else:
                raise ValueError(f"Invalid ICE server format: {item}")
        ICE_SERVERS = ice_servers
    else:
        raise ValueError("Invalid ICE_SERVERS_JSON format")
except (json.JSONDecodeError, ValueError) as e:
    logger.error(f"Failed to parse ICE_SERVERS_JSON: {_ICE_SERVERS_JSON}, error: {e}")
    ICE_SERVERS = [RTCIceServer(urls="stun:stun.l.google.com:19302")]


@router.post("/offer")
async def webrtc_offer(body: OfferBody):
    """
    Handle WebRTC offer and return answer.
    
    This endpoint establishes a WebRTC peer connection for streaming
    analysis frames to the browser.
    """
    if body.type != "offer":
        raise HTTPException(status_code=400, detail="SDP type must be 'offer'")
    
    # Verify the analysis exists and is active
    try:
        analysis_info = video_service.get_analysis_status(body.analysis_id)
        if analysis_info["status"] not in ["pending", "processing"]:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis {body.analysis_id} is not active"
            )
    except Exception as exc:
        logger.error(f"Error validating analysis {body.analysis_id}: {exc}")
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {body.analysis_id} not found or invalid"
        )
    
    # Create peer connection with RTCConfiguration
    config = RTCConfiguration(iceServers=ICE_SERVERS)
    pc = RTCPeerConnection(configuration=config)
    
    # Add video track
    track = AnalysisVideoTrack(body.analysis_id, fps=30)
    pc.addTrack(track)
    
    # Handle connection state changes
    @pc.on("connectionstatechange")
    async def on_state_change():
        logger.info(
            f"WebRTC connection state for {body.analysis_id}: {pc.connectionState}"
        )
        if pc.connectionState in ("failed", "closed", "disconnected"):
            latest_frame_bus.close(body.analysis_id)
            await pc.close()
    
    try:
        # Set remote description (offer)
        offer = RTCSessionDescription(sdp=body.sdp, type=body.type)
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    except Exception as exc:
        logger.error(f"Error handling WebRTC offer: {exc}")
        await pc.close()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to establish WebRTC connection: {str(exc)}"
        )



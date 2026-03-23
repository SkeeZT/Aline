# AI Trainer API

A FastAPI-based REST API for AI-powered exercise analysis, specifically designed for squat form evaluation using computer vision and velocity-based training metrics.

## Features

- **Video Analysis**: Upload and analyze exercise videos for form evaluation
- **Real-time Webcam Analysis**: Live analysis using webcam feed
- **Velocity-Based Training (VBT)**: Detailed velocity metrics for each rep
- **Voice Feedback**: Real-time audio feedback during analysis
- **Person Segmentation**: Advanced person segmentation with overlay
- **Comprehensive Metrics**: Anthropometric measurements and form analysis
- **Multiple Experience Levels**: Beginner, intermediate, and advanced analysis modes

## Quick Start

### Prerequisites

- Python 3.8+
- OpenCV
- YOLO models (pose and segmentation)
- FFmpeg
- Pygame (for audio)

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Ensure you have the required models in the correct paths:
   - `./assets/models/yolo11x-pose.pt` (pose detection)
   - `./assets/models/yolo11s-seg.pt` (segmentation)

3. Start the API server:
```bash
uv run python api/start_api.py
```

The API will be available at `http://localhost:8000`

### Using Docker (Optional)

```bash
# Build the image
docker build -t ai-trainer-api .

# Run the container
docker run -p 8000:8000 ai-trainer-api
```

## API Endpoints

### Health Check
- `GET /api/v1/health/` - Check API health and dependencies
- `GET /api/v1/health/ready` - Check if service is ready
- `GET /api/v1/health/live` - Check if service is alive

### Video Analysis
- `POST /api/v1/video/analyze` - Upload and analyze video file
- `POST /api/v1/video/webcam/start` - Start webcam analysis
- `GET /api/v1/video/analysis/{id}/status` - Get analysis status
- `POST /api/v1/video/analysis/{id}/stop` - Stop analysis
- `DELETE /api/v1/video/analysis/{id}` - Delete analysis
- `GET /api/v1/video/analysis/{id}/download` - Download processed video

### Analysis Results
- `GET /api/v1/analysis/{id}` - Get complete analysis results
- `GET /api/v1/analysis/{id}/metrics` - Get velocity metrics
- `GET /api/v1/analysis/{id}/summary` - Get summary statistics
- `GET /api/v1/analysis/{id}/exercise` - Get exercise results
- `GET /api/v1/analysis/{id}/files/{type}` - Download specific files
- `GET /api/v1/analysis/` - List all analyses
- `POST /api/v1/analysis/cleanup` - Clean up old files

## Usage Examples

### 1. Analyze Video File

```bash
curl -X POST "http://localhost:8000/api/v1/video/analyze" \
  -F "file=@squat_video.mp4" \
  -F "exercise_type=squat" \
  -F "experience_level=intermediate" \
  -F "enable_voice_feedback=true"
```

### 2. Start Webcam Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/video/webcam/start" \
  -H "Content-Type: application/json" \
  -d '{
    "webcam_id": 0,
    "exercise_type": "squat",
    "experience_level": "intermediate",
    "show_visualization": true
  }'
```

### 3. Get Analysis Results

```bash
curl "http://localhost:8000/api/v1/analysis/{analysis_id}"
```

### 4. Download Processed Video

```bash
curl "http://localhost:8000/api/v1/video/analysis/{analysis_id}/download" \
  -o processed_video.mp4
```

## Configuration

The API uses the same configuration file as the main application (`config.yaml`). Key settings:

```yaml
# API Settings
paths:
  model: "./assets/models/yolo11x-pose.pt"
  segmentation_model: "./assets/models/yolo11s-seg.pt"
  voice_messages: "./assets/voice_messages/Tone_A"
  output_dir: "./assets/output"

# Experience Levels
experience:
  level: intermediate  # beginner | intermediate | advanced

# Voice Settings
voice:
  enabled: true
  volume: 0.7
```

## Environment Variables

- `HOST` - API host (default: 0.0.0.0)
- `PORT` - API port (default: 8000)
- `DEBUG` - Debug mode (default: false)
- `MAX_FILE_SIZE` - Max upload size in MB (default: 100)
- `UPLOAD_DIR` - Upload directory (default: ./uploads)
- `OUTPUT_DIR` - Output directory (default: ./assets/output)

## Response Formats

### Analysis Response
```json
{
  "analysis_id": "uuid",
  "status": "pending|processing|completed|failed",
  "exercise_type": "squat",
  "experience_level": "intermediate",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Complete Results
```json
{
  "analysis_id": "uuid",
  "status": "completed",
  "exercise_results": {
    "successful_reps": 8,
    "unsuccessful_reps": 2,
    "total_reps": 10,
    "facing_side": "right"
  },
  "velocity_metrics": [...],
  "summary_statistics": {...},
  "output_files": {
    "processed_video": "path/to/video.mp4",
    "velocity_data": "path/to/data.json"
  }
}
```

## Error Handling

The API returns structured error responses:

```json
{
  "error": "Error Type",
  "message": "Error description",
  "details": {...},
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Common error codes:
- `400` - Bad Request (invalid file, parameters)
- `404` - Not Found (analysis not found)
- `413` - Payload Too Large (file too big)
- `500` - Internal Server Error

## Testing

### Using Postman

Import the provided `postman_collection.json` file into Postman for easy testing.

### Using curl

See the usage examples above for curl commands.

### Automated Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=api tests/
```

## Development

### Project Structure

```
api/
├── main.py                 # FastAPI application
├── models/                 # Pydantic models
│   ├── requests.py        # Request schemas
│   └── responses.py       # Response schemas
├── services/              # Business logic
│   ├── video_service.py  # Video processing
│   └── analysis_service.py # Results handling
├── endpoints/             # API endpoints
│   ├── video.py          # Video analysis
│   ├── analysis.py       # Results
│   └── health.py         # Health checks
├── core/                  # Core utilities
│   ├── config.py         # Configuration
│   └── exceptions.py     # Custom exceptions
└── requirements.txt       # Dependencies
```

### Adding New Endpoints

1. Create endpoint in appropriate router file
2. Add Pydantic models if needed
3. Implement business logic in services
4. Add tests
5. Update documentation

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Use async/await for I/O operations
- Handle exceptions properly

## Performance Considerations

- Video processing is CPU-intensive
- Use background tasks for long operations
- Implement proper file cleanup
- Consider rate limiting for production
- Monitor memory usage with large files

## Security

- Validate all file uploads
- Implement proper CORS settings
- Use trusted host middleware in production
- Sanitize file paths
- Implement authentication if needed

## Monitoring

- Health check endpoints for monitoring
- Structured logging with loguru
- Request/response logging middleware
- Error tracking and reporting

## Squat Analysis Logic

### 1. Correct Rep Calculation
A rep is counted as correct (Successful) if it meets the following criteria:

*   **Significant Descent**: The rep starts when the user bends their knees effectively (angle < baseline - 5°) and velocity is downward. This prevents "ghost reps" from micro-movements.
*   **Depth Achieved**: The user reaches a bottom position where the Range of Motion (ROM) percent exceeds the `success_threshold` (default 0.25).
    *   ROM % is calculated relative to the user's standing knee angle (0%) and their deep squat position (100%).
*   **Return to Standing**: The user returns to a standing position (Knee Angle > Max Baseline - 10°).
*   **No Major Form Issues**: The rep contains none of the major form errors listed below.

### 2. Error Detection
The engine monitors for specific form errors throughout the movement:

*   **Insufficient Depth**:
    *   **Logic**: The lowest point of the squat did not reach the required ROM threshold (e.g., < 40% ROM).
    *   **Feedback**: "Go deeper" or specific depth correction.

*   **Excessive Forward Lean (Descent)**:
    *   **Logic**: Measures the synchronization between Hip and Knee angles during descent.
    *   **Condition**: `(Knee Angle - Hip Angle) > relative_angle_difference` (default 30°).
    *   **Meaning**: The torso is leaning forward significantly faster/more than the knees are bending, indicating a collapse at the hips.

*   **Good Morning Squat (Ascent)**:
    *   **Logic**: Measures synchronization during ascent.
    *   **Condition**: `(Knee Angle - Hip Angle) > relative_angle_difference` (default 30°).
    *   **Meaning**: The legs are straightening (Knees extending) while the back remains horizontal (Hips stayed flexed), causing the hips to "shoot up" first.

*   **Knees Over Toes**:
    *   **Logic**: Checks if the knee horizontal position passes the toes (Foot Index or estimated from Heel) by more than a tolerance buffer (50px).
    *   **Side Specific**: Correctly tracks the Left Knee vs Left Foot/Heel and Right Knee vs Right Foot/Heel.

*   **Hyperextension**:
    *   **Logic**: Uses the segmentation model to analyze the spine's contour.
    *   **Condition**: `back_curvature > back_curvature_threshold` (default 1.5).
    *   **Fallback**: If segmentation is unavailable, falls back to hip angle calculation (> 190°).

## License

This project is part of the AI Trainer system. See the main project for license information.

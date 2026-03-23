import os
import sys
import argparse
from loguru import logger

# Add the backend root directory to sys.path to allow imports from core and engine
backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_root not in sys.path:
    sys.path.append(backend_root)

# Initialize logging before other local imports
from core.config import settings
log_file = settings.get_config_value("paths.log_file", "./assets/logs/api.log")
# Ensure we log to a specific batch processing log file for clarity if desired, 
# or just use the configured api log. 
# Here we'll use a dedicated batch processing log for better separation.
batch_log_path = os.path.join(os.path.dirname(log_file), "batch_processing.log")
logger.add(batch_log_path, rotation="10 MB", level="INFO")
logger.info(f"Batch processing logging initialized. Saving to: {batch_log_path}")

from core.config import Config
from engine.video_processor import VideoProcessor

def process_batch(input_path, experience_level="beginner", limit=None):
    """
    Process videos in the specified directory or a single file.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input path not found: {input_path}")
        return

    # Load configuration
    config_path = os.path.join(backend_root, "config.yaml")
    config = Config.load_config(config_path)
    
    # Disable visualization for batch processing
    if "video" not in config:
        config["video"] = {}
    config["video"]["show_visualize"] = False

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    if os.path.isfile(input_path):
        # Handle single file
        if os.path.splitext(input_path)[1].lower() not in video_extensions:
            logger.error(f"File is not a supported video format: {input_path}")
            return
        input_dir = os.path.dirname(input_path)
        video_files = [os.path.basename(input_path)]
        logger.info(f"Processing single file: {input_path}")
    else:
        # Handle directory
        input_dir = input_path
        video_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in video_extensions]
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return

        logger.info(f"Found {len(video_files)} videos in {input_dir}")
        
        if limit:
            video_files = video_files[:limit]
            logger.info(f"Limiting processing to first {limit} videos")

    processed_count = 0
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        
        logger.info(f"[{processed_count + 1}/{len(video_files)}] Processing: {video_file}")
        
        try:
            processor = VideoProcessor(
                analysis_id=video_id,
                config=config,
                exercise_type="squat",
                video_path=video_path,
                use_webcam=False,
                video_id=video_id,
                experience_level=experience_level
            )
            processor.process()
            processed_count += 1
            logger.success(f"Successfully processed: {video_file}")
        except Exception as e:
            logger.error(f"Failed to process {video_file}: {e}")

    logger.info(f"Batch processing complete. Total processed: {processed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process videos for squat analysis.")
    parser.add_argument("--input", type=str, required=True, help="Directory containing raw videos")
    parser.add_argument("--level", type=str, default="beginner", choices=["beginner", "intermediate", "advanced"], help="Experience level")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of videos to process")
    
    args = parser.parse_args()
    
    process_batch(args.input, args.level, args.limit)

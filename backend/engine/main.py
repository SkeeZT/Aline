import os
import sys
import uuid
import argparse
from loguru import logger


from core.config import Config
from engine.video_processor import VideoProcessor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Trainer - Exercise Pose Estimation"
    )
    parser.add_argument(
        "--video", "-v", help="Path to input video file (overrides config)"
    )
    parser.add_argument(
        "--config", "-c", default="./config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--exercise",
        "-e",
        choices=["squat"],  # Only squat supported in new implementation
        default="squat",
        help="Exercise type to analyze",
    )
    parser.add_argument(
        "--webcam", "-w", action="store_true", help="Use webcam instead of video file"
    )
    parser.add_argument(
        "--webcam-id",
        "-wid",
        type=int,
        default=0,
        help="Webcam device ID to use (default: 0)",
    )
    parser.add_argument(
        "--front-view",
        action="store_true",
        help="Force front-view analysis (skip side positioning)",
    )
    parser.add_argument(
        "--experience",
        choices=["beginner", "intermediate", "advanced"],
        help="User experience level to set rep success thresholds",
    )

    return parser.parse_args()


# Configure loguru to save logs to files
def setup_logging():
    """Setup loguru logging configuration."""
    # Remove default console handler
    logger.remove()

    # Add console handler with color and formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # Add file handler for all logs
    log_dir = "./assets/logs"
    os.makedirs(log_dir, exist_ok=True)

    # General log file
    logger.add(
        f"{log_dir}/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    # Error log file (only errors)
    logger.add(
        f"{log_dir}/errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="5 MB",
        retention="30 days",
        compression="zip",
    )

    logger.info("Logging system initialized")


def main():
    """Main entry point for the application."""
    # Setup logging first
    setup_logging()

    args = parse_arguments()

    try:
        # Load configuration
        config = Config.load_config(args.config)

        # Validate exercise type
        if args.exercise != "squat":
            logger.warning("Only 'squat' exercise is supported.")
            logger.warning("Falling back to squat analysis.")
        exercise_type = "squat"

        # Determine video source
        use_webcam = args.webcam if args.webcam else config["video"]["use_webcam"]
        webcam_id = (
            args.webcam_id
            if args.webcam_id is not None
            else config["video"]["webcam_id"]
        )
        video_path = args.video if args.video else config["paths"]["input_video"]

        # Determine view mode (CLI overrides config)
        force_front_view = bool(args.front_view) or bool(
            config.get("view", {}).get("force_front_view", False)
        )

        # Experience level
        experience_level = (
            args.experience
            if args.experience is not None
            else config.get("experience", {}).get("level", "intermediate")
        )

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        logger.info(f"Starting analysis with ID: {analysis_id}")

        # Create and run video processor
        processor = VideoProcessor(
            analysis_id=analysis_id,
            config=config,
            exercise_type=exercise_type,
            use_webcam=use_webcam,
            webcam_id=webcam_id,
            video_path=video_path,
            force_front_view=force_front_view,
            experience_level=experience_level,
        )

        processor.process()

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

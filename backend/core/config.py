"""
Configuration management for the FastAPI application.
"""

import os
import yaml
from loguru import logger
from typing import Dict, Any, Optional


class Settings:
    """Application settings."""

    def __init__(self):
        self.config_path = os.getenv("CONFIG_PATH", "./config.yaml")
        self.config = self._load_config()

        # API settings
        self.api_title = self.get_config_value("api.title", "AI Trainer API")
        self.api_version = self.get_config_value("api.version", "1.0.0")
        self.api_description = self.get_config_value(
            "api.description",
            "AI-powered exercise analysis API for squat form evaluation"
        )

        # Server settings
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # File upload settings
        self.max_file_size = (
            int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024
        )  # 100MB
        self.allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        self.upload_dir = self.get_config_value("paths.upload_dir", "./assets/uploads")
        self.output_dir = self.get_config_value("paths.output_dir", "./assets/output")

        # Analysis settings
        self.max_concurrent_analyses = int(os.getenv("MAX_CONCURRENT_ANALYSES", "3"))
        self.analysis_timeout = int(os.getenv("ANALYSIS_TIMEOUT", "3600"))  # 1 hour

        # WebSocket settings
        self.websocket_timeout = int(os.getenv("WEBSOCKET_TIMEOUT", "300"))  # 5 minutes

        # Create necessary directories
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as file:
                    config = yaml.safe_load(file)
                    logger.info(f"Configuration loaded from: {self.config_path}")
                    return config
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}

    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.upload_dir,
            self.output_dir,
            os.path.join(self.output_dir, "videos"),
            os.path.join(self.output_dir, "audio"),
            os.path.join(self.output_dir, "velocity_calculations"),
            "./assets/logs",
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'paths.model')."""
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def update_config(self, key: str, value: Any):
        """Update configuration value."""
        keys = key.split(".")
        config = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, "w") as file:
                yaml.dump(self.config, file, default_flow_style=False)
            logger.info(f"Configuration saved to: {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


# Global settings instance
settings = Settings()


class Config:
    """Unified configuration loader for YAML files - used by both CLI and API."""

    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file.
                        If None, uses default './config.yaml'

        Returns:
            Dictionary containing configuration settings

        Raises:
            FileNotFoundError: If configuration file is not found
            ValueError: If configuration file cannot be parsed
        """
        if config_path is None:
            config_path = "./config.yaml"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from: {config_path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")

    @staticmethod
    def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'paths.model')."""
        keys = key.split(".")
        value = config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

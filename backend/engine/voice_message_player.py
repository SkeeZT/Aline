"""
Voice message utilities for AI Trainer application.

This module provides a dedicated thread voice message player for exercise feedback.
The system uses pre-recorded audio files for real-time feedback during workouts.

Available voice messages:
- Corrections/start_workout.mp3 - Played at the beginning of a workout
- Corrections/generic.mp3 - Generic form correction message
- Corrections/end_workout.mp3 - Played at the end of a workout

Specific correction messages:
- Corrections/knee_valgus.mp3 - For knees caving inward
- Corrections/heel_lift.mp3 - For lifting heels
- Corrections/forward_lean.mp3 - For leaning too far forward
- Corrections/limited_depth.mp3 - For not going deep enough
- Corrections/hyperextension.mp3 - For over-arching the back

Additional voice messages are available in the voice_messages directory for future expansion.
"""

import os
import queue
import pygame
import threading
from typing import Dict, Any
from loguru import logger


class VoiceMessagePlayer:
    """Dedicated thread voice message player for exercise feedback."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the voice message player.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.voice_path = config["paths"]["voice_messages"]
        self.enabled = config["voice"].get("enabled", True)
        self.volume = config["voice"].get("volume", 0.7)

        # Dedicated audio thread with queue
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.stop_event = threading.Event()

        # Initialize pygame mixer if available
        if self.enabled:
            try:
                pygame.mixer.init()
                pygame.mixer.music.set_volume(self.volume)
                logger.info(f"Voice messages initialized. Path: {self.voice_path}")

                # Start dedicated audio thread
                self.audio_thread = threading.Thread(
                    target=self._audio_worker, daemon=True
                )
                self.audio_thread.start()
                logger.info("Dedicated audio thread started")
            except Exception as e:
                logger.warning(f"Failed to initialize voice messages: {e}")
                self.enabled = False
        else:
            logger.info("Voice messages disabled")

    def _audio_worker(self) -> None:
        """Dedicated thread worker for playing audio files."""
        while not self.stop_event.is_set():
            try:
                # Wait for audio file to play (with timeout to check stop_event)
                filename = self.audio_queue.get(timeout=0.1)

                if filename is None:  # Special signal to stop
                    break

                self._play_audio_now(filename)
                self.audio_queue.task_done()

            except queue.Empty:
                continue  # Check stop_event again
            except Exception as e:
                logger.error(f"Error in audio worker: {e}")

    def _play_audio_now(self, filename: str) -> None:
        """
        Play an audio file immediately (blocking).

        Args:
            filename: Name of the audio file to play
        """
        if not self.enabled:
            return

        filepath = os.path.join(self.voice_path, filename)

        if not os.path.exists(filepath):
            logger.warning(f"Voice file not found: {filepath}")
            return

        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()

            # Wait for playback to finish (non-blocking wait)
            while pygame.mixer.music.get_busy() and not self.stop_event.is_set():
                pygame.time.wait(10)  # Shorter wait time for more responsive stopping

        except Exception as e:
            logger.error(f"Error playing voice file {filename}: {e}")

    def _play_audio_file(self, filename: str) -> None:
        """
        Queue an audio file for playback.

        Args:
            filename: Name of the audio file to play
        """
        if not self.enabled:
            return

        filepath = os.path.join(self.voice_path, filename)

        if not os.path.exists(filepath):
            logger.warning(f"Voice file not found: {filepath}")
            return

        # Add to queue for dedicated thread to handle
        try:
            self.audio_queue.put_nowait(filename)
            logger.debug(f"Queued voice message: {filename}")
        except queue.Full:
            logger.warning(f"Audio queue full, dropping voice message: {filename}")

    def play_start_workout(self) -> None:
        """Play workout start message."""
        self._play_audio_file("Corrections/start_workout.mp3")

    def play_end_workout_and_wait(self) -> None:
        """Play workout end message and wait for it to finish."""
        if not self.enabled:
            return

        filepath = os.path.join(self.voice_path, "Corrections/end_workout.mp3")

        if not os.path.exists(filepath):
            logger.warning(f"Voice file not found: {filepath}")
            return

        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            logger.info("Playing end workout message - waiting to finish...")

            # Wait for playback to finish (blocking call)
            while pygame.mixer.music.get_busy():
                pygame.time.wait(10)
            logger.info("End workout message finished")
        except Exception as e:
            logger.error(f"Error playing end workout message: {e}")

    def play_form_correction(self, correction_type: str = "generic") -> None:
        """
        Play form correction message for unsuccessful rep.

        Args:
            correction_type: Type of correction message to play.
                           Options include: "knee_valgus", "heel_lift", "forward_lean", etc.
        """
        # Map correction types to specific audio files
        correction_files = {
            "knee_valgus": "Corrections/knee_valgus.mp3",
            "heel_lift": "Corrections/heel_lift.mp3",
            "forward_lean": "Corrections/forward_lean.mp3",
            "knee_caving": "Corrections/knee_valgus.mp3",
            "limited_depth": "Corrections/limited_depth.mp3",
            "weight_drifting": "Corrections/forward_lean.mp3",
            "lifting_heels": "Corrections/heel_lift.mp3",
            "knees_over_toes": "Corrections/knees_over_toes.mp3",
            "incomplete_depth": "Corrections/limited_depth.mp3",
            "insufficient_knee_bend": "Corrections/limited_depth.mp3",
            "hyperextension": "Corrections/hyperextension.mp3",
            "insufficient_hold": "Corrections/generic.mp3",
            "generic": "Corrections/generic.mp3",
        }

        filename = correction_files.get(correction_type, correction_files["generic"])
        self._play_audio_file(filename)

    def cleanup(self) -> None:
        """Clean up voice thread and pygame mixer."""
        if not self.enabled:
            return

        # Signal audio thread to stop
        self.stop_event.set()

        # Wait for audio thread to finish
        if self.audio_thread and self.audio_thread.is_alive():
            try:
                # Send stop signal through queue
                self.audio_queue.put_nowait(None)
                self.audio_thread.join(timeout=1.0)
            except Exception as e:
                logger.error(f"Error stopping audio thread: {e}")

        # Clean up pygame
        try:
            pygame.mixer.quit()
        except Exception as e:
            logger.error(f"Error cleaning up pygame: {e}")

        finally:
            self.audio_thread = None
            self.audio_queue = None
            self.stop_event = None
            self.enabled = False
            self.volume = 0.0
            self.voice_path = None
            self.config = None

"""
Synchronized output utilities for video, audio, and data.
This module handles saving video, audio, and velocity data with synchronized timestamps.
"""

import os
import json
import pygame
import ffmpeg
import numpy as np
from loguru import logger
from datetime import datetime
from pydub import AudioSegment
from typing import List, Dict, Any, Optional, Tuple


class AudioMixer:
    """Advanced audio mixer for combining voice messages at precise timestamps."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the audio mixer.

        Args:
            sample_rate: Sample rate for audio processing (default: 44100 Hz)
        """
        self.sample_rate = sample_rate
        self.pygame_initialized = False

        # Try to initialize pygame for audio handling
        try:
            pygame.mixer.pre_init(
                frequency=sample_rate, size=-16, channels=2, buffer=512
            )
            pygame.mixer.init()
            self.pygame_initialized = True
        except Exception as e:
            logger.warning(f"Could not initialize pygame mixer: {e}")

    def create_silent_track(self, duration: float) -> np.ndarray:
        """
        Create a silent audio track of specified duration.

        Args:
            duration: Duration in seconds

        Returns:
            Numpy array representing silent audio track
        """
        samples = int(duration * self.sample_rate)
        return np.zeros(samples, dtype=np.int16)

    def load_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load an audio file and convert it to numpy array.

        Args:
            file_path: Path to audio file

        Returns:
            Numpy array of audio data, or None if failed
        """
        try:
            # Use pydub to load various audio formats
            audio = AudioSegment.from_file(file_path)

            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Resample to our target sample rate if needed
            if audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)

            # Convert to numpy array
            audio_data = np.array(audio.get_array_of_samples())

            # Normalize to int16 range if needed
            if audio.sample_width == 4:  # 32-bit
                audio_data = (audio_data / (2**31)).astype(np.float32)
                audio_data = (audio_data * (2**15)).astype(np.int16)
            elif audio.sample_width == 1:  # 8-bit
                audio_data = (audio_data * 256).astype(np.int16)

            return audio_data

        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            return None

    def overlay_audio_at_timestamp(
        self, base_audio: np.ndarray, overlay_audio: np.ndarray, timestamp: float
    ) -> np.ndarray:
        """
        Overlay audio at a specific timestamp in the base audio.

        Args:
            base_audio: Base audio track (numpy array)
            overlay_audio: Audio to overlay (numpy array)
            timestamp: Timestamp in seconds where to place the overlay

        Returns:
            Modified base audio with overlay
        """
        # Calculate start sample position
        start_sample = int(timestamp * self.sample_rate)

        # Ensure we don't exceed the base audio length
        if start_sample >= len(base_audio):
            return base_audio

        # Calculate end sample position
        end_sample = min(start_sample + len(overlay_audio), len(base_audio))
        overlay_end = end_sample - start_sample

        # Overlay the audio (simple mixing)
        base_audio[start_sample:end_sample] += overlay_audio[:overlay_end]

        # Prevent clipping by normalizing if needed
        max_val = np.max(np.abs(base_audio))
        if max_val > 32767:  # Max int16 value
            base_audio = (base_audio * (32767 / max_val)).astype(np.int16)

        return base_audio

    def create_mixed_audio_track(
        self,
        duration: float,
        voice_messages: List[Tuple[float, str]],
        voice_messages_dir: str,
    ) -> Optional[np.ndarray]:
        """
        Create a mixed audio track with voice messages at specified timestamps.

        Args:
            duration: Total duration of the audio track in seconds
            voice_messages: List of (timestamp, message_file) tuples
            voice_messages_dir: Directory containing voice message files

        Returns:
            Mixed audio track as numpy array, or None if failed
        """
        if not self.pygame_initialized:
            logger.warning("Pygame not initialized, cannot create mixed audio track")
            return None

        try:
            # Create base silent track
            mixed_audio = self.create_silent_track(duration)

            # Sort voice messages by timestamp
            sorted_messages = sorted(voice_messages, key=lambda x: x[0])

            # Overlay each voice message
            for timestamp, message_file in sorted_messages:
                message_path = os.path.join(voice_messages_dir, message_file)

                if os.path.exists(message_path):
                    # Load the voice message audio
                    voice_audio = self.load_audio_file(message_path)

                    if voice_audio is not None:
                        # Overlay the voice message at the specified timestamp
                        mixed_audio = self.overlay_audio_at_timestamp(
                            mixed_audio, voice_audio, timestamp
                        )
                        logger.debug(
                            f"Overlaid voice message {message_file} at {timestamp:.2f}s"
                        )
                    else:
                        logger.warning(f"Could not load voice message: {message_file}")
                else:
                    logger.warning(f"Voice message file not found: {message_path}")

            return mixed_audio

        except Exception as e:
            logger.error(f"Error creating mixed audio track: {e}")
            return None

    def save_audio_track(self, audio_data: np.ndarray, output_path: str) -> bool:
        """
        Save audio data to a WAV file.

        Args:
            audio_data: Audio data as numpy array
            output_path: Path where to save the audio file

        Returns:
            True if successful, False otherwise
        """
        try:
            import wave

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save to WAV file
            with wave.open(output_path, "w") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            logger.info(f"Audio track saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving audio track to {output_path}: {e}")
            return False


class SynchronizedOutputManager:
    """Manager for synchronized video, audio, and data outputs with common timestamps."""

    def __init__(self, config: Dict[str, Any], base_output_dir: str):
        """
        Initialize the synchronized output manager.

        Args:
            config: Configuration dictionary
            base_output_dir: Base directory for all outputs
        """
        self.config = config
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create synchronized output directories
        self.video_dir = os.path.join(base_output_dir, "videos")
        self.audio_dir = os.path.join(base_output_dir, "audio")
        self.data_dir = os.path.join(base_output_dir, "velocity_calculations")

        for directory in [self.video_dir, self.audio_dir, self.data_dir]:
            os.makedirs(directory, exist_ok=True)

        # Audio recording setup
        self.voice_messages = []  # List of (timestamp, message_file) tuples
        self.voice_path = config["paths"]["voice_messages"]

        # Initialize audio mixer
        self.audio_mixer = AudioMixer()

    def get_base_filename(self, prefix: str = "") -> str:
        """
        Get base filename with timestamp.

        Args:
            prefix: Optional prefix for the filename

        Returns:
            Base filename with timestamp
        """
        if prefix:
            return f"{prefix}_{self.timestamp}"
        return self.timestamp

    def get_video_output_path(self, prefix: str = "", suffix: str = "") -> str:
        """
        Get path for video output file.

        Args:
            prefix: Optional prefix for the filename
            suffix: Optional suffix to add before the extension

        Returns:
            Full path for video output
        """
        filename = self.get_base_filename(prefix)
        if suffix:
            filename = f"{filename}_{suffix}"
        return os.path.join(self.video_dir, f"{filename}.mp4")

    def get_audio_output_path(self, prefix: str = "") -> str:
        """
        Get path for audio output file.

        Args:
            prefix: Optional prefix for the filename

        Returns:
            Full path for audio output
        """
        filename = self.get_base_filename(prefix)
        return os.path.join(self.audio_dir, f"{filename}.wav")

    def get_data_output_path(self, prefix: str = "", extension: str = "json") -> str:
        """
        Get path for data output file.

        Args:
            prefix: Optional prefix for the filename
            extension: File extension (default: json)

        Returns:
            Full path for data output
        """
        filename = self.get_base_filename(prefix)
        return os.path.join(self.data_dir, f"{filename}.{extension}")

    def record_voice_message(self, message_type: str, timestamp: float) -> None:
        """
        Record a voice message with timestamp.

        Args:
            message_type: Type of voice message (e.g., "start_workout", "form_correction")
            timestamp: Timestamp in seconds
        """
        # Map message types to actual audio files
        # Currently supported messages:
        # - "start_workout": Played at beginning of workout
        # - "form_correction": Played when form issues detected
        # - "form_correction_knee_valgus": Knee caving inward
        # - "form_correction_heel_lift": Lifting heels
        # - "form_correction_forward_lean": Leaning too far forward
        # - "form_correction_knee_caving": Knees caving inward
        # - "form_correction_limited_depth": Not going deep enough
        # - "form_correction_weight_drifting": Weight shifting forward
        # - "form_correction_lifting_heels": Lifting heels
        # - "form_correction_incomplete_depth": Not reaching bottom position
        # - "form_correction_insufficient_knee_bend": Not bending knees enough
        # - "form_correction_hyperextension": Over-arching back
        # - "form_correction_generic": Generic form correction
        # - "end_workout": Played at end of workout
        message_files = {
            "start_workout": "Corrections/start_workout.mp3",
            "form_correction_generic": "Corrections/generic.mp3",
            "form_correction_knee_valgus": "Corrections/knee_valgus.mp3",
            "form_correction_heel_lift": "Corrections/heel_lift.mp3",
            "form_correction_forward_lean": "Corrections/forward_lean.mp3",
            "form_correction_knee_caving": "Corrections/knee_valgus.mp3",
            "form_correction_limited_depth": "Corrections/limited_depth.mp3",
            "form_correction_weight_drifting": "Corrections/forward_lean.mp3",
            "form_correction_lifting_heels": "Corrections/heel_lift.mp3",
            "form_correction_knees_over_toes": "Corrections/knees_over_toes.mp3",
            "form_correction_incomplete_depth": "Corrections/limited_depth.mp3",
            "form_correction_insufficient_knee_bend": "Corrections/limited_depth.mp3",
            "form_correction_hyperextension": "Corrections/hyperextension.mp3",
            "end_workout": "Corrections/end_workout.mp3",
        }

        if message_type in message_files:
            message_file = message_files[message_type]
            self.voice_messages.append((timestamp, message_file))
            logger.debug(f"Recorded voice message: {message_type} at {timestamp:.2f}s")

    def create_audio_track(
        self, duration: float, voice_messages: List[Tuple[float, str]] = None
    ) -> Optional[str]:
        """
        Create an audio track with all recorded voice messages.

        Args:
            duration: Duration of the audio track in seconds
            voice_messages: List of (timestamp, message_type) tuples. If None, uses internal voice_messages.

        Returns:
            Path to the created audio file, or None if failed
        """
        # Use provided voice messages or fall back to internal ones
        messages_to_process = (
            voice_messages if voice_messages is not None else self.voice_messages
        )

        if not messages_to_process:
            logger.warning("No voice messages to process")
            return None

        try:
            # Create audio output path
            audio_output_path = self.get_audio_output_path("voice_track")

            # Create mixed audio track with all voice messages
            mixed_audio = self.audio_mixer.create_mixed_audio_track(
                duration, messages_to_process, self.voice_path
            )

            if mixed_audio is not None:
                # Save the mixed audio track
                if self.audio_mixer.save_audio_track(mixed_audio, audio_output_path):
                    logger.info(
                        f"Created audio track with {len(messages_to_process)} voice messages: {audio_output_path}"
                    )
                    return audio_output_path
                else:
                    logger.error("Failed to save mixed audio track")
                    return None
            else:
                logger.error("Failed to create mixed audio track")
                return None

        except Exception as e:
            logger.error(f"Error creating audio track: {e}")
            return None

    def merge_video_audio(
        self, video_path: str, audio_path: str, output_path: str
    ) -> bool:
        """
        Merge video and audio files using ffmpeg with CRF compression.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path for merged output

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use ffmpeg to merge video and audio
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(audio_path)

            # Merge video and audio with CRF compression
            output = ffmpeg.output(
                video_input["v"],  # Video stream
                audio_input["a"],  # Audio stream
                output_path,
                vcodec="libx264",  # Use H.264 encoder
                crf=23,  # CRF value for compression (18-28, lower = better quality)
                acodec="aac",  # Use AAC audio codec
            )

            # Run the ffmpeg command
            ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)

            logger.info(f"Successfully merged video and audio to: {output_path}")
            return True

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode('utf8') if e.stderr else str(e)
            logger.error(f"FFmpeg error during merge: {error_msg}")
            return False
        except Exception as e:
            logger.error(f"Error merging video and audio: {e}")
            return False

    def save_synchronized_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Save synchronized metadata with common timestamp.

        Args:
            metadata: Metadata to save

        Returns:
            Path to saved metadata file
        """
        metadata_path = self.get_data_output_path("metadata", "json")
        metadata["timestamp"] = self.timestamp
        metadata["created_at"] = datetime.now().isoformat()

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Synchronized metadata saved to: {metadata_path}")
        return metadata_path

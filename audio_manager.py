"""Audio manager module for industrial sound feedback.

Uses base64-encoded tones generated programmatically.
No external sound files needed.
"""

import base64
import io
import numpy as np
from typing import Optional


# Sound frequencies and durations for industrial tones
SOUNDS = {
    "click": {"freq": 800, "duration": 0.03, "volume": 0.15},
    "tick": {"freq": 600, "duration": 0.04, "volume": 0.12},
    "start": {"freq": 440, "duration": 0.15, "volume": 0.2},
    "success": {"freq": 880, "duration": 0.2, "volume": 0.25},
    "error": {"freq": 220, "duration": 0.3, "volume": 0.25},
    "complete": {"freq": 1200, "duration": 0.1, "volume": 0.2},
    "warning": {"freq": 350, "duration": 0.15, "volume": 0.2},
}


def generate_tone(frequency: float, duration: float, volume: float = 0.2,
                 sample_rate: int = 44100) -> bytes:
    """Generate a sine wave tone as WAV bytes.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        volume: Volume (0.0 to 1.0)
        sample_rate: Sample rate

    Returns:
        WAV file as bytes
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Sine wave with slight decay envelope
    envelope = np.exp(-t * 3)
    tone = np.sin(2 * np.pi * frequency * t) * envelope * volume
    # Convert to 16-bit integers
    tone = (tone * 32767).astype(np.int16)

    # Create WAV file
    buffer = io.BytesIO()
    import wave
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(tone.tobytes())

    buffer.seek(0)
    return buffer.read()


def get_sound_base64(sound_name: str) -> str:
    """Get base64 encoded sound for HTML audio.

    Args:
        sound_name: Name of sound (click, tick, start, success, error, etc.)

    Returns:
        Base64 encoded WAV data URI
    """
    if sound_name not in SOUNDS:
        sound_name = "click"

    params = SOUNDS[sound_name]
    wav_bytes = generate_tone(params["freq"], params["duration"], params["volume"])
    b64 = base64.b64encode(wav_bytes).decode()
    return f"data:audio/wav;base64,{b64}"


class AudioManager:
    """Audio manager for industrial sound feedback."""

    def __init__(self):
        self.muted = False
        self.volume_level = "medium"  # off, low, medium
        self.volume_multiplier = {"off": 0, "low": 0.3, "medium": 0.7}

    def set_volume(self, level: str):
        """Set volume level."""
        self.volume_level = level

    def toggle_mute(self):
        """Toggle mute state."""
        self.muted = not self.muted
        return self.muted

    def is_muted(self) -> bool:
        """Check if audio is muted."""
        return self.muted

    def play(self, sound_name: str) -> Optional[str]:
        """Get base64 sound data for playing.

        Args:
            sound_name: Name of sound to play

        Returns:
            Base64 data URI or None if muted
        """
        if self.muted:
            return None

        return get_sound_base64(sound_name)

    def get_all_sounds(self) -> dict:
        """Get all sounds as base64 dict."""
        if self.muted:
            return {}

        return {name: get_sound_base64(name) for name in SOUNDS.keys()}


# Global instance
_audio_manager = AudioManager()


def get_audio_manager() -> AudioManager:
    """Get global audio manager instance."""
    return _audio_manager


def play_sound(sound_name: str) -> Optional[str]:
    """Play a sound by name. Returns base64 data URI."""
    return _audio_manager.play(sound_name)


def toggle_audio() -> bool:
    """Toggle mute and return new state."""
    return _audio_manager.toggle_mute()
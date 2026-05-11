"""Real-time microphone capture for live transcription.

Uses ``sounddevice`` to pull audio from the default input device and
buffers it for the transcriber.
"""
from __future__ import annotations

import logging
import queue
import sys
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class MicrophoneListener:
    """Listens to the default microphone and yields audio chunks.

    Parameters
    ----------
    sample_rate:
        Target sample rate. Whisper expects 16000.
    block_duration:
        Duration of each audio block in seconds.
    """

    def __init__(self, sample_rate: int = 16_000, block_duration: float = 0.5):
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_duration)
        self.queue = queue.Queue()
        self._stream = None

    def _callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackStatus):
        """This is called (from a separate thread) for each audio block."""
        if status:
            logger.warning("Stream status: %s", status)
        # indata is (frames, channels). We want mono, so take the first channel.
        self.queue.put(indata[:, 0].copy())

    def start(self):
        """Start the audio stream."""
        if self._stream is not None:
            return
        
        logger.info("Starting microphone stream at %d Hz...", self.sample_rate)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback
        )
        self._stream.start()

    def stop(self):
        """Stop the audio stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Microphone stream stopped.")

    def listen(self, chunk_duration: float = 3.0):
        """Generator that yields audio chunks of *chunk_duration* seconds.

        This will block until enough data is collected for a chunk.
        """
        blocks_per_chunk = int(chunk_duration / (self.block_size / self.sample_rate))
        if blocks_per_chunk < 1:
            blocks_per_chunk = 1

        buffer = []
        while True:
            try:
                # Block until we get a piece of audio
                block = self.queue.get()
                buffer.append(block)

                if len(buffer) >= blocks_per_chunk:
                    # Concatenate blocks into a single chunk
                    chunk = np.concatenate(buffer)
                    yield chunk
                    buffer = []
            except KeyboardInterrupt:
                break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

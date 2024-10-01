import numpy as np
import sounddevice as sd
import time
import threading

from rich.console import Console
from queue import Queue
from whisper.model import Whisper


def record_audio(stop_event: threading.Event, data_queue: Queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    Args:
        stop_event: An event that, when set, signals the function to stop recording.
        data_queue: A queue to which the recorded audio data will be added.
    Returns:
        None
    """

    def callback(indata, frames, time, status):
        # if status:
        #     console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray, speech_to_text: Whisper) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.
    Args:
        audio_np: The audio data to be transcribed.
    Returns:
        str: The transcribed text.
    """
    result = speech_to_text.transcribe(
        audio_np, fp16=False
    )  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


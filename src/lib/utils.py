
import time
from typing import Any, BinaryIO, Callable, Tuple

import librosa
import numpy as np
from numpy import ndarray


def time_function(func: Callable[..., Any], *args, **kwargs) -> Tuple[Any, float]:
    """
    Times how long a function takes to execute and returns its result and the time taken.

    Args:
        func: The function to be timed.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        A tuple containing:
            - The result of the executed function.
            - The time taken for the function to execute, in seconds.
    """
    start_time = time.perf_counter()  # Use perf_counter for more precise timing
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    return result, time_taken


def get_sample_rate(
    audio: np.ndarray
) -> float:
    """
    Gets the sample rate of a given audio
    """

    return librosa.get_samplerate(
        audio
    )

def get_audio_from_file(file: BinaryIO) -> tuple[np.ndarray, int]:
    """
    Gets the raw audio and sample rate from a .wav file
    """
    return librosa.load(
        file,
        mono=False
    )

def resample_audio(
    audio: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int,
) -> ndarray:
    """
    Takes in audio at a given sample rate then
    """

    return librosa.resample(
        audio,
        orig_sr=original_sample_rate,
        target_sr=target_sample_rate
    )

def get_duration(
    audio: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Gets the duration of the audio provided
    """

    return librosa.get_duration(
        y=audio, sr=sample_rate
    )
import io
import os
from pathlib import Path
import pickle
from typing import IO, Generator, Tuple
import librosa
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
import pyloudnorm as pyln
import bentoml
import numpy as np
import torch
import logging 
from silero_vad import load_silero_vad

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NORMALIZATION_FILES_DIR = os.getcwd() + '/common/normalization'
MED_NORM_PICKLE = pickle.load(open(os.path.join(NORMALIZATION_FILES_DIR, 'MED_HumBug_paper1_paper2_norm_mel.pickle'), 'rb'))
MSC_NORM_PICKLE = pickle.load(open(os.path.join(NORMALIZATION_FILES_DIR, 'MSC_norm_mel_20250305_new2new4_species_9_class.pickle'), 'rb'))
MEAN_STD_STORE = {
    "MED": {"mean": MED_NORM_PICKLE['mean'], "std": MED_NORM_PICKLE['std']},
    "MSC": {"mean": MSC_NORM_PICKLE['mean'], "std": MSC_NORM_PICKLE['std']},
}
VAD_MODEL = load_silero_vad()

def _move_to_device(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """Helper to move numpy array to a torch tensor on the correct device."""
    if 'float' in str(x.dtype):
        return torch.Tensor(x).to(device)
    if 'int' in str(x.dtype):
        return torch.LongTensor(x).to(device)
    return torch.tensor(x).to(device)


def _reshape_feat(feats: np.ndarray, frames_per_second: int, mel_bins: int) -> np.ndarray:
    """
    Stateless version of reshape_feat.
    Windows a feature array.
    """
    if feats.shape[0] < frames_per_second:
        logger.warning(f"Feature length ({feats.shape[0]}) is shorter than frames_per_second ({frames_per_second}). Returning empty array.")
        return np.array([])
        
    # Original function expected a list of features, we adapt for a single feature array
    feats_windowed = np.lib.stride_tricks.as_strided(
        feats,
        shape=(
            (feats.shape[0] - frames_per_second) // frames_per_second + 1,
            frames_per_second,
            mel_bins
        ),
        strides=(
            frames_per_second * feats.strides[0],
            feats.strides[0],
            feats.strides[1]
        )
    )
    return feats_windowed[:, np.newaxis, :, :] # Add channel dimension


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax function."""
    if x.ndim == 2:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    return x # Should not happen

class Preprocessor:
    SUPPORT_CPU_MULTI_THREADING = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else "mps" if torch.mps.is_available() else 'cpu')

    """
    Instantiated per model to handle audio preprocessing tasks.
    """
    def __init__(self, sample_rate: int, mel_bins: int = 64, normalization_type: str = '12db', vad_threshold = 0.2):
        self.normalization_type = normalization_type

        self.window_size = int(np.floor(1024 * (sample_rate / 32000)))
        self.step_size = int(np.floor(320 * (sample_rate / 32000)))

        self.vad_threshold = vad_threshold

        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.spectrogram_extractor = Spectrogram(
            n_fft=self.window_size, hop_length=self.step_size, win_length=self.window_size,
            window='hann', center=True, pad_mode='reflect', freeze_parameters=True
        ).eval().to(self.device)

        self.logmel_extractor = LogmelFilterBank(
            sr=self.sample_rate, n_fft=self.window_size, n_mels=self.mel_bins,
            fmin=50, fmax=int(self.sample_rate / 2), freeze_parameters=True, top_db=None # type: ignore
        ).eval().to(self.device)

    def extract_features(self, resampled_audio: np.ndarray) -> np.ndarray:
        """
        Extracts features from the input audio tensor.
        This is a GPU-bound task.
        """
        logger.info(f"Extracting features from input tensor with shape: {resampled_audio.shape}, dtype: {resampled_audio.dtype}, device: {resampled_audio.device}")
        normalized_audio = self._normalize_loudness(resampled_audio, normalization_type='12db')

        logmel_features = self._extract_logmel(normalized_audio) 
        logger.info(f"Extracted logmel features with shape: {logmel_features.shape}, dtype: {logmel_features.dtype}, device: {logmel_features.device}")

        return logmel_features
    
    def _reshape_features(self, features: np.ndarray) -> np.ndarray:   
        logger.info(f"Reshaping features with shape: {features.shape}, dtype: {features.dtype}")

        """
        For a sample rate of 16000 Hz, the preprocessing parameters are:

        window_size = 512 samples (calculated as 1024 * (16000/32000))
        step_size = 160 samples (calculated as 320 * (16000/32000))
        The step_size determines how many audio samples we move for each frame of the spectrogram:

        Each frame represents 160/16000 = 0.01 seconds (10ms) of audio
        Therefore, in the spectrogram/feature domain:

        100 frames * 10ms/frame = 1000ms = 1 second
        """
        
        frames_per_second = int(self.sample_rate / self.step_size) # Ensures that regardless of the sample rate, we get a second per window
        reshaped = _reshape_feat(features, frames_per_second=frames_per_second, mel_bins=self.mel_bins)
        if reshaped.size == 0:
            logger.warning("Reshaped features are empty. Returning empty array.")
            return np.array([])
        
        logger.info(f"Reshaped features to shape: {reshaped.shape}, dtype: {reshaped.dtype}")
        return reshaped
    
    def prepare_for_event_detection(self, extracted_features: np.ndarray) -> torch.Tensor:
        reshaped_feature_window = self._reshape_features(extracted_features)

        mean, std = MEAN_STD_STORE["MED"]['mean'], MEAN_STD_STORE["MED"]['std']
        med_batch =  (reshaped_feature_window - mean) / std
        return _move_to_device(med_batch, self.device)  

    def detect_voices(self, input: np.ndarray, logmel_features: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects voices in the input audio tensor.
        This is a GPU-bound task.
        """
        assert sample_rate == 16000, "Sample rate must be 16000 for VAD model compatibility"

        logger.info(f"Detecting voices in input tensor with shape: {input.shape}, dtype: {input.dtype}, device: {input.device}")

        vad_mask = np.ones(logmel_features.shape[0], dtype=np.int8) 
        wav_tensor = _move_to_device(input, self.device)

        vad_predictions: np.ndarray = VAD_MODEL.audio_forward(wav_tensor, sr=sample_rate)[0].numpy()
        logger.info(f"VAD predictions shape: {vad_predictions.shape}, dtype: {vad_predictions.dtype}")
        vad_binary = (vad_predictions >= self.vad_threshold).astype(np.int8)

        mel_frames, vad_frames = logmel_features.shape[0], len(vad_binary)
        if vad_frames == 0:
            return vad_predictions, np.zeros(mel_frames, dtype=np.int8)

        time_ratio = mel_frames / vad_frames
        repeats = (np.ceil(np.arange(1, vad_frames + 1) * time_ratio) - 
           np.ceil(np.arange(vad_frames) * time_ratio)).astype(np.int8)
        vad_mask = np.repeat(vad_binary, repeats)
        
        if len(vad_mask) > mel_frames:
            vad_mask = vad_mask[:mel_frames]
        elif len(vad_mask) < mel_frames:
            vad_mask = np.pad(vad_mask, (0, mel_frames - len(vad_mask)), 'edge')
            
        return vad_predictions, vad_mask
    
    def remove_voice_from_med_predictions(self, med_predictions: np.ndarray, vad_mask: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Finds voice predictions in MED model predictions based on VAD mask.

        This is done by iterating through batches of the MED predictions and checking the corresponding VAD mask.
        So for each MED prediction, there will be an array of VAD predictions [onset:offset] that corresponds to the time period of the MED prediction.

        We are adding 

        Returns a boolean array indicating where voices are detected, the number of elements as in med_predictions.
        """

        time_ratio = len(vad_mask) / len(med_predictions)
        vad_med_mask = []
        med_results_with_voices: list[bool] = []
        onset = 0
        for i, pred in enumerate(med_predictions):
            offset = int(np.ceil(time_ratio * (i + 1)))
            if len(vad_mask[onset:offset]) > 0 and np.mean(vad_mask[onset:offset]) >= 0.5:  # VAD Alignment threshold
                vad_med_mask.append(float(0))
                med_results_with_voices.append(True)
            else:
                vad_med_mask.append(pred[0])
                med_results_with_voices.append(False)
            onset = offset

        vad_med_mask = np.array(vad_med_mask, dtype=np.float32)
        return vad_med_mask, med_results_with_voices

    def does_file_require_resampling(self, audio_path: Path, target_sr: int) -> bool:
        path_str = str(audio_path)
        original_sample_rate = librosa.get_samplerate(path=path_str)
        logger.info(f"Detected original sample rate: {original_sample_rate}")
        return original_sample_rate != target_sr

    def load_audio(self, audio_path: Path, sample_rate: float, chunk_duration_seconds: int) -> Generator[np.ndarray]:
        frame_length = int(sample_rate * chunk_duration_seconds )
        hop_length = frame_length  # No overlap between chunks
        audio_stream = librosa.stream(
            str(audio_path),
            block_length=1,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        
        logger.info(f"Loading audio from {audio_path} with sample rate: {sample_rate}")
        yield from audio_stream

    def resample(self, audio_path: Path, target_sr: int, chunk_duration_seconds: int) -> Generator[np.ndarray]:
        """
        Loads audio from a file path, resamples it, and saves it to a new temporary file.
        Returns the path to the resampled file.
        """
        original_sample_rate = librosa.get_samplerate(str(audio_path))
        audio_stream = self.load_audio(audio_path, sample_rate=original_sample_rate, chunk_duration_seconds=chunk_duration_seconds)

        logger.info(f"Resampling audio with target sample rate: {target_sr}")
        for i, audio_chunk in enumerate(audio_stream):
            resampled_chunk = audio_chunk

            try: 
                resampled_chunk = librosa.resample(
                    y=audio_chunk, orig_sr=original_sample_rate, target_sr=target_sr, res_type='kaiser_fast'
                )
                logger.debug(f"Resampled chunk {i} to target sample rate: {target_sr}")
                yield resampled_chunk
            except librosa.util.exceptions.ParameterError as e:
                logger.error(f"Parameter error resampling audio chunk {i}: {e}")
                raise e
            except Exception as e:
                logger.error(f"Error resampling audio chunk {i}: {e}")
                raise e
            
    def _normalize_loudness(self, audio_data: np.ndarray, normalization_type: str) -> np.ndarray:
        if normalization_type == 'peak':
            return pyln.normalize.peak(audio_data, -1.0)
        if normalization_type == '12db':
            meter = pyln.Meter(self.sample_rate)
            loudness = meter.integrated_loudness(audio_data)
            return pyln.normalize.loudness(audio_data, loudness, -12.0)
        return audio_data

    def _extract_logmel(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extracts log-mel spectrogram features from the audio data.
        """

        # Ensure audio data is in the correct shape (2D array with two channels)
        if audio_data.ndim == 1:
            audio_data = np.array([audio_data, audio_data])
    
        with torch.no_grad():
            x_gpu = _move_to_device(audio_data, self.device)
            spectrogram = self.spectrogram_extractor(x_gpu)
            logmel = self.logmel_extractor(spectrogram)
            logmel = logmel[:, 0].data.cpu().numpy()
        return logmel[0]


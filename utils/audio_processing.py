import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr

def load_audio(file_path, target_sr=16000):
    """Load audio file and resample if needed"""
    audio, sr = sf.read(file_path)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

def reduce_noise(audio, sr):
    """Apply noise reduction to audio"""
    return nr.reduce_noise(y=audio, sr=sr)

def normalize_audio(audio):
    """Normalize audio to [-1, 1] range"""
    return audio / np.max(np.abs(audio))

def trim_silence(audio, top_db=20):
    """Trim leading/trailing silence from audio"""
    return librosa.effects.trim(audio, top_db=top_db)[0]

def preprocess_audio(file_path, target_sr=16000):
    """Full audio preprocessing pipeline"""
    audio, sr = load_audio(file_path, target_sr)
    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sr)
    audio = trim_silence(audio)
    return audio, sr
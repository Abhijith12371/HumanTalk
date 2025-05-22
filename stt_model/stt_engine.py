import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from utils.logger import setup_logger

logger = setup_logger(__name__)

class STTEngine:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", audio_params=None, decoding_params=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing STT on device: {self.device}")
        
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        
        # Audio parameters
        self.audio_params = audio_params or {
            "sample_rate": 16000,
            "chunk_length": 10,
            "stride_length": 4
        }
        
        # Decoding parameters
        self.decoding_params = decoding_params or {
            "beam_width": 5,
            "lm_weight": 1.0,
            "word_score": -1.0
        }

    def speech_to_text(self, audio_path):
        """Transcribe speech audio to text"""
        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != self.audio_params["sample_rate"]:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.audio_params["sample_rate"]
                )
                waveform = resampler(waveform)
            
            # Process audio
            inputs = self.processor(
                waveform.squeeze().numpy(), 
                sampling_rate=self.audio_params["sample_rate"], 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Run through model
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(
                predicted_ids,
                beam_width=self.decoding_params["beam_width"],
                lm_weight=self.decoding_params["lm_weight"],
                word_score=self.decoding_params["word_score"]
            )
            return transcription[0]
        
        except Exception as e:
            logger.error(f"Error in speech recognition: {str(e)}")
            return ""
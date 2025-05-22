from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.logger import setup_logger

logger = setup_logger(__name__)

class EmotionDetector:
    def __init__(self, model_name="bert-base-uncased", num_labels=7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Emotion Detector on device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        self.labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    def detect_emotion(self, text):
        """Detect emotion from text with confidence score"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probabilities = torch.softmax(outputs.logits, dim=1)
            emotion_idx = torch.argmax(probabilities).item()
            return self.labels[emotion_idx], probabilities[0][emotion_idx].item()
        
        except Exception as e:
            logger.error(f"Error detecting emotion: {str(e)}")
            return "neutral", 0.0
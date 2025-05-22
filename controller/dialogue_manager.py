# controller/dialogue_manager.py
from typing import Optional
import time

class DialogueManager:
    def __init__(self):
        from nlp_model.llm_chat import ConversationalLLM
        from stt_model.stt_engine import STTEngine
        from tts_model.tts_engine import TTSEngine
        from emotion_classifier.emotion_detector import EmotionDetector
        from context_memory.memory_manager import MemoryManager
        
        # Initialize components
        self.llm = ConversationalLLM()
        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.emotion_detector = EmotionDetector()
        self.memory = MemoryManager()
        
        # Conversation state
        self.current_user = None
        self.conversation_history = []
    
    def process_input(self, audio_path: str, user_id: str = "default"):
        """Main pipeline: voice → text → emotion → response → voice"""
        self.current_user = user_id
        
        # Step 1: Speech to Text
        user_text = self.stt.speech_to_text(audio_path)
        
        # Step 2: Detect Emotion
        emotion, confidence = self.emotion_detector.detect_emotion(user_text)
        
        # Step 3: Retrieve Context
        relevant_context = self.memory.retrieve_context(user_id, user_text)
        
        # Step 4: Generate Response
        prompt = self._format_prompt(user_text, emotion, relevant_context)
        ai_response = self.llm.generate_response(prompt)
        
        # Step 5: Text to Speech with emotion
        speech_output = self.tts.text_to_speech(ai_response, emotion=emotion)
        
        # Store conversation
        self._store_conversation(user_text, ai_response, emotion)
        
        return {
            "user_text": user_text,
            "ai_response": ai_response,
            "emotion": emotion,
            "confidence": confidence,
            "speech_output": speech_output
        }
    
    def _format_prompt(self, user_text, emotion, context):
        """Format the prompt with context and emotion"""
        context_str = "\n".join([f"Previous conversation: {text}" for text, _ in context])
        return f"""
        [Context]
        {context_str}
        
        [Emotion]
        The user seems to be feeling {emotion}.
        
        [Current Message]
        User: {user_text}
        
        Please respond naturally to the user's message, taking into account their apparent emotion and the conversation context.
        Response:
        """
    
    def _store_conversation(self, user_text, ai_response, emotion):
        """Store the conversation in memory"""
        timestamp = time.time()
        self.memory.store_conversation(self.current_user, user_text, timestamp)
        self.memory.store_conversation(self.current_user, ai_response, timestamp + 1)
        
        # Also keep local history
        self.conversation_history.append({
            "user": user_text,
            "ai": ai_response,
            "emotion": emotion,
            "timestamp": timestamp
        })
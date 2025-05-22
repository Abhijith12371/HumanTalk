#!/usr/bin/env python3
from controller.dialogue_manager import DialogueManager
import sounddevice as sd
import numpy as np
import argparse
import time
import os
from utils.logger import setup_logger

logger = setup_logger(__name__)

class HumanTalkerAI:
    def __init__(self, config_path="configs/"):
        self.dialogue_manager = DialogueManager(config_path)
        self.sample_rate = 22050  # Should match TTS output sample rate
        self.audio_temp_dir = "data/speech/temp/"
        os.makedirs(self.audio_temp_dir, exist_ok=True)

    def run_interactive(self):
        """Run in interactive mode with microphone input"""
        print("\nHumanTalker AI - Interactive Mode")
        print("Press Ctrl+C to exit\n")
        
        try:
            while True:
                # Record audio from microphone
                print("Listening... (press Ctrl+C to stop)")
                audio = self._record_audio()
                
                # Save temporary audio file
                temp_path = os.path.join(self.audio_temp_dir, f"temp_{int(time.time())}.wav")
                self._save_audio(audio, temp_path)
                
                # Process the audio
                result = self.dialogue_manager.process_input(temp_path)
                
                # Output results
                print(f"\nUser: {result['user_text']}")
                print(f"AI: {result['ai_response']}")
                print(f"Detected Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
                
                # Play the AI response
                print("Playing response...")
                sd.play(result['speech_output'], self.sample_rate)
                sd.wait()
                
        except KeyboardInterrupt:
            print("\nExiting HumanTalker AI...")
            logger.info("Application closed by user")

    def _record_audio(self, max_duration=10, sample_rate=16000):
        """Record audio from microphone"""
        logger.info("Starting audio recording...")
        audio = sd.rec(int(max_duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype='float32')
        sd.wait()  # Wait until recording is finished
        return audio

    def _save_audio(self, audio, path, sample_rate=16000):
        """Save audio to file"""
        import soundfile as sf
        sf.write(path, audio, sample_rate)
        logger.info(f"Audio saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HumanTalker AI System")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode with microphone")
    parser.add_argument("--config", default="configs/", help="Path to config directory")
    args = parser.parse_args()
    
    ai = HumanTalkerAI(args.config)
    
    if args.interactive:
        ai.run_interactive()
    else:
        print("Please specify a mode (--interactive)")
        logger.warning("No run mode specified")
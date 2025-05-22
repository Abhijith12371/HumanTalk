import unittest
from nlp_model.llm_chat import ConversationalLLM
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TestConversationalLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = ConversationalLLM(use_lora=False)  # Use smaller model for testing
    
    def test_response_generation(self):
        """Test that the LLM generates a response"""
        prompt = "Hello, how are you?"
        response = self.llm.generate_response(prompt, max_length=50)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        logger.info(f"Generated response: {response}")
    
    def test_disfluencies(self):
        """Test that disfluencies are sometimes added"""
        prompt = "What's your opinion on AI?"
        responses = [self.llm._add_disfluencies("I think AI is amazing") for _ in range(10)]
        has_disfluency = any("uh" in r or "um" in r or "you know" in r for r in responses)
        self.assertTrue(has_disfluency)  # At least one should have disfluency
        logger.info(f"Disfluency test responses: {responses}")

if __name__ == "__main__":
    unittest.main()
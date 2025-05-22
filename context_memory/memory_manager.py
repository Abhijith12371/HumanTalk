import redis
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MemoryManager:
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            self.redis = redis.Redis(host=host, port=port, db=db)
            self.redis.ping()  # Test connection
            logger.info("Connected to Redis server")
        except redis.ConnectionError:
            logger.error("Failed to connect to Redis")
            raise
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def store_conversation(self, user_id, text, timestamp=None):
        """Store conversation with embedding in Redis"""
        try:
            if not text.strip():
                return
            
            timestamp = timestamp or datetime.now().timestamp()
            embedding = self.encoder.encode(text)
            
            # Store as JSON
            data = {
                "text": text,
                "timestamp": timestamp,
                "embedding": embedding.tolist()
            }
            
            self.redis.rpush(f"user:{user_id}:conversations", json.dumps(data))
            logger.debug(f"Stored conversation for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")

    def retrieve_context(self, user_id, query_text, top_k=3):
        """Retrieve most relevant conversations using vector similarity"""
        try:
            # Encode query
            query_embedding = self.encoder.encode(query_text)
            
            # Get all conversations for user
            conversations = self.redis.lrange(f"user:{user_id}:conversations", 0, -1)
            if not conversations:
                return []
            
            # Parse and calculate similarities
            parsed_conversations = []
            similarities = []
            
            for conv in conversations:
                data = json.loads(conv)
                embedding = np.array(data["embedding"])
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                parsed_conversations.append(data)
                similarities.append(similarity)
            
            # Get top-k most similar
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [(parsed_conversations[i]["text"], similarities[i]) for i in top_indices]
        
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
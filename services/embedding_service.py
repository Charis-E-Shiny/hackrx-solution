import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List
import os

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode list of texts to embeddings"""
        try:
            if not texts:
                return np.array([])
            
            # Convert texts to embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to embedding"""
        try:
            embedding = self.model.encode([query], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to encode query: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()

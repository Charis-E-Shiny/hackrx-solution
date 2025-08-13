import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List
import re

logger = logging.getLogger(__name__)

class SimpleEmbeddingService:
    """
    Free alternative to sentence-transformers using TF-IDF vectorization
    This provides good semantic similarity without requiring external ML models
    """
    
    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            lowercase=True,
            strip_accents='ascii'
        )
        self.is_fitted = False
        self.document_vectors = None
        
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)
        return text.lower().strip()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit the vectorizer on texts and transform them to vectors"""
        try:
            if not texts:
                return np.array([])
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Fit and transform
            self.document_vectors = self.vectorizer.fit_transform(processed_texts)
            self.is_fitted = True
            
            logger.info(f"Fitted TF-IDF vectorizer on {len(texts)} documents with {self.document_vectors.shape[1]} features")
            
            return self.document_vectors.toarray()
            
        except Exception as e:
            logger.error(f"Failed to fit and transform texts: {str(e)}")
            raise
    
    def transform_query(self, query: str) -> np.ndarray:
        """Transform a query to vector using fitted vectorizer"""
        try:
            if not self.is_fitted:
                raise ValueError("Vectorizer not fitted yet")
            
            processed_query = self.preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
            
            return query_vector.toarray()[0]
            
        except Exception as e:
            logger.error(f"Failed to transform query: {str(e)}")
            raise
    
    def compute_similarity(self, query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents"""
        try:
            if document_vectors.ndim == 1:
                document_vectors = document_vectors.reshape(1, -1)
            
            query_vector = query_vector.reshape(1, -1)
            
            similarities = cosine_similarity(query_vector, document_vectors)[0]
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the feature vectors"""
        if self.is_fitted and self.document_vectors is not None:
            return self.document_vectors.shape[1]
        return self.max_features
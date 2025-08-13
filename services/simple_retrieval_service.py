import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Tuple
from models import RetrievalResult

logger = logging.getLogger(__name__)

class SimpleRetrievalService:
    """
    Free alternative to FAISS using basic numpy operations and TF-IDF
    This provides good retrieval performance without external dependencies
    """
    
    def __init__(self, embedding_service, index_path: str = "./simple_index"):
        self.embedding_service = embedding_service
        self.index_path = index_path
        self.document_vectors = None
        self.documents = []
        self.is_trained = False
        
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to the index"""
        try:
            if not chunks:
                logger.warning("No chunks provided to add to index")
                return
            
            # Extract text content for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate TF-IDF vectors
            self.document_vectors = self.embedding_service.fit_transform(texts)
            
            # Store document chunks with metadata
            self.documents = []
            for i, chunk in enumerate(chunks):
                chunk_with_metadata = {
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'vector_index': i
                }
                self.documents.append(chunk_with_metadata)
            
            self.is_trained = True
            logger.info(f"Added {len(chunks)} documents to index")
            
        except Exception as e:
            logger.error(f"Failed to add documents to index: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for relevant documents given a query"""
        try:
            if not self.is_trained:
                logger.warning("Index not trained, cannot perform search")
                return []
            
            # Generate query vector
            query_vector = self.embedding_service.transform_query(query)
            
            # Compute similarities with all documents
            similarities = self.embedding_service.compute_similarity(
                query_vector, 
                self.document_vectors
            )
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:min(top_k, len(self.documents))]
            
            # Prepare results
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include relevant results
                    doc = self.documents[idx]
                    result = RetrievalResult(
                        content=doc['content'],
                        score=float(similarities[idx]),
                        metadata=doc['metadata']
                    )
                    results.append(result)
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            raise
    
    def save_index(self, path: str = None):
        """Save the index and documents to disk"""
        try:
            save_path = path or self.index_path
            os.makedirs(save_path, exist_ok=True)
            
            # Save document vectors
            if self.document_vectors is not None:
                np.save(os.path.join(save_path, "document_vectors.npy"), self.document_vectors)
            
            # Save documents metadata
            with open(os.path.join(save_path, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
            
            # Save vectorizer state
            with open(os.path.join(save_path, "vectorizer.pkl"), "wb") as f:
                pickle.dump(self.embedding_service.vectorizer, f)
            
            logger.info(f"Saved index to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise
    
    def load_index(self, path: str = None):
        """Load the index and documents from disk"""
        try:
            load_path = path or self.index_path
            vectors_file = os.path.join(load_path, "document_vectors.npy")
            docs_file = os.path.join(load_path, "documents.pkl")
            vectorizer_file = os.path.join(load_path, "vectorizer.pkl")
            
            if all(os.path.exists(f) for f in [vectors_file, docs_file, vectorizer_file]):
                self.document_vectors = np.load(vectors_file)
                
                with open(docs_file, "rb") as f:
                    self.documents = pickle.load(f)
                
                with open(vectorizer_file, "rb") as f:
                    self.embedding_service.vectorizer = pickle.load(f)
                    self.embedding_service.is_fitted = True
                
                self.is_trained = True
                logger.info(f"Loaded index from {load_path}")
                return True
            else:
                logger.info("No existing index found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return False
    
    def clear_index(self):
        """Clear the current index and documents"""
        self.document_vectors = None
        self.documents = []
        self.is_trained = False
        self.embedding_service.is_fitted = False
        logger.info("Cleared index and documents")
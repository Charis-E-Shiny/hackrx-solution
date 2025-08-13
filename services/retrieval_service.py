import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Tuple
from models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, embedding_service, index_path: str = "./faiss_index"):
        self.embedding_service = embedding_service
        self.index_path = index_path
        self.index = None
        self.documents = []
        self.is_trained = False
        
    def _create_index(self, dimension: int):
        """Create a new FAISS index"""
        # Using IndexFlatIP for cosine similarity (Inner Product)
        self.index = faiss.IndexFlatIP(dimension)
        logger.info(f"Created new FAISS index with dimension {dimension}")
    
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to the index"""
        try:
            if not chunks:
                logger.warning("No chunks provided to add to index")
                return
            
            # Extract text content for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_service.encode_texts(texts)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create index if it doesn't exist
            if self.index is None:
                self._create_index(embeddings.shape[1])
            
            # Add embeddings to index
            self.index.add(embeddings.astype('float32'))
            
            # Store document chunks
            for i, chunk in enumerate(chunks):
                chunk_with_embedding = {
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'embedding': embeddings[i].tolist()
                }
                self.documents.append(chunk_with_embedding)
            
            self.is_trained = True
            logger.info(f"Added {len(chunks)} documents to FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to add documents to index: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for relevant documents given a query"""
        try:
            if not self.is_trained:
                logger.warning("Index not trained, cannot perform search")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_service.encode_query(query)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search in index
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):  # Valid index
                    doc = self.documents[idx]
                    result = RetrievalResult(
                        content=doc['content'],
                        score=float(score),
                        metadata=doc['metadata']
                    )
                    results.append(result)
            
            # Sort by score (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            raise
    
    def save_index(self, path: str = None):
        """Save the FAISS index and documents to disk"""
        try:
            save_path = path or self.index_path
            os.makedirs(save_path, exist_ok=True)
            
            if self.index is not None:
                faiss.write_index(self.index, os.path.join(save_path, "index.faiss"))
            
            with open(os.path.join(save_path, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Saved index to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise
    
    def load_index(self, path: str = None):
        """Load the FAISS index and documents from disk"""
        try:
            load_path = path or self.index_path
            index_file = os.path.join(load_path, "index.faiss")
            docs_file = os.path.join(load_path, "documents.pkl")
            
            if os.path.exists(index_file) and os.path.exists(docs_file):
                self.index = faiss.read_index(index_file)
                
                with open(docs_file, "rb") as f:
                    self.documents = pickle.load(f)
                
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
        self.index = None
        self.documents = []
        self.is_trained = False
        logger.info("Cleared index and documents")

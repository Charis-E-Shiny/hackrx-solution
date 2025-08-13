import os

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    
    # Authentication
    BEARER_TOKEN = "94adf1bd4f8978d3029fd88cb36c7f95a255d331301920caa50830f5f6c3ba26"
    
    # Model Configuration
    EMBEDDING_MODEL = "tfidf"  # Using TF-IDF for free embedding alternative
    LLM_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
    
    # Retrieval Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 5
    
    # Index Configuration
    FAISS_INDEX_PATH = "./simple_index"
    
    # PDF Processing
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

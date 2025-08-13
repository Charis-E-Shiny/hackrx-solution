from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    documents: str  # Blob URL for the PDF document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DocumentChunk(BaseModel):
    content: str
    metadata: dict
    embedding: Optional[List[float]] = None

class RetrievalResult(BaseModel):
    content: str
    score: float
    metadata: dict

class LLMResponse(BaseModel):
    answer: str
    reasoning: str
    sources: List[str]

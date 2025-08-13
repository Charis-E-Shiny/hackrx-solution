/////DEEPSEEK////
import os
import re
import time
import json
import uuid
import requests
import numpy as np
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
import hashlib
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import PhraseMatcher
import pinecone
import neo4j
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline

# --------------------------
# Configuration & Constants
# --------------------------

class Config:
    EMBEDDING_MODEL = "all-mpnet-base-v2"  # High-quality sentence embeddings
    SIMILARITY_THRESHOLD = 0.82  # Minimum similarity score for matches
    MAX_CONTEXT_LENGTH = 1000  # Characters for context window
    MIN_ANSWER_LENGTH = 15  # Minimum characters for valid answer
    MAX_PARALLEL_REQUESTS = 5
    CACHE_EXPIRY_SECONDS = 3600
    NER_MODEL = "en_core_web_lg"  # spaCy large model for NER
    QA_MODEL = "deepset/roberta-base-squad2"  # QA model
    PINECONE_INDEX = "document-knowledge"
    NEO4J_LABELS = {"ENTITY": "Entity", "DOCUMENT": "Document", "CLAUSE": "Clause"}
    DOCUMENT_TYPES = ["policy", "contract", "claim", "report"]

# Initialize models early to load them once
nlp = spacy.load(Config.NER_MODEL)
embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
qa_pipeline = pipeline('question-answering', model=Config.QA_MODEL)

# --------------------------
# Data Models
# --------------------------

class DocumentMetadata(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    doc_type: str
    source: str
    effective_date: Optional[str]
    expiration_date: Optional[str]
    entities: List[Dict] = []
    sections: List[Dict] = []
    version: str = "1.0"

class QuestionRequest(BaseModel):
    documents: List[str]  # URLs or document IDs
    questions: List[str]
    context: Optional[Dict] = None  # Additional context for the query
    user_id: Optional[str] = None  # For personalization

class AnswerResponse(BaseModel):
    answers: List[Dict]
    confidence_scores: List[float]
    sources: List[List[Dict]]
    processing_time_ms: float
    debug_info: Optional[Dict] = None

class DocumentProcessingResult(BaseModel):
    success: bool
    doc_id: str
    metadata: DocumentMetadata
    chunks: List[Dict]
    processing_time: float
    error: Optional[str] = None

# --------------------------
# Core System Components
# --------------------------

class KnowledgeGraph:
    def __init__(self):
        self.driver = neo4j.GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )
    
    def create_document_node(self, metadata: DocumentMetadata):
        with self.driver.session() as session:
            result = session.write_transaction(
                self._create_document_node_tx, metadata
            )
        return result
    
    @staticmethod
    def _create_document_node_tx(tx, metadata: DocumentMetadata):
        query = """
        MERGE (d:Document {doc_id: $doc_id})
        ON CREATE SET 
            d.title = $title,
            d.type = $doc_type,
            d.source = $source,
            d.effective_date = $effective_date,
            d.expiration_date = $expiration_date,
            d.version = $version,
            d.created_at = datetime()
        RETURN d
        """
        return tx.run(query, **metadata.dict()).single()
    
    def link_entities_to_document(self, doc_id: str, entities: List[Dict]):
        with self.driver.session() as session:
            result = session.write_transaction(
                self._link_entities_tx, doc_id, entities
            )
        return result
    
    @staticmethod
    def _link_entities_tx(tx, doc_id: str, entities: List[Dict]):
        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {id: entity.id})
        SET e.label = entity.label, e.type = entity.type
        WITH e, entity
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (e)-[r:APPEARS_IN]->(d)
        SET r.count = entity.count, r.context = entity.context
        RETURN count(e) as entities_linked
        """
        return tx.run(query, entities=entities, doc_id=doc_id).single()

class VectorStore:
    def __init__(self):
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV", "us-west1-gcp")
        )
        self.index = pinecone.Index(Config.PINECONE_INDEX)
    
    def upsert_document_chunks(self, doc_id: str, chunks: List[Dict]):
        vectors = []
        for chunk in chunks:
            vector = {
                "id": chunk["chunk_id"],
                "values": chunk["embedding"],
                "metadata": {
                    "doc_id": doc_id,
                    "text": chunk["text"],
                    "section": chunk.get("section", ""),
                    "page": chunk.get("page", 0)
                }
            }
            vectors.append(vector)
        
        self.index.upsert(vectors=vectors)
        return len(vectors)
    
    def semantic_search(self, query_embedding: List[float], top_k: int = 5, doc_ids: List[str] = None):
        filter = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        return results

class DocumentProcessor:
    def __init__(self):
        self.kg = KnowledgeGraph()
        self.vector_store = VectorStore()
        self.executor = ThreadPoolExecutor(max_workers=Config.MAX_PARALLEL_REQUESTS)
    
    async def process_document(self, file_source: str, is_url: bool = True) -> DocumentProcessingResult:
        start_time = time.time()
        result = DocumentProcessingResult(success=False, doc_id="", metadata=None, chunks=[])
        
        try:
            # Step 1: Extract text from document
            if is_url:
                raw_text = await self._download_and_extract_text(file_source)
            else:
                raw_text = await self._extract_text_from_file(file_source)
            
            if not raw_text:
                raise ValueError("No text extracted from document")
            
            # Step 2: Extract metadata and structure
            metadata = self._extract_metadata(raw_text, file_source)
            doc_id = metadata.doc_id
            
            # Step 3: Process document into chunks with embeddings
            chunks = self._chunk_and_embed_document(raw_text, doc_id)
            
            # Step 4: Extract entities and relationships
            entities = self._extract_entities(raw_text)
            
            # Step 5: Store in knowledge base
            self.kg.create_document_node(metadata)
            self.kg.link_entities_to_document(doc_id, entities)
            self.vector_store.upsert_document_chunks(doc_id, chunks)
            
            # Update result
            result.success = True
            result.doc_id = doc_id
            result.metadata = metadata
            result.chunks = chunks
            result.processing_time = time.time() - start_time
            
        except Exception as e:
            result.error = str(e)
            result.processing_time = time.time() - start_time
        
        return result
    
    async def _download_and_extract_text(self, url: str) -> str:
        """Enhanced text extraction with multi-modal support"""
        try:
            # Download the document
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Process based on content type
            content_type = response.headers.get('content-type', '')
            
            if 'pdf' in content_type.lower():
                return await self._extract_pdf_text(BytesIO(response.content))
            elif any(t in content_type.lower() for t in ['text', 'html', 'xml']):
                return response.text
            else:
                # Try OCR for images or unknown types
                try:
                    image = Image.open(BytesIO(response.content))
                    return pytesseract.image_to_string(image)
                except:
                    return response.text[:100000]  # Safety limit
            
        except Exception as e:
            print(f"Error downloading/extracting document: {e}")
            return ""
    
    async def _extract_text_from_file(self, file: UploadFile) -> str:
        """Extract text from uploaded file"""
        content = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            return await self._extract_pdf_text(BytesIO(content))
        else:
            # Try to decode as text
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                # Try OCR if text decoding fails
                try:
                    image = Image.open(BytesIO(content))
                    return pytesseract.image_to_string(image)
                except:
                    return ""
    
    async def _extract_pdf_text(self, pdf_stream) -> str:
        """Extract text from PDF with layout awareness"""
        text = ""
        try:
            with fitz.open(stream=pdf_stream) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    
                    # Extract text from embedded images
                    for img in page.get_images():
                        base_image = doc.extract_image(img[0])
                        image = Image.open(BytesIO(base_image["image"]))
                        text += pytesseract.image_to_string(image) + "\n"
        except Exception as e:
            print(f"PDF extraction error: {e}")
        return text
    
    def _extract_metadata(self, text: str, source: str) -> DocumentMetadata:
        """Extract document metadata using NLP and pattern matching"""
        doc = nlp(text[:5000])  # Only process first part for metadata
        
        # Detect document type
        doc_type = "unknown"
        for label in Config.DOCUMENT_TYPES:
            if label in text[:500].lower():
                doc_type = label
                break
        
        # Extract dates
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        effective_date = dates[0] if dates else None
        expiration_date = dates[1] if len(dates) > 1 else None
        
        # Extract title (first non-empty line)
        title = text.split('\n')[0].strip() or "Untitled Document"
        
        return DocumentMetadata(
            title=title,
            doc_type=doc_type,
            source=source,
            effective_date=effective_date,
            expiration_date=expiration_date
        )
    
    def _chunk_and_embed_document(self, text: str, doc_id: str) -> List[Dict]:
        """Intelligent document chunking preserving semantic structure"""
        chunks = []
        
        # First split by major sections
        sections = self._split_by_sections(text)
        
        for section_idx, section in enumerate(sections):
            # Split section into paragraphs
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            
            for para_idx, paragraph in enumerate(paragraphs):
                # Generate chunk ID
                chunk_id = f"{doc_id}_s{section_idx}_p{para_idx}"
                
                # Generate embedding
                embedding = embedding_model.encode(paragraph)
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": paragraph,
                    "embedding": embedding.tolist(),
                    "section": f"Section {section_idx}",
                    "page": 0,  # Would be actual page number in real implementation
                    "doc_id": doc_id
                })
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split document by sections using heading patterns"""
        # Common heading patterns
        heading_patterns = [
            r'\n\s*[A-Z][A-Z0-9\s]+\n',  # ALL CAPS headings
            r'\n\s*\d+\.\d+\s+[A-Z][^\n]+',  # Numbered headings
            r'\n\s*[A-Z][^\n]+\n\s*[-=]+\s*\n',  # Underlined headings
            r'\n\s*SECTION\s+\d+:',  # Explicit section markers
            r'\n\s*Article\s+\d+:',  # Legal document markers
        ]
        
        # Combine patterns
        pattern = '|'.join(heading_patterns)
        sections = re.split(pattern, text)
        
        # Filter empty sections
        return [s.strip() for s in sections if s.strip()]
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities with context"""
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LAW", "DATE", "TIME", "PERCENT", "MONEY", "PRODUCT"]:
                # Get surrounding context (50 chars before and after)
                start = max(0, ent.start_char - 50)
                end = min(len(text), ent.end_char + 50)
                context = text[start:end].replace('\n', ' ')
                
                entities.append({
                    "id": f"{ent.label_}_{hashlib.md5(ent.text.encode()).hexdigest()}",
                    "text": ent.text,
                    "label": ent.label_,
                    "type": "entity",
                    "count": 1,
                    "context": context
                })
        
        # Also extract key phrases (like "grace period")
        phrase_matcher = PhraseMatcher(nlp.vocab)
        phrases = ["grace period", "pre-existing condition", "sum insured", 
                  "waiting period", "no claim bonus", "policy holder"]
        patterns = [nlp(text) for text in phrases]
        phrase_matcher.add("InsuranceTerms", patterns)
        
        matches = phrase_matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            entities.append({
                "id": f"TERM_{hashlib.md5(span.text.encode()).hexdigest()}",
                "text": span.text,
                "label": "INSURANCE_TERM",
                "type": "term",
                "count": 1,
                "context": span.sent.text if span.sent else ""
            })
        
        return entities

class QueryEngine:
    def __init__(self):
        self.kg = KnowledgeGraph()
        self.vector_store = VectorStore()
        self.cache = {}  # In production, use Redis
    
    async def answer_questions(self, request: QuestionRequest) -> AnswerResponse:
        start_time = time.time()
        answers = []
        confidence_scores = []
        sources = []
        
        # Process each question
        for question in request.questions:
            answer, confidence, source = await self._answer_question(
                question, 
                request.documents,
                request.context
            )
            
            answers.append({"question": question, "answer": answer})
            confidence_scores.append(confidence)
            sources.append(source)
        
        return AnswerResponse(
            answers=answers,
            confidence_scores=confidence_scores,
            sources=sources,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    async def _answer_question(self, question: str, doc_ids: List[str], context: Dict = None) -> Tuple[str, float, List[Dict]]:
        """Answer a single question using multi-stage retrieval"""
        # Stage 1: Semantic search for relevant chunks
        query_embedding = embedding_model.encode(question).tolist()
        vector_results = self.vector_store.semantic_search(query_embedding, top_k=5, doc_ids=doc_ids)
        
        # Stage 2: Knowledge graph lookup for entities
        kg_results = await self._query_knowledge_graph(question, doc_ids)
        
        # Stage 3: Combine results and rank
        candidates = self._combine_results(vector_results, kg_results)
        
        if not candidates:
            return "I couldn't find an answer in the provided documents.", 0.0, []
        
        # Stage 4: Answer extraction using QA model
        best_answer, confidence, source = self._extract_answer(question, candidates)
        
        # Stage 5: Post-processing and validation
        validated_answer = self._validate_answer(best_answer, question)
        
        return validated_answer, confidence, source
    
    async def _query_knowledge_graph(self, question: str, doc_ids: List[str]) -> List[Dict]:
        """Query knowledge graph for entities and relationships"""
        doc = nlp(question)
        entities = [ent.text for ent in doc.ents]
        
        with self.kg.driver.session() as session:
            result = session.read_transaction(
                self._query_entities_tx, entities, doc_ids
            )
        
        return result
    
    @staticmethod
    def _query_entities_tx(tx, entities: List[str], doc_ids: List[str]):
        query = """
        MATCH (e:Entity)-[r:APPEARS_IN]->(d:Document)
        WHERE d.doc_id IN $doc_ids AND e.text IN $entities
        RETURN e.text as entity, e.type as type, r.context as context, 
               d.doc_id as doc_id, d.title as doc_title
        ORDER BY r.count DESC
        LIMIT 5
        """
        result = tx.run(query, entities=entities, doc_ids=doc_ids)
        return [dict(record) for record in result]
    
    def _combine_results(self, vector_results, kg_results) -> List[Dict]:
        """Combine and re-rank results from different retrieval methods"""
        combined = []
        
        # Add vector search results
        for match in vector_results.get('matches', []):
            combined.append({
                "text": match['metadata']['text'],
                "score": match['score'],
                "source": "vector",
                "doc_id": match['metadata']['doc_id'],
                "section": match['metadata'].get('section', ''),
                "page": match['metadata'].get('page', 0)
            })
        
        # Add knowledge graph results
        for result in kg_results:
            combined.append({
                "text": result['context'],
                "score": 0.9,  # Fixed high score for KG results
                "source": "knowledge_graph",
                "doc_id": result['doc_id'],
                "section": "Entity Context",
                "page": 0
            })
        
        # Sort by score
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        return combined
    
    def _extract_answer(self, question: str, candidates: List[Dict]) -> Tuple[str, float, List[Dict]]:
        """Use QA model to extract precise answer from candidates"""
        # Try each candidate until we get a good answer
        best_answer = ""
        best_score = 0.0
        best_source = []
        
        for candidate in candidates[:3]:  # Only top 3 candidates
            context = candidate['text']
            
            # Skip if context is too short
            if len(context) < Config.MIN_ANSWER_LENGTH:
                continue
            
            # Get QA model prediction
            qa_result = qa_pipeline({
                'question': question,
                'context': context
            })
            
            # Update best answer if score improves
            if qa_result['score'] > best_score:
                best_answer = qa_result['answer']
                best_score = qa_result['score']
                best_source = [{
                    "text": context[:500] + "..." if len(context) > 500 else context,
                    "doc_id": candidate['doc_id'],
                    "section": candidate['section'],
                    "page": candidate['page'],
                    "source": candidate['source'],
                    "score": candidate['score']
                }]
        
        return best_answer, best_score, best_source
    
    def _validate_answer(self, answer: str, question: str) -> str:
        """Validate and format the answer"""
        # Basic validation
        if not answer or len(answer) < Config.MIN_ANSWER_LENGTH:
            return "I couldn't find a precise answer in the documents."
        
        # Clean up answer
        answer = answer.strip()
        answer = re.sub(r'\s+', ' ', answer)  # Remove extra whitespace
        
        # Add question-specific formatting
        if any(q_word in question.lower() for q_word in ["how much", "what is the amount"]):
            if not any(c in answer for c in ['$', '€', '£', '¥', '%']):
                answer = f"The amount is {answer}"
        
        return answer

# --------------------------
# API Endpoints
# --------------------------

app = FastAPI(
    title="Next-Gen Document Processing API",
    description="Enterprise-grade document intelligence system",
    version="2.0.0"
)

security = HTTPBearer()

# Initialize components
processor = DocumentProcessor()
query_engine = QueryEngine()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # In production, validate JWT or API key
    return credentials.credentials

@app.post("/documents/process", response_model=DocumentProcessingResult)
async def process_document(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """Process and index a document"""
    return await processor.process_document(file, is_url=False)

@app.post("/documents/process-url", response_model=DocumentProcessingResult)
async def process_document_url(
    request: Dict,  # {"url": "http://example.com/doc.pdf"}
    token: str = Depends(verify_token)
):
    """Process and index a document from URL"""
    return await processor.process_document(request["url"], is_url=True)

@app.post("/query", response_model=AnswerResponse)
async def answer_questions(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    """Answer questions based on processed documents"""
    return await query_engine.answer_questions(request)

@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# --------------------------
# Main Execution
# --------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        workers=int(os.getenv("WEB_CONCURRENCY", 2))
    )






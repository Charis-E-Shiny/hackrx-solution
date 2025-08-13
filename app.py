from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from contextlib import asynccontextmanager

from models import QueryRequest, QueryResponse
from services.simple_document_processor import SimpleDocumentProcessor as DocumentProcessor
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

# Initialize document processor
document_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global document_processor
    logger.info("Initializing HackRx LLM Query-Retrieval System")
    try:
        document_processor = DocumentProcessor(config)
        logger.info("Document processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="HackRx LLM Query-Retrieval System",
    description="Intelligent document analysis system for insurance, legal, HR, and compliance domains",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the bearer token"""
    if credentials.credentials != config.BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HackRx LLM Query-Retrieval System is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        return {
            "status": "healthy",
            "services": {
                "document_processor": document_processor is not None,
                "embedding_service": document_processor.embedding_service.vectorizer is not None if document_processor else False,
                "llm_service": bool(config.OPENAI_API_KEY),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Main endpoint to process document queries
    
    This endpoint:
    1. Downloads and processes the PDF document from the provided URL
    2. Creates embeddings and builds a searchable index
    3. Uses LLM to answer questions based on retrieved document context
    4. Returns structured JSON responses with precise answers
    """
    try:
        logger.info(f"Received query request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        if not document_processor:
            raise HTTPException(
                status_code=500,
                detail="Document processor not initialized"
            )
        
        # Validate input
        if not request.documents:
            raise HTTPException(
                status_code=400,
                detail="Document URL is required"
            )
        
        if not request.questions:
            raise HTTPException(
                status_code=400,
                detail="At least one question is required"
            )
        
        # Process the request
        answers = await document_processor.process_document_and_questions(
            document_url=request.documents,
            questions=request.questions
        )
        
        if len(answers) != len(request.questions):
            logger.warning(f"Mismatch in questions ({len(request.questions)}) and answers ({len(answers)})")
        
        logger.info("Successfully processed query request")
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/v1/status")
async def get_status(token: str = Depends(verify_token)):
    """Get system status and statistics"""
    try:
        return {
            "system_status": "operational",
            "document_processor": {
                "initialized": document_processor is not None,
                "embedding_model": config.EMBEDDING_MODEL,
                "llm_model": config.LLM_MODEL,
                "chunk_size": config.CHUNK_SIZE,
                "top_k_retrieval": config.TOP_K_RETRIEVAL
            },
            "configuration": {
                "max_file_size_mb": config.MAX_FILE_SIZE // (1024 * 1024),
                "chunk_overlap": config.CHUNK_OVERLAP
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(
        status_code=500,
        detail="An unexpected error occurred. Please try again later."
    )

# Entry point for gunicorn
# Run with: gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app

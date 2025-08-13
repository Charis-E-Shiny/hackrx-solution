import logging
from typing import List, Dict
from utils.pdf_parser import PDFParser
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.pdf_parser = PDFParser(max_file_size=config.MAX_FILE_SIZE)
        self.embedding_service = EmbeddingService(model_name=config.EMBEDDING_MODEL)
        self.retrieval_service = RetrievalService(
            embedding_service=self.embedding_service,
            index_path=config.FAISS_INDEX_PATH
        )
        self.llm_service = LLMService(
            api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL
        )
    
    async def process_document_and_questions(self, document_url: str, questions: List[str]) -> List[str]:
        """Main processing pipeline"""
        try:
            logger.info("Starting document processing pipeline")
            
            # Step 1: Process PDF document
            logger.info("Step 1: Processing PDF document")
            chunks = await self.pdf_parser.process_document(
                url=document_url,
                chunk_size=self.config.CHUNK_SIZE,
                overlap=self.config.CHUNK_OVERLAP
            )
            
            if not chunks:
                raise ValueError("No content extracted from document")
            
            # Step 2: Clear existing index and add new document
            logger.info("Step 2: Building document index")
            self.retrieval_service.clear_index()
            self.retrieval_service.add_documents(chunks)
            
            # Step 3: Process questions
            logger.info("Step 3: Processing questions with LLM")
            answers = self.llm_service.batch_answer_questions(
                questions=questions,
                retrieval_service=self.retrieval_service
            )
            
            logger.info(f"Successfully processed {len(questions)} questions")
            return answers
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}")
            raise

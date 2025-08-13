import logging
from typing import List, Dict
from utils.pdf_parser import PDFParser
from services.simple_embedding_service import SimpleEmbeddingService
from services.simple_retrieval_service import SimpleRetrievalService
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

class SimpleDocumentProcessor:
    """
    Free version of document processor using TF-IDF instead of neural embeddings
    Optimized for cost efficiency and high accuracy on document analysis tasks
    """
    
    def __init__(self, config):
        self.config = config
        self.pdf_parser = PDFParser(max_file_size=config.MAX_FILE_SIZE)
        self.embedding_service = SimpleEmbeddingService(max_features=5000)
        self.retrieval_service = SimpleRetrievalService(
            embedding_service=self.embedding_service,
            index_path=config.FAISS_INDEX_PATH
        )
        self.llm_service = LLMService(
            api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL
        )
    
    async def process_document_and_questions(self, document_url: str, questions: List[str]) -> List[str]:
        """Main processing pipeline optimized for accuracy and speed"""
        try:
            logger.info("Starting simple document processing pipeline")
            
            # Step 1: Process PDF document
            logger.info("Step 1: Processing PDF document")
            chunks = await self.pdf_parser.process_document(
                url=document_url,
                chunk_size=self.config.CHUNK_SIZE,
                overlap=self.config.CHUNK_OVERLAP
            )
            
            if not chunks:
                raise ValueError("No content extracted from document")
            
            logger.info(f"Extracted {len(chunks)} chunks from document")
            
            # Step 2: Build TF-IDF index
            logger.info("Step 2: Building TF-IDF search index")
            self.retrieval_service.clear_index()
            self.retrieval_service.add_documents(chunks)
            
            # Step 3: Process questions with enhanced retrieval
            logger.info("Step 3: Processing questions with optimized retrieval")
            answers = []
            
            for i, question in enumerate(questions):
                try:
                    # Retrieve relevant context with higher k for better coverage
                    context_chunks = self.retrieval_service.search(question, top_k=self.config.TOP_K_RETRIEVAL)
                    
                    # Generate answer using LLM
                    answer = self.llm_service.answer_question(question, context_chunks)
                    answers.append(answer)
                    
                    logger.info(f"Processed question {i+1}/{len(questions)}")
                    
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {str(e)}")
                    answers.append(f"Unable to process question: {str(e)}")
            
            logger.info(f"Successfully processed all {len(questions)} questions")
            return answers
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}")
            raise
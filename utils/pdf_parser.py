import fitz  # PyMuPDF
import httpx
import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self, max_file_size: int = 50 * 1024 * 1024):
        self.max_file_size = max_file_size

    async def download_pdf(self, url: str) -> bytes:
        """Download PDF from blob URL"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                if len(response.content) > self.max_file_size:
                    raise ValueError(f"File size exceeds maximum limit of {self.max_file_size} bytes")
                
                return response.content
        except Exception as e:
            logger.error(f"Failed to download PDF from {url}: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        # Remove multiple consecutive periods
        text = re.sub(r'\.{3,}', '...', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Split text into overlapping chunks"""
        clean_text = self.clean_text(text)
        chunks = []
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {
                        "chunk_id": len(chunks),
                        "char_count": len(current_chunk)
                    }
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk)
            else:
                current_chunk += " " + sentence
                current_size += sentence_size
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {
                    "chunk_id": len(chunks),
                    "char_count": len(current_chunk)
                }
            })
        
        return chunks

    async def process_document(self, url: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Main method to process PDF document from URL"""
        logger.info(f"Processing document from URL: {url}")
        
        # Download PDF
        pdf_content = await self.download_pdf(url)
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_content)
        
        # Create chunks
        chunks = self.chunk_text(text, chunk_size, overlap)
        
        logger.info(f"Successfully processed document into {len(chunks)} chunks")
        return chunks

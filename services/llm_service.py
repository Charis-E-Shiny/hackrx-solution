import openai
import json
import logging
from typing import List, Dict
from models import RetrievalResult

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def answer_question(self, question: str, context_chunks: List[RetrievalResult]) -> str:
        """Generate answer for a question using retrieved context"""
        try:
            # Prepare context from retrieved chunks
            context_text = "\n\n".join([
                f"[Chunk {i+1} - Score: {chunk.score:.3f}]\n{chunk.content}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Create optimized prompt for token efficiency
            system_prompt = """You are an expert insurance policy analyst. Your task is to answer questions about insurance policies with high precision and provide explainable reasoning.

Instructions:
1. Answer based ONLY on the provided context
2. Be specific and cite relevant policy clauses
3. If the information is not in the context, state clearly "Information not available in the provided document"
4. Provide clear, concise answers that directly address the question
5. Include specific details like time periods, amounts, conditions when available"""

            user_prompt = f"""Context from Insurance Policy Document:
{context_text}

Question: {question}

Please provide a precise answer based on the context above. Include specific policy details and conditions when relevant."""

            # Call OpenAI API with optimized parameters
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=300,   # Limit tokens for cost efficiency
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Log token usage for monitoring
            usage = response.usage
            logger.info(f"OpenAI API call - Prompt tokens: {usage.prompt_tokens}, "
                       f"Completion tokens: {usage.completion_tokens}, "
                       f"Total tokens: {usage.total_tokens}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def batch_answer_questions(self, questions: List[str], retrieval_service) -> List[str]:
        """Process multiple questions efficiently"""
        answers = []
        
        for question in questions:
            try:
                # Retrieve relevant context for each question
                context_chunks = retrieval_service.search(question, top_k=5)
                
                # Generate answer
                answer = self.answer_question(question, context_chunks)
                answers.append(answer)
                
                logger.info(f"Processed question: {question[:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        return answers

"""
Groq LLM Service for high-performance inference.

Provides unified interface to Groq API for rapid agent decision-making
and processing across all agents in the multi-agent system.
"""
import asyncio
from typing import Dict, List, Any, Optional
import logging
from groq import AsyncGroq
import os

logger = logging.getLogger(__name__)


class GroqLLMService:
    """
    High-performance LLM service using Groq API for rapid inference.
    
    Designed to handle 5,000+ requests daily with sub-second response times
    for agent decision-making and content generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        self.client = AsyncGroq(api_key=self.api_key)
        self.model = model
        self.default_params = {
            "temperature": 0.1,
            "max_tokens": 2048,
            "top_p": 1.0,
            "stream": False
        }
        
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response using Groq API with high-speed inference.
        
        Args:
            prompt: User prompt or query
            context: Additional context for the query
            system_prompt: System instructions for the model
            **kwargs: Additional parameters for the API call
            
        Returns:
            Generated response text
        """
        try:
            # Prepare messages
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            elif context:
                system_content = f"Context: {context}"
                messages.append({"role": "system", "content": system_content})
            
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Merge parameters
            params = {**self.default_params, **kwargs}
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **params
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.warning("Empty response from Groq API")
                return ""
                
        except Exception as e:
            logger.error(f"Error in Groq API call: {e}")
            raise
    
    async def generate_structured_response(
        self,
        prompt: str,
        schema: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured response following a specific schema.
        
        Args:
            prompt: User prompt
            schema: Expected response schema
            context: Additional context
            **kwargs: Additional API parameters
            
        Returns:
            Structured response as dictionary
        """
        schema_prompt = f"""
        {prompt}
        
        Please respond with a JSON object that follows this schema:
        {schema}
        
        Ensure your response is valid JSON and follows the schema exactly.
        """
        
        response_text = await self.generate_response(
            schema_prompt, 
            context=context,
            **kwargs
        )
        
        # Parse JSON response
        try:
            import json
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            # Return a fallback structure
            return {"error": "Failed to parse structured response", "raw_response": response_text}
    
    async def generate_batch_responses(
        self,
        prompts: List[str],
        context: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple responses concurrently for high throughput.
        
        Args:
            prompts: List of prompts to process
            context: Shared context for all prompts
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional API parameters
            
        Returns:
            List of generated responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_prompt(prompt: str) -> str:
            async with semaphore:
                return await self.generate_response(prompt, context=context, **kwargs)
        
        tasks = [process_prompt(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def analyze_code(
        self,
        code: str,
        language: str,
        analysis_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Specialized method for code analysis using Groq.
        
        Args:
            code: Code to analyze
            language: Programming language
            analysis_type: Type of analysis (general, security, performance, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Analysis results
        """
        analysis_prompt = f"""
        Analyze the following {language} code for {analysis_type} aspects:
        
        ```{language}
        {code}
        ```
        
        Provide analysis in the following JSON format:
        {{
            "language": "{language}",
            "analysis_type": "{analysis_type}",
            "summary": "Brief summary of the code",
            "issues": ["list of identified issues"],
            "suggestions": ["list of improvement suggestions"],
            "complexity_score": 0.0,
            "maintainability": "high|medium|low",
            "functions": ["list of function names"],
            "classes": ["list of class names"],
            "dependencies": ["list of imports/dependencies"]
        }}
        """
        
        return await self.generate_structured_response(
            analysis_prompt,
            schema={
                "language": "string",
                "analysis_type": "string", 
                "summary": "string",
                "issues": ["string"],
                "suggestions": ["string"],
                "complexity_score": "number",
                "maintainability": "string",
                "functions": ["string"],
                "classes": ["string"],
                "dependencies": ["string"]
            },
            **kwargs
        )
    
    async def generate_documentation(
        self,
        code: str,
        language: str,
        doc_type: str = "api",
        **kwargs
    ) -> str:
        """
        Generate documentation for code using Groq.
        
        Args:
            code: Code to document
            language: Programming language
            doc_type: Type of documentation (api, readme, comments, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Generated documentation
        """
        doc_prompt = f"""
        Generate {doc_type} documentation for the following {language} code:
        
        ```{language}
        {code}
        ```
        
        Create comprehensive, clear, and well-structured documentation that includes:
        - Purpose and functionality description
        - Parameter documentation
        - Return value documentation
        - Usage examples
        - Any important notes or warnings
        
        Format the documentation appropriately for {doc_type} style.
        """
        
        return await self.generate_response(doc_prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model": self.model,
            "provider": "groq",
            "default_params": self.default_params,
            "capabilities": [
                "text_generation",
                "code_analysis", 
                "structured_output",
                "batch_processing",
                "high_speed_inference"
            ]
        } 
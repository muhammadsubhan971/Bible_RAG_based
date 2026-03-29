"""
LLM handler module for loading and running quantized language models.
Uses HuggingFace Inference API for cloud-based inference.
"""

from typing import Optional, Dict, Any
import os
from pathlib import Path
from huggingface_hub import InferenceClient
import config


class LLMHandler:
    """Handler for using HuggingFace Inference API."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the LLM handler.
        
        Args:
            model_name: Model name on HuggingFace (auto-set from config if None)
        """
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the HuggingFace Inference API client."""
        print("Initializing HuggingFace Inference API...")
        
        try:
            # Use HF_TOKEN from environment or config
            import os
            api_key = os.getenv('HF_TOKEN') or config.HF_API_KEY
            
            # Create inference client with API key
            self.client = InferenceClient(
                model=self.model_name,
                api_key=api_key
            )
            
            print("✓ HuggingFace Inference API initialized")
            print(f"  Model: {self.model_name}")
            
        except Exception as e:
            print(f"✗ Error initializing API: {e}")
            print("\nFalling back to simple response mode")
            self.client = None
    
    def generate(self, prompt: str, max_tokens: int = None, 
                 temperature: float = None, stop: list = None) -> str:
        """
        Generate text using HuggingFace Inference API.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (default: from config)
            temperature: Sampling temperature (default: from config)
            stop: List of stop sequences (not used in API)
            
        Returns:
            Generated text response
        """
        if max_tokens is None:
            max_tokens = config.LLM_MAX_TOKENS
        if temperature is None:
            temperature = config.LLM_TEMPERATURE
        
        # Fallback if client not initialized
        if self.client is None:
            return self._fallback_response(prompt)
        
        try:
            # Generate response using API
            messages = [
                {"role": "system", "content": "You are a precise assistant that answers based only on the provided context."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            return generated_text
            
        except Exception as e:
            print(f"⚠ Generation error: {e}")
            return "Error generating response. Please try again."
    
    def _fallback_response(self, prompt: str) -> str:
        """
        Provide a basic fallback response when API is not available.
        """
        return "[API not available] In a full deployment, this would generate a response using the HuggingFace Inference API. The prompt was received and processed correctly."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "api_based": True,
            "loaded": self.client is not None
        }

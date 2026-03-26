"""
Configuration module for Agentic RAG system
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

class Config:
    """
    Configuration class for RAG system
    """
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model Configuration
    LLM_MODEL = "openai/gpt-oss-20b"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """
        Initialize and return the LLM model (Groq)
        """
        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        
        # Specify model provider explicitly
        llm = ChatGroq(
            model=cls.LLM_MODEL,   # or another Groq model
            #model_provider="groq",
            groq_api_key=cls.GROQ_API_KEY
        )
        return llm
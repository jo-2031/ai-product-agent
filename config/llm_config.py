import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq


class LLMConfig:
    """Centralized LLM configuration supporting multiple providers (OpenAI, Groq)"""
    
    def __init__(self, model_name=None, embedding_model=None, embedding_dim=None, provider=None, temperature=None):
        """
        Initialize LLM Config with environment-based settings
        
        Args:
            model_name: Override model name (defaults to env var)
            embedding_model: Override embedding model (defaults to env var)
            embedding_dim: Override embedding dimensions (defaults to env var)
            provider: Override provider ('openai' or 'groq', defaults to env var)
            temperature: Override temperature (defaults to env var)
        """
        load_dotenv()
        
        # Get provider from env or parameter
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()
        
        # Provider-specific model defaults
        default_models = {
            "openai": "gpt-4o-mini",
            "groq": "llama-3.1-70b-versatile"
        }
        
        # Get configuration from environment variables with fallbacks
        self.model_name = model_name or os.getenv("LLM_MODEL") or default_models.get(self.provider, "gpt-4o-mini")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", "1024"))
        self.temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0"))
        
        # Validate API keys
        self._validate_api_keys()
    
    def _validate_api_keys(self):
        """Validate that required API keys are present"""
        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Please set it in your .env file."
                )
        elif self.provider == "groq":
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError(
                    "GROQ_API_KEY not found in environment variables. "
                    "Please set it in your .env file."
                )
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: openai, groq"
            )
    
    def get_chat_model(self):
        """
        Returns a configured chat model instance based on provider
        
        Returns:
            ChatOpenAI or ChatGroq instance
        """
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.provider == "groq":
            return ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("GROQ_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def get_embedding_model(self):
        """
        Returns a configured embedding model instance
        Currently only supports OpenAI embeddings
        
        Returns:
            OpenAIEmbeddings instance
        """
        return OpenAIEmbeddings(
            model=self.embedding_model,
            dimensions=self.embedding_dim,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def get_config_info(self):
        """
        Returns current configuration information
        
        Returns:
            dict: Configuration details
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "temperature": self.temperature,
            "api_key_configured": bool(os.getenv(f"{self.provider.upper()}_API_KEY"))
        }
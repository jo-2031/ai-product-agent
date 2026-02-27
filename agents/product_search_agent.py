import os 
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from utils.logging import logger
from config.llm_config import LLMConfig
from typing import List
import json

class ProductSearchAgent:
    """Product Search Agent - Retrieves and recommends products using RAG"""
    
    def __init__(self):
        """Initialize Product Search Agent"""
        load_dotenv()
        
        # Get persist directory from environment or use default
        persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        
        # Initialize LLM configuration
        self.llm_config = LLMConfig()
        
        # Log configuration info
        config_info = self.llm_config.get_config_info()
        logger.info(f"ProductSearchAgent initialized with provider: {config_info['provider']}, model: {config_info['model_name']}")
        
        # Embeddings and vector store
        self.embeddings = self.llm_config.get_embedding_model()
        self.vector_store = Chroma(
            collection_name="product_collection",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Get chat model from config
        self.model = self.llm_config.get_chat_model()
        self.agent = None
        self.retriever = None

    def load_and_process_data(self, file_path):
        """Load CSV and create vector store"""
        loader = CSVLoader(
            file_path=file_path,
            csv_args={
                'delimiter': ',',
                'quotechar': '"',
                'fieldnames': [
                    'Product ID', 'Product', 'Category', 'Provider', 'Selling Price', 'Brands',
                    'Discount', 'Rating', 'Bought Count', 'MRP Price',
                    'Delivery Date', 'Product Image URL'
                ]
            }
        )  
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)
        self.vector_store.add_documents(split_documents)
        logger.info(f"Loaded {len(split_documents)} chunks into Chroma DB")

    def _create_retrieval_tool(self):
        """Create RAG retrieval tool"""
        @tool
        def retrieve_product_context(query: str) -> str:
            """Retrieve top 3 relevant products from the database based on the query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=3)
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")  

            products = []
            for doc in retrieved_docs:
                product = dict(doc.metadata)
                product['description'] = doc.page_content[:500]
                products.append(product)
            return json.dumps(products, indent=2, ensure_ascii=False)    
        return retrieve_product_context
    
    def _check_relevance(self, query: str, products: List[dict]) -> bool:
        """Use LLM to check if retrieved products are relevant to the query"""
        if not products:
            return False
        
        # Extract categories from retrieved products
        categories = [p.get('Category', '') for p in products if p.get('Category')]
        product_names = [p.get('Product', '') for p in products[:3] if p.get('Product')]
        
        # Create a quick relevance check prompt
        relevance_prompt = f"""User is searching for: "{query}"
        Retrieved products are in categories: {', '.join(set(categories))}
        Product examples: {', '.join(product_names[:3])}

        Are these products relevant to what the user is looking for?
        Answer only 'yes' or 'no'."""

        try:
            response = self.model.invoke(relevance_prompt)
            answer = response.content.strip().lower()
            is_relevant = 'yes' in answer
            logger.info(f"Relevance check for '{query}': {is_relevant} (LLM response: {answer})")
            return is_relevant
        except Exception as e:
            logger.error(f"Relevance check failed: {e}")
            # On error, be conservative and show products
            return True
    
    def get_products_data(self, query: str) -> List[dict]:
        """Get product data using optimized similarity search
        
        Returns:
            List of product dictionaries, or empty list if not relevant
        """
        # Retrieve potential products
        retrieved_docs = self.vector_store.similarity_search(query, k=3)
        logger.info(f"Retrieved {len(retrieved_docs)} products")
        
        products = []
        for doc in retrieved_docs:
            # Extract all fields from page_content
            product_data = {}
            for line in doc.page_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    product_data[key.strip()] = value.strip()
            
            products.append(product_data)
        
        # Check if products are relevant to the query
        if not self._check_relevance(query, products):
            logger.info(f"Products not relevant to query: {query}")
            return []
        
        return products
    
    def product_search_agent(self):
        """Create Product Search Agent"""
        tools = [self._create_retrieval_tool()]
        system_prompt = """You are a shopping assistant. Use the retrieval tool to fetch products.
            Show top 3 products with: Name, Brand, Price, Rating.
            Keep responses clean and simple.
            """
        
        self.agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt
        )
        logger.info("ProductSearchAgent created")
        return self.agent
    
    def run(self, state):
        """Run agent for LangGraph workflow"""
        if not self.agent:
            self.product_search_agent()
        
        messages = state["messages"]
        query = messages[0].content
        
        logger.info(f"ProductSearchAgent processing: {query}")
        result = self.agent.invoke({"messages": [HumanMessage(content=query)]})
        return {"messages": [result["messages"][-1]]}
    
    def run_agent(self, query: str):
        """Standalone execution method"""
        if not self.agent:
            self.product_search_agent()
        result = self.agent.invoke({"messages": [HumanMessage(content=query)]})
        return result["messages"][-1].content

    

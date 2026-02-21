import os 
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from utils.logging import logger
import json

class ProductCollectionAgent:
    """Complete RAG Agent using new create_agent API"""
    
    def __init__(self, persist_directory="./chroma_db", model_name="gpt-4o-mini"):
        load_dotenv()
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
        # Embeddings and vector store
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
        self.vector_store = Chroma(
            collection_name="product_collection",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.agent = None

    def load_and_process_data(self, file_path):
            """Load CSV and create vector store"""
            loader = CSVLoader(
                file_path=file_path,
                csv_args={
                    'delimiter': ',',
                    'quotechar': '"',
                    'fieldnames': [
                        'Product ID', 'Product', 'Brands', 'Discount', 
                        'Rating', 'Bought Count', 'MRP Price', 'Selling Price', 
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
                """Retrieve relevant product information from the database."""
                retrieved_docs = self.vector_store.similarity_search(query, k=3)
                logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")  

                products = []

                for doc in retrieved_docs:
                    product = dict(doc.metadata)
                    product['description'] = doc.page_content[:500]
                    products.append(product)
                return json.dumps(products, indent=2, ensure_ascii=False)    
        return retrieve_product_context
    
    def product_search_agent(self):
        """Create RAG retrieval tool"""
        tools = [self._create_retrieval_tool()]
        system_prompt = (
            "You are a helpful assistant that retrieves product information based on user queries. "
            "Use the provided retrieval tool to fetch relevant product details from the database."
        )
        self.agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt
        )
        logger.info("Agent created with RAG retrieval tool")
        return self.agent
    
    #test agent with a query
    def run_agent(self, query: str):
        if not self.agent:
            self.product_search_agent()
        result = self.agent.invoke({"messages": [HumanMessage(content=query)]})
        return result["messages"][-1].content
    
# if __name__ == "__main__":
#     agent = ProductCollectionAgent()
#     logger.info("Initialized ProductCollectionAgent")
#     agent.load_and_process_data("/Users/jothikaravichandran/Documents/Self_Projects/personal_project/ai_geeks_product_agent/ai-product-agent/source_product_input/merged_product_data.csv")
  
#     logger.info("Agent ready! Testing queries...")
#     queries = [
#         "What laptops are available under 70000?",
#         # "Best rated products with good discount?",
#         # "Show me phones from Samsung under 30000"
#     ]   
#     for query in queries:
#         logger.info(f"Running agent for query: {query}")
#         response = agent.run_agent(query)
#         logger.info(f"Response for query '{query}': {response}")
    
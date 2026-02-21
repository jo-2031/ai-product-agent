from typing import Dict, Any, TypedDict, Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from .product_search_agent import ProductCollectionAgent
from utils.logging import logger


class ProductOrchestratorAgent:
    """Agent to orchestrate product search and recommendation"""
    
    def __init__(self, rag_agent: ProductCollectionAgent):
        self.rag_agent = rag_agent
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        

    def route_query(self, query: str) -> Dict[str, Any]:
        """route the query to the appropriate agent and return results"""

        router_prompt = f""" 
        Analyze this user query and decide routing:
        
        ROUTE TO "rag" if it's about:
        - Products (laptop, phone, shoes, etc.)
        - Prices, discounts, ratings, brands
        - Buying, recommendations, specifications
        - Any shopping/product related question
        
        ROUTE TO "general" if it's:
        - Greetings (hi, hello)
        - General chat, jokes, weather, time
        - Non-product questions
        
        Query: "{query}"
        
        Respond ONLY with: "rag" or "general"
        """

        route_decision = self.llm.invoke([HumanMessage(content=router_prompt)])
        route = route_decision.content.strip().lower()

        if "rag" in route:
            logger.info("Routing to Product RAG Agent")
            rag_result = self.rag_agent.run_agent(query)
            return {"route":"rag", "result": rag_result}
        else:
            logger.info("Handling as general query")
            response = self.llm.invoke([HumanMessage(content=query)])
            return {"route": "general", "result": response.content}
        
        

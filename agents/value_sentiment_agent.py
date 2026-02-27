"""
ValueSentimentAgent - Analyzes customer sentiment and popularity
No recommendation - only analysis
"""
from langchain_core.messages import HumanMessage, SystemMessage
from config.llm_config import LLMConfig
from utils.logging import logger

class ValueSentimentAgent:
    """Customer Sentiment Analyst - Analyzes ratings and popularity"""
    
    def __init__(self):
        """Initialize Sentiment Agent"""
        self.llm_config = LLMConfig()
        self.model = self.llm_config.get_chat_model()
        logger.info("ValueSentimentAgent initialized")
    
    def analyze(self, products_text: str) -> str:
        """Analyze products for customer sentiment
        
        Args:
            products_text: String containing product data
        
        Returns:
            Sentiment analysis in table format
        """
        system_prompt =  """You are a customer sentiment analyst.

Analyze customer sentiment and create a CLEAR TABLE.

Show:
1. Product name
2. Rating (out of 5)
3. Bought Count
4. Customer Preference (High/Medium/Low)
5. Market Traction (Strong/Good/Weak)
6. Trust Score (High/Medium/Low)
7. Key Review Themes  ← e.g., battery, screen, comfort, fit, durability

Format as a markdown table:
| Product | Rating | Bought Count | Customer Preference | Market Traction | Trust Score | Key Review Themes |
|---------|--------|--------------|---------------------|-----------------|-------------|-------------------|

After table, add 2–3 lines about customer trends and common complaints/praise.

Do NOT recommend. Just analyze."""


        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze these products:\n\n{products_text}")
        ]
        
        logger.info("ValueSentimentAgent analyzing products...")
        result = self.model.invoke(messages)
        return result.content
    
    def run(self, state):
        """Run agent for LangGraph workflow"""
        messages = state["messages"]
        products_data = messages[-1].content
        
        analysis = self.analyze(products_data)
        return {"messages": [HumanMessage(content=analysis)]}

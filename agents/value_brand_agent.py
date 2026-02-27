"""
ValueBrandAgent - Analyzes brand reputation and positioning
No recommendation - only analysis
"""
from langchain_core.messages import HumanMessage, SystemMessage
from config.llm_config import LLMConfig
from utils.logging import logger

class ValueBrandAgent:
    """Brand Specialist - Analyzes brand strength and positioning"""
    
    def __init__(self):
        """Initialize Brand Agent"""
        self.llm_config = LLMConfig()
        self.model = self.llm_config.get_chat_model()
        logger.info("ValueBrandAgent initialized")
    
    def analyze(self, products_text: str) -> str:
        """Analyze products for brand value
        
        Args:
            products_text: String containing product data
        
        Returns:
            Brand analysis in table format
        """
        system_prompt = """You are a brand specialist.

Analyze the brands and create a CLEAR TABLE.

Show:
1. Product name
2. Brand
3. Brand Category (Premium/Mid-Range/Budget)
4. Brand Reputation (Excellent/Good/Average)
5. Market Position
6. Trust Level (High/Medium/Low)
7. Key Brand Strengths  ← e.g., reliability, customer service, innovation
8. provider

Format as a markdown table:
| Product | Brand | Category | Reputation | Market Position | Trust Level | Key Brand Strengths |
|---------|-------|----------|------------|-----------------|-------------|---------------------|

After table, add 2–3 lines explaining why these brands are strong or weak overall.

Do NOT recommend. Just analyze."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze these products:\n\n{products_text}")
        ]
        
        logger.info("ValueBrandAgent analyzing products...")
        result = self.model.invoke(messages)
        return result.content
    
    def run(self, state):
        """Run agent for LangGraph workflow"""
        messages = state["messages"]
        products_data = messages[-1].content
        
        analysis = self.analyze(products_data)
        return {"messages": [HumanMessage(content=analysis)]}

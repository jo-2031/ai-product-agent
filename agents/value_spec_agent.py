"""
SpecAgent - Analyzes product value based on price, discount, and rating
No recommendation - only analysis
"""
from langchain_core.messages import HumanMessage, SystemMessage
from config.llm_config import LLMConfig
from utils.logging import logger

class ValueSpecAgent:
    """Product Value Analyst - Analyzes price, discount, rating"""
    
    def __init__(self):
        """Initialize Spec Agent"""
        self.llm_config = LLMConfig()
        self.model = self.llm_config.get_chat_model()
        logger.info("ValueSpecAgent initialized")
    
    def analyze(self, products_text: str) -> str:
        """Analyze products for value
        
        Args:
            products_text: String containing product data
        
        Returns:
            Value analysis in table format
        """
        system_prompt = system_prompt = """You are a product value analyst.

Analyze the given products and create a CLEAR TABLE.

Always show at least these columns:
1. Product name
2. MRP Price
3. Selling Price
4. Discount %
5. Rating (out of 5)
6. Bought Count
7. Value Score (your assessment: Excellent / Good / Average)
8. Description  ← one key description column

For each product:
- "Description" must summarize:
  - What this product is and what it's best at
  - Main strengths
  - Main weaknesses
  - Who it suits (e.g., office, gaming, casual, travel)

Format as a markdown table:

| Product | MRP | Selling Price | Discount | Rating | Bought Count | Value Score | Description |
|---------|-----|---------------|----------|--------|--------------|-------------|-------------|
| ...     | ... | ...           | ...      | ...    | ...          | ...         | ...         |

After the table, add 2–3 lines explaining:
- Which product offers the best value overall
- Why it stands out (price vs features, rating, reviews, etc.)

Do NOT recommend which one to buy. Just analyze and explain value."""


        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze these products:\n\n{products_text}")
        ]
        
        logger.info("ValueSpecAgent analyzing products...")
        result = self.model.invoke(messages)
        return result.content
    
    def run(self, state):
        """Run agent for LangGraph workflow"""
        messages = state["messages"]
        products_data = messages[-1].content
        
        analysis = self.analyze(products_data)
        return {"messages": [HumanMessage(content=analysis)]}

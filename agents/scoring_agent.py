
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.logging import logger


class ScoringAgent:
    """Scores products on price-performance ratio via LLM."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def score(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = []
        for product in products:
            prompt = f"""Score this product from 0 to 10 based on price-performance ratio.
Consider: discount vs MRP, rating, and bought count.

- Name         : {product.get('Product', 'N/A')}
- MRP Price    : {product.get('MRP Price', 'N/A')}
- Selling Price: {product.get('Selling Price', 'N/A')}
- Discount     : {product.get('Discount', 'N/A')}
- Rating       : {product.get('Rating', 'N/A')}
- Bought Count : {product.get('Bought Count', 'N/A')}

Respond with ONLY a number between 0 and 10."""

            try:
                score = float(self.llm.invoke([HumanMessage(content=prompt)]).content.strip())
            except ValueError:
                score = 5.0
            product = dict(product)
            product["performance_score"] = score
            scored.append(product)
            logger.info("ScoringAgent: '%s' → %.1f", product.get("Product", "?"), score)
        return scored

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.logging import logger


class RecommendationAgent:
    """Generates a human-readable recommendation from the top ranked product."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def recommend(self, ranked_products: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        if not ranked_products:
            return {"recommendation": "Sorry, no products found.", "top_product": {}}

        top = ranked_products[0]
        ranked_summary = "\n".join(
            f"{i}. {p.get('Product','N/A')} | {p.get('Selling Price','N/A')} | "
            f"⭐{p.get('Rating','N/A')} | Score: {p.get('final_score','N/A')}"
            for i, p in enumerate(ranked_products, 1)
        )

        prompt = f"""You are a helpful product recommendation assistant.

User query: "{query}"

Ranked products:
{ranked_summary}

Top recommended: {top.get('Product','N/A')}
- Brand: {top.get('Brands','N/A')}
- Price: {top.get('Selling Price','N/A')}
- Discount: {top.get('Discount','N/A')}
- Rating: {top.get('Rating','N/A')}
- Score: {top.get('final_score','N/A')}/10
- Sentiment: {top.get('sentiment_label','N/A')}

Write a concise friendly recommendation with:
1. Top pick with reason
2. Key pros (2-3 points)
3. Price summary
4. Ask: "Would you like to proceed with this product?" """

        logger.info("RecommendationAgent: recommending '%s'", top.get("Product", "?"))
        return {
            "recommendation": self.llm.invoke([HumanMessage(content=prompt)]).content,
            "top_product": top
        }

import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.logging import logger


class SentimentAgent:
    """Estimates customer sentiment from rating and bought count."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def analyze(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        analyzed = []
        for product in products:
            prompt = f"""Estimate customer sentiment for this product.

- Name        : {product.get('Product', 'N/A')}
- Rating      : {product.get('Rating', 'N/A')} / 5
- Bought Count: {product.get('Bought Count', 'N/A')}

Respond in JSON only: {{"label": "positive|neutral|negative", "score": <0-10>}}"""

            try:
                data  = json.loads(self.llm.invoke([HumanMessage(content=prompt)]).content.strip())
                label = data.get("label", "neutral").lower()
                score = float(data.get("score", 5.0))
            except (json.JSONDecodeError, ValueError):
                label, score = "neutral", 5.0

            product = dict(product)
            product["sentiment_label"] = label
            product["sentiment_score"] = score
            analyzed.append(product)
            logger.info("SentimentAgent: '%s' → %s (%.1f)", product.get("Product", "?"), label, score)
        return analyzed

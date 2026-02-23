import csv
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.logging import logger


class BrandingAgent:
    """Scores brand reliability and market positioning.

    Known brands: fast CSV lookup.
    Unknown brands: LLM fallback for intelligent scoring.
    """

    def __init__(self, known_brands: set, model_name: str = "gpt-4o-mini"):
        self._known_brands = known_brands
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    @staticmethod
    def load_known_brands(file_path: str) -> set:
        with open(file_path, newline="", encoding="utf-8") as f:
            return {
                word
                for row in csv.DictReader(f)
                for word in row.get("Brands", "").lower().split()
                if word.isalpha() and len(word) > 2
            }

    def _is_known(self, brand_field: str) -> bool:
        return bool(set(brand_field.lower().split()) & self._known_brands)

    def _score_with_llm(self, product: Dict[str, Any]) -> tuple[float, str]:
        prompt = f"""Evaluate the brand of this product for reliability and market positioning.

- Product: {product.get('Product', 'N/A')}
- Brand  : {product.get('Brands', 'N/A')}

Respond in JSON only:
{{"score": <0-10>, "label": "established|niche", "reason": "<one line>"}}"""

        try:
            data = json.loads(self.llm.invoke([HumanMessage(content=prompt)]).content.strip())
            return float(data.get("score", 5.0)), data.get("label", "niche").lower()
        except (json.JSONDecodeError, ValueError):
            return 5.0, "niche"

    def analyze(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for product in products:
            brand_field = product.get("Brands", "")

            if self._is_known(brand_field):
                brand_score, brand_label = 7.0, "established"
                logger.info("BrandingAgent: '%s' → known brand, skipping LLM", product.get("Product", "?"))
            else:
                brand_score, brand_label = self._score_with_llm(product)
                logger.info("BrandingAgent: '%s' → LLM scored %s (%.2f)", product.get("Product", "?"), brand_label, brand_score)

            result.append({**product, "brand_score": brand_score, "brand_label": brand_label})
        return result

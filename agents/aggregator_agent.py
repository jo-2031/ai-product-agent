from typing import List, Dict, Any
from utils.logging import logger


class AggregatorAgent:
    """Merges scores from Scoring + Sentiment + Branding agents and ranks products.

    Weights: performance 40% | sentiment 40% | brand 20%
    """

    WEIGHTS = {"performance_score": 0.4, "sentiment_score": 0.4, "brand_score": 0.2}

    def aggregate(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for product in products:
            final = sum(
                product.get(k, 5.0) * w for k, w in self.WEIGHTS.items()
            )
            product["final_score"] = round(final, 2)
            logger.info("Aggregator: '%s' → final_score=%.2f", product.get("Product", "?"), final)
        return sorted(products, key=lambda p: p["final_score"], reverse=True)



from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.logging import logger


class HumanValidationNode:
    """Interprets YES / NO / feedback from the user."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def _detect_intent(self, user_response: str) -> str:
        """Use LLM to detect if response is a confirmation, greeting/reset, or refinement feedback."""
        prompt = f"""Classify this user response in the context of a product purchase confirmation.

Response: "{user_response}"

Respond with ONLY one word:
- confirmed : user agrees to proceed (e.g. yes, sure, okay, go ahead, sounds good)
- greet     : user wants to start over or is greeting (e.g. hi, reset, new search)
- refine    : user wants changes or has feedback (e.g. too expensive, show cheaper ones)"""

        return self.llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()

    def validate(self, user_response: str, top_product: Dict[str, Any],
                 original_query: str, memory_manager=None) -> Dict[str, Any]:

        intent = self._detect_intent(user_response)

        if intent == "confirmed":
            logger.info("HumanValidation: CONFIRMED '%s'", top_product.get("Product", "?"))
            if memory_manager:
                memory_manager.write_confirmation(original_query, top_product)
            return {
                "status": "confirmed",
                "final_answer": self._confirmation_message(top_product),
                "refined_query": None,
            }
        elif intent == "greet":
            logger.info("HumanValidation: greet during validation — resetting")
            return {
                "status": "confirmed",
                "final_answer": "No problem! What product are you looking for next?",
                "refined_query": None,
            }
        else:
            logger.info("HumanValidation: REJECTED — refining with '%s'", user_response)
            return {
                "status": "refine",
                "final_answer": None,
                "refined_query": self._build_refined_query(original_query, user_response),
            }

    def _confirmation_message(self, top_product: Dict[str, Any]) -> str:
        prompt = (
            f"User confirmed purchase of: {top_product.get('Product', 'N/A')} "
            f"at {top_product.get('Selling Price', 'N/A')} by {top_product.get('Brands', 'N/A')}.\n"
            f"Write a short friendly confirmation. Mention preference is saved."
        )
        return self.llm.invoke([HumanMessage(content=prompt)]).content

    def _build_refined_query(self, original_query: str, feedback: str) -> str:
        prompt = (
            f"Original query: \"{original_query}\"\n"
            f"User feedback: \"{feedback}\"\n"
            f"Combine into a single clear search query. Return ONLY the query."
        )
        return self.llm.invoke([HumanMessage(content=prompt)]).content.strip()

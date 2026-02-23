
from langgraph.checkpoint.memory import MemorySaver
from utils.logging import logger

PREF_BRAND = "preferred_brand"
PREF_BUDGET = "last_budget"
CTX_BRAND = "brand"


class MemoryManager:
    """Manages LangGraph MemorySaver checkpointer + user preferences."""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.preferences = {}
        self.confirmed_selections = []
        self.conv_context = {}
        logger.info("MemoryManager initialized")

    def get_checkpointer(self) -> MemorySaver:
        return self.checkpointer

    def get_config(self, session_id: str = "default") -> dict:
        return {"configurable": {"thread_id": session_id}}

    def read(self) -> dict:
        return {
            "preferences": self.preferences,
            "confirmed_selections": self.confirmed_selections,
            "conv_context": self.conv_context,
        }

    def write_confirmation(self, query: str, product: dict):
        self.confirmed_selections.append({"query": query, "product": product})
        if "Brands" in product:
            self.preferences[PREF_BRAND] = product["Brands"]
            self.conv_context[CTX_BRAND] = product["Brands"]
        if "Selling Price" in product:
            self.preferences[PREF_BUDGET] = product["Selling Price"]
        logger.info("Memory: confirmed '%s'", product.get("Product", "?"))

    def clear(self):
        self.preferences = {}
        self.confirmed_selections = []
        self.conv_context = {}
        logger.info("Memory cleared")

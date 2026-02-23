def parse_price(price_str: str) -> float | None:
    """Parses a price string like '₹72,990' or '72990' into a float."""
    cleaned = str(price_str).replace(",", "").replace("₹", "").replace("rs.", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None

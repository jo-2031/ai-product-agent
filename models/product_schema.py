"""
Product Schema - Pydantic models for product data
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List
from datetime import date

class Product(BaseModel):
    """Product data model with validation"""
    
    product_id: str = Field(alias="Product ID", description="Unique product identifier")
    product_name: str = Field(alias="Product", description="Product name")
    category: str = Field(alias="Category", default="Other", description="Product category")
    brand: str = Field(alias="Brands", description="Brand name")
    discount: str = Field(alias="Discount", description="Discount percentage")
    rating: str = Field(alias="Rating", description="Product rating out of 5")
    bought_count: str = Field(alias="Bought Count", description="Number of purchases")
    mrp_price: str = Field(alias="MRP Price", description="Maximum retail price")
    selling_price: str = Field(alias="Selling Price", description="Current selling price")
    delivery_date: str = Field(alias="Delivery Date", description="Expected delivery date")
    image_url: Optional[str] = Field(alias="Product Image URL", default="", description="Product image URL")
    provider: str = Field(alias="Provider", description="Provider/platform name")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Product ID": "LP001",
                "Product": "Dell Inspiron 14 Laptop",
                "Brands": "Dell",
                "Discount": "10%",
                "Rating": "4.3",
                "Bought Count": "2456",
                "MRP Price": "75000",
                "Selling Price": "67500",
                "Delivery Date": "2026-03-01",
                "Product Image URL": "https://example.com/image.jpg",
                "Provider": "Amazon"
            }
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary with original field names"""
        return {
            "Product ID": self.product_id,
            "Product": self.product_name,
            "Category": self.category,
            "Brands": self.brand,
            "Discount": self.discount,
            "Rating": self.rating,
            "Bought Count": self.bought_count,
            "MRP Price": self.mrp_price,
            "Selling Price": self.selling_price,
            "Delivery Date": self.delivery_date,
            "Product Image URL": self.image_url,
            "Provider": self.provider
        }


class ProductList(BaseModel):
    """List of products"""
    products: List[Product]
    total_count: int = Field(description="Total number of products")
    
    class Config:
        json_schema_extra = {
            "example": {
                "products": [],
                "total_count": 3
            }
        }


class ProductSearchResponse(BaseModel):
    """Response from product search"""
    products: List[Product]
    query: str = Field(description="Original search query")
    total_found: int = Field(description="Total products found")
    message: str = Field(description="Response message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "products": [],
                "query": "laptop under 70000",
                "total_found": 3,
                "message": "Found 3 laptops matching your criteria"
            }
        }

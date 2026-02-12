from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Hackathon Book Backend", version="1.0.0")

# Import routers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define a simple route for testing
@app.get("/")
async def read_root():
    return {"message": "Hackathon Book Backend API is running!"}

# Include a simple books router without database dependency
from fastapi import APIRouter

books_router = APIRouter(prefix="/api/books", tags=["books"])

# Mock data for testing
mock_books = [
    {
        "id": "1",
        "title": "Test Book",
        "author": "Test Author",
        "description": "This is a test book",
        "aiSummary": "This is a test summary",
        "aiSentiment": [{"label": "POSITIVE", "score": 0.9}]
    }
]

@books_router.get("/")
async def get_books():
    return mock_books

@books_router.get("/{id}")
async def get_book(id: str):
    book = next((book for book in mock_books if book["id"] == id), None)
    if book is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@books_router.post("/")
async def create_book(book: dict):
    book["id"] = str(len(mock_books) + 1)
    mock_books.append(book)
    return book

app.include_router(books_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
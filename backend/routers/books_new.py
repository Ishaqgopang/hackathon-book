from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
import uuid

# Import the storage functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage import get_all_books, get_book_by_id, create_book, update_book, delete_book

# Create router
router = APIRouter(prefix="/api/books", tags=["books"])

# Pydantic models
from pydantic import BaseModel

class Book(BaseModel):
    title: str
    author: str
    description: Optional[str] = None
    isbn: Optional[str] = None
    publishedDate: Optional[str] = None
    pageCount: Optional[int] = None
    categories: Optional[List[str]] = []
    thumbnail: Optional[str] = None
    aiSummary: Optional[str] = None
    aiSentiment: Optional[List[dict]] = []

class BookResponse(Book):
    id: str
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

# Helper function to convert book to response format
def book_to_response_format(book: dict):
    # Create a BookResponse instance with the dictionary values
    return BookResponse(
        id=book.get('id'),
        title=book.get('title', ''),
        author=book.get('author', ''),
        description=book.get('description'),
        isbn=book.get('isbn'),
        publishedDate=book.get('publishedDate'),
        pageCount=book.get('pageCount'),
        categories=book.get('categories', []),
        thumbnail=book.get('thumbnail'),
        aiSummary=book.get('aiSummary'),
        aiSentiment=book.get('aiSentiment', []),
        createdAt=book.get('createdAt'),
        updatedAt=book.get('updatedAt')
    )

# GET /api/books - Get all books
@router.get("/")
async def get_books(skip: int = 0, limit: int = 100):
    books = get_all_books(skip, limit)
    return [book_to_response_format(book) for book in books]

# GET /api/books/{id} - Get a specific book
@router.get("/{id}")
async def get_book(id: str):
    book = get_book_by_id(id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return book_to_response_format(book)

# POST /api/books - Create a new book
@router.post("/")
async def create_book_endpoint(book: Book):
    book_data = book.dict()
    created_book = create_book(book_data)
    return book_to_response_format(created_book)

# PUT /api/books/{id} - Update a book
@router.put("/{id}")
async def update_book_endpoint(id: str, book: Book):
    book_data = book.dict()
    updated_book = update_book(id, book_data)
    if updated_book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return book_to_response_format(updated_book)

# DELETE /api/books/{id} - Delete a book
@router.delete("/{id}")
async def delete_book_endpoint(id: str):
    success = delete_book(id)
    if not success:
        raise HTTPException(status_code=404, detail="Book not found")
    return {"msg": "Book deleted successfully"}
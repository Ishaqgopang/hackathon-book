"""In-memory storage for the backend"""

import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import uuid

# Global in-memory storage
books_storage: List[Dict] = []

def get_all_books(skip: int = 0, limit: int = 100) -> List[Dict]:
    """Get all books with pagination"""
    return books_storage[skip:skip+limit]

def get_book_by_id(book_id: str) -> Optional[Dict]:
    """Get a book by its ID"""
    for book in books_storage:
        if book.get('id') == book_id:
            return book
    return None

def create_book(book_data: Dict) -> Dict:
    """Create a new book"""
    book = book_data.copy()
    book['id'] = str(uuid.uuid4())
    book['createdAt'] = datetime.utcnow()
    book['updatedAt'] = datetime.utcnow()
    books_storage.append(book)
    return book

def update_book(book_id: str, book_data: Dict) -> Optional[Dict]:
    """Update a book"""
    for i, book in enumerate(books_storage):
        if book.get('id') == book_id:
            # Update the book with new data
            updated_book = book.copy()
            updated_book.update(book_data)
            updated_book['id'] = book_id  # Ensure ID stays the same
            updated_book['updatedAt'] = datetime.utcnow()
            books_storage[i] = updated_book
            return updated_book
    return None

def delete_book(book_id: str) -> bool:
    """Delete a book"""
    for i, book in enumerate(books_storage):
        if book.get('id') == book_id:
            del books_storage[i]
            return True
    return False

def search_books(query: str) -> List[Dict]:
    """Search books by title or author"""
    results = []
    query_lower = query.lower()
    for book in books_storage:
        if (query_lower in book.get('title', '').lower() or 
            query_lower in book.get('author', '').lower()):
            results.append(book)
    return results
from fastapi import APIRouter, HTTPException
from typing import List
from bson import ObjectId
from datetime import datetime

# Import models and database
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Book, BookCreate, BookUpdate
from database import get_database

# Create router
router = APIRouter(prefix="/api/books", tags=["books"])

# Helper function to convert MongoDB document to dict
def book_helper(book) -> dict:
    if book is None:
        return None
    return {
        "id": str(book["_id"]) if "_id" in book else book.get("id"),
        "title": book.get("title"),
        "author": book.get("author"),
        "description": book.get("description"),
        "isbn": book.get("isbn"),
        "publishedDate": book.get("publishedDate"),
        "pageCount": book.get("pageCount"),
        "categories": book.get("categories", []),
        "thumbnail": book.get("thumbnail"),
        "aiSummary": book.get("aiSummary"),
        "aiSentiment": book.get("aiSentiment", []),
        "createdAt": book.get("createdAt"),
        "updatedAt": book.get("updatedAt")
    }

# GET /api/books - Get all books
@router.get("/", response_model=List[Book])
async def get_books(skip: int = 0, limit: int = 100):
    from ..database import database  # Get the global database instance
    if database is None:
        # Return empty list if database is not connected
        return []
    books = []
    async for book in database["books"].find().skip(skip).limit(limit):
        books.append(book_helper(book))
    return [Book(**book) for book in books if book is not None]

# GET /api/books/{id} - Get a specific book
@router.get("/{id}", response_model=Book)
async def get_book(id: str):
    from ..database import database  # Get the global database instance
    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="Invalid book ID")
    
    if database is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    book = await database["books"].find_one({"_id": ObjectId(id)})
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    
    return Book(**book_helper(book))

# POST /api/books - Create a new book
@router.post("/", response_model=Book)
async def create_book(book: BookCreate):
    from ..database import database  # Get the global database instance
    if database is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    book_dict = book.dict()
    book_dict["_id"] = ObjectId()
    book_dict["createdAt"] = datetime.utcnow()
    book_dict["updatedAt"] = datetime.utcnow()

    # Process with AI if description is provided
    if book_dict.get("description"):
        try:
            # Import and use AI service
            from ..services.ai_service import ai_service
            # Generate AI summary
            ai_summary = await ai_service.summarize_text(book_dict["description"])
            book_dict["aiSummary"] = ai_summary

            # Analyze sentiment
            ai_sentiment = await ai_service.classify_sentiment(book_dict["description"])
            book_dict["aiSentiment"] = ai_sentiment
        except ImportError:
            # AI service not available
            print("AI service not available")
        except Exception as e:
            print(f"Error processing with AI: {str(e)}")

    result = await database["books"].insert_one(book_dict)
    created_book = await database["books"].find_one({"_id": result.inserted_id})

    return Book(**book_helper(created_book))

# PUT /api/books/{id} - Update a book
@router.put("/{id}", response_model=Book)
async def update_book(id: str, book: BookUpdate):
    from ..database import database  # Get the global database instance
    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="Invalid book ID")

    if database is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    book_dict = {k: v for k, v in book.dict().items() if v is not None}
    book_dict["updatedAt"] = datetime.utcnow()

    # Process with AI if description is provided
    if book_dict.get("description"):
        try:
            # Import and use AI service
            from ..services.ai_service import ai_service
            # Generate AI summary
            ai_summary = await ai_service.summarize_text(book_dict["description"])
            book_dict["aiSummary"] = ai_summary

            # Analyze sentiment
            ai_sentiment = await ai_service.classify_sentiment(book_dict["description"])
            book_dict["aiSentiment"] = ai_sentiment
        except ImportError:
            # AI service not available
            print("AI service not available")
        except Exception as e:
            print(f"Error processing with AI: {str(e)}")

    result = await database["books"].update_one(
        {"_id": ObjectId(id)}, {"$set": book_dict}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Book not found")

    updated_book = await database["books"].find_one({"_id": ObjectId(id)})
    return Book(**book_helper(updated_book))

# DELETE /api/books/{id} - Delete a book
@router.delete("/{id}")
async def delete_book(id: str):
    from ..database import database  # Get the global database instance
    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="Invalid book ID")
    
    if database is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    result = await database["books"].delete_one({"_id": ObjectId(id)})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Book not found")
    
    return {"msg": "Book deleted successfully"}
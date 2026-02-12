import os
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# In-memory storage for books (for Hugging Face Spaces compatibility)
books_storage = []

# Pydantic models
class BookBase(BaseModel):
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

class Book(BookBase):
    id: str
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Application starting for Hugging Face Spaces...")
    yield
    # Shutdown
    print("Application shutting down...")

app = FastAPI(
    title="Hackathon Book Backend (Hugging Face Spaces Version)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Hackathon Book Backend API is running on Hugging Face Spaces!"}

# GET /api/books - Get all books
@app.get("/api/books", response_model=List[Book])
async def get_books(skip: int = 0, limit: int = 100):
    return books_storage[skip:skip+limit]

# GET /api/books/{id} - Get a specific book
@app.get("/api/books/{id}", response_model=Book)
async def get_book(id: str):
    for book in books_storage:
        if book.id == id:
            return book
    raise HTTPException(status_code=404, detail="Book not found")

# POST /api/books - Create a new book
@app.post("/api/books", response_model=Book)
async def create_book(book: BookBase):
    book_obj = Book(
        id=str(uuid.uuid4()),
        **book.dict(),
        createdAt=datetime.utcnow(),
        updatedAt=datetime.utcnow()
    )
    
    # Simple AI processing - just basic text manipulation if description exists
    if book.description:
        # Simple "AI" processing - just create a summary by taking first 100 chars
        if len(book.description) > 100:
            book_obj.aiSummary = book.description[:100] + "..."
        else:
            book_obj.aiSummary = book.description
        
        # Simple sentiment analysis - just a placeholder
        book_obj.aiSentiment = [{"label": "NEUTRAL", "score": 0.5}]
    
    books_storage.append(book_obj)
    return book_obj

# PUT /api/books/{id} - Update a book
@app.put("/api/books/{id}", response_model=Book)
async def update_book(id: str, book_update: BookBase):
    for i, book in enumerate(books_storage):
        if book.id == id:
            # Update the book
            updated_book = Book(
                id=book.id,
                **book_update.dict(),
                createdAt=book.createdAt,
                updatedAt=datetime.utcnow()
            )
            
            # Simple AI processing - just basic text manipulation if description exists
            if book_update.description:
                # Simple "AI" processing - just create a summary by taking first 100 chars
                if len(book_update.description) > 100:
                    updated_book.aiSummary = book_update.description[:100] + "..."
                else:
                    updated_book.aiSummary = book_update.description
                
                # Simple sentiment analysis - just a placeholder
                updated_book.aiSentiment = [{"label": "NEUTRAL", "score": 0.5}]
            
            books_storage[i] = updated_book
            return updated_book
    
    raise HTTPException(status_code=404, detail="Book not found")

# DELETE /api/books/{id} - Delete a book
@app.delete("/api/books/{id}")
async def delete_book(id: str):
    for i, book in enumerate(books_storage):
        if book.id == id:
            del books_storage[i]
            return {"msg": "Book deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Book not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))  # Hugging Face Spaces uses port 7860
    uvicorn.run(app, host="0.0.0.0", port=port)
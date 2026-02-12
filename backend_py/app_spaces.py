import os
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# In-memory storage for books
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
    print("Application starting...")
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
    
    # Process with AI if description is provided
    if book.description:
        try:
            # Dynamically import AI service to avoid startup issues
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from services.ai_service import ai_service
            # Generate AI summary
            ai_summary = await ai_service.summarize_text(book.description)
            book_obj.aiSummary = ai_summary

            # Analyze sentiment
            ai_sentiment = await ai_service.classify_sentiment(book.description)
            book_obj.aiSentiment = ai_sentiment
        except ImportError:
            print("AI service not available - skipping AI processing")
        except Exception as e:
            print(f"Error processing with AI: {str(e)}")
    
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
            
            # Process with AI if description is provided
            if book_update.description:
                try:
                    # Dynamically import AI service to avoid startup issues
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    from services.ai_service import ai_service
                    # Generate AI summary
                    ai_summary = await ai_service.summarize_text(book_update.description)
                    updated_book.aiSummary = ai_summary

                    # Analyze sentiment
                    ai_sentiment = await ai_service.classify_sentiment(book_update.description)
                    updated_book.aiSentiment = ai_sentiment
                except ImportError:
                    print("AI service not available - skipping AI processing")
                except Exception as e:
                    print(f"Error processing with AI: {str(e)}")
            
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
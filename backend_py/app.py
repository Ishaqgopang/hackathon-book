import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
import asyncio

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize FastAPI app with lifespan to handle startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo_connection()

app = FastAPI(
    title="Hackathon Book Backend", 
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

# Import routers after app initialization to avoid circular imports
from routers.books import router as books_router
app.include_router(books_router)

# MongoDB connection parameters
MONGODB_URL = os.getenv("MONGODB_URI", "mongodb://localhost:27017/hackathon-book")

# Global variables for MongoDB connection
mongodb_client = None
database = None

async def connect_to_mongo():
    """Connect to MongoDB"""
    global mongodb_client, database
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        mongodb_client = AsyncIOMotorClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=2000,  # 2 second timeout
            connectTimeoutMS=2000,
        )
        # Extract database name from the connection string
        db_name = MONGODB_URL.split("/")[-1] if "/" in MONGODB_URL else "hackathon-book"
        database = mongodb_client[db_name]
        # Test the connection
        await database.command('ping')
        print("Connected to MongoDB successfully")
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
        print("Running in offline mode - data will not persist")
        # Set database to None so the app knows it's not connected
        database = None

async def close_mongo_connection():
    """Close MongoDB connection"""
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        print("Disconnected from MongoDB")

@app.get("/")
async def read_root():
    return {"message": "Hackathon Book Backend API is running!"}

# Pydantic models
from pydantic import BaseModel as PydanticBaseModel
from typing import List, Optional

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return handler(str)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(str(v)):
            raise ValueError('Invalid ObjectId')
        return ObjectId(str(v))

class Book(PydanticBaseModel):
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
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

# For Hugging Face Spaces, we need to make sure the app runs on port 7860
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))  # Hugging Face Spaces uses port 7860
    uvicorn.run(app, host="0.0.0.0", port=port)
"""MongoDB connection and models for the backend"""

from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB URI from environment, default to local instance
MONGODB_URL = os.getenv("MONGODB_URI", "mongodb://localhost:27017/hackathon-book")

# MongoDB client
client: Optional[AsyncIOMotorClient] = None

# Global variables for MongoDB connection
mongodb_client = None
database = None

async def get_database():
    """Get the database instance"""
    global database
    return database

async def connect_to_mongo():
    """Connect to MongoDB"""
    global mongodb_client, database
    try:
        mongodb_client = AsyncIOMotorClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=1000,  # 1 second timeout
            connectTimeoutMS=1000,
            retryWrites=False  # Disable retry writes for faster failure
        )
        # Extract database name from the connection string
        db_name = MONGODB_URL.split("/")[-1] if "/" in MONGODB_URL else "hackathon-book"
        database = mongodb_client[db_name]
        # Test the connection
        await database.command('ping')
        print("Connected to MongoDB successfully")
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
        print("Make sure MongoDB is running")
        # Set database to None so the app knows it's not connected
        database = None

async def close_mongo_connection():
    """Close MongoDB connection"""
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        print("Disconnected from MongoDB")
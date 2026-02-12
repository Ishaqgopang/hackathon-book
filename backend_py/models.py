from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from pydantic import Field

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
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

class BookCreate(BookBase):
    pass

class BookUpdate(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    isbn: Optional[str] = None
    publishedDate: Optional[str] = None
    pageCount: Optional[int] = None
    categories: Optional[List[str]] = None
    thumbnail: Optional[str] = None
    aiSummary: Optional[str] = None
    aiSentiment: Optional[List[dict]] = None
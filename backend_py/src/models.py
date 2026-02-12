from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: dict, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(dict(type='string'))
    
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
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

class BookInDB(Book):
    id: PyObjectId
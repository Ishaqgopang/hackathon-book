"""Mock database implementation for when MongoDB is not available"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

class MockDatabase:
    def __init__(self):
        self.collections = {}
        self.collections['books'] = []
    
    async def find(self, query=None, skip=0, limit=100):
        """Mock find method"""
        if query is None:
            query = {}
        # Simple filtering based on query
        results = self.collections['books']
        # Apply skip and limit
        return results[skip:skip+limit]
    
    async def find_many(self, query=None, skip=0, limit=100):
        """Mock find_many method to iterate through results"""
        if query is None:
            query = {}
        results = self.collections['books']
        # Apply skip and limit
        for book in results[skip:skip+limit]:
            yield book
    
    async def find_one(self, query):
        """Mock find_one method"""
        if '_id' in query:
            book_id = query['_id']
            for book in self.collections['books']:
                if str(book['_id']) == str(book_id):
                    return book
        return None
    
    async def insert_one(self, document):
        """Mock insert_one method"""
        # Generate a unique ID if not provided
        if '_id' not in document:
            document['_id'] = str(uuid.uuid4())
        self.collections['books'].append(document)
        return MockInsertResult(document['_id'])
    
    async def update_one(self, filter_query, update_data):
        """Mock update_one method"""
        for i, book in enumerate(self.collections['books']):
            if str(book['_id']) == str(filter_query['_id']):
                # Apply the update ($set operation)
                if '$set' in update_data:
                    book.update(update_data['$set'])
                    book['updatedAt'] = datetime.utcnow()
                else:
                    book.update(update_data)
                return MockUpdateResult(1, True)
        return MockUpdateResult(0, False)
    
    async def delete_one(self, filter_query):
        """Mock delete_one method"""
        for i, book in enumerate(self.collections['books']):
            if str(book['_id']) == str(filter_query['_id']):
                del self.collections['books'][i]
                return MockDeleteResult(1)
        return MockDeleteResult(0)
    
    async def command(self, cmd):
        """Mock command method"""
        if cmd == 'ping':
            return {'ok': 1}
        return {}

class MockInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id

class MockUpdateResult:
    def __init__(self, matched_count, modified_count):
        self.matched_count = matched_count
        self.modified_count = modified_count

class MockDeleteResult:
    def __init__(self, deleted_count):
        self.deleted_count = deleted_count

# Global instance
mock_db = MockDatabase()
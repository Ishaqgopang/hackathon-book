import json
from urllib.parse import urlparse, parse_qs

# Import your main application logic here
# For Vercel serverless functions, we need to reimplement the API endpoints

def handler(event, context):
    """
    Vercel serverless function handler
    """
    # Parse the incoming request
    http_method = event['httpMethod']
    path = event['path']
    query_params = parse_qs(event.get('queryStringParameters', {}) or {})
    body = event.get('body')
    
    # Route based on path and method
    if path == '/api/books':
        if http_method == 'GET':
            return get_books(query_params)
        elif http_method == 'POST':
            body_json = json.loads(body) if body else {}
            enable_ai_processing = query_params.get('enable_ai_processing', [True])[0] in [True, 'true']
            return create_book(body_json, enable_ai_processing)
    elif path.startswith('/api/books/') and path.count('/') == 3:  # /api/books/{id}
        book_id = path.split('/')[-1]
        if http_method == 'GET':
            return get_book(book_id)
        elif http_method == 'PUT':
            body_json = json.loads(body) if body else {}
            return update_book(book_id, body_json)
        elif http_method == 'DELETE':
            return delete_book(book_id)
    elif path == '/' and http_method == 'GET':
        return health_check()
    
    # Return 404 if route not found
    return {
        'statusCode': 404,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': '*'
        },
        'body': json.dumps({'error': 'Route not found'})
    }

def health_check():
    """Health check endpoint"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': '*'
        },
        'body': json.dumps({'message': 'Hackathon Book Backend API is running!'})
    }

def get_books(query_params):
    """Get all books"""
    # This would connect to MongoDB in a real implementation
    skip = int(query_params.get('skip', [0])[0])
    limit = int(query_params.get('limit', [100])[0])
    
    # Mock response
    books = [
        {
            "id": "1",
            "title": "Sample Book",
            "author": "Sample Author",
            "description": "This is a sample book description",
            "aiSummary": "This is a sample AI summary",
            "aiSentiment": [{"label": "POSITIVE", "score": 0.9}]
        }
    ]
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': '*'
        },
        'body': json.dumps(books)
    }

def get_book(book_id):
    """Get a specific book by ID"""
    # Mock response
    book = {
        "id": book_id,
        "title": "Sample Book",
        "author": "Sample Author",
        "description": "This is a sample book description",
        "aiSummary": "This is a sample AI summary",
        "aiSentiment": [{"label": "POSITIVE", "score": 0.9}]
    }
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': '*'
        },
        'body': json.dumps(book)
    }

def create_book(book_data, enable_ai_processing=True):
    """Create a new book"""
    # In a real implementation, this would:
    # 1. Validate the input
    # 2. Process with AI if enabled
    # 3. Save to database
    
    # Mock response - assign an ID and return
    book_data["id"] = "new_id_123"
    
    # If AI processing is enabled and there's a description, process it
    if enable_ai_processing and book_data.get("description"):
        # This would call the AI service in a real implementation
        book_data["aiSummary"] = f"AI summary of: {book_data['description'][:50]}..."
        book_data["aiSentiment"] = [{"label": "NEUTRAL", "score": 0.5}]
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': '*'
        },
        'body': json.dumps(book_data)
    }

def update_book(book_id, book_data):
    """Update a book"""
    book_data["id"] = book_id
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': '*'
        },
        'body': json.dumps(book_data)
    }

def delete_book(book_id):
    """Delete a book"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': '*'
        },
        'body': json.dumps({"msg": "Book deleted successfully"})
    }
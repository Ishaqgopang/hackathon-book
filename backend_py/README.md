# Hackathon Book Backend (Python/FastAPI)

This is the Python backend for the hackathon book project, built with FastAPI and MongoDB.

## Setup

1. Install Python 3.8 or higher

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root of the backend_py directory with the following:
```env
MONGODB_URI=mongodb://localhost:27017/hackathon-book
PORT=5000
HUGGING_FACE_TOKEN=your_huggingface_token_here
```

5. Start the development server:
```bash
python -m src.main
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --port 5000
```

## Available Scripts

- `python -m src.main` - Start the production server
- `uvicorn main:app --reload` - Start the development server with hot reload

## API Routes

- `GET /` - Health check
- `GET /api/books` - Get all books
- `GET /api/books/{id}` - Get a specific book
- `POST /api/books` - Create a new book (supports AI processing with `enable_ai_processing` query param)
- `PUT /api/books/{id}` - Update a book
- `DELETE /api/books/{id}` - Delete a book

## AI Features

The backend includes AI-powered features using Hugging Face integration:
- Text summarization of book descriptions
- Sentiment analysis of book content
- Enhanced book management with AI insights

To enable AI processing when creating a book, include `enable_ai_processing=true` as a query parameter.
# Hackathon Book Backend (Python/FastAPI) - Vercel Deployment

This is the Python backend for the hackathon book project, built with Python and designed for Vercel deployment using serverless functions.

## Vercel Deployment

### Prerequisites
- Vercel account (sign up at https://vercel.com)
- Vercel CLI installed: `npm i -g vercel`

### Deployment Steps

1. Navigate to the backend directory:
```bash
cd backend_py
```

2. Set up environment variables in Vercel:
   - `MONGODB_URI`: Your MongoDB connection string
   - `HUGGING_FACE_TOKEN`: Your Hugging Face API token (optional)

3. Deploy to Vercel:
```bash
vercel --prod
```

Or connect your GitHub repository to Vercel for automatic deployments.

### Alternative: Deploy via Vercel Dashboard

1. Go to your [Vercel dashboard](https://vercel.com/dashboard)
2. Click "Add New..." → "Project"
3. Import your repository
4. Set the root directory to `backend_py`
5. Add the following build command: `pip install -r requirements.txt`
6. Set the output directory appropriately
7. Add environment variables:
   - `MONGODB_URI`: Your MongoDB connection string
   - `HUGGING_FACE_TOKEN`: Your Hugging Face API token (optional)
8. Deploy!

## Local Development

For local development, you can still run the traditional FastAPI server:

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your environment variables:
```env
MONGODB_URI=your_mongodb_connection_string
HUGGING_FACE_TOKEN=your_huggingface_token
```

4. Run with Uvicorn:
```bash
uvicorn main:app --reload --port 5000
```

## API Routes

- `GET /api/` - Health check
- `GET /api/books` - Get all books
- `GET /api/books/{id}` - Get a specific book
- `POST /api/books?enable_ai_processing=true` - Create a new book (supports AI processing)
- `PUT /api/books/{id}` - Update a book
- `DELETE /api/books/{id}` - Delete a book

## Architecture

This backend uses Vercel's serverless functions to provide API endpoints. Each endpoint is implemented as a separate Python function that can be deployed independently. The backend connects to MongoDB for data persistence and integrates with Hugging Face for AI features.

## Notes

- The serverless function approach means each request initializes the application context
- For production use, consider connection pooling and caching strategies
- The Hugging Face integration works best when the token is provided as an environment variable
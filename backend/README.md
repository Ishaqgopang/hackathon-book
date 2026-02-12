# Hackathon Book Backend

This is the backend for the hackathon book project with AI enhancement capabilities via Hugging Face integration.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file in the root of the backend directory with the following:
```env
MONGODB_URI=mongodb://localhost:27017/hackathon-book
PORT=5000
HUGGING_FACE_TOKEN=your_huggingface_token_here
```

3. Start the development server:
```bash
npm run dev
```

## Available Scripts

- `npm start` - Start the production server
- `npm run dev` - Start the development server with nodemon
- `npm test` - Run tests

## API Routes

- `GET /` - Health check
- `GET /api/books` - Get all books
- `POST /api/books` - Create a new book (supports AI processing with `enableAIProcessing` flag)

## AI Features

The backend includes AI-powered features using Hugging Face integration:
- Text summarization of book descriptions
- Sentiment analysis of book content
- Enhanced book management with AI insights

To enable AI processing when creating a book, include `"enableAIProcessing": true` in your POST request body.
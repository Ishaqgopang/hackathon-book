# Hackathon Book Backend on Hugging Face

This is the backend for the Hackathon Book project, deployed on Hugging Face Spaces. It provides a RESTful API for managing books with AI-powered features using Hugging Face models.

## Features

- Book management (CRUD operations)
- AI-powered text summarization of book descriptions
- Sentiment analysis of book content
- Integration with Hugging Face models for NLP tasks

## API Endpoints

- `GET /` - Health check
- `GET /api/books` - Get all books
- `GET /api/books/{id}` - Get a specific book
- `POST /api/books` - Create a new book (with AI processing)
- `PUT /api/books/{id}` - Update a book
- `DELETE /api/books/{id}` - Delete a book

## Hugging Face Integration

The backend uses Hugging Face models for:
- Text summarization (using facebook/bart-large-cnn)
- Sentiment analysis (using cardiffnlp/twitter-roberta-base-sentiment-latest)
- Text generation (using gpt2)

## Environment Variables

To run this application, you'll need to set the following environment variables:

- `HUGGING_FACE_TOKEN`: Your Hugging Face API token
- `MONGODB_URI`: MongoDB connection string (optional, for persistent storage)

## Usage

The API is accessible at the root URL of this Space. You can make HTTP requests to interact with the book management system.

When creating or updating books with descriptions, the system will automatically process the text using Hugging Face models to generate summaries and analyze sentiment.
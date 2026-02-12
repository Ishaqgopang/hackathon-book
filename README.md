# Hackathon Book Project

A comprehensive full-stack application featuring book management with AI enhancements and modern deployment capabilities.

## Project Structure

- `backend_py/` - Python/FastAPI API server with MongoDB for book management
- `frontend_book/` - Docusaurus-based documentation site for the Physical AI & Humanoid Robotics course
- `mcp-server/` - GitHub Model Context Protocol server for AI context management

## Features

### Backend API
- Full CRUD operations for book management
- MongoDB integration for data persistence
- AI-powered features using Hugging Face integration:
  - Text summarization of book descriptions
  - Sentiment analysis
  - Text generation capabilities

### Frontend Documentation Site
- Built with Docusaurus
- Comprehensive course materials on Physical AI & Humanoid Robotics
- Responsive design and easy navigation

### MCP Server
- GitHub Model Context Protocol implementation
- Standardized API for AI systems to access book context
- Tools for searching, retrieving, and summarizing book information

### Deployment Options
- Vercel deployment configuration for frontend
- Standard Node.js deployment for backend
- MCP server for AI context protocols

## Setup Instructions

### Python Backend Setup

1. Navigate to the Python backend directory:
```bash
cd backend_py
```

2. Install Python and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the backend_py directory with the following:
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

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend_book
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

### MCP Server Setup

1. Navigate to the MCP server directory:
```bash
cd mcp-server
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file in the mcp-server directory with the following:
```env
MCP_PORT=8080
```

4. Start the MCP server:
```bash
npm run dev
```

## Environment Variables

### Backend
- `MONGODB_URI` - MongoDB connection string
- `PORT` - Port for the backend server
- `HUGGING_FACE_TOKEN` - Token for Hugging Face API access

### MCP Server
- `MCP_PORT` - Port for the MCP server

## Deployment

### Vercel Deployment

The frontend is configured for deployment on Vercel. To deploy:

1. Install the Vercel CLI:
```bash
npm i -g vercel
```

2. Navigate to the frontend directory and run:
```bash
vercel
```

Follow the prompts to configure and deploy your site.

### Backend Deployment

The backend can be deployed to any Node.js hosting platform (Heroku, AWS, etc.) with access to a MongoDB database.

## API Endpoints

### Books API
- `GET /api/books` - Get all books
- `GET /api/books/{id}` - Get a specific book
- `POST /api/books` - Create a new book (supports AI processing with `enable_ai_processing` query parameter)
- `PUT /api/books/{id}` - Update a book
- `DELETE /api/books/{id}` - Delete a book

## MCP Server Endpoints

### Health Check
- `GET /mcp/health` - Check server health status

### Tools
- `GET /mcp/tools/list` - List available tools
- `POST /mcp/tools/get_book_context` - Retrieve context for a specific book
- `POST /mcp/tools/search_books` - Search for books based on query
- `POST /mcp/tools/summarize_book_content` - Generate summary of book content

### Resources
- `GET /mcp/resources/list` - List available resources
- `POST /mcp/resources/subscribe` - Subscribe to resource updates
- `POST /mcp/resources/unsubscribe` - Unsubscribe from resource updates

## Docker Deployment

The project includes Docker support for easy deployment of all services.

### Prerequisites

- Docker
- Docker Compose

### Setup

1. Create a `.env` file in the root directory with your Hugging Face token:
```env
HUGGING_FACE_TOKEN=your_huggingface_token_here
```

2. Build and start all services:
```bash
docker-compose up --build
```

3. Access the services:
   - Backend API: http://localhost:5000
   - MCP Server: http://localhost:8080
   - Frontend: http://localhost:3000
   - MongoDB: localhost:27017 (for external access)

### Services

The docker-compose configuration includes:
- MongoDB database
- Backend API server with Hugging Face integration
- MCP (Model Context Protocol) server
- Frontend Docusaurus site

### Development

For development, you can start individual services:
```bash
# Start only the backend and database
docker-compose up --build backend mongodb

# Start all services in detached mode
docker-compose up --build -d
```

## Cloud Deployment

### Deploy Frontend to Vercel

The frontend can be deployed to Vercel as a static site:

1. Fork this repository
2. Sign up for a Vercel account at https://vercel.com
3. Install the Vercel CLI: `npm i -g vercel`
4. Navigate to the frontend directory: `cd frontend_book`
5. Run `vercel` and follow the prompts
6. Connect your GitHub repository to Vercel for automatic deployments

Alternatively, you can import the project directly from GitHub in the Vercel dashboard:
1. Go to your Vercel dashboard
2. Click "New Project"
3. Import your forked repository
4. Select the `frontend_book` directory
5. Add the following build command: `npm run build`
6. Set the output directory to `build`
7. Deploy!

### Deploy Backend to Vercel

The backend can also be deployed to Vercel using serverless functions:

1. Fork this repository
2. Navigate to the backend directory: `cd backend_py`
3. Create a Vercel project and link it to your repository
4. Set the root directory to `backend_py`
5. Add environment variables:
   - `MONGODB_URI`: Your MongoDB connection string
   - `HUGGING_FACE_TOKEN`: Your Hugging Face API token (optional)
6. Deploy!

For more details, see the backend Vercel documentation in `backend_py/README.vercel.md`.

### Deploy Frontend to Hugging Face Spaces

The frontend can also be deployed to Hugging Face Spaces as a static site:

1. Fork this repository
2. Create a new Space on Hugging Face
3. Select "Docker" SDK and "CPU" hardware
4. Add the following to your `Dockerfile` (already included as `Dockerfile.hf`):

```
FROM nginx:alpine

# Install Node.js and npm
RUN apk add --no-cache nodejs npm

WORKDIR /app

# Copy package files
COPY frontend_book/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY frontend_book/ .

# Build the Docusaurus site
RUN npm run build

# Copy the built site to nginx directory
RUN rm -rf /usr/share/nginx/html/*
RUN cp -r build/* /usr/share/nginx/html/

# Expose port
EXPOSE 7860

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
```

5. Push your code to the Space repository

### Deploy Backend to Cloud Platforms

#### Heroku Deployment

1. Create a Heroku account and install the Heroku CLI
2. Fork this repository
3. Create a new Heroku app
4. Link your repository to Heroku
5. Set the following config vars in Heroku:
   - `MONGODB_URI`: Your MongoDB connection string
   - `HUGGING_FACE_TOKEN`: Your Hugging Face API token
6. Deploy the app

#### Railway Deployment

1. Create a Railway account
2. Fork this repository
3. Create a new Railway project
4. Link your repository to Railway
5. Set the following variables:
   - `MONGODB_URI`: Your MongoDB connection string
   - `HUGGING_FACE_TOKEN`: Your Hugging Face API token
6. Deploy the app using the `railway.yml` configuration

#### Manual Deployment

For other platforms, simply:
1. Clone the repository
2. Install dependencies: `npm install`
3. Set environment variables:
   - `MONGODB_URI`: Your MongoDB connection string
   - `PORT`: Desired port (defaults to 5000)
   - `HUGGING_FACE_TOKEN`: Your Hugging Face API token
4. Start the server: `npm start`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
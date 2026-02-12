const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.MCP_PORT || 8080;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// In-memory storage for contexts (in production, use a database)
let contexts = new Map();
let currentRequestId = 0;

// MCP Protocol Routes
// https://modelcontextprotocol.io/docs/spec

// Root route for health check
app.get('/', (req, res) => {
  res.json({
    message: 'GitHub Model Context Protocol Server is running',
    protocol_version: '1.0',
    endpoints: [
      '/mcp/health',
      '/mcp/tools/list',
      '/mcp/resources/list',
      '/mcp/resources/subscribe',
      '/mcp/resources/unsubscribe'
    ]
  });
});

// Health check endpoint
app.get('/mcp/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Tools endpoints
app.get('/mcp/tools/list', (req, res) => {
  res.json({
    tools: [
      {
        name: 'get_book_context',
        description: 'Retrieve context information about a specific book',
        input_schema: {
          type: 'object',
          properties: {
            book_id: { type: 'string', description: 'The ID of the book to retrieve context for' }
          },
          required: ['book_id']
        }
      },
      {
        name: 'search_books',
        description: 'Search for books based on query',
        input_schema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query for books' },
            limit: { type: 'number', description: 'Maximum number of results to return' }
          },
          required: ['query']
        }
      },
      {
        name: 'summarize_book_content',
        description: 'Generate a summary of book content',
        input_schema: {
          type: 'object',
          properties: {
            book_id: { type: 'string', description: 'The ID of the book to summarize' },
            length: { type: 'number', description: 'Preferred summary length in sentences' }
          },
          required: ['book_id']
        }
      }
    ]
  });
});

// Resources endpoints
app.get('/mcp/resources/list', (req, res) => {
  res.json({
    resources: [
      {
        uri: 'resource://hackathon-book/book/{id}',
        name: 'book',
        description: 'Book resource with detailed information',
        schema: {
          type: 'object',
          properties: {
            id: { type: 'string' },
            title: { type: 'string' },
            author: { type: 'string' },
            description: { type: 'string' },
            aiSummary: { type: 'string' },
            aiSentiment: { type: 'array' }
          }
        }
      },
      {
        uri: 'resource://hackathon-book/search-results',
        name: 'search_results',
        description: 'Results from book searches',
        schema: {
          type: 'object',
          properties: {
            query: { type: 'string' },
            results: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  id: { type: 'string' },
                  title: { type: 'string' },
                  author: { type: 'string' }
                }
              }
            }
          }
        }
      }
    ]
  });
});

// Tool implementations
app.post('/mcp/tools/get_book_context', async (req, res) => {
  try {
    const { book_id } = req.body;
    
    // In a real implementation, this would fetch from the backend API
    // For now, we'll simulate the response
    const mockBookData = {
      id: book_id,
      title: `Mock Book Title for ID: ${book_id}`,
      author: 'Mock Author',
      description: 'This is a mock description for demonstration purposes. In a real implementation, this would fetch actual book data from the backend API.',
      aiSummary: 'This is a mock AI summary generated for demonstration purposes.',
      aiSentiment: [
        { label: 'POSITIVE', score: 0.85 },
        { label: 'NEGATIVE', score: 0.10 },
        { label: 'NEUTRAL', score: 0.05 }
      ]
    };
    
    res.json({
      result: {
        content: [
          {
            type: 'text',
            text: JSON.stringify(mockBookData, null, 2)
          }
        ]
      }
    });
  } catch (error) {
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message
      }
    });
  }
});

app.post('/mcp/tools/search_books', async (req, res) => {
  try {
    const { query, limit = 10 } = req.body;
    
    // Mock search results
    const mockResults = Array.from({ length: Math.min(limit, 5) }, (_, i) => ({
      id: `mock-book-id-${i + 1}`,
      title: `Mock Book Result ${i + 1} for query: ${query}`,
      author: 'Mock Author',
      description: `This is a mock book description for the search result ${i + 1}`
    }));
    
    res.json({
      result: {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              query,
              results: mockResults
            }, null, 2)
          }
        ]
      }
    });
  } catch (error) {
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message
      }
    });
  }
});

app.post('/mcp/tools/summarize_book_content', async (req, res) => {
  try {
    const { book_id, length = 3 } = req.body;
    
    // Mock summary
    const mockSummary = `This is a mock summary for book ID ${book_id}. In a real implementation, this would call the backend API to get the book content and generate an AI-powered summary. The summary would be approximately ${length} sentences long.`;
    
    res.json({
      result: {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              book_id,
              summary: mockSummary
            }, null, 2)
          }
        ]
      }
    });
  } catch (error) {
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message
      }
    });
  }
});

// Resource subscription endpoints (simplified for demo)
const subscriptions = new Set();

app.post('/mcp/resources/subscribe', (req, res) => {
  const { uri } = req.body;
  subscriptions.add(uri);
  
  res.json({
    status: 'subscribed',
    uri
  });
});

app.post('/mcp/resources/unsubscribe', (req, res) => {
  const { uri } = req.body;
  subscriptions.delete(uri);
  
  res.json({
    status: 'unsubscribed',
    uri
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: {
      code: 'INTERNAL_ERROR',
      message: 'An internal server error occurred'
    }
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: {
      code: 'NOT_FOUND',
      message: `Endpoint ${req.method} ${req.path} not found`
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`MCP Server is running on port ${PORT}`);
  console.log(`Access the server at: http://localhost:${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/mcp/health`);
});

module.exports = app;
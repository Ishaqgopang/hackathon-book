# GitHub Model Context Protocol (MCP) Server

This is a Model Context Protocol server that integrates with the hackathon book project to provide AI context management capabilities.

## Overview

The Model Context Protocol (MCP) server provides a standardized way for AI models and assistants to access contextual information about books in the hackathon book system. It implements the [MCP specification](https://modelcontextprotocol.io/) to allow AI systems to retrieve, search, and summarize book information.

## Features

- Standardized API endpoints following MCP specification
- Tools for retrieving book context, searching books, and generating summaries
- Resource management for book data
- Integration-ready with AI systems and assistants

## Endpoints

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

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file in the root of the mcp-server directory with the following:
```env
MCP_PORT=8080
```

3. Start the development server:
```bash
npm run dev
```

## Configuration

The server can be configured using environment variables:

- `MCP_PORT` - Port number for the server (default: 8080)

## Integration

This MCP server can be integrated with AI systems that support the Model Context Protocol, allowing them to access book information from the hackathon book project in a standardized way.
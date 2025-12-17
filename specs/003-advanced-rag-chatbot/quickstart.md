# Quickstart: Consolidated RAG Chatbot for AI Textbook

## Setup

1. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn qdrant-client openai python-dotenv requests beautifulsoup4 langchain sentence-transformers
   ```

2. **Set up environment variables**:
   ```bash
   # Create .env file with:
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   DATABASE_URL=your_neon_postgres_connection_string
   SECRET_KEY=your_secret_key_for_session_management
   RATE_LIMIT_REQUESTS=100
   RATE_LIMIT_WINDOW=3600  # 1 hour in seconds
   ```

3. **Initialize Qdrant collection**:
   The system will automatically create a collection named "rag_embedding" on first run.

## Running the Service

1. **Start the backend**:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Ingest textbook content**:
   ```bash
   curl -X POST "http://localhost:8000/ingest-book" \
     -H "Content-Type: application/json" \
     -d '{"base_url": "https://myphysicalaihumanoidtextbook.vercel.app/"}'
   ```

3. **Test the API**:
   ```bash
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Physical AI?"}'
   ```

## Integration with Docusaurus

To embed the chat widget in your Docusaurus site:

1. Add the ChatKit React component to your pages
2. Use the `/chat` endpoint for conversation functionality
3. Implement text selection feature to use the "ask about selected text" mode

## API Endpoints

The following v2 endpoints are available (advanced capabilities):

- `POST /query-v2` - Submit a query and receive a response with citations
- `POST /chat-v2` - Submit a chat message in a session context
- `POST /rerank` - Rerank retrieved chunks based on relevance to the query
- `GET /history/{session_id}` - Retrieve chat history for a session

## Testing

Run backend tests:
```bash
cd backend
python -m pytest tests/
```

## Architecture Overview

The system consists of:

1. **Backend**: FastAPI application handling RAG logic, session management, and API endpoints
2. **Vector Storage**: Qdrant for storing and retrieving textbook embeddings
3. **Session Storage**: Neon Postgres for persistent chat sessions
4. **Frontend**: ChatKit integration in Docusaurus site for user interaction

## Key Features

- Basic RAG: Initial implementation with whole-book semantic search
- Multi-section RAG: Queries span multiple textbook chapters/sections
- Persistent sessions: Chat history preserved across visits
- Structured citations: Sources attributed in "Section > Page > Paragraph" format
- Selection override: Questions about highlighted text bypass global retrieval
- Rate limiting: Middleware prevents system abuse
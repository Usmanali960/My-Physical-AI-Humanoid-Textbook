# Research: Consolidated RAG Chatbot for AI Textbook

## Decision: Backend Framework Selection
**Rationale**: FastAPI was chosen as the backend framework because it provides high performance, built-in async support, automatic API documentation (Swagger UI), and excellent integration with Python data science libraries needed for RAG implementation.

**Alternatives considered**:
- Flask: More familiar but less performant for async operations
- Django: Too heavy for this specific use case
- Express.js: Would require switching to JavaScript ecosystem

## Decision: Vector Database Selection
**Rationale**: Qdrant was selected as the vector database due to its excellent performance, Python SDK, open-source availability, and efficient similarity search capabilities essential for RAG implementations.

**Alternatives considered**:
- Pinecone: Cloud-only, commercial
- Weaviate: Good alternative but Qdrant has better performance benchmarks for this use case
- FAISS: Good for similarity search but lacks built-in API layer

## Decision: Embedding Model Selection
**Rationale**: Using OpenAI's embedding API for its proven quality and reliability, though we could use open-source alternatives like Sentence Transformers as fallback.

**Alternatives considered**:
- Sentence Transformers (all-MiniLM-L6-v2): Free but potentially less accurate
- Cohere embeddings: Good quality but vendor lock-in
- HuggingFace models: Self-hosted option but more complex deployment

## Decision: Textbook Content Retrieval
**Rationale**: Using the deployed textbook URL (https://myphysicalaihumanoidtextbook.vercel.app/) to extract content programmatically, ensuring we always have the latest version of the textbook.

**Approach**: Implement get_all_urls to crawl the site and extract_text_from_url to parse content from each page.

## Decision: Frontend Integration
**Rationale**: Implementing a React chat widget that can be embedded in Docusaurus pages to provide the "ask about selected text" functionality.

**Technical approach**: Using React components that can capture selected text and send it to the backend API with appropriate context.

## Decision: Multi-section RAG Architecture
**Rationale**: The system needs to retrieve information from 3-6 sections of the textbook per query to provide comprehensive answers. This requires modifying the existing RAG system to perform multiple retrievals and merge the context.

**Alternatives Considered**:
- Single section retrieval (insufficient for complex queries)
- Full-text search without section boundaries (less precise)
- Vector database with section metadata (complex implementation)

## Decision: Persistent Session Storage with Neon Postgres
**Rationale**: Neon Postgres provides serverless PostgreSQL with built-in branching and storage optimization, making it ideal for persistent chat session storage.

**Alternatives Considered**:
- In-memory storage (not persistent across sessions)
- File-based storage (not scalable)
- Redis (additional infrastructure component)

## Decision: ChatKit Integration for Frontend
**Rationale**: Using ChatKit for the frontend UI ensures a consistent, well-designed chat interface that integrates well with the existing Docusaurus site.

**Alternatives Considered**:
- Custom-built chat UI (time-consuming)
- Alternative chat libraries (would diverge from spec requirements)

## Decision: Rate Limiting and Safety Middleware
**Rationale**: Implementing middleware for rate limiting and safety ensures service availability and prevents abuse of the system.

**Alternatives Considered**:
- No rate limiting (vulnerable to abuse)
- Client-side rate limiting (easily bypassed)
- Different middleware frameworks (FastAPI's built-in middleware is sufficient)

## Decision: Citation Format Implementation
**Rationale**: Citations in "Section > Page > Paragraph" format provide precise attribution that helps students verify information and locate original content.

**Alternatives Considered**:
- General chapter citations (less precise)
- No citations (violates academic requirements)
- URL-based citations (not appropriate for textbook content)

## Decision: User Selection Override Mode
**Rationale**: The selection-only mode allows users to ask questions about specific text they've highlighted, bypassing global retrieval when appropriate.

**Alternatives Considered**:
- No selection mode (reduces functionality)
- Toggle-based mode (less user-friendly)
- Context-aware selection (more complex implementation)
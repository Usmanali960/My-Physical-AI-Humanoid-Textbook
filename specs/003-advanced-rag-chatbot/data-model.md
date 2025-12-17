# Data Model: Consolidated RAG Chatbot for AI Textbook

## Entity: DocumentChunk

Represents a segment of text from the textbook that has been processed for vector search, with metadata for proper citation and retrieval.

**Fields**:
- id: UUID (Primary Key)
- text: Text (The actual text content of the chunk)
- source_url: String (URL where this content originated from)
- source_section: String (Which section the content comes from)
- source_page: String (Which page the content comes from)
- source_paragraph: String (Which paragraph identifier)
- embedding: Vector representation of the text content
- embedding_id: String (Reference to the vector in Qdrant)
- relevance_score: Float (Score from 0.0 to 1.0, for retrieval ranking)
- metadata: JSON object with additional information about the content
- created_at: DateTime
- updated_at: DateTime

**Validation Rules**:
- text must not be empty
- relevance_score must be between 0.0 and 1.0
- At least one of source_url, source_section, source_page, or source_paragraph must be provided
- embedding_id must be unique
- created_at and updated_at are automatically managed by the system

## Entity: ChatSession

Represents a user's conversation with the RAG chatbot, containing the history of interactions that persists across visits and maintains context for follow-up questions.

**Fields**:
- id: UUID (Primary Key)
- user_id: String (Optional, for logged-in users)
- session_token: String (For anonymous users)
- created_at: DateTime
- updated_at: DateTime
- metadata: JSON object for additional session data

**Validation Rules**:
- session_token must be unique
- created_at and updated_at are automatically managed by the system
- Either user_id or session_token must be provided

## Entity: Message

Represents a single message in a chat session, either from the user or the AI.

**Fields**:
- id: UUID (Primary Key)
- session_id: UUID (Foreign Key to ChatSession)
- message_id: String (Unique identifier for the message)
- content: Text (The actual query or response text)
- role: String (Either "user" or "assistant")
- timestamp: DateTime
- citations: List of CitationReference objects
- selected_text: Text (Optional selected text that was the basis for this message)

**Validation Rules**:
- role must be either "user" or "assistant"
- content cannot be empty
- timestamp is automatically set when the message is created
- session_id must reference an existing ChatSession

## Entity: CitationReference

Structured reference that links specific information in a response back to its original location in the textbook (Section > Page > Paragraph format).

**Fields**:
- id: UUID (Primary Key)
- message_id: UUID (Reference to a Message)
- source_section: String
- source_page: String
- source_paragraph: String
- source_url: String
- text_excerpt: Text (The specific text that's being cited)
- created_at: DateTime

**Validation Rules**:
- source_section is required
- At least one of source_page, source_paragraph, or source_url must be provided
- text_excerpt must not exceed 500 characters
- message_id must reference an existing Message

## Entity: TextbookSection

A discrete part of the AI textbook that can be independently referenced and retrieved as part of the multi-section RAG process.

**Fields**:
- id: UUID (Primary Key)
- section_title: String
- section_path: String (Path in the textbook structure)
- section_content: Text (The full text of the section)
- embedding_metadata: JSON (Metadata for vector storage)
- created_at: DateTime
- updated_at: DateTime

**Validation Rules**:
- section_title cannot be empty
- section_path must be unique
- section_content cannot be empty

## Entity: RateLimitToken

Entity that manages user access patterns to prevent system abuse while ensuring fair access for legitimate users.

**Fields**:
- id: UUID (Primary Key)
- user_id: String (User identifier, can be IP address for anonymous users)
- endpoint: String (Which API endpoint is being limited)
- request_count: Integer (Number of requests in the current window)
- window_start: DateTime (When the current rate limit window started)
- created_at: DateTime

**Validation Rules**:
- request_count cannot be negative
- window_start must be in the past
- endpoint must be one of the defined API endpoints

## Relationships

- ChatSession has many Messages
- Message has many CitationReferences
- DocumentChunk has many CitationReferences (as source of citations)
- TextbookSection connects to DocumentChunk (as the parent section)
# Personal Knowledge Assistant

A RAG (Retrieval-Augmented Generation) powered knowledge base copilot that lets you chat with your documents.

## Features

- **Document Upload**: Support for PDF, Markdown, and TXT files
- **Smart Chunking**: Configurable text splitting with overlap strategies
- **Vector Search**: FAISS-based semantic search with configurable similarity thresholds
- **Streaming Responses**: Real-time AI responses with SSE streaming
- **Chat Sessions**: Persistent conversation history with session management
- **Dark/Light Theme**: System-aware theming with manual toggle
- **Multi-format Support**: Handles complex PDFs, Markdown documents, and plain text

## Architecture

```
Personal-knowledge-assistant/
├── knowledge-copilot/
│   ├── backend/              # FastAPI Python backend
│   │   ├── app/
│   │   │   ├── api/         # API endpoints
│   │   │   ├── core/        # Config & error handling
│   │   │   └── services/    # Business logic
│   │   ├── data/            # Uploaded files & vector store
│   │   └── venv/            # Python virtual environment
│   │
│   └── frontend/            # Next.js React frontend
│       ├── app/             # Next.js app router
│       ├── components/      # UI components
│       ├── hooks/           # Custom React hooks
│       └── lib/             # API client
```

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **FAISS** - Vector similarity search
- **LangChain** - LLM orchestration
- **Sentence Transformers** - Local embeddings (all-MiniLM-L6-v2)
- **PyPDF** - PDF document parsing
- **SlowAPI** - Rate limiting

### Frontend
- **Next.js 16** - React framework
- **React 19** - UI library
- **Tailwind CSS** - Styling
- **react-markdown** - Markdown rendering

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Ollama (for local LLM) or OpenAI API key

### Backend Setup

```bash
cd knowledge-copilot/backend

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Create .env file with your configuration
# Example .env:
cat > .env << 'EOF'
# Embedding settings
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_LOCAL=all-MiniLM-L6-v2

# LLM settings (Ollama)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:latest

# Or use OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your-api-key
# LLM_MODEL=gpt-3.5-turbo
EOF

# Start the backend server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd knowledge-copilot/frontend

# Install dependencies
npm install

# Create .env.local file
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000
EOF

# Start the development server
npm run dev
```

### Access the Application

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Configuration

### Environment Variables

#### Backend (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | Knowledge Copilot | Application name |
| `DEBUG` | true | Enable debug mode |
| `EMBEDDING_PROVIDER` | local | Embedding provider: `local` or `openai` |
| `EMBEDDING_MODEL_LOCAL` | all-MiniLM-L6-v2 | Local embedding model |
| `EMBEDDING_MODEL_OPENAI` | text-embedding-3-small | OpenAI embedding model |
| `OPENAI_API_KEY` | - | OpenAI API key (required if using OpenAI embeddings) |
| `VECTOR_STORE_PROVIDER` | faiss | Vector store: `faiss` or `chroma` |
| `LLM_PROVIDER` | ollama | LLM provider: `openai` or `ollama` |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama server URL |
| `OLLAMA_MODEL` | llama3:latest | Ollama model name |
| `LLM_TEMPERATURE` | 0.2 | LLM response creativity |
| `LLM_MAX_TOKENS` | 1024 | Maximum response length |

#### Frontend (`.env.local`)

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend API URL (default: http://localhost:8000) |

### Ollama Setup

Install and run Ollama for local LLM inference:

```bash
# Install Ollama
# Visit https://ollama.ai for installation instructions

# Pull a model
ollama pull llama3:latest

# Start Ollama server (usually runs automatically)
ollama serve
```

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/sessions` | Create new chat session |
| `GET` | `/api/v1/sessions/{id}` | Get session details |
| `GET` | `/api/v1/sessions` | List all sessions |
| `DELETE` | `/api/v1/sessions/{id}` | Delete a session |
| `POST` | `/api/v1/documents` | Upload and index document |
| `GET` | `/api/v1/documents/status` | Get indexing status |
| `POST` | `/api/v1/ask` | Ask a question |

### Ask Request Example

```json
{
  "session_id": "uuid-here",
  "query": "What are the main conclusions?",
  "k": 5,
  "score_threshold": 0.3,
  "stream": true
}
```

### Ask Response (Non-streaming)

```json
{
  "session_id": "uuid-here",
  "query": "What are the main conclusions?",
  "answer": "Based on the documents...",
  "sources": [
    {
      "file_name": "document.pdf",
      "page": 1,
      "score": 0.85,
      "preview": "The main conclusions are..."
    }
  ],
  "context_used": true,
  "meta": {
    "chunks_retrieved": 3,
    "model": "llama3:latest",
    "provider": "ollama"
  }
}
```

### Document Upload

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@document.pdf" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200" \
  -F "strategy=recursive"
```

## Usage Guide

1. **Upload Documents**: Drag and drop PDF, MD, or TXT files into the sidebar
2. **Wait for Indexing**: Files are automatically chunked and indexed
3. **Ask Questions**: Type questions in the chat input to query your documents
4. **View Sources**: Click on source citations to see which documents informed the answer
5. **New Conversation**: Click "New conversation" to start fresh while keeping documents indexed

### Starter Questions

The app suggests quick questions when documents are loaded:
- "Summarise this document in 3 key points"
- "What are the main conclusions?"
- "List any important dates or figures mentioned"
- "What questions does this document answer?"

## Development

### Running Tests

```bash
# Backend tests (if available)
cd backend
pytest

# Frontend linting
cd frontend
npm run lint
```

### Project Structure Details

```
backend/
├── app/
│   ├── api/
│   │   ├── v1.py          # Main API router
│   │   ├── chat.py       # Chat endpoints (legacy)
│   │   ├── embed.py      # Embedding endpoints (legacy)
│   │   ├── ingest.py     # Document ingestion (legacy)
│   │   ├── retriever.py  # Retrieval endpoints (legacy)
│   │   └── vectorstore.py # Vector store endpoints (legacy)
│   ├── core/
│   │   ├── config.py     # Settings management
│   │   └── errors.py     # Error handlers
│   └── services/
│       ├── chunker.py       # Text chunking
│       ├── chat_session.py  # Session management
│       ├── document_loader.py # File parsing
│       ├── embedder.py      # Embedding generation
│       ├── llm.py           # LLM integration
│       ├── retriever.py     # Vector search
│       └── vector_store.py  # Vector database
├── data/
│   ├── sessions/         # Chat session data
│   ├── uploads/          # Uploaded documents
│   └── vector_store/    # FAISS/Chroma index
└── requirements.txt
```

## License

MIT

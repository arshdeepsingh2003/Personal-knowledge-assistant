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
- **Sentence Transformers** - Local embeddings (BAAI/bge-large-en-v1.5)
- **pdfplumber** - PDF document parsing (table-aware)
- **Groq** - Fast LLM inference (default)
- **MongoDB** - Chat history & user persistence
- **SlowAPI** - Rate limiting

### Frontend
- **Next.js 16** - React framework
- **React 19** - UI library
- **Tailwind CSS** - Styling
- **Clerk** - Authentication (Google OAuth + email/password)
- **react-markdown** - Markdown rendering

## Getting Started

### Prerequisites

Before you begin, ensure you have the following:

- **Git** - To clone the repository
- **Python 3.11+** - For the backend
- **Node.js 18+** - For the frontend
- **MongoDB** - A running instance (local or [MongoDB Atlas](https://www.mongodb.com/atlas) free tier)
- **Groq API key** - Get one free at [console.groq.com](https://console.groq.com)
- **Ollama** (optional) - For local LLM as an alternative to Groq

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Personal-knowledge-assistant.git
cd Personal-knowledge-assistant
```

---

### 2. MongoDB Setup

You need a running MongoDB instance:

**Option A: MongoDB Atlas (cloud, recommended)**
1. Create a free account at [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Deploy a free M0 cluster
3. Go to **Database Access** → create a database user with password
4. Go to **Network Access** → add your IP (or `0.0.0.0/0` for development)
5. Click **Connect** → **Drivers** → copy the connection string (looks like: `mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/`)

**Option B: Local MongoDB**
```bash
# Install MongoDB Community Edition
# https://www.mongodb.com/docs/manual/administration/install-community/

# Start MongoDB (default port 27017)
mongod
```

---

### 3. Backend Setup

```bash
cd knowledge-copilot/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy language model (required for text processing)
python -m spacy download en_core_web_sm
```

#### Create Backend `.env` File

Create `knowledge-copilot/backend/.env` with the following content:

```bash
# --- MongoDB ---
MONGODB_URL=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net
MONGODB_DB_NAME=personal_knowledge_copilot

# --- JWT Authentication ---
JWT_SECRET_KEY=generate_a_random_64_char_string_here_change_this_in_production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=10080

# --- Clerk (optional, for Google OAuth) ---
CLERK_SECRET_KEY=sk_test_xxxxx

# --- LLM Provider (Groq is default) ---
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxx                             # Get from https://console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1500

# --- OR use Ollama (local) ---
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2

# --- OR use OpenAI ---
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-xxxxx
# LLM_MODEL=gpt-3.5-turbo

# --- Embeddings (local by default) ---
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_LOCAL=BAAI/bge-large-en-v1.5

# --- Retrieval ---
RETRIEVAL_K=10
RETRIEVAL_SCORE_THRESHOLD=0.20
RETRIEVAL_MAX_CONTEXT_CHARS=25000

# --- Reranker ---
RERANKER_PROVIDER=bge
RERANKER_MODEL=BAAI/bge-reranker-large
```

#### Start the Backend Server

```bash
uvicorn app.main:app --reload --port 8000
```

The backend will start at `http://localhost:8000`. It will:
- Auto-create MongoDB indexes on first startup
- Download the embedding model on first use (~1.3GB for BGE models)
- Create `data/uploads/` and `data/vector_store/` directories

---

### 4. Frontend Setup

#### Clerk Account (for Authentication)

This project uses [Clerk](https://clerk.com) for authentication (Google OAuth + email/password).

1. Sign up at [clerk.com](https://clerk.com)
2. Create a new application
3. Choose **Email/Password** and **Google** as sign-in methods
4. Go to **API Keys** — copy the **Publishable Key** (`pk_test_xxxxx`) and **Secret Key** (`sk_test_xxxxx`)

```bash
cd knowledge-copilot/frontend

# Install dependencies
npm install

# Create .env.local file
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_xxxxx
EOF

# Start the development server
npm run dev
```

The frontend will start at `http://localhost:3000`.

---

### 5. Access the Application

Open [http://localhost:3000](http://localhost:3000) in your browser.

1. **Sign up** using email/password or Google OAuth
2. **Upload documents** via drag-and-drop in the sidebar
3. **Ask questions** about your documents in the chat

---

### 6. (Optional) Ollama for Local LLM

If you want to run the LLM locally instead of using Groq:

```bash
# Download and install Ollama from https://ollama.ai

# Pull the model specified in your .env
ollama pull llama3.2

# Ollama runs as a background service on port 11434
# Then set LLM_PROVIDER=ollama in backend/.env
```

---

## Configuration

### Environment Variables

#### Backend (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URL` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB_NAME` | `knowledge_copilot` | MongoDB database name |
| `JWT_SECRET_KEY` | `CHANGE_THIS_TO_A_RANDOM_64_CHAR_STRING` | Secret for JWT token signing |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_EXPIRE_MINUTES` | `10080` | JWT token expiry (7 days) |
| `CLERK_SECRET_KEY` | - | Clerk API secret key (for OAuth) |
| `APP_NAME` | Knowledge Copilot | Application name |
| `DEBUG` | true | Enable debug mode |
| `EMBEDDING_PROVIDER` | local | Embedding provider: `local` or `openai` |
| `EMBEDDING_MODEL_LOCAL` | BAAI/bge-large-en-v1.5 | Local embedding model |
| `EMBEDDING_MODEL_OPENAI` | text-embedding-3-large | OpenAI embedding model |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `VECTOR_STORE_PROVIDER` | faiss | Vector store: `faiss` or `chroma` |
| `LLM_PROVIDER` | groq | LLM provider: `groq`, `openai`, or `ollama` |
| `GROQ_API_KEY` | - | Groq API key |
| `GROQ_MODEL` | llama-3.3-70b-versatile | Groq model name |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama server URL |
| `OLLAMA_MODEL` | llama3.2 | Ollama model name |
| `LLM_TEMPERATURE` | 0.1 | LLM response creativity |
| `LLM_MAX_TOKENS` | 1500 | Maximum response length |
| `RETRIEVAL_K` | 10 | Number of chunks to retrieve |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.20 | Minimum similarity score |
| `RETRIEVAL_MAX_CONTEXT_CHARS` | 25000 | Max context characters for LLM |
| `RERANKER_PROVIDER` | bge | Reranker provider: `bge` or `cohere` |
| `RERANKER_MODEL` | BAAI/bge-reranker-large | Reranker model name |

#### Frontend (`.env.local`)

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend API URL (default: http://localhost:8000) |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk publishable key for auth |

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

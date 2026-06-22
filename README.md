# Personal Knowledge Assistant

A RAG (Retrieval-Augmented Generation) powered knowledge base copilot that lets you chat with your documents.

## Features

- **Document Upload**: Support for PDF, Markdown, and TXT files
- **Smart Chunking**: Semantic text splitting with configurable strategies
- **Vector Search**: Qdrant Cloud-based semantic search with hybrid (BM25 + vector) retrieval
- **Streaming Responses**: Real-time AI responses with SSE streaming
- **Chat Sessions**: Persistent conversation history with MongoDB persistence
- **Table QA**: Table-aware parsing and question answering
- **Multi-Hop Reasoning**: Complex query decomposition and evidence aggregation
- **Answer Completeness**: Automated completeness checking with regeneration
- **Confidence Scoring**: Per-response confidence estimates with citation grounding
- **Conversation Memory**: Entity tracking, summarization, and context compression
- **Dark/Light Theme**: System-aware theming with manual toggle
- **Multi-format Support**: Handles complex PDFs, Markdown documents, and plain text

## Architecture

```
Personal-knowledge-assistant/
└── knowledge-copilot/
    ├── backend/                  # FastAPI Python backend
    │   ├── app/
    │   │   ├── api/             # API endpoints (v1, auth, files, legacy)
    │   │   ├── core/            # Config, error handling, security
    │   │   ├── middleware/      # Auth middleware
    │   │   ├── models/          # Database models & schemas
    │   │   └── services/        # Business logic (11+ services)
    │   ├── data/                # Local data artifacts
    │   ├── tests/               # RAG evaluation tests
    │   ├── main.py              # Uvicorn runner (port 8001)
    │   ├── app/main.py          # FastAPI app factory
    │   ├── requirements.txt     # Python dependencies
    │   └── .env.example         # Environment template
    │
    └── frontend/                # Next.js React frontend
        ├── app/                 # Next.js app router
        ├── components/          # UI components
        ├── hooks/               # Custom React hooks
        ├── lib/                 # API client
        ├── middleware.js        # Clerk auth middleware
        ├── package.json         # Node dependencies
        └── .env                 # Frontend environment
```

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Qdrant Cloud** - Vector similarity search (with BM25 hybrid index)
- **LangChain** - LLM orchestration framework
- **Sentence Transformers** - Local embeddings (BAAI/bge-large-en-v1.5, 1024d)
- **pdfplumber** - Table-aware PDF parsing
- **Groq** - Fast LLM inference (default, llama-4-scout-17b)
- **MongoDB** - Chat history & file metadata persistence
- **Supabase Storage** - File storage backend
- **SlowAPI** - Rate limiting
- **Pydantic Settings** - Environment configuration

### Frontend
- **Next.js 16** - React framework
- **React 19** - UI library
- **Tailwind CSS v4** - Styling
- **Clerk** - Authentication (Google OAuth + email/password)

## Prerequisites

Before you begin, ensure you have the following installed:

| Tool | Version | Check Command |
|------|---------|---------------|
| **Git** | Any recent | `git --version` |
| **Python** | 3.11+ | `python --version` |
| **Node.js** | 18+ | `node --version` |
| **npm** | 9+ (ships with Node) | `npm --version` |

You will also need accounts for these cloud services:

| Service | Required? | Purpose | Cost |
|---------|-----------|---------|------|
| [Qdrant Cloud](https://cloud.qdrant.io) | **Yes** | Vector database (free 1GB cluster) | Free tier available |
| [MongoDB Atlas](https://www.mongodb.com/atlas) | **Yes** | Chat/file metadata storage | Free M0 cluster |
| [Supabase](https://supabase.com) | **Yes** | File storage backend | Free tier available |
| [Groq](https://console.groq.com) | **Yes** | LLM inference | Free tier available |
| [Clerk](https://clerk.com) | Yes | Authentication (Google OAuth + email) | Free tier available |

---

## Full Setup Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Personal-knowledge-assistant.git
cd Personal-knowledge-assistant
```

---

### Step 2: Qdrant Cloud Setup (Vector Store)

1. Go to [cloud.qdrant.io](https://cloud.qdrant.io) and sign up
2. Create a new cluster (free tier gives 1GB)
3. Once the cluster is active, go to the **Overview** tab and copy:
   - **Cluster URL** (ends in `.qdrant.io`) — this is `QDRANT_URL`
   - Go to **Data Access** → **Create API Key** → copy the key — this is `QDRANT_API_KEY`
4. Choose a collection name (e.g., `knowledge_copilot`) — this is `QDRANT_COLLECTION`

---

### Step 3: Supabase Setup (File Storage)

1. Go to [supabase.com](https://supabase.com) and sign up
2. Create a new project
3. Once created, go to **Project Settings** → **API** and copy:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`) — this is `SUPABASE_URL`
   - **Service Role Key** (`eyJ...`) — this is `SUPABASE_KEY` (use the `service_role` key, NOT the anon key)
4. Go to **Storage** in the left sidebar
5. Create a new bucket called `documents` (make it private)
6. Optionally configure RLS policies for the bucket

---

### Step 4: MongoDB Atlas Setup (Metadata Storage)

1. Go to [mongodb.com/atlas](https://www.mongodb.com/atlas) and sign up
2. Deploy a free M0 cluster
3. Go to **Database Access** → create a database user with password
4. Go to **Network Access** → add your IP (or `0.0.0.0/0` for development)
5. Click **Connect** → **Drivers** → copy the connection string:
   `mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/?appName=Cluster0`

---

### Step 5: Groq Setup (LLM Provider)

1. Go to [console.groq.com](https://console.groq.com) and sign up
2. Go to **API Keys** → create a new key → copy it (starts with `gsk_`)
3. Note the model name: `meta-llama/llama-4-scout-17b-16e-instruct` (recommended)

---

### Step 6: Clerk Setup (Authentication)

1. Go to [clerk.com](https://clerk.com) and sign up
2. Create a new application
3. Choose **Email/Password** and **Google** as sign-in methods
4. Go to **API Keys** → copy:
   - **Publishable Key** (`pk_test_xxxxx`)
   - **Secret Key** (`sk_test_xxxxx`)

---

### Step 7: Backend Setup

```bash
cd knowledge-copilot/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On macOS / Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model (required for text processing)
python -m spacy download en_core_web_sm
```

#### Create Backend `.env` File

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual values. At minimum, you must set:

```bash
# --- Qdrant Cloud (Vector Store) ---
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIs...
QDRANT_COLLECTION=knowledge_copilot

# --- Supabase Storage (File Storage) ---
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_BUCKET=documents

# --- MongoDB (Chat History & Metadata) ---
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?appName=Cluster0
MONGODB_DB_NAME=personal_knowledge_copilot

# --- Authentication (JWT) ---
JWT_SECRET_KEY=your_super_secret_key_change_this_in_production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# --- Clerk (Optional, for OAuth) ---
CLERK_SECRET_KEY=sk_test_xxxxx
CLERK_PUBLISHABLE_KEY=pk_test_xxxxx

# --- LLM Provider (Groq) ---
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxx
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# --- CORS ---
CORS_ORIGINS=http://localhost:3000
```

Refer to `.env.example` for all available configuration options.

#### Start the Backend Server

```bash
# Using the uvicorn runner (recommended, runs on port 8001):
python main.py

# Or directly with uvicorn (runs on port 8000):
uvicorn app.main:app --reload --port 8000
```

The backend will start. It will:
- Auto-create the Qdrant collection on first startup
- Download the embedding model (~1.3GB for BAAI/bge-large-en-v1.5) on first use
- Create MongoDB indexes on first request

---

### Step 8: Frontend Setup

```bash
cd knowledge-copilot/frontend

# Install dependencies
npm install
```

#### Create Frontend `.env` File

Create `knowledge-copilot/frontend/.env` (note: `.env`, not `.env.local`):

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_xxxxx
```

#### Start the Frontend Development Server

```bash
npm run dev
```

The frontend will start at `http://localhost:3000`.

---

### Step 9: Access the Application

Open [http://localhost:3000](http://localhost:3000) in your browser.

1. **Sign up** using email/password or Google OAuth
2. **Upload documents** via drag-and-drop in the sidebar
3. **Ask questions** about your documents in the chat

---

### Step 10: (Optional) Ollama for Local LLM

If you want to run the LLM locally instead of using Groq:

```bash
# Download and install Ollama from https://ollama.ai

# Pull the model
ollama pull llama3.2

# Ollama runs as a background service on port 11434
```

Then in `backend/.env`, set:
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

---

## Configuration Reference

### Backend Environment Variables (`.env`)

#### Required
| Variable | Description |
|----------|-------------|
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | Qdrant collection name (e.g., `knowledge_copilot`) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service_role key |
| `SUPABASE_BUCKET` | Supabase storage bucket name (e.g., `documents`) |
| `MONGODB_URL` | MongoDB connection string |
| `JWT_SECRET_KEY` | Secret for JWT token signing |
| `GROQ_API_KEY` | Groq API key |
| `CORS_ORIGINS` | Allowed origins (e.g., `http://localhost:3000`) |

#### LLM Providers
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | One of: `groq`, `openai`, `ollama` |
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model name |
| `LLM_TEMPERATURE` | `0.3` | Response creativity |
| `LLM_MAX_TOKENS` | `8192` | Maximum response tokens |
| `OPENAI_API_KEY` | - | OpenAI API key (if provider=openai) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |

#### Embeddings
| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `local` | One of: `local`, `openai` |
| `EMBEDDING_MODEL_LOCAL` | `BAAI/bge-large-en-v1.5` | Local embedding model |
| `EMBEDDING_MODEL_OPENAI` | `text-embedding-3-large` | OpenAI embedding model |

#### Chunking
| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNKING_DEFAULT_STRATEGY` | `semantic` | One of: `semantic`, `recursive` |
| `CHUNKING_DEFAULT_SIZE` | `700` | Chunk size in characters |
| `CHUNKING_DEFAULT_OVERLAP` | `150` | Chunk overlap |

#### Retrieval
| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_K` | `15` | Top-k chunks to retrieve |
| `RETRIEVAL_FETCH_K` | `200` | Fetch size before MMR |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.15` | Minimum similarity score |
| `RETRIEVAL_HYBRID_SEARCH` | `true` | Enable BM25 + vector hybrid search |
| `RETRIEVAL_MMR_LAMBDA` | `0.3` | MMR diversity (0=max diversity) |
| `RETRIEVAL_MAX_CONTEXT_CHARS` | `16000` | Max context for LLM |

#### MongoDB Connection Pool
| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_MAX_POOL_SIZE` | `100` | Max connections in pool |
| `MONGODB_MIN_POOL_SIZE` | `10` | Min connections in pool |
| `MONGODB_SERVER_SELECTION_TIMEOUT_MS` | `5000` | Server selection timeout |

### Frontend Environment Variables (`.env`)

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend API URL (default: `http://localhost:8000`) |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk publishable key for auth |

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/register` | Register new user |
| `POST` | `/auth/login` | Login (returns JWT) |
| `POST` | `/api/v1/sessions` | Create new chat session |
| `GET` | `/api/v1/sessions` | List all sessions |
| `GET` | `/api/v1/sessions/{id}` | Get session details |
| `DELETE` | `/api/v1/sessions/{id}` | Delete a session |
| `POST` | `/api/v1/documents` | Upload and index document |
| `GET` | `/api/v1/documents/status` | Get indexing status |
| `POST` | `/api/v1/ask` | Ask a question (with streaming) |
| `POST` | `/api/v1/files/upload` | Upload file to Supabase Storage |

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

### Document Upload

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@document.pdf" \
  -F "chunk_size=700" \
  -F "chunk_overlap=150" \
  -F "strategy=semantic"
```

---

## Usage Guide

1. **Upload Documents**: Drag and drop PDF, MD, or TXT files into the sidebar
2. **Wait for Indexing**: Files are automatically chunked and indexed to Qdrant
3. **Ask Questions**: Type questions in the chat input to query your documents
4. **View Sources**: Click on source citations to see which documents informed the answer
5. **New Conversation**: Click "New conversation" to start fresh while keeping documents indexed

### Starter Questions

The app suggests quick questions when documents are loaded:
- "Summarise this document in 3 key points"
- "What are the main conclusions?"
- "List any important dates or figures mentioned"
- "What questions does this document answer?"

---

## Development

### Running Tests

```bash
# Backend tests
cd knowledge-copilot/backend
pytest

# Frontend linting
cd knowledge-copilot/frontend
npm run lint
```

### Project Structure Details

```
backend/
├── app/
│   ├── api/
│   │   ├── v1.py              # Main API router (sessions, ask, documents)
│   │   ├── auth.py            # Authentication endpoints
│   │   ├── files.py           # File upload/download endpoints
│   │   ├── chat.py            # Chat endpoints (legacy)
│   │   ├── embed.py           # Embedding endpoints (legacy)
│   │   ├── ingest.py          # Document ingestion (legacy)
│   │   ├── retriever.py       # Retrieval endpoints (legacy)
│   │   └── vectorstore.py     # Vector store endpoints (legacy)
│   ├── core/
│   │   ├── config.py          # Pydantic Settings management
│   │   ├── errors.py          # Error handlers
│   │   └── security.py        # Password hashing, JWT utilities
│   ├── middleware/
│   │   └── auth_middleware.py  # JWT auth middleware
│   ├── models/
│   │   ├── database.py        # MongoDB connection & indexes
│   │   ├── models.py          # Pydantic request/response models
│   │   └── schema_notes.py    # Document schema mapping
│   └── services/
│       ├── vector_store.py    # Qdrant client (hybrid search, MMR, caching)
│       ├── embedder.py        # Embedding model (BGE local / OpenAI)
│       ├── chunker.py         # Semantic & recursive text chunking
│       ├── document_loader.py # PDF/MD/TXT file parsing
│       ├── llm.py             # LLM integration (Groq, OpenAI, Ollama)
│       ├── chat_session.py    # Session CRUD
│       ├── chat_history.py    # Chat message persistence
│       ├── auth_service.py    # User registration, login, JWT
│       ├── supabase_storage.py # Supabase file upload/download
│       ├── retriever.py       # Section-aware retrieval pipeline
│       ├── synthesis.py       # Pre-generation context synthesis
│       ├── summarizer.py      # Document summarization
│       ├── confidence.py      # Confidence scoring & citation grounding
│       ├── completeness.py    # Answer completeness checking
│       ├── memory_manager.py  # Conversation memory & entity tracking
│       ├── query_analyzer.py  # Query ambiguity & adversarial detection
│       ├── metrics.py         # Retrieval quality metrics
│       └── special_handling.py # Edge case handling
├── tests/
│   └── test_rag_evaluation.py # RAG pipeline evaluation tests
├── data/                      # Local data (uploads, vector store cache)
├── main.py                    # Uvicorn runner (port 8001)
├── app/main.py                # FastAPI app factory
├── requirements.txt           # Python dependencies
└── .env.example               # Environment template
```

## Troubleshooting

### Backend won't start
- Verify all required env vars are set in `.env`
- Check Qdrant Cloud cluster is running
- Ensure MongoDB Atlas IP whitelist includes your IP
- Check Supabase project is active and bucket exists

### Embedding model download fails
- The BGE model is ~1.3GB. Ensure stable internet on first run
- Alternatively, switch to `EMBEDDING_PROVIDER=openai` and set `OPENAI_API_KEY`

### Frontend can't reach backend
- Verify `NEXT_PUBLIC_API_URL` in frontend `.env` matches backend port
- Check `CORS_ORIGINS` in backend `.env` includes the frontend URL
- Ensure both servers are running

### "Collection does not exist" error
- The app auto-creates the Qdrant collection on startup
- Check `QDRANT_COLLECTION` name and that the Qdrant cluster is active

## License

MIT

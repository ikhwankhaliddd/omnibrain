# OmniBrain — Design Specification

**Version:** 0.1 (MVP)
**Date:** 2026-03-23

## Version Roadmap

| Version | Codename | Focus | Status |
|---------|----------|-------|--------|
| **v0.1** | **MVP** | Text/PDF upload, RAG chat, simple UI | **Current** |
| v0.2 | Multimodal | Audio/video ingestion (Whisper, Vision API), long-term memory (cross-session) | Planned |
| v0.3 | Agents | AI Agents with autonomous tools (web search, code execution) | Planned |
| v0.4 | Privacy | Open-source/local model fallback (Ollama) | Planned |
| v0.5 | Enterprise | Guardrails, moderation, multi-user collaboration & sharing | Planned |
| v0.6 | Smart RAG | Semantic chunking upgrade, reranking, parent-child retrieval | Planned |
| v0.7 | Memory | Long-term memory full (semantic + episodic + procedural) | Planned |
| v0.8 | Hardening | Rate limiting, monitoring, logging, TruLens real-time eval dashboard | Planned |
| v0.9 | Scale | CI/CD pipeline, cloud deployment, horizontal scaling | Planned |
| v1.0 | Launch | Production-ready release, security audit, documentation, onboarding | Planned |

---

## 1. Overview

OmniBrain is a B2B AI-powered workspace where users can upload documents (text, PDF), organize them into workspaces, and chat with their data using Retrieval-Augmented Generation (RAG). The MVP (v0.1) focuses on delivering a solid "chat with your docs" experience for general knowledge workers.

**v0.1 Scope:**
- Text & PDF document upload only (no audio/video/image)
- Multi-workspace per user (no sharing/collaboration)
- Single-user auth (simple JWT)
- API-only backend + simple chat UI frontend

**Out of Scope (v0.2+):**
- Multimodal ingestion (audio, video, image) → v0.2
- AI Agents with autonomous tools (web search, code execution) → v0.3
- Open-source/local model fallback (Ollama) → v0.4
- Enterprise guardrails (moderation API, prompt injection detection) → v0.5
- Multi-user collaboration & sharing → v0.5
- Long-term memory (cross-session) → v0.2

---

## 2. Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12+, FastAPI |
| Frontend | Next.js (App Router), Tailwind CSS |
| Database | PostgreSQL (relational data) |
| Vector Store | Qdrant (embeddings, 4096 dimensions) |
| Task Queue | Celery + Redis |
| LLM | GPT-4o via OpenRouter |
| Embeddings | Octen-Embedding-8B (4096d) via octen.ai API |
| RAG Framework | LangChain + LangGraph |
| Auth | Simple JWT (HS256) |
| Orchestration | Docker Compose (PostgreSQL, Qdrant, Redis) |

---

## 3. Architecture

**Pattern:** Modular Monolith — single FastAPI application with strict module boundaries. Each domain module has its own models, services, and routes. Designed to split into microservices later if needed.

```
omnibrain/
├── backend/
│   ├── modules/
│   │   ├── auth/          — JWT authentication
│   │   │   ├── models.py      (User model)
│   │   │   ├── service.py     (login, token generation/validation)
│   │   │   └── routes.py      (POST /auth/register, POST /auth/login)
│   │   │
│   │   ├── workspace/     — workspace management
│   │   │   ├── models.py      (Workspace model)
│   │   │   ├── service.py     (CRUD workspace)
│   │   │   └── routes.py      (GET/POST/PUT/DELETE /workspaces)
│   │   │
│   │   ├── document/      — file upload, chunking, embedding
│   │   │   ├── models.py      (Document, DocumentChunk models)
│   │   │   ├── service.py     (upload handling, status tracking)
│   │   │   ├── tasks.py       (Celery: parse → chunk → embed → store)
│   │   │   └── routes.py      (POST upload, GET status)
│   │   │
│   │   └── chat/          — RAG pipeline & conversation
│   │       ├── models.py      (ChatSession, Message models)
│   │       ├── service.py     (RAG: retrieve → generate)
│   │       ├── routes.py      (POST chat, GET history)
│   │       └── streaming.py   (SSE streaming response)
│   │
│   ├── core/              — shared infrastructure
│   │   ├── config.py          (env vars, Pydantic Settings)
│   │   ├── database.py        (SQLAlchemy async engine + session)
│   │   ├── dependencies.py    (FastAPI dependency injection)
│   │   ├── qdrant.py          (Qdrant client singleton)
│   │   └── security.py        (JWT utilities)
│   │
│   └── main.py            — FastAPI app, router registration
│
├── frontend/              — Next.js chat UI
├── eval/                  — RAGAS evaluation (offline)
│   ├── datasets/              (test Q&A pairs)
│   ├── run_eval.py            (evaluation script)
│   └── results/               (eval score outputs)
│
└── docker-compose.yml     — PostgreSQL, Qdrant, Redis
```

**Module boundary rules:**
- Modules only import from `core/`, never directly from other modules
- Cross-module data access via dependency injection or events
- Each module owns its ORM models — no shared models

---

## 4. Data Models

### 4.1 PostgreSQL Schema

**users**
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| email | VARCHAR | UNIQUE, NOT NULL |
| hashed_password | VARCHAR | NOT NULL |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |

**workspaces**
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| user_id | UUID | FK → users, NOT NULL |
| name | VARCHAR(100) | NOT NULL |
| description | TEXT | NULL |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |
| updated_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |

**documents**
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| workspace_id | UUID | FK → workspaces, NOT NULL |
| filename | VARCHAR | NOT NULL |
| file_type | VARCHAR | NOT NULL (pdf/txt) |
| file_size | BIGINT | NOT NULL |
| file_path | VARCHAR | NOT NULL |
| status | VARCHAR | NOT NULL (pending/processing/completed/failed) |
| chunk_count | INT | DEFAULT 0 |
| error_message | TEXT | NULL |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |

**chat_sessions**
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| workspace_id | UUID | FK → workspaces, NOT NULL |
| title | VARCHAR | NULL |
| context_summary | TEXT | NULL (summarized old messages) |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |
| updated_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |

**messages**
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| session_id | UUID | FK → chat_sessions, NOT NULL |
| role | VARCHAR | NOT NULL (user/assistant) |
| content | TEXT | NOT NULL |
| sources | JSONB | NULL [{doc_id, chunk_text_preview, similarity_score}] |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |

### 4.2 Qdrant Collection

**Collection: document_chunks**
| Field | Details |
|-------|---------|
| Vector | 4096 dimensions (Octen-Embedding-8B) |
| Distance | Cosine similarity |

**Payload:**
| Field | Type | Purpose |
|-------|------|---------|
| document_id | UUID | Link back to PostgreSQL |
| workspace_id | UUID | Filtered search per workspace |
| chunk_index | INT | Position in document |
| chunk_text | STRING | Raw text content |
| metadata | JSON | {page_number, source_file} |

**Indexes:**
- `workspace_id` — keyword index for filtered vector search

---

## 5. Document Ingestion Pipeline

```
User Upload (PDF/TXT)
    │
    ▼
[FastAPI endpoint] ── validate file type & size (max 20MB)
    │                   ── save file to local storage (uploads/{workspace_id}/{doc_id}/)
    │                   ── create Document record (status: pending)
    │                   ── dispatch Celery task
    │                   ── return document_id + status
    ▼
[Celery Task: process_document]
    │
    ├─ 1. Parse file
    │     ├─ TXT → read directly
    │     └─ PDF → extract text (PyPDF2 / pdfplumber)
    │
    ├─ 2. Chunk text
    │     └─ Pluggable chunking interface
    │        Default: LangChain RecursiveCharacterTextSplitter
    │        (chunk_size=2000 chars, overlap=200 chars)
    │
    ├─ 3. Generate embeddings
    │     └─ Batch call to Octen.ai API (batch size: 32)
    │
    ├─ 4. Store to Qdrant
    │     └─ Upsert vectors + payload
    │
    └─ 5. Update Document status
          ├─ Success → status: completed, chunk_count: N
          └─ Failure → status: failed, error_message: "..."
```

**Chunking strategy:**
- MVP uses `RecursiveCharacterTextSplitter` (chunk_size=2000 characters (~500 tokens), chunk_overlap=200 characters)
- Chunking is behind a pluggable interface — can swap to semantic chunking (ClusterSemanticChunker) or parent-child strategy in future iterations without changing the rest of the pipeline

**File storage:**
- Local filesystem for MVP (`uploads/{workspace_id}/{doc_id}/`)
- Migrate to S3/MinIO for production deployment

**Retry policy:**
- Octen API timeout/error → Celery retry (max 3x, exponential backoff)
- Qdrant connection error → Celery retry (max 3x, exponential backoff)

**Idempotency:**
- Re-upload detected by matching filename within the same workspace → delete existing chunks from Qdrant first, then re-process

---

## 6. RAG Pipeline & Chat Flow

```
User sends question
    │
    ▼
[FastAPI endpoint: POST /workspaces/{id}/chat]
    │
    ├─ 1. Validate session (create new / continue existing)
    │
    ├─ 2. Save user message to DB
    │
    ├─ 3. Retrieval (LangChain)
    │     ├─ Embed user question via Octen.ai API
    │     ├─ Vector search in Qdrant (filtered by workspace_id)
    │     ├─ Top-K chunks (k=5) returned with scores
    │     └─ Score threshold filtering (min similarity 0.7)
    │
    ├─ 4. Prompt Construction
    │     ├─ System prompt (role, behavior rules)
    │     ├─ Context summary (if exists, from previous summarization)
    │     ├─ Retrieved chunks as context (with source attribution)
    │     ├─ Chat history (last 10 individual messages from session, i.e. ~5 user-assistant turn pairs)
    │     └─ User question
    │
    ├─ 5. LLM Generation via OpenRouter
    │     ├─ Model: GPT-4o
    │     ├─ stream: true
    │     └─ SSE streaming to frontend
    │
    └─ 6. Post-processing
          ├─ Save assistant message + sources (JSONB) to DB
          └─ Sources = [{doc_id, chunk_text_preview, similarity_score}]
```

**Key parameters:**
- Top-K = 5 (retrieve 5 most relevant chunks)
- Score threshold = 0.7 (filter irrelevant chunks)
- Chat history = last 10 individual messages per session (~5 user-assistant turn pairs)
- SSE streaming for real-time response

**Edge cases:**
- No relevant chunks found (all below 0.7) → respond "Saya tidak menemukan informasi yang relevan di dokumen Anda untuk menjawab pertanyaan ini"
- Empty workspace (no documents) → respond "Belum ada dokumen di workspace ini. Silakan upload dokumen terlebih dahulu"

---

## 7. Memory Strategy

### MVP: Short-Term Memory (within session)

**Strategy:** Summarization via LangChain/LangGraph

When total chat history exceeds 3000 tokens (measured by tiktoken with `cl100k_base` encoding for GPT-4o compatibility), trigger summarization:
1. Summarize all messages except the last 10 via a separate LLM call
2. Store summary in `chat_sessions.context_summary`
3. Keep last 10 messages intact + summary of older messages
4. Inject summary into prompt construction (step 4 in RAG pipeline)

### Post-MVP: Long-Term Memory (cross-session)

Future addition of `user_memories` table:
- Semantic memory (facts/preferences about user)
- Background task extracts memory from conversations asynchronously
- Memory injected into system prompt at start of each new session

---

## 8. API Endpoints

All endpoints (except auth) require `Authorization: Bearer <token>` header.

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login, return JWT token |

### Workspace
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/workspaces` | List all user workspaces |
| POST | `/workspaces` | Create new workspace |
| GET | `/workspaces/{id}` | Workspace detail |
| PUT | `/workspaces/{id}` | Update name/description |
| DELETE | `/workspaces/{id}` | Delete workspace (cascade) |

### Document
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/workspaces/{id}/documents` | Upload file (multipart/form-data, max 20MB) |
| GET | `/workspaces/{id}/documents` | List documents in workspace |
| GET | `/documents/{id}` | Document detail & status |
| DELETE | `/documents/{id}` | Delete document + Qdrant chunks |

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/workspaces/{id}/chat` | Send message, stream response (SSE) |
| GET | `/workspaces/{id}/chat/sessions` | List chat sessions |
| GET | `/chat/sessions/{id}/messages` | Message history |
| DELETE | `/chat/sessions/{id}` | Delete session |

---

## 9. Frontend

### Pages
1. **Login/Register** — simple form, store JWT in localStorage
2. **Dashboard** — list workspaces, create/delete workspace
3. **Workspace View** — two panels:
   - **Left panel:** document list (upload, status indicator, delete)
   - **Right panel:** chat interface (input, messages, streaming response, source citations)

### Tech
- Next.js (App Router) + Tailwind CSS
- `react-markdown` or `marked` for rendering LLM responses as formatted HTML (no raw markdown symbols visible to user)
- `EventSource` / `fetch` with ReadableStream for SSE streaming

### Not included in MVP
- Document viewer/preview
- Drag & drop upload
- Dark/light mode toggle
- Mobile responsive (desktop-first)
- Rich text formatting in chat

---

## 10. Evaluation (RAGAS)

**Approach:** Offline evaluation script, not real-time monitoring.

**Metrics:**
- **Faithfulness** — are LLM answers grounded in retrieved context? (reference-free)
- **Context Precision** — are retrieved chunks relevant to the question?
- **Answer Relevancy** — does the answer actually address the question?

**Structure:**
```
eval/
├── datasets/          ← test Q&A pairs (JSON/YAML, 20-30 pairs)
├── run_eval.py        ← script to run RAGAS evaluation
└── results/           ← output eval scores per run
```

**Workflow:**
1. Create test dataset from sample documents (question + expected relevant context)
2. Run `python -m eval.run_eval` after any pipeline change (chunking, prompt, model)
3. Compare scores across runs to validate improvements

**Post-MVP:**
- Log every request (question, retrieved chunks, answer) for analysis
- User feedback loop (thumbs up/down) as human eval signal
- TruLens dashboard for production quality monitoring

---

## 11. Error Handling

### Document Ingestion
- File corrupt / parse failure → status: `failed`, store error message, user can re-upload
- Octen API timeout → Celery retry (max 3x, exponential backoff)
- Qdrant down → Celery retry (max 3x, exponential backoff)

### Chat/RAG
- No relevant chunks → inform user, don't hallucinate
- OpenRouter API error → return error message via SSE, don't crash
- Empty workspace → inform user to upload documents first

### Input Validation
- File type: `.pdf` and `.txt` only
- File size: max 20MB
- File MIME type validation (not just extension)
- Chat message: max 4000 characters
- Workspace name: max 100 characters, not empty

### Security
- JWT validation on every request
- SQL injection: prevented by SQLAlchemy ORM (parameterized queries)
- Ownership check: user can only access their own workspaces
- Rate limiting: not in MVP, add when deploying publicly

---

## 12. Infrastructure & Development Setup

### Docker Compose Services
| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Relational data |
| Qdrant | 6333 | Vector store |
| Redis | 6379 | Celery broker + result backend |

### Local Services (not in Docker, for faster dev)
| Service | Port | Purpose |
|---------|------|---------|
| FastAPI | 8000 | Backend API |
| Celery worker | — | Background task processing |
| Next.js | 3000 | Frontend dev server |

### Environment Variables (.env)
```
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/omnibrain

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys
OPENROUTER_API_KEY=sk-...
OCTEN_API_KEY=...

# JWT
JWT_SECRET_KEY=...
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=1440
```

### Deployment Strategy
- Hybrid: develop locally via Docker Compose, deployment config ready from start
- All services can be fully Dockerized for production deployment later

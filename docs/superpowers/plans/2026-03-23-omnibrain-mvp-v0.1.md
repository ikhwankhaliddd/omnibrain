# OmniBrain MVP v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a "chat with your docs" workspace where users upload text/PDF files and chat with their data via RAG.

**Architecture:** Modular monolith with FastAPI backend (4 modules: auth, workspace, document, chat), Next.js frontend, PostgreSQL + Qdrant for storage, Celery + Redis for background processing. LangChain for RAG pipeline, OpenRouter for LLM, Octen.ai for embeddings.

**Tech Stack:** Python 3.12+, FastAPI, SQLAlchemy (async), Next.js, Tailwind CSS, PostgreSQL, Qdrant, Redis, Celery, LangChain, RAGAS

**Spec:** `docs/superpowers/specs/2026-03-23-omnibrain-mvp-design.md`

---

## File Structure

```
omnibrain/
├── backend/
│   ├── modules/
│   │   ├── auth/
│   │   │   ├── __init__.py
│   │   │   ├── models.py          — User SQLAlchemy model
│   │   │   ├── schemas.py         — Pydantic request/response schemas
│   │   │   ├── service.py         — register, login, hash password, verify
│   │   │   └── routes.py          — POST /auth/register, POST /auth/login
│   │   ├── workspace/
│   │   │   ├── __init__.py
│   │   │   ├── models.py          — Workspace SQLAlchemy model
│   │   │   ├── schemas.py         — Pydantic request/response schemas
│   │   │   ├── service.py         — CRUD operations
│   │   │   └── routes.py          — GET/POST/PUT/DELETE /workspaces
│   │   ├── document/
│   │   │   ├── __init__.py
│   │   │   ├── models.py          — Document SQLAlchemy model
│   │   │   ├── schemas.py         — Pydantic request/response schemas
│   │   │   ├── service.py         — upload handling, status tracking
│   │   │   ├── tasks.py           — Celery: parse → chunk → embed → store
│   │   │   ├── chunking.py        — Pluggable chunking interface + default impl
│   │   │   └── routes.py          — POST upload, GET status, DELETE
│   │   └── chat/
│   │       ├── __init__.py
│   │       ├── models.py          — ChatSession, Message SQLAlchemy models
│   │       ├── schemas.py         — Pydantic request/response schemas
│   │       ├── service.py         — RAG: retrieve → generate
│   │       ├── memory.py          — Summarization logic (short-term memory)
│   │       ├── prompts.py         — System prompt templates
│   │       └── routes.py          — POST chat (SSE), GET history, DELETE
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              — Pydantic Settings, env vars
│   │   ├── database.py            — SQLAlchemy async engine + session factory
│   │   ├── dependencies.py        — FastAPI Depends (get_db, get_current_user)
│   │   ├── qdrant.py              — Qdrant client singleton + collection setup
│   │   ├── security.py            — JWT encode/decode, password hashing
│   │   └── celery_app.py          — Celery instance configuration
│   ├── main.py                    — FastAPI app, CORS, router registration
│   ├── alembic.ini                — Alembic config
│   ├── alembic/
│   │   ├── env.py
│   │   └── versions/              — migration files
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── next.config.ts
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx         — root layout
│   │   │   ├── page.tsx           — redirect to login or dashboard
│   │   │   ├── login/
│   │   │   │   └── page.tsx       — login/register form
│   │   │   ├── dashboard/
│   │   │   │   └── page.tsx       — workspace list
│   │   │   └── workspace/
│   │   │       └── [id]/
│   │   │           └── page.tsx   — document panel + chat panel
│   │   ├── components/
│   │   │   ├── ChatMessage.tsx    — single message with markdown rendering
│   │   │   ├── ChatInput.tsx      — message input box
│   │   │   ├── DocumentList.tsx   — document list with status & upload
│   │   │   ├── WorkspaceCard.tsx  — workspace card for dashboard
│   │   │   └── SourceCitation.tsx — expandable source references
│   │   └── lib/
│   │       ├── api.ts             — fetch wrapper with JWT auth
│   │       └── auth.ts            — JWT storage, login/logout helpers
│   └── Dockerfile
├── eval/
│   ├── datasets/
│   │   └── sample_qa.json         — 20-30 test Q&A pairs
│   ├── run_eval.py                — RAGAS evaluation script
│   └── results/                   — output scores per run
├── tests/
│   ├── conftest.py                — shared fixtures (db, client, auth)
│   ├── test_auth.py
│   ├── test_workspace.py
│   ├── test_document.py
│   ├── test_chat.py
│   └── test_chunking.py
├── docker-compose.yml             — PostgreSQL, Qdrant, Redis
├── .env.example
├── .gitignore
└── README.md
```

---

## Task 1: Project Scaffolding & Infrastructure

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `backend/requirements.txt`
- Create: `backend/core/__init__.py`
- Create: `backend/core/config.py`
- Create: `backend/main.py`

- [ ] **Step 1: Create docker-compose.yml**

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: omnibrain
      POSTGRES_PASSWORD: omnibrain_dev
      POSTGRES_DB: omnibrain
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
  qdrant_data:
```

- [ ] **Step 2: Create .env.example**

```env
# Database
DATABASE_URL=postgresql+asyncpg://omnibrain:omnibrain_dev@localhost:5432/omnibrain

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys
OPENROUTER_API_KEY=sk-your-openrouter-key
OCTEN_API_KEY=your-octen-api-key

# JWT
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=1440
```

- [ ] **Step 3: Create backend/requirements.txt**

```txt
# Web framework
fastapi==0.115.*
uvicorn[standard]==0.34.*

# Database
sqlalchemy[asyncio]==2.0.*
asyncpg==0.30.*
alembic==1.14.*

# Vector store
qdrant-client==1.12.*

# Task queue
celery[redis]==5.4.*
redis==5.2.*

# AI/ML
langchain==0.3.*
langchain-openai==0.3.*
langchain-text-splitters==0.3.*
openai==1.59.*
tiktoken==0.8.*

# Auth
python-jose[cryptography]==3.3.*
passlib[bcrypt]==1.7.*

# File processing
pdfplumber==0.11.*
python-multipart==0.0.*

# Validation
pydantic==2.10.*
pydantic-settings==2.7.*

# Eval
ragas==0.2.*

# Testing
pytest==8.3.*
pytest-asyncio==0.24.*
httpx==0.28.*
```

- [ ] **Step 4: Create backend/core/config.py**

```python
# backend/core/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # API Keys
    openrouter_api_key: str
    octen_api_key: str

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 1440

    # File upload
    max_upload_size_bytes: int = 20 * 1024 * 1024  # 20MB
    upload_dir: str = "uploads"

    # RAG
    rag_top_k: int = 5
    rag_score_threshold: float = 0.7
    chat_history_limit: int = 10
    summarization_token_threshold: int = 3000

    # Embedding
    embedding_batch_size: int = 32
    embedding_dimensions: int = 4096

    # Chunking
    chunk_size: int = 2000
    chunk_overlap: int = 200

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

- [ ] **Step 5: Create backend/main.py (minimal)**

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="OmniBrain", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
```

- [ ] **Step 6: Create __init__.py files**

Create empty `__init__.py` in:
- `backend/__init__.py`
- `backend/core/__init__.py`
- `backend/modules/__init__.py`
- `backend/modules/auth/__init__.py`
- `backend/modules/workspace/__init__.py`
- `backend/modules/document/__init__.py`
- `backend/modules/chat/__init__.py`

- [ ] **Step 7: Start Docker Compose and verify services**

```bash
docker compose up -d
# Verify all 3 services are running:
docker compose ps
# Expected: postgres, qdrant, redis all "running"
```

- [ ] **Step 8: Copy .env.example to .env, fill in API keys**

```bash
cp .env.example .env
# Edit .env with real OPENROUTER_API_KEY and OCTEN_API_KEY
```

- [ ] **Step 9: Verify FastAPI starts**

```bash
cd backend && uvicorn main:app --reload --port 8000
# Visit http://localhost:8000/health
# Expected: {"status": "ok", "version": "0.1.0"}
```

- [ ] **Step 10: Commit**

```bash
git add docker-compose.yml .env.example .gitignore backend/
git commit -m "feat: project scaffolding with Docker Compose, FastAPI, and config"
```

---

## Task 2: Database Setup (SQLAlchemy + Alembic)

**Files:**
- Create: `backend/core/database.py`
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/modules/auth/models.py`
- Create: `backend/modules/workspace/models.py`
- Create: `backend/modules/document/models.py`
- Create: `backend/modules/chat/models.py`

- [ ] **Step 1: Create backend/core/database.py**

```python
# backend/core/database.py
import uuid
from datetime import datetime, timezone
from collections.abc import AsyncGenerator

from sqlalchemy import DateTime, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from backend.core.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class UpdateTimestampMixin(TimestampMixin):
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
```

- [ ] **Step 2: Create all 4 module models**

`backend/modules/auth/models.py`:
```python
import uuid

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
```

`backend/modules/workspace/models.py`:
```python
import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base, UpdateTimestampMixin


class Workspace(Base, UpdateTimestampMixin):
    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
```

`backend/modules/document/models.py`:
```python
import uuid

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base, TimestampMixin


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
```

`backend/modules/chat/models.py`:
```python
import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base, TimestampMixin, UpdateTimestampMixin


class ChatSession(Base, UpdateTimestampMixin):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class Message(Base, TimestampMixin):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
```

- [ ] **Step 3: Initialize Alembic**

```bash
cd backend
pip install -r requirements.txt
alembic init alembic
```

- [ ] **Step 4: Configure alembic/env.py**

Update `backend/alembic/env.py` to use async engine and import all models:

```python
# backend/alembic/env.py
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from backend.core.config import settings
from backend.core.database import Base
from backend.modules.auth.models import User
from backend.modules.workspace.models import Workspace
from backend.modules.document.models import Document
from backend.modules.chat.models import ChatSession, Message

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 5: Generate and run initial migration**

```bash
cd backend
alembic revision --autogenerate -m "initial schema: users, workspaces, documents, chat_sessions, messages"
alembic upgrade head
```

- [ ] **Step 6: Verify tables exist**

```bash
docker compose exec postgres psql -U omnibrain -d omnibrain -c "\dt"
# Expected: users, workspaces, documents, chat_sessions, messages tables listed
```

- [ ] **Step 7: Commit**

```bash
git add backend/
git commit -m "feat: database models and Alembic migrations for all 4 modules"
```

---

## Task 3: Qdrant Client & Collection Setup

**Files:**
- Create: `backend/core/qdrant.py`
- Create: `tests/test_qdrant.py`

- [ ] **Step 1: Write failing test for Qdrant client**

```python
# tests/test_qdrant.py
import pytest
from backend.core.qdrant import get_qdrant_client, ensure_collection_exists

COLLECTION_NAME = "document_chunks"


def test_get_qdrant_client_returns_client():
    client = get_qdrant_client()
    assert client is not None


def test_ensure_collection_creates_collection():
    client = get_qdrant_client()
    ensure_collection_exists(client)
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    assert COLLECTION_NAME in names


def test_collection_has_correct_vector_config():
    client = get_qdrant_client()
    ensure_collection_exists(client)
    info = client.get_collection(COLLECTION_NAME)
    vector_config = info.config.params.vectors
    assert vector_config.size == 4096
    assert vector_config.distance.name == "COSINE"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_qdrant.py -v
# Expected: FAIL — ModuleNotFoundError
```

- [ ] **Step 3: Implement Qdrant client**

```python
# backend/core/qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from backend.core.config import settings

COLLECTION_NAME = "document_chunks"

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


def ensure_collection_exists(client: QdrantClient) -> None:
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_qdrant.py -v
# Expected: 3 passed
```

- [ ] **Step 5: Commit**

```bash
git add backend/core/qdrant.py tests/test_qdrant.py
git commit -m "feat: Qdrant client singleton and collection setup"
```

---

## Task 4: JWT Security & Auth Module

**Files:**
- Create: `backend/core/security.py`
- Create: `backend/modules/auth/schemas.py`
- Create: `backend/modules/auth/service.py`
- Create: `backend/modules/auth/routes.py`
- Create: `backend/core/dependencies.py`
- Create: `tests/conftest.py`
- Create: `tests/test_auth.py`

- [ ] **Step 1: Write failing tests for auth**

```python
# tests/conftest.py
import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.core.config import settings
from backend.core.database import Base, get_async_session
from backend.main import app

TEST_DATABASE_URL = settings.database_url + "_test"
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def setup_database():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session():
    async with test_session_factory() as session:
        yield session


@pytest.fixture
async def client():
    async def override_get_session():
        async with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_async_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
async def auth_headers(client: AsyncClient):
    await client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "testpassword123"
    })
    response = await client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "testpassword123"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
```

```python
# tests/test_auth.py
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    response = await client.post("/auth/register", json={
        "email": "new@example.com",
        "password": "password123"
    })
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "new@example.com"
    assert "id" in data
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    await client.post("/auth/register", json={
        "email": "dup@example.com",
        "password": "password123"
    })
    response = await client.post("/auth/register", json={
        "email": "dup@example.com",
        "password": "password456"
    })
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient):
    await client.post("/auth/register", json={
        "email": "login@example.com",
        "password": "password123"
    })
    response = await client.post("/auth/login", json={
        "email": "login@example.com",
        "password": "password123"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient):
    await client.post("/auth/register", json={
        "email": "wrongpw@example.com",
        "password": "password123"
    })
    response = await client.post("/auth/login", json={
        "email": "wrongpw@example.com",
        "password": "wrongpassword"
    })
    assert response.status_code == 401
```

- [ ] **Step 2: Create test database**

```bash
docker compose exec postgres psql -U omnibrain -c "CREATE DATABASE omnibrain_test;"
# Expected: CREATE DATABASE
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_auth.py -v
# Expected: FAIL — routes not registered
```

- [ ] **Step 4: Implement security utilities**

```python
# backend/core/security.py
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiration_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload.get("sub")
    except JWTError:
        return None
```

- [ ] **Step 4: Implement auth schemas, service, routes**

```python
# backend/modules/auth/schemas.py
import uuid
from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterResponse(BaseModel):
    id: uuid.UUID
    email: str

    model_config = {"from_attributes": True}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

```python
# backend/modules/auth/service.py
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.security import hash_password, verify_password
from backend.modules.auth.models import User


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create_user(session: AsyncSession, email: str, password: str) -> User:
    user = User(id=uuid.uuid4(), email=email, hashed_password=hash_password(password))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def authenticate_user(session: AsyncSession, email: str, password: str) -> User | None:
    user = await get_user_by_email(session, email)
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user
```

```python
# backend/modules/auth/routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_async_session
from backend.core.security import create_access_token
from backend.modules.auth.schemas import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse
from backend.modules.auth.service import authenticate_user, create_user, get_user_by_email

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, session: AsyncSession = Depends(get_async_session)):
    existing = await get_user_by_email(session, body.email)
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user = await create_user(session, body.email, body.password)
    return user


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, session: AsyncSession = Depends(get_async_session)):
    user = await authenticate_user(session, body.email, body.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(subject=str(user.id))
    return LoginResponse(access_token=token)
```

- [ ] **Step 6: Create dependencies (get_current_user)**

```python
# backend/core/dependencies.py
import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_async_session
from backend.core.security import decode_access_token
from backend.modules.auth.models import User

security_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    session: AsyncSession = Depends(get_async_session),
) -> User:
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    result = await session.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
```

- [ ] **Step 7: Register auth router in main.py**

Update `backend/main.py` to include:
```python
from backend.modules.auth.routes import router as auth_router
app.include_router(auth_router)
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
pytest tests/test_auth.py -v
# Expected: 4 passed
```

- [ ] **Step 9: Commit**

```bash
git add backend/ tests/
git commit -m "feat: auth module with JWT register/login endpoints"
```

---

## Task 5: Workspace Module

**Files:**
- Create: `backend/modules/workspace/schemas.py`
- Create: `backend/modules/workspace/service.py`
- Create: `backend/modules/workspace/routes.py`
- Create: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for workspace CRUD**

```python
# tests/test_workspace.py
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_workspace(client: AsyncClient, auth_headers: dict):
    response = await client.post("/workspaces", json={
        "name": "My Research",
        "description": "Research workspace"
    }, headers=auth_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "My Research"
    assert "id" in data


@pytest.mark.asyncio
async def test_list_workspaces(client: AsyncClient, auth_headers: dict):
    await client.post("/workspaces", json={"name": "WS1"}, headers=auth_headers)
    await client.post("/workspaces", json={"name": "WS2"}, headers=auth_headers)
    response = await client.get("/workspaces", headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json()) == 2


@pytest.mark.asyncio
async def test_get_workspace(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post("/workspaces", json={"name": "Detail WS"}, headers=auth_headers)
    ws_id = create_resp.json()["id"]
    response = await client.get(f"/workspaces/{ws_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["name"] == "Detail WS"


@pytest.mark.asyncio
async def test_update_workspace(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post("/workspaces", json={"name": "Old Name"}, headers=auth_headers)
    ws_id = create_resp.json()["id"]
    response = await client.put(f"/workspaces/{ws_id}", json={"name": "New Name"}, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["name"] == "New Name"


@pytest.mark.asyncio
async def test_delete_workspace(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post("/workspaces", json={"name": "To Delete"}, headers=auth_headers)
    ws_id = create_resp.json()["id"]
    response = await client.delete(f"/workspaces/{ws_id}", headers=auth_headers)
    assert response.status_code == 204
    get_resp = await client.get(f"/workspaces/{ws_id}", headers=auth_headers)
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_workspace_name_too_long(client: AsyncClient, auth_headers: dict):
    response = await client.post("/workspaces", json={
        "name": "x" * 101
    }, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_unauthorized_access(client: AsyncClient):
    response = await client.get("/workspaces")
    assert response.status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_workspace.py -v
# Expected: FAIL
```

- [ ] **Step 3: Implement workspace schemas**

```python
# backend/modules/workspace/schemas.py
import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None


class WorkspaceUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None


class WorkspaceResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
```

- [ ] **Step 4: Implement workspace service**

```python
# backend/modules/workspace/service.py
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.workspace.models import Workspace
from backend.modules.workspace.schemas import WorkspaceCreate, WorkspaceUpdate


async def create_workspace(session: AsyncSession, user_id: uuid.UUID, data: WorkspaceCreate) -> Workspace:
    workspace = Workspace(id=uuid.uuid4(), user_id=user_id, name=data.name, description=data.description)
    session.add(workspace)
    await session.commit()
    await session.refresh(workspace)
    return workspace


async def list_workspaces(session: AsyncSession, user_id: uuid.UUID) -> list[Workspace]:
    result = await session.execute(
        select(Workspace).where(Workspace.user_id == user_id).order_by(Workspace.created_at.desc())
    )
    return list(result.scalars().all())


async def get_workspace(session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID) -> Workspace | None:
    result = await session.execute(
        select(Workspace).where(Workspace.id == workspace_id, Workspace.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def update_workspace(
    session: AsyncSession, workspace: Workspace, data: WorkspaceUpdate
) -> Workspace:
    if data.name is not None:
        workspace.name = data.name
    if data.description is not None:
        workspace.description = data.description
    await session.commit()
    await session.refresh(workspace)
    return workspace


async def delete_workspace(session: AsyncSession, workspace: Workspace) -> None:
    await session.delete(workspace)
    await session.commit()
```

- [ ] **Step 5: Implement workspace routes**

```python
# backend/modules/workspace/routes.py
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_async_session
from backend.core.dependencies import get_current_user
from backend.modules.auth.models import User
from backend.modules.workspace.schemas import WorkspaceCreate, WorkspaceResponse, WorkspaceUpdate
from backend.modules.workspace.service import (
    create_workspace,
    delete_workspace,
    get_workspace,
    list_workspaces,
    update_workspace,
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create(
    body: WorkspaceCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    return await create_workspace(session, user.id, body)


@router.get("", response_model=list[WorkspaceResponse])
async def list_all(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    return await list_workspaces(session, user.id)


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_one(
    workspace_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    workspace = await get_workspace(session, workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    return workspace


@router.put("/{workspace_id}", response_model=WorkspaceResponse)
async def update(
    workspace_id: uuid.UUID,
    body: WorkspaceUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    workspace = await get_workspace(session, workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    return await update_workspace(session, workspace, body)


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(
    workspace_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    workspace = await get_workspace(session, workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    await delete_workspace(session, workspace)
```

- [ ] **Step 6: Register workspace router in main.py**

Add to `backend/main.py`:
```python
from backend.modules.workspace.routes import router as workspace_router
app.include_router(workspace_router)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_workspace.py -v
# Expected: 7 passed
```

- [ ] **Step 8: Commit**

```bash
git add backend/modules/workspace/ tests/test_workspace.py backend/main.py
git commit -m "feat: workspace module with full CRUD and ownership checks"
```

---

## Task 6: Document Upload & Ingestion Pipeline

**Files:**
- Create: `backend/modules/document/schemas.py`
- Create: `backend/modules/document/service.py`
- Create: `backend/modules/document/chunking.py`
- Create: `backend/modules/document/tasks.py`
- Create: `backend/modules/document/routes.py`
- Create: `backend/core/celery_app.py`
- Create: `tests/test_document.py`
- Create: `tests/test_chunking.py`

- [ ] **Step 1: Write failing tests for chunking**

```python
# tests/test_chunking.py
from backend.modules.document.chunking import ChunkingStrategy, DefaultChunkingStrategy


def test_default_chunking_implements_interface():
    strategy = DefaultChunkingStrategy(chunk_size=2000, chunk_overlap=200)
    assert isinstance(strategy, ChunkingStrategy)


def test_chunk_short_text():
    strategy = DefaultChunkingStrategy(chunk_size=2000, chunk_overlap=200)
    text = "This is a short text."
    chunks = strategy.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == "This is a short text."


def test_chunk_long_text_produces_multiple_chunks():
    strategy = DefaultChunkingStrategy(chunk_size=100, chunk_overlap=20)
    text = "Word " * 200  # 1000 chars
    chunks = strategy.chunk(text)
    assert len(chunks) > 1


def test_chunk_preserves_all_content():
    strategy = DefaultChunkingStrategy(chunk_size=100, chunk_overlap=0)
    text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 5
    chunks = strategy.chunk(text)
    reassembled = " ".join(chunks)
    # All original words should appear
    for word in text.split():
        assert word in reassembled
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_chunking.py -v
# Expected: FAIL
```

- [ ] **Step 3: Implement chunking interface + default strategy**

```python
# backend/modules/document/chunking.py
from abc import ABC, abstractmethod

from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        pass


class DefaultChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> list[str]:
        return self._splitter.split_text(text)
```

- [ ] **Step 4: Run chunking tests to verify they pass**

```bash
pytest tests/test_chunking.py -v
# Expected: 4 passed
```

- [ ] **Step 5: Implement Celery app config**

```python
# backend/core/celery_app.py
from celery import Celery

from backend.core.config import settings

celery_app = Celery(
    "omnibrain",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery_app.autodiscover_tasks(["backend.modules.document"])
```

- [ ] **Step 6: Implement document schemas**

```python
# backend/modules/document/schemas.py
import uuid
from datetime import datetime
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
```

- [ ] **Step 7: Implement document service**

```python
# backend/modules/document/service.py
import os
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.modules.document.models import Document


async def save_uploaded_file(
    workspace_id: uuid.UUID, doc_id: uuid.UUID, filename: str, content: bytes
) -> str:
    dir_path = os.path.join(settings.upload_dir, str(workspace_id), str(doc_id))
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, filename)
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path


async def create_document(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    filename: str,
    file_type: str,
    file_size: int,
    file_path: str,
) -> Document:
    doc = Document(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        file_path=file_path,
        status="pending",
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return doc


async def get_document(session: AsyncSession, doc_id: uuid.UUID) -> Document | None:
    result = await session.execute(select(Document).where(Document.id == doc_id))
    return result.scalar_one_or_none()


async def list_documents(session: AsyncSession, workspace_id: uuid.UUID) -> list[Document]:
    result = await session.execute(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


async def find_existing_document(
    session: AsyncSession, workspace_id: uuid.UUID, filename: str
) -> Document | None:
    result = await session.execute(
        select(Document).where(
            Document.workspace_id == workspace_id,
            Document.filename == filename,
        )
    )
    return result.scalar_one_or_none()


async def delete_document(session: AsyncSession, doc: Document) -> None:
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    await session.delete(doc)
    await session.commit()
```

- [ ] **Step 8: Implement Celery task (process_document)**

```python
# backend/modules/document/tasks.py
import uuid

import pdfplumber
from celery import shared_task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import httpx
from qdrant_client.models import PointStruct

from backend.core.config import settings
from backend.core.celery_app import celery_app
from backend.core.database import async_session_factory
from backend.core.qdrant import COLLECTION_NAME, get_qdrant_client
from backend.modules.document.chunking import DefaultChunkingStrategy
from backend.modules.document.models import Document


def _parse_file(file_path: str, file_type: str) -> str:
    if file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_type == "pdf":
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _generate_embeddings(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[i : i + settings.embedding_batch_size]
        response = httpx.post(
            "https://api.octen.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.octen_api_key}"},
            json={"model": "Octen-Embedding-8B", "input": batch},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        batch_embeddings = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


async def _update_document_status(
    doc_id: uuid.UUID, status: str, chunk_count: int = 0, error_message: str | None = None
) -> None:
    async with async_session_factory() as session:
        result = await session.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one()
        doc.status = status
        doc.chunk_count = chunk_count
        doc.error_message = error_message
        await session.commit()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_document(self, doc_id: str, workspace_id: str) -> None:
    doc_uuid = uuid.UUID(doc_id)
    workspace_uuid = uuid.UUID(workspace_id)

    try:
        asyncio.run(_update_document_status(doc_uuid, "processing"))

        # 1. Get document info
        async def _get_doc():
            async with async_session_factory() as session:
                result = await session.execute(select(Document).where(Document.id == doc_uuid))
                return result.scalar_one()

        doc = asyncio.run(_get_doc())

        # 2. Parse file
        text = _parse_file(doc.file_path, doc.file_type)
        if not text.strip():
            asyncio.run(_update_document_status(doc_uuid, "failed", error_message="File is empty or unreadable"))
            return

        # 3. Chunk text
        chunker = DefaultChunkingStrategy(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        chunks = chunker.chunk(text)

        # 4. Generate embeddings
        embeddings = _generate_embeddings(chunks)

        # 5. Delete existing chunks for this document (idempotency)
        qdrant = get_qdrant_client()
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector={"filter": {"must": [{"key": "document_id", "match": {"value": doc_id}}]}},
        )

        # 6. Store to Qdrant
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "document_id": doc_id,
                    "workspace_id": workspace_id,
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "metadata": {"source_file": doc.filename},
                },
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        # 7. Update status
        asyncio.run(_update_document_status(doc_uuid, "completed", chunk_count=len(chunks)))

    except Exception as exc:
        asyncio.run(
            _update_document_status(doc_uuid, "failed", error_message=str(exc)[:500])
        )
        raise self.retry(exc=exc)
```

- [ ] **Step 9: Implement document routes**

```python
# backend/modules/document/routes.py
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.database import get_async_session
from backend.core.dependencies import get_current_user
from backend.core.qdrant import COLLECTION_NAME, get_qdrant_client
from backend.modules.auth.models import User
from backend.modules.document.schemas import DocumentResponse
from backend.modules.document.service import (
    create_document,
    delete_document,
    find_existing_document,
    get_document,
    list_documents,
    save_uploaded_file,
)
from backend.modules.document.tasks import process_document
from backend.modules.workspace.service import get_workspace

router = APIRouter(tags=["documents"])

ALLOWED_TYPES = {"application/pdf": "pdf", "text/plain": "txt"}


@router.post(
    "/workspaces/{workspace_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    workspace_id: uuid.UUID,
    file: UploadFile,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    # Verify workspace ownership
    workspace = await get_workspace(session, workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Validate MIME type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=422, detail="Only PDF and TXT files are allowed")

    # Read and validate size
    content = await file.read()
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=422, detail="File exceeds 20MB limit")

    file_type = ALLOWED_TYPES[file.content_type]
    doc_id = uuid.uuid4()

    # Check for existing document with same filename (idempotency)
    existing = await find_existing_document(session, workspace_id, file.filename)
    if existing is not None:
        # Delete old document and its Qdrant chunks
        qdrant = get_qdrant_client()
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector={"filter": {"must": [{"key": "document_id", "match": {"value": str(existing.id)}}]}},
        )
        await delete_document(session, existing)

    # Save file
    file_path = await save_uploaded_file(workspace_id, doc_id, file.filename, content)

    # Create DB record
    doc = await create_document(session, workspace_id, file.filename, file_type, len(content), file_path)

    # Dispatch Celery task
    process_document.delay(str(doc.id), str(workspace_id))

    return doc


@router.get("/workspaces/{workspace_id}/documents", response_model=list[DocumentResponse])
async def list_docs(
    workspace_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    workspace = await get_workspace(session, workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return await list_documents(session, workspace_id)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_doc(
    document_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    doc = await get_document(session, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    # Verify ownership via workspace
    workspace = await get_workspace(session, doc.workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_doc(
    document_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    doc = await get_document(session, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    workspace = await get_workspace(session, doc.workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Document not found")
    # Delete Qdrant chunks
    qdrant = get_qdrant_client()
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector={"filter": {"must": [{"key": "document_id", "match": {"value": str(doc.id)}}]}},
    )
    await delete_document(session, doc)
```

- [ ] **Step 10: Register document router in main.py**

Add to `backend/main.py`:
```python
from backend.modules.document.routes import router as document_router
app.include_router(document_router)
```

- [ ] **Step 11: Write integration tests for document endpoints**

```python
# tests/test_document.py
import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_txt_document(client: AsyncClient, auth_headers: dict):
    # Create workspace first
    ws_resp = await client.post("/workspaces", json={"name": "Doc WS"}, headers=auth_headers)
    ws_id = ws_resp.json()["id"]

    files = {"file": ("test.txt", io.BytesIO(b"Hello, this is a test document."), "text/plain")}
    response = await client.post(f"/workspaces/{ws_id}/documents", files=files, headers=auth_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["file_type"] == "txt"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_upload_invalid_file_type(client: AsyncClient, auth_headers: dict):
    ws_resp = await client.post("/workspaces", json={"name": "Doc WS2"}, headers=auth_headers)
    ws_id = ws_resp.json()["id"]

    files = {"file": ("test.jpg", io.BytesIO(b"fake image"), "image/jpeg")}
    response = await client.post(f"/workspaces/{ws_id}/documents", files=files, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_documents(client: AsyncClient, auth_headers: dict):
    ws_resp = await client.post("/workspaces", json={"name": "Doc WS3"}, headers=auth_headers)
    ws_id = ws_resp.json()["id"]

    files = {"file": ("doc1.txt", io.BytesIO(b"Content 1"), "text/plain")}
    await client.post(f"/workspaces/{ws_id}/documents", files=files, headers=auth_headers)

    response = await client.get(f"/workspaces/{ws_id}/documents", headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json()) == 1


@pytest.mark.asyncio
async def test_delete_document(client: AsyncClient, auth_headers: dict):
    ws_resp = await client.post("/workspaces", json={"name": "Doc WS4"}, headers=auth_headers)
    ws_id = ws_resp.json()["id"]

    files = {"file": ("to_delete.txt", io.BytesIO(b"Delete me"), "text/plain")}
    doc_resp = await client.post(f"/workspaces/{ws_id}/documents", files=files, headers=auth_headers)
    doc_id = doc_resp.json()["id"]

    response = await client.delete(f"/documents/{doc_id}", headers=auth_headers)
    assert response.status_code == 204
```

- [ ] **Step 12: Run tests to verify they pass**

```bash
pytest tests/test_document.py tests/test_chunking.py -v
# Expected: all passed
```

- [ ] **Step 13: Commit**

```bash
git add backend/modules/document/ backend/core/celery_app.py tests/test_document.py tests/test_chunking.py backend/main.py
git commit -m "feat: document module with upload, chunking pipeline, and Celery tasks"
```

---

## Task 7: Chat Module & RAG Pipeline

**Files:**
- Create: `backend/modules/chat/schemas.py`
- Create: `backend/modules/chat/prompts.py`
- Create: `backend/modules/chat/memory.py`
- Create: `backend/modules/chat/service.py`
- Create: `backend/modules/chat/routes.py`
- Create: `tests/test_chat.py`

- [ ] **Step 1: Implement chat schemas**

```python
# backend/modules/chat/schemas.py
import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: uuid.UUID | None = None


class SourceReference(BaseModel):
    document_id: str
    chunk_text_preview: str
    similarity_score: float


class MessageResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    sources: list[SourceReference] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
```

- [ ] **Step 2: Implement prompt templates**

```python
# backend/modules/chat/prompts.py

SYSTEM_PROMPT = """You are OmniBrain, an AI assistant that helps users understand and extract insights from their documents.

RULES:
- Answer ONLY based on the provided context from the user's documents.
- If the context does not contain enough information to answer, say so clearly.
- Always cite which document the information comes from when possible.
- Be concise and direct in your answers.
- Format your responses with clear structure (headings, bullet points) when appropriate.

CONTEXT FROM DOCUMENTS:
{context}

{summary_section}"""

SUMMARY_SECTION = """PREVIOUS CONVERSATION SUMMARY:
{summary}"""

SUMMARIZATION_PROMPT = """Summarize the following conversation concisely, preserving key facts, decisions, and context that would be needed to continue the conversation:

{messages}

Provide a concise summary in 2-3 paragraphs."""
```

- [ ] **Step 3: Implement memory (summarization)**

```python
# backend/modules/chat/memory.py
import tiktoken
import httpx

from backend.core.config import settings
from backend.modules.chat.prompts import SUMMARIZATION_PROMPT

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def count_messages_tokens(messages: list[dict[str, str]]) -> int:
    return sum(count_tokens(m.get("content", "")) for m in messages)


def should_summarize(messages: list[dict[str, str]]) -> bool:
    return count_messages_tokens(messages) > settings.summarization_token_threshold


async def generate_summary(messages: list[dict[str, str]]) -> str:
    messages_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    prompt = SUMMARIZATION_PROMPT.format(messages=messages_text)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Implement RAG service**

```python
# backend/modules/chat/service.py
import uuid

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.qdrant import COLLECTION_NAME, get_qdrant_client
from backend.modules.chat.memory import count_messages_tokens, generate_summary, should_summarize
from backend.modules.chat.models import ChatSession, Message
from backend.modules.chat.prompts import SUMMARY_SECTION, SYSTEM_PROMPT


async def get_or_create_session(
    session: AsyncSession, workspace_id: uuid.UUID, session_id: uuid.UUID | None
) -> ChatSession:
    if session_id is not None:
        result = await session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        chat_session = result.scalar_one_or_none()
        if chat_session is not None:
            return chat_session

    chat_session = ChatSession(id=uuid.uuid4(), workspace_id=workspace_id)
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return chat_session


async def save_message(
    session: AsyncSession,
    session_id: uuid.UUID,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> Message:
    message = Message(
        id=uuid.uuid4(),
        session_id=session_id,
        role=role,
        content=content,
        sources=sources,
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


async def get_chat_history(session: AsyncSession, session_id: uuid.UUID) -> list[Message]:
    result = await session.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    return list(result.scalars().all())


async def list_sessions(session: AsyncSession, workspace_id: uuid.UUID) -> list[ChatSession]:
    result = await session.execute(
        select(ChatSession)
        .where(ChatSession.workspace_id == workspace_id)
        .order_by(ChatSession.updated_at.desc())
    )
    return list(result.scalars().all())


async def delete_session(session: AsyncSession, chat_session: ChatSession) -> None:
    await session.delete(chat_session)
    await session.commit()


def retrieve_relevant_chunks(workspace_id: str, query_embedding: list[float]) -> list[dict]:
    qdrant = get_qdrant_client()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter={"must": [{"key": "workspace_id", "match": {"value": workspace_id}}]},
        limit=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )
    return [
        {
            "document_id": point.payload["document_id"],
            "chunk_text": point.payload["chunk_text"],
            "chunk_index": point.payload["chunk_index"],
            "score": point.score,
            "metadata": point.payload.get("metadata", {}),
        }
        for point in results
    ]


def embed_query(text: str) -> list[float]:
    response = httpx.post(
        "https://api.octen.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.octen_api_key}"},
        json={"model": "Octen-Embedding-8B", "input": [text]},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant documents found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source_file", "Unknown")
        parts.append(f"[Source {i}: {source}]\n{chunk['chunk_text']}")
    return "\n\n---\n\n".join(parts)


async def build_prompt_messages(
    db_session: AsyncSession,
    chat_session: ChatSession,
    context: str,
    user_message: str,
) -> list[dict[str, str]]:
    # Build summary section
    summary_section = ""
    if chat_session.context_summary:
        summary_section = SUMMARY_SECTION.format(summary=chat_session.context_summary)

    system_content = SYSTEM_PROMPT.format(context=context, summary_section=summary_section)

    messages = [{"role": "system", "content": system_content}]

    # Get recent chat history
    history = await get_chat_history(db_session, chat_session.id)
    recent = history[-(settings.chat_history_limit):]

    history_dicts = [{"role": m.role, "content": m.content} for m in recent]

    # Check if summarization is needed
    if should_summarize(history_dicts) and len(history) > settings.chat_history_limit:
        old_messages = [{"role": m.role, "content": m.content} for m in history[:-settings.chat_history_limit]]
        summary = await generate_summary(old_messages)
        chat_session.context_summary = summary
        await db_session.commit()
        # Rebuild system prompt with new summary
        summary_section = SUMMARY_SECTION.format(summary=summary)
        system_content = SYSTEM_PROMPT.format(context=context, summary_section=summary_section)
        messages = [{"role": "system", "content": system_content}]

    messages.extend(history_dicts)
    messages.append({"role": "user", "content": user_message})
    return messages


async def stream_llm_response(prompt_messages: list[dict[str, str]]):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o",
                "messages": prompt_messages,
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    yield data
```

- [ ] **Step 5: Implement chat routes with SSE streaming**

```python
# backend/modules/chat/routes.py
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_async_session
from backend.core.dependencies import get_current_user
from backend.modules.auth.models import User
from backend.modules.chat.schemas import ChatRequest, ChatSessionResponse, MessageResponse
from backend.modules.chat.service import (
    build_context,
    build_prompt_messages,
    delete_session,
    embed_query,
    get_chat_history,
    get_or_create_session,
    list_sessions,
    retrieve_relevant_chunks,
    save_message,
    stream_llm_response,
)
from backend.modules.workspace.service import get_workspace

router = APIRouter(tags=["chat"])


@router.post("/workspaces/{workspace_id}/chat")
async def chat(
    workspace_id: uuid.UUID,
    body: ChatRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    workspace = await get_workspace(session, workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Get or create chat session
    chat_session = await get_or_create_session(session, workspace_id, body.session_id)

    # Save user message
    await save_message(session, chat_session.id, "user", body.message)

    # Retrieve relevant chunks
    query_embedding = embed_query(body.message)
    chunks = retrieve_relevant_chunks(str(workspace_id), query_embedding)

    # Build context and prompt
    context = build_context(chunks)
    prompt_messages = await build_prompt_messages(session, chat_session, context, body.message)

    # Prepare sources for saving later
    sources = [
        {
            "document_id": c["document_id"],
            "chunk_text_preview": c["chunk_text"][:200],
            "similarity_score": c["score"],
        }
        for c in chunks
    ]

    async def event_stream():
        full_response = []
        # Send session_id as first event
        yield f"data: {json.dumps({'type': 'session', 'session_id': str(chat_session.id)})}\n\n"

        try:
            async for chunk_data in stream_llm_response(prompt_messages):
                try:
                    parsed = json.loads(chunk_data)
                    delta = parsed.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_response.append(content)
                        yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                except json.JSONDecodeError:
                    continue

            # Save complete response
            complete_text = "".join(full_response)
            await save_message(session, chat_session.id, "assistant", complete_text, sources)

            # Send sources
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/workspaces/{workspace_id}/chat/sessions", response_model=list[ChatSessionResponse])
async def get_sessions(
    workspace_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    workspace = await get_workspace(session, workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return await list_sessions(session, workspace_id)


@router.get("/chat/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    session_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    from backend.modules.chat.models import ChatSession as ChatSessionModel
    from sqlalchemy import select

    result = await session.execute(select(ChatSessionModel).where(ChatSessionModel.id == session_id))
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    workspace = await get_workspace(session, chat_session.workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return await get_chat_history(session, session_id)


@router.delete("/chat/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    from backend.modules.chat.models import ChatSession as ChatSessionModel
    from sqlalchemy import select

    result = await session.execute(select(ChatSessionModel).where(ChatSessionModel.id == session_id))
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    workspace = await get_workspace(session, chat_session.workspace_id, user.id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Session not found")

    await delete_session(session, chat_session)
```

- [ ] **Step 6: Register chat router in main.py**

Add to `backend/main.py`:
```python
from backend.modules.chat.routes import router as chat_router
app.include_router(chat_router)
```

- [ ] **Step 7: Write integration tests**

```python
# tests/test_chat.py
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_sessions_empty(client: AsyncClient, auth_headers: dict):
    ws_resp = await client.post("/workspaces", json={"name": "Chat WS"}, headers=auth_headers)
    ws_id = ws_resp.json()["id"]
    response = await client.get(f"/workspaces/{ws_id}/chat/sessions", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_delete_session_not_found(client: AsyncClient, auth_headers: dict):
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await client.delete(f"/chat/sessions/{fake_id}", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_workspace_not_found(client: AsyncClient, auth_headers: dict):
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await client.post(
        f"/workspaces/{fake_id}/chat",
        json={"message": "Hello"},
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_message_too_long(client: AsyncClient, auth_headers: dict):
    ws_resp = await client.post("/workspaces", json={"name": "Chat WS2"}, headers=auth_headers)
    ws_id = ws_resp.json()["id"]
    response = await client.post(
        f"/workspaces/{ws_id}/chat",
        json={"message": "x" * 4001},
        headers=auth_headers,
    )
    assert response.status_code == 422
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
pytest tests/test_chat.py -v
# Expected: 4 passed
```

- [ ] **Step 9: Commit**

```bash
git add backend/modules/chat/ tests/test_chat.py backend/main.py
git commit -m "feat: chat module with RAG pipeline, SSE streaming, and memory summarization"
```

---

## Task 8: Frontend — Next.js Chat UI

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/next.config.ts`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/lib/auth.ts`
- Create: `frontend/src/app/layout.tsx`
- Create: `frontend/src/app/page.tsx`
- Create: `frontend/src/app/login/page.tsx`
- Create: `frontend/src/app/dashboard/page.tsx`
- Create: `frontend/src/app/workspace/[id]/page.tsx`
- Create: `frontend/src/components/ChatMessage.tsx`
- Create: `frontend/src/components/ChatInput.tsx`
- Create: `frontend/src/components/DocumentList.tsx`
- Create: `frontend/src/components/WorkspaceCard.tsx`
- Create: `frontend/src/components/SourceCitation.tsx`

- [ ] **Step 1: Initialize Next.js project**

```bash
cd frontend
npx create-next-app@latest . --typescript --tailwind --app --src-dir --no-import-alias --use-npm
npm install react-markdown remark-gfm
```

- [ ] **Step 2: Create API client and auth helpers**

`frontend/src/lib/auth.ts` — JWT storage with login/logout helpers
`frontend/src/lib/api.ts` — fetch wrapper that attaches `Authorization: Bearer <token>` header, handles errors

- [ ] **Step 3: Create Login/Register page**

`frontend/src/app/login/page.tsx` — form with email + password, toggle between login/register mode, redirect to dashboard on success

- [ ] **Step 4: Create Dashboard page**

`frontend/src/app/dashboard/page.tsx` — list workspaces, create new workspace button, click workspace to navigate to `/workspace/[id]`

`frontend/src/components/WorkspaceCard.tsx` — workspace card with name, description, delete button

- [ ] **Step 5: Create Workspace View page**

`frontend/src/app/workspace/[id]/page.tsx` — two-panel layout:
- Left: `DocumentList` component
- Right: chat interface with `ChatMessage` list + `ChatInput`

- [ ] **Step 6: Create DocumentList component**

`frontend/src/components/DocumentList.tsx` — file upload input, list documents with status badges (pending=yellow, processing=blue, completed=green, failed=red), delete button

- [ ] **Step 7: Create Chat components**

`frontend/src/components/ChatMessage.tsx` — renders message with `react-markdown` + `remark-gfm` for proper formatting. User messages styled differently from assistant messages.

`frontend/src/components/ChatInput.tsx` — text input with send button, max 4000 chars, disabled while streaming

`frontend/src/components/SourceCitation.tsx` — expandable section showing source document references for each assistant message

- [ ] **Step 8: Implement SSE streaming in chat**

In workspace page, handle chat submission:
1. POST to `/workspaces/{id}/chat` with `fetch` (not EventSource, since we need POST)
2. Read response as `ReadableStream`
3. Parse SSE events: `session` → save session_id, `content` → append to message, `sources` → show citations, `[DONE]` → finalize

- [ ] **Step 9: Verify frontend runs**

```bash
cd frontend && npm run dev
# Visit http://localhost:3000
# Expected: login page renders
```

- [ ] **Step 10: Commit**

```bash
git add frontend/
git commit -m "feat: Next.js frontend with login, dashboard, workspace view, and chat UI"
```

---

## Task 9: RAGAS Evaluation Setup

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/datasets/sample_qa.json`
- Create: `eval/run_eval.py`

- [ ] **Step 1: Create sample evaluation dataset**

```json
// eval/datasets/sample_qa.json
{
  "test_cases": [
    {
      "question": "What is the main topic of the document?",
      "ground_truth_context": "The document discusses...",
      "expected_answer": "The main topic is..."
    }
  ],
  "metadata": {
    "description": "Sample QA pairs for OmniBrain RAG evaluation",
    "version": "0.1",
    "note": "Replace with real Q&A pairs from your actual test documents"
  }
}
```

- [ ] **Step 2: Create evaluation script**

```python
# eval/run_eval.py
"""
RAGAS evaluation script for OmniBrain RAG pipeline.

Usage:
    python -m eval.run_eval --dataset eval/datasets/sample_qa.json

Evaluates: Faithfulness, Context Precision, Answer Relevancy
Saves results to eval/results/
"""
import argparse
import json
import os
from datetime import datetime, timezone

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from backend.modules.chat.service import build_context, embed_query, retrieve_relevant_chunks


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["test_cases"]


def run_evaluation(dataset_path: str, workspace_id: str) -> dict:
    test_cases = load_dataset(dataset_path)
    questions = []
    contexts = []
    answers = []
    ground_truths = []

    for case in test_cases:
        question = case["question"]
        query_embedding = embed_query(question)
        chunks = retrieve_relevant_chunks(workspace_id, query_embedding)
        context = build_context(chunks)

        questions.append(question)
        contexts.append([c["chunk_text"] for c in chunks])
        ground_truths.append(case.get("expected_answer", ""))
        # Note: actual LLM answer would need to be generated here
        # TODO: Wire up actual RAG pipeline call here to generate real answers
        # For now, placeholder — run_eval will only measure context_precision without answers
        answers.append("")

    result = evaluate(
        dataset={
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        },
        metrics=[faithfulness, context_precision, answer_relevancy],
    )
    return result.to_pandas().to_dict()


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--dataset", required=True, help="Path to QA dataset JSON")
    parser.add_argument("--workspace-id", required=True, help="Workspace UUID to evaluate against")
    args = parser.parse_args()

    results = run_evaluation(args.dataset, args.workspace_id)

    os.makedirs("eval/results", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = f"eval/results/eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify eval script imports correctly**

```bash
python -c "from eval.run_eval import load_dataset; print('OK')"
# Expected: OK
```

- [ ] **Step 4: Commit**

```bash
git add eval/
git commit -m "feat: RAGAS evaluation scaffold with sample dataset and runner script"
```

---

## Task 10: End-to-End Integration Test & Final Verification

**Files:**
- Modify: `backend/main.py` (ensure all routers registered)
- Modify: `docker-compose.yml` (if needed)

- [ ] **Step 1: Verify all routers are registered in main.py**

```python
# backend/main.py should have:
from backend.modules.auth.routes import router as auth_router
from backend.modules.workspace.routes import router as workspace_router
from backend.modules.document.routes import router as document_router
from backend.modules.chat.routes import router as chat_router

app.include_router(auth_router)
app.include_router(workspace_router)
app.include_router(document_router)
app.include_router(chat_router)
```

- [ ] **Step 2: Start all services and verify**

```bash
# Terminal 1: Infrastructure
docker compose up -d

# Terminal 2: Backend
cd backend && uvicorn main:app --reload --port 8000

# Terminal 3: Celery worker
cd backend && celery -A core.celery_app worker --loglevel=info

# Terminal 4: Frontend
cd frontend && npm run dev
```

- [ ] **Step 3: Manual smoke test via Swagger UI**

Visit `http://localhost:8000/docs`:
1. POST `/auth/register` — create user
2. POST `/auth/login` — get JWT token
3. Authorize in Swagger with Bearer token
4. POST `/workspaces` — create workspace
5. POST `/workspaces/{id}/documents` — upload a .txt file
6. GET `/documents/{id}` — poll until status is "completed"
7. POST `/workspaces/{id}/chat` — send a question about the document

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v --tb=short
# Expected: all tests pass
```

- [ ] **Step 5: Verify frontend flow**

Visit `http://localhost:3000`:
1. Register/login
2. Create workspace
3. Upload document
4. Chat with document — verify streaming works, markdown renders properly, sources show

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: OmniBrain MVP v0.1 complete — end-to-end integration verified"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Project scaffolding & infra | docker-compose.yml, config.py, main.py |
| 2 | Database models & Alembic | modules/*/models.py, alembic/ |
| 3 | Qdrant client & collection | core/qdrant.py |
| 4 | Auth module (JWT) | modules/auth/*, core/security.py |
| 5 | Workspace module (CRUD) | modules/workspace/* |
| 6 | Document module (upload + Celery pipeline) | modules/document/*, core/celery_app.py |
| 7 | Chat module (RAG + SSE streaming) | modules/chat/* |
| 8 | Frontend (Next.js chat UI) | frontend/src/* |
| 9 | RAGAS evaluation | eval/* |
| 10 | End-to-end integration test | All files verified |

# OmniBrain MVP v0.1 — Testing Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Comprehensive testing coverage for OmniBrain MVP v0.1 across API (unit + integration), UI (component + E2E), and manual end-to-end verification.

**Architecture:** Three testing layers — (1) Backend pytest suite for unit & API integration tests, (2) Frontend Jest + Playwright for component & E2E browser tests, (3) Manual E2E test script for human verification of the full flow including streaming, Celery, and RAG quality.

**Tech Stack:** pytest, pytest-asyncio, httpx, unittest.mock, Jest, React Testing Library, Playwright, Docker Compose

**Spec:** `docs/superpowers/specs/2026-03-23-omnibrain-mvp-design.md`
**Implementation Plan:** `docs/superpowers/plans/2026-03-23-omnibrain-mvp-v0.1.md`

---

## File Structure

```
omnibrain/
├── tests/                              — Backend tests (pytest)
│   ├── conftest.py                         — shared fixtures (db, client, auth, mocks)
│   ├── unit/
│   │   ├── test_security.py                — JWT encode/decode, password hashing
│   │   ├── test_chunking.py                — chunking interface & strategies
│   │   ├── test_memory.py                  — token counting, summarization trigger
│   │   ├── test_prompts.py                 — prompt template construction
│   │   └── test_config.py                  — settings validation
│   ├── integration/
│   │   ├── test_auth_api.py                — register/login endpoints
│   │   ├── test_workspace_api.py           — workspace CRUD endpoints
│   │   ├── test_document_api.py            — document upload/list/delete endpoints
│   │   ├── test_document_pipeline.py       — Celery task: parse → chunk → embed → store
│   │   ├── test_chat_api.py                — chat endpoints, session management
│   │   ├── test_qdrant_integration.py      — Qdrant client, collection, vector ops
│   │   └── test_rag_pipeline.py            — retrieval → prompt → generation flow
│   └── fixtures/
│       ├── sample.txt                      — test text file (~500 words)
│       └── sample.pdf                      — test PDF file (2 pages)
│
├── frontend/
│   ├── __tests__/                      — Frontend tests (Jest + RTL)
│   │   ├── components/
│   │   │   ├── ChatMessage.test.tsx         — markdown rendering, role styling
│   │   │   ├── ChatInput.test.tsx           — input validation, submit, disabled state
│   │   │   ├── DocumentList.test.tsx        — upload, status badges, delete
│   │   │   ├── WorkspaceCard.test.tsx       — display, delete action
│   │   │   └── SourceCitation.test.tsx      — expand/collapse, content display
│   │   ├── lib/
│   │   │   ├── api.test.ts                  — fetch wrapper, auth header injection
│   │   │   └── auth.test.ts                 — JWT storage, login/logout
│   │   └── pages/
│   │       ├── login.test.tsx               — form submit, validation, redirect
│   │       └── dashboard.test.tsx           — workspace list, create, navigate
│   └── e2e/                            — E2E browser tests (Playwright)
│       ├── playwright.config.ts
│       ├── auth.spec.ts                     — register → login flow
│       ├── workspace.spec.ts                — create/delete workspace
│       ├── document.spec.ts                 — upload, status polling, delete
│       └── chat.spec.ts                     — send message, streaming, sources
│
└── docs/
    └── superpowers/
        └── plans/
            └── 2026-03-23-omnibrain-mvp-v0.1-testing.md  — this file
```

---

## Task 1: Test Infrastructure Setup

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/fixtures/sample.txt`
- Create: `tests/fixtures/sample.pdf`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Modify: `backend/requirements.txt` (add test dependencies)

- [ ] **Step 1: Add test dependencies to requirements.txt**

Append to `backend/requirements.txt`:
```txt
# Testing
pytest==8.3.*
pytest-asyncio==0.24.*
pytest-cov==6.0.*
httpx==0.28.*
factory-boy==3.3.*
```

- [ ] **Step 2: Create conftest.py with shared fixtures**

```python
# tests/conftest.py
import asyncio
import io
import os
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.config import settings
from backend.core.database import Base, get_async_session
from backend.main import app

TEST_DATABASE_URL = settings.database_url.replace("/omnibrain", "/omnibrain_test")
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
async def registered_user(client: AsyncClient) -> dict:
    """Register a user and return {email, password}."""
    email = f"test-{uuid.uuid4().hex[:8]}@example.com"
    password = "testpassword123"
    await client.post("/auth/register", json={"email": email, "password": password})
    return {"email": email, "password": password}


@pytest.fixture
async def auth_headers(client: AsyncClient, registered_user: dict) -> dict:
    """Login and return auth headers."""
    response = await client.post("/auth/login", json=registered_user)
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def workspace_id(client: AsyncClient, auth_headers: dict) -> str:
    """Create a workspace and return its ID."""
    response = await client.post(
        "/workspaces",
        json={"name": "Test Workspace", "description": "For testing"},
        headers=auth_headers,
    )
    return response.json()["id"]


@pytest.fixture
def sample_txt_content() -> bytes:
    """Return sample text content for upload tests."""
    return (
        b"Artificial intelligence (AI) is transforming how businesses operate. "
        b"Machine learning models can analyze vast datasets to uncover patterns. "
        b"Natural language processing enables computers to understand human language. "
        b"Computer vision allows machines to interpret images and videos. "
        b"Reinforcement learning trains agents through trial and error. "
        b"Deep learning uses neural networks with multiple layers. "
        b"Transfer learning applies knowledge from one domain to another. "
        b"AI ethics ensures responsible development and deployment of AI systems. "
        b"Generative AI creates new content including text, images, and code. "
        b"RAG combines retrieval with generation for more accurate AI responses."
    )


@pytest.fixture
def sample_txt_file(sample_txt_content: bytes):
    """Return a file-like object for upload."""
    return {"file": ("test_document.txt", io.BytesIO(sample_txt_content), "text/plain")}


@pytest.fixture
def mock_celery_task():
    """Mock Celery task dispatch to avoid needing a running worker."""
    with patch("backend.modules.document.routes.process_document") as mock_task:
        mock_task.delay = MagicMock()
        yield mock_task


@pytest.fixture
def mock_embeddings():
    """Mock Octen.ai embedding API calls."""
    fake_embedding = [0.1] * 4096
    with patch("backend.modules.chat.service.embed_query", return_value=fake_embedding) as mock:
        yield mock


@pytest.fixture
def mock_llm_stream():
    """Mock OpenRouter LLM streaming response."""
    import json

    async def fake_stream(*args, **kwargs):
        chunk = json.dumps({"choices": [{"delta": {"content": "This is a test AI response."}}]})
        yield chunk

    with patch("backend.modules.chat.service.stream_llm_response", side_effect=fake_stream) as mock:
        yield mock


@pytest.fixture
def mock_qdrant_search():
    """Mock Qdrant vector search results."""
    mock_results = [
        MagicMock(
            payload={
                "document_id": str(uuid.uuid4()),
                "chunk_text": "AI is transforming how businesses operate.",
                "chunk_index": 0,
                "metadata": {"source_file": "test_document.txt"},
            },
            score=0.92,
        ),
        MagicMock(
            payload={
                "document_id": str(uuid.uuid4()),
                "chunk_text": "RAG combines retrieval with generation.",
                "chunk_index": 1,
                "metadata": {"source_file": "test_document.txt"},
            },
            score=0.85,
        ),
    ]
    with patch("backend.modules.chat.service.retrieve_relevant_chunks") as mock:
        mock.return_value = [
            {
                "document_id": r.payload["document_id"],
                "chunk_text": r.payload["chunk_text"],
                "chunk_index": r.payload["chunk_index"],
                "score": r.score,
                "metadata": r.payload["metadata"],
            }
            for r in mock_results
        ]
        yield mock
```

- [ ] **Step 3: Create test fixture files**

`tests/fixtures/sample.txt`:
```
Artificial intelligence (AI) is a broad field of computer science focused on building smart machines capable of performing tasks that typically require human intelligence. AI systems can learn from data, identify patterns, and make decisions with minimal human intervention.

Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. Deep learning, a further subset, uses neural networks with many layers to analyze complex patterns in large amounts of data.

Natural language processing (NLP) allows computers to understand, interpret, and generate human language. Applications include chatbots, translation services, and sentiment analysis tools.

Retrieval-Augmented Generation (RAG) is a technique that enhances AI responses by retrieving relevant information from a knowledge base before generating an answer. This approach reduces hallucination and improves accuracy by grounding responses in factual data.

Vector databases store data as high-dimensional vectors, enabling efficient similarity search. They are essential for RAG systems, where user queries are converted to vectors and matched against stored document embeddings to find the most relevant information.
```

For `tests/fixtures/sample.pdf`: create programmatically in a test helper or commit a simple 2-page PDF.

- [ ] **Step 4: Create __init__.py files**

```bash
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/fixtures/__init__.py
```

- [ ] **Step 5: Verify test infrastructure**

```bash
pytest tests/ --collect-only
# Expected: test collection succeeds, 0 tests collected (no test files yet)
```

- [ ] **Step 6: Commit**

```bash
git add tests/ backend/requirements.txt
git commit -m "test: setup test infrastructure with shared fixtures and mocks"
```

---

## Task 2: Unit Tests — Core Utilities

**Files:**
- Create: `tests/unit/test_security.py`
- Create: `tests/unit/test_chunking.py`
- Create: `tests/unit/test_memory.py`
- Create: `tests/unit/test_prompts.py`
- Create: `tests/unit/test_config.py`

These tests have NO external dependencies (no DB, no Redis, no Qdrant, no API calls).

- [ ] **Step 1: Write security unit tests**

```python
# tests/unit/test_security.py
from backend.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_password_returns_different_string(self):
        hashed = hash_password("mypassword")
        assert hashed != "mypassword"
        assert len(hashed) > 20

    def test_verify_correct_password(self):
        hashed = hash_password("mypassword")
        assert verify_password("mypassword", hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("mypassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_same_password_produces_different_hashes(self):
        hash1 = hash_password("mypassword")
        hash2 = hash_password("mypassword")
        assert hash1 != hash2  # bcrypt salts


class TestJWT:
    def test_create_and_decode_token(self):
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        token = create_access_token(subject=user_id)
        decoded = decode_access_token(token)
        assert decoded == user_id

    def test_decode_invalid_token_returns_none(self):
        result = decode_access_token("invalid.token.here")
        assert result is None

    def test_decode_empty_token_returns_none(self):
        result = decode_access_token("")
        assert result is None

    def test_token_is_string(self):
        token = create_access_token(subject="test")
        assert isinstance(token, str)
        assert len(token) > 50
```

- [ ] **Step 2: Write chunking unit tests**

```python
# tests/unit/test_chunking.py
from backend.modules.document.chunking import ChunkingStrategy, DefaultChunkingStrategy


class TestChunkingInterface:
    def test_default_strategy_implements_interface(self):
        strategy = DefaultChunkingStrategy(chunk_size=2000, chunk_overlap=200)
        assert isinstance(strategy, ChunkingStrategy)

    def test_cannot_instantiate_abstract_interface(self):
        import pytest
        with pytest.raises(TypeError):
            ChunkingStrategy()


class TestDefaultChunkingStrategy:
    def test_short_text_single_chunk(self):
        strategy = DefaultChunkingStrategy(chunk_size=2000, chunk_overlap=200)
        chunks = strategy.chunk("This is a short text.")
        assert len(chunks) == 1
        assert chunks[0] == "This is a short text."

    def test_empty_text_returns_empty_list(self):
        strategy = DefaultChunkingStrategy(chunk_size=2000, chunk_overlap=200)
        chunks = strategy.chunk("")
        assert chunks == []

    def test_long_text_produces_multiple_chunks(self):
        strategy = DefaultChunkingStrategy(chunk_size=100, chunk_overlap=20)
        text = "This is a sentence with some words. " * 50  # ~1800 chars
        chunks = strategy.chunk(text)
        assert len(chunks) > 1

    def test_each_chunk_respects_max_size(self):
        chunk_size = 200
        strategy = DefaultChunkingStrategy(chunk_size=chunk_size, chunk_overlap=20)
        text = "Word " * 500  # 2500 chars
        chunks = strategy.chunk(text)
        for chunk in chunks:
            assert len(chunk) <= chunk_size + 50  # small tolerance for split boundaries

    def test_chunks_cover_all_content(self):
        strategy = DefaultChunkingStrategy(chunk_size=100, chunk_overlap=0)
        text = "Alpha. Bravo. Charlie. Delta. Echo. Foxtrot. Golf. Hotel. India. Juliet."
        chunks = strategy.chunk(text)
        combined = " ".join(chunks)
        for word in ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet"]:
            assert word in combined

    def test_overlap_creates_redundancy(self):
        strategy = DefaultChunkingStrategy(chunk_size=100, chunk_overlap=30)
        text = "Sentence one about topic A. " * 20
        chunks = strategy.chunk(text)
        if len(chunks) >= 2:
            # With overlap, end of chunk N should appear in start of chunk N+1
            assert len(set(chunks)) == len(chunks)  # all chunks are unique despite overlap

    def test_paragraph_boundary_splitting(self):
        strategy = DefaultChunkingStrategy(chunk_size=200, chunk_overlap=0)
        text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three content here."
        chunks = strategy.chunk(text)
        # Should prefer splitting at paragraph boundaries (\n\n)
        assert len(chunks) >= 1
```

- [ ] **Step 3: Write memory unit tests**

```python
# tests/unit/test_memory.py
from unittest.mock import patch

from backend.modules.chat.memory import count_tokens, count_messages_tokens, should_summarize


class TestTokenCounting:
    def test_count_tokens_empty_string(self):
        assert count_tokens("") == 0

    def test_count_tokens_simple_text(self):
        count = count_tokens("Hello, world!")
        assert count > 0
        assert count < 10

    def test_count_tokens_long_text(self):
        text = "word " * 1000
        count = count_tokens(text)
        assert count > 500  # roughly 1 token per word

    def test_count_messages_tokens(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]
        count = count_messages_tokens(messages)
        assert count > 0

    def test_count_messages_tokens_empty_list(self):
        assert count_messages_tokens([]) == 0


class TestShouldSummarize:
    def test_short_conversation_no_summarize(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        assert should_summarize(messages) is False

    def test_long_conversation_triggers_summarize(self):
        # Each message ~100 tokens, 40 messages = ~4000 tokens > 3000 threshold
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "word " * 100}
            for i in range(40)
        ]
        assert should_summarize(messages) is True

    @patch("backend.modules.chat.memory.settings")
    def test_respects_custom_threshold(self, mock_settings):
        mock_settings.summarization_token_threshold = 10
        messages = [{"role": "user", "content": "This is a test message"}]
        # With threshold of 10, even a short message should trigger
        assert should_summarize(messages) is True
```

- [ ] **Step 4: Write prompt template tests**

```python
# tests/unit/test_prompts.py
from backend.modules.chat.prompts import SYSTEM_PROMPT, SUMMARY_SECTION, SUMMARIZATION_PROMPT


class TestPromptTemplates:
    def test_system_prompt_has_context_placeholder(self):
        assert "{context}" in SYSTEM_PROMPT

    def test_system_prompt_has_summary_placeholder(self):
        assert "{summary_section}" in SYSTEM_PROMPT

    def test_system_prompt_renders_with_context(self):
        rendered = SYSTEM_PROMPT.format(
            context="Document content here",
            summary_section=""
        )
        assert "Document content here" in rendered
        assert "{context}" not in rendered

    def test_summary_section_has_summary_placeholder(self):
        assert "{summary}" in SUMMARY_SECTION

    def test_summary_section_renders(self):
        rendered = SUMMARY_SECTION.format(summary="Previous conversation about AI")
        assert "Previous conversation about AI" in rendered

    def test_summarization_prompt_has_messages_placeholder(self):
        assert "{messages}" in SUMMARIZATION_PROMPT

    def test_system_prompt_includes_key_rules(self):
        assert "ONLY based on the provided context" in SYSTEM_PROMPT
        assert "cite" in SYSTEM_PROMPT.lower()
```

- [ ] **Step 5: Write config tests**

```python
# tests/unit/test_config.py
from backend.core.config import Settings


class TestConfig:
    def test_default_values(self):
        """Verify default config values match spec requirements."""
        # Don't instantiate Settings (needs env vars), just check class defaults
        assert Settings.model_fields["rag_top_k"].default == 5
        assert Settings.model_fields["rag_score_threshold"].default == 0.7
        assert Settings.model_fields["chat_history_limit"].default == 10
        assert Settings.model_fields["summarization_token_threshold"].default == 3000
        assert Settings.model_fields["chunk_size"].default == 2000
        assert Settings.model_fields["chunk_overlap"].default == 200
        assert Settings.model_fields["embedding_dimensions"].default == 4096
        assert Settings.model_fields["max_upload_size_bytes"].default == 20 * 1024 * 1024
        assert Settings.model_fields["embedding_batch_size"].default == 32
        assert Settings.model_fields["jwt_algorithm"].default == "HS256"
        assert Settings.model_fields["jwt_expiration_minutes"].default == 1440
```

- [ ] **Step 6: Run all unit tests**

```bash
pytest tests/unit/ -v --tb=short
# Expected: all tests pass
```

- [ ] **Step 7: Check coverage**

```bash
pytest tests/unit/ --cov=backend.core.security --cov=backend.modules.document.chunking --cov=backend.modules.chat.memory --cov=backend.modules.chat.prompts --cov-report=term-missing
# Expected: 80%+ coverage on these modules
```

- [ ] **Step 8: Commit**

```bash
git add tests/unit/
git commit -m "test: unit tests for security, chunking, memory, prompts, and config"
```

---

## Task 3: Integration Tests — Auth API

**Files:**
- Create: `tests/integration/test_auth_api.py`

- [ ] **Step 1: Write auth API integration tests**

```python
# tests/integration/test_auth_api.py
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestRegister:
    async def test_register_success(self, client: AsyncClient):
        response = await client.post("/auth/register", json={
            "email": "newuser@example.com",
            "password": "securepassword123"
        })
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert "id" in data
        assert "hashed_password" not in data
        assert "password" not in data

    async def test_register_duplicate_email_returns_409(self, client: AsyncClient):
        payload = {"email": "dup@example.com", "password": "password123"}
        await client.post("/auth/register", json=payload)
        response = await client.post("/auth/register", json=payload)
        assert response.status_code == 409

    async def test_register_invalid_email_returns_422(self, client: AsyncClient):
        response = await client.post("/auth/register", json={
            "email": "not-an-email",
            "password": "password123"
        })
        assert response.status_code == 422

    async def test_register_missing_password_returns_422(self, client: AsyncClient):
        response = await client.post("/auth/register", json={
            "email": "valid@example.com"
        })
        assert response.status_code == 422

    async def test_register_empty_password_returns_422(self, client: AsyncClient):
        response = await client.post("/auth/register", json={
            "email": "valid@example.com",
            "password": ""
        })
        assert response.status_code == 422


@pytest.mark.asyncio
class TestLogin:
    async def test_login_success(self, client: AsyncClient, registered_user: dict):
        response = await client.post("/auth/login", json=registered_user)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 50

    async def test_login_wrong_password_returns_401(self, client: AsyncClient, registered_user: dict):
        response = await client.post("/auth/login", json={
            "email": registered_user["email"],
            "password": "wrongpassword"
        })
        assert response.status_code == 401

    async def test_login_nonexistent_email_returns_401(self, client: AsyncClient):
        response = await client.post("/auth/login", json={
            "email": "nobody@example.com",
            "password": "password123"
        })
        assert response.status_code == 401

    async def test_login_invalid_email_format_returns_422(self, client: AsyncClient):
        response = await client.post("/auth/login", json={
            "email": "bad-format",
            "password": "password123"
        })
        assert response.status_code == 422


@pytest.mark.asyncio
class TestAuthProtection:
    async def test_protected_endpoint_without_token_returns_403(self, client: AsyncClient):
        response = await client.get("/workspaces")
        assert response.status_code == 403

    async def test_protected_endpoint_with_invalid_token_returns_401(self, client: AsyncClient):
        response = await client.get("/workspaces", headers={
            "Authorization": "Bearer invalid.token.here"
        })
        assert response.status_code == 401

    async def test_protected_endpoint_with_valid_token_succeeds(self, client: AsyncClient, auth_headers: dict):
        response = await client.get("/workspaces", headers=auth_headers)
        assert response.status_code == 200
```

- [ ] **Step 2: Run auth integration tests**

```bash
pytest tests/integration/test_auth_api.py -v
# Expected: all tests pass
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_auth_api.py
git commit -m "test: auth API integration tests (register, login, protection)"
```

---

## Task 4: Integration Tests — Workspace API

**Files:**
- Create: `tests/integration/test_workspace_api.py`

- [ ] **Step 1: Write workspace API integration tests**

```python
# tests/integration/test_workspace_api.py
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestCreateWorkspace:
    async def test_create_workspace_success(self, client: AsyncClient, auth_headers: dict):
        response = await client.post("/workspaces", json={
            "name": "My Research",
            "description": "Research workspace"
        }, headers=auth_headers)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "My Research"
        assert data["description"] == "Research workspace"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    async def test_create_workspace_without_description(self, client: AsyncClient, auth_headers: dict):
        response = await client.post("/workspaces", json={
            "name": "Minimal WS"
        }, headers=auth_headers)
        assert response.status_code == 201
        assert response.json()["description"] is None

    async def test_create_workspace_empty_name_returns_422(self, client: AsyncClient, auth_headers: dict):
        response = await client.post("/workspaces", json={
            "name": ""
        }, headers=auth_headers)
        assert response.status_code == 422

    async def test_create_workspace_name_too_long_returns_422(self, client: AsyncClient, auth_headers: dict):
        response = await client.post("/workspaces", json={
            "name": "x" * 101
        }, headers=auth_headers)
        assert response.status_code == 422

    async def test_create_workspace_name_exactly_100_chars_succeeds(self, client: AsyncClient, auth_headers: dict):
        response = await client.post("/workspaces", json={
            "name": "x" * 100
        }, headers=auth_headers)
        assert response.status_code == 201


@pytest.mark.asyncio
class TestListWorkspaces:
    async def test_list_empty(self, client: AsyncClient, auth_headers: dict):
        response = await client.get("/workspaces", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_multiple(self, client: AsyncClient, auth_headers: dict):
        await client.post("/workspaces", json={"name": "WS1"}, headers=auth_headers)
        await client.post("/workspaces", json={"name": "WS2"}, headers=auth_headers)
        await client.post("/workspaces", json={"name": "WS3"}, headers=auth_headers)
        response = await client.get("/workspaces", headers=auth_headers)
        assert response.status_code == 200
        assert len(response.json()) == 3

    async def test_list_only_own_workspaces(self, client: AsyncClient, auth_headers: dict):
        # Create workspace with user 1
        await client.post("/workspaces", json={"name": "User1 WS"}, headers=auth_headers)

        # Register and login as user 2
        await client.post("/auth/register", json={"email": "user2@example.com", "password": "pass123"})
        login_resp = await client.post("/auth/login", json={"email": "user2@example.com", "password": "pass123"})
        user2_headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

        # User 2 should see 0 workspaces
        response = await client.get("/workspaces", headers=user2_headers)
        assert response.json() == []


@pytest.mark.asyncio
class TestGetWorkspace:
    async def test_get_workspace_success(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        response = await client.get(f"/workspaces/{workspace_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == workspace_id

    async def test_get_nonexistent_workspace_returns_404(self, client: AsyncClient, auth_headers: dict):
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/workspaces/{fake_id}", headers=auth_headers)
        assert response.status_code == 404

    async def test_get_other_users_workspace_returns_404(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        # Register user 2
        await client.post("/auth/register", json={"email": "other@example.com", "password": "pass123"})
        login_resp = await client.post("/auth/login", json={"email": "other@example.com", "password": "pass123"})
        other_headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

        response = await client.get(f"/workspaces/{workspace_id}", headers=other_headers)
        assert response.status_code == 404


@pytest.mark.asyncio
class TestUpdateWorkspace:
    async def test_update_name(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        response = await client.put(f"/workspaces/{workspace_id}", json={
            "name": "Updated Name"
        }, headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Name"

    async def test_update_description(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        response = await client.put(f"/workspaces/{workspace_id}", json={
            "description": "New description"
        }, headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["description"] == "New description"

    async def test_update_nonexistent_returns_404(self, client: AsyncClient, auth_headers: dict):
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.put(f"/workspaces/{fake_id}", json={"name": "X"}, headers=auth_headers)
        assert response.status_code == 404


@pytest.mark.asyncio
class TestDeleteWorkspace:
    async def test_delete_success(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        response = await client.delete(f"/workspaces/{workspace_id}", headers=auth_headers)
        assert response.status_code == 204
        # Verify it's gone
        get_resp = await client.get(f"/workspaces/{workspace_id}", headers=auth_headers)
        assert get_resp.status_code == 404

    async def test_delete_nonexistent_returns_404(self, client: AsyncClient, auth_headers: dict):
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.delete(f"/workspaces/{fake_id}", headers=auth_headers)
        assert response.status_code == 404

    async def test_delete_cascades_documents(self, client: AsyncClient, auth_headers: dict, workspace_id: str, mock_celery_task):
        # Upload a document first
        files = {"file": ("doc.txt", b"test content", "text/plain")}
        await client.post(f"/workspaces/{workspace_id}/documents", files=files, headers=auth_headers)

        # Delete workspace
        await client.delete(f"/workspaces/{workspace_id}", headers=auth_headers)

        # Documents should be gone too
        docs_resp = await client.get(f"/workspaces/{workspace_id}/documents", headers=auth_headers)
        assert docs_resp.status_code == 404
```

- [ ] **Step 2: Run workspace integration tests**

```bash
pytest tests/integration/test_workspace_api.py -v
# Expected: all tests pass
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_workspace_api.py
git commit -m "test: workspace API integration tests (CRUD, ownership, cascade)"
```

---

## Task 5: Integration Tests — Document API

**Files:**
- Create: `tests/integration/test_document_api.py`
- Create: `tests/integration/test_document_pipeline.py`

- [ ] **Step 1: Write document API integration tests**

```python
# tests/integration/test_document_api.py
import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestUploadDocument:
    async def test_upload_txt_success(self, client: AsyncClient, auth_headers: dict, workspace_id: str, mock_celery_task):
        files = {"file": ("report.txt", io.BytesIO(b"Test document content"), "text/plain")}
        response = await client.post(
            f"/workspaces/{workspace_id}/documents", files=files, headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "report.txt"
        assert data["file_type"] == "txt"
        assert data["status"] == "pending"
        assert data["file_size"] > 0
        assert "id" in data
        # Verify Celery task was dispatched
        mock_celery_task.delay.assert_called_once()

    async def test_upload_pdf_success(self, client: AsyncClient, auth_headers: dict, workspace_id: str, mock_celery_task):
        # Minimal valid PDF bytes
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"
        files = {"file": ("report.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        response = await client.post(
            f"/workspaces/{workspace_id}/documents", files=files, headers=auth_headers
        )
        assert response.status_code == 201
        assert response.json()["file_type"] == "pdf"

    async def test_upload_invalid_type_returns_422(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        files = {"file": ("image.jpg", io.BytesIO(b"fake image data"), "image/jpeg")}
        response = await client.post(
            f"/workspaces/{workspace_id}/documents", files=files, headers=auth_headers
        )
        assert response.status_code == 422
        assert "Only PDF and TXT" in response.json()["detail"]

    async def test_upload_wrong_mime_type_with_valid_extension_returns_422(
        self, client: AsyncClient, auth_headers: dict, workspace_id: str
    ):
        """File has .txt extension but image/jpeg MIME type — should be rejected based on MIME, not extension."""
        files = {"file": ("sneaky.txt", io.BytesIO(b"fake image data"), "image/jpeg")}
        response = await client.post(
            f"/workspaces/{workspace_id}/documents", files=files, headers=auth_headers
        )
        assert response.status_code == 422

    async def test_upload_exceeds_size_limit_returns_422(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        # 21MB file
        large_content = b"x" * (21 * 1024 * 1024)
        files = {"file": ("huge.txt", io.BytesIO(large_content), "text/plain")}
        response = await client.post(
            f"/workspaces/{workspace_id}/documents", files=files, headers=auth_headers
        )
        assert response.status_code == 422
        assert "20MB" in response.json()["detail"]

    async def test_upload_to_nonexistent_workspace_returns_404(self, client: AsyncClient, auth_headers: dict):
        fake_id = "00000000-0000-0000-0000-000000000000"
        files = {"file": ("doc.txt", io.BytesIO(b"content"), "text/plain")}
        response = await client.post(
            f"/workspaces/{fake_id}/documents", files=files, headers=auth_headers
        )
        assert response.status_code == 404

    async def test_reupload_same_filename_replaces_document(
        self, client: AsyncClient, auth_headers: dict, workspace_id: str, mock_celery_task
    ):
        files1 = {"file": ("same_name.txt", io.BytesIO(b"version 1"), "text/plain")}
        resp1 = await client.post(f"/workspaces/{workspace_id}/documents", files=files1, headers=auth_headers)
        id1 = resp1.json()["id"]

        files2 = {"file": ("same_name.txt", io.BytesIO(b"version 2"), "text/plain")}
        resp2 = await client.post(f"/workspaces/{workspace_id}/documents", files=files2, headers=auth_headers)
        id2 = resp2.json()["id"]

        # Should be different document IDs
        assert id1 != id2

        # Only 1 document should exist
        list_resp = await client.get(f"/workspaces/{workspace_id}/documents", headers=auth_headers)
        assert len(list_resp.json()) == 1


@pytest.mark.asyncio
class TestListDocuments:
    async def test_list_empty(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        response = await client.get(f"/workspaces/{workspace_id}/documents", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_multiple_documents(
        self, client: AsyncClient, auth_headers: dict, workspace_id: str, mock_celery_task
    ):
        for name in ["doc1.txt", "doc2.txt", "doc3.txt"]:
            files = {"file": (name, io.BytesIO(b"content"), "text/plain")}
            await client.post(f"/workspaces/{workspace_id}/documents", files=files, headers=auth_headers)

        response = await client.get(f"/workspaces/{workspace_id}/documents", headers=auth_headers)
        assert len(response.json()) == 3


@pytest.mark.asyncio
class TestGetDocument:
    async def test_get_document_detail(
        self, client: AsyncClient, auth_headers: dict, workspace_id: str, mock_celery_task
    ):
        files = {"file": ("detail.txt", io.BytesIO(b"detail content"), "text/plain")}
        create_resp = await client.post(f"/workspaces/{workspace_id}/documents", files=files, headers=auth_headers)
        doc_id = create_resp.json()["id"]

        response = await client.get(f"/documents/{doc_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["filename"] == "detail.txt"

    async def test_get_nonexistent_document_returns_404(self, client: AsyncClient, auth_headers: dict):
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/documents/{fake_id}", headers=auth_headers)
        assert response.status_code == 404


@pytest.mark.asyncio
class TestDeleteDocument:
    async def test_delete_document_success(
        self, client: AsyncClient, auth_headers: dict, workspace_id: str, mock_celery_task
    ):
        files = {"file": ("to_delete.txt", io.BytesIO(b"delete me"), "text/plain")}
        create_resp = await client.post(f"/workspaces/{workspace_id}/documents", files=files, headers=auth_headers)
        doc_id = create_resp.json()["id"]

        response = await client.delete(f"/documents/{doc_id}", headers=auth_headers)
        assert response.status_code == 204

        # Verify it's gone
        get_resp = await client.get(f"/documents/{doc_id}", headers=auth_headers)
        assert get_resp.status_code == 404
```

- [ ] **Step 2: Write document pipeline integration tests (Celery task)**

```python
# tests/integration/test_document_pipeline.py
import os
import pytest
from unittest.mock import patch, MagicMock

from backend.modules.document.tasks import _parse_file


class TestParseFile:
    def test_parse_txt_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, this is a test document.", encoding="utf-8")
        result = _parse_file(str(file_path), "txt")
        assert result == "Hello, this is a test document."

    def test_parse_txt_file_utf8(self, tmp_path):
        file_path = tmp_path / "unicode.txt"
        file_path.write_text("Café résumé naïve", encoding="utf-8")
        result = _parse_file(str(file_path), "txt")
        assert "Café" in result

    def test_parse_empty_txt_file(self, tmp_path):
        file_path = tmp_path / "empty.txt"
        file_path.write_text("", encoding="utf-8")
        result = _parse_file(str(file_path), "txt")
        assert result == ""

    def test_parse_unsupported_type_raises(self, tmp_path):
        file_path = tmp_path / "test.jpg"
        file_path.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported file type"):
            _parse_file(str(file_path), "jpg")

    def test_parse_pdf_file(self):
        """Test PDF parsing with the fixture file."""
        fixture_path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample.pdf")
        if not os.path.exists(fixture_path):
            pytest.skip("sample.pdf fixture not found")
        result = _parse_file(fixture_path, "pdf")
        assert len(result) > 0
```

- [ ] **Step 3: Run document tests**

```bash
pytest tests/integration/test_document_api.py tests/integration/test_document_pipeline.py -v
# Expected: all tests pass
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_document_api.py tests/integration/test_document_pipeline.py
git commit -m "test: document API and pipeline integration tests"
```

---

## Task 6: Integration Tests — Chat API & RAG Pipeline

**Files:**
- Create: `tests/integration/test_chat_api.py`
- Create: `tests/integration/test_rag_pipeline.py`
- Create: `tests/integration/test_qdrant_integration.py`

- [ ] **Step 1: Write chat API integration tests**

```python
# tests/integration/test_chat_api.py
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestChatSessions:
    async def test_list_sessions_empty(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        response = await client.get(f"/workspaces/{workspace_id}/chat/sessions", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == []

    async def test_chat_creates_new_session(
        self, client: AsyncClient, auth_headers: dict, workspace_id: str,
        mock_embeddings, mock_qdrant_search
    ):
        response = await client.post(
            f"/workspaces/{workspace_id}/chat",
            json={"message": "What is AI?"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    async def test_chat_nonexistent_workspace_returns_404(self, client: AsyncClient, auth_headers: dict):
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.post(
            f"/workspaces/{fake_id}/chat",
            json={"message": "Hello"},
            headers=auth_headers,
        )
        assert response.status_code == 404

    async def test_chat_message_too_long_returns_422(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        response = await client.post(
            f"/workspaces/{workspace_id}/chat",
            json={"message": "x" * 4001},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_chat_empty_message_returns_422(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        response = await client.post(
            f"/workspaces/{workspace_id}/chat",
            json={"message": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_delete_session(self, client: AsyncClient, auth_headers: dict, workspace_id: str):
        # First, need a session — we test deletion of a non-existent one
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.delete(f"/chat/sessions/{fake_id}", headers=auth_headers)
        assert response.status_code == 404

    async def test_get_messages_nonexistent_session_returns_404(self, client: AsyncClient, auth_headers: dict):
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/chat/sessions/{fake_id}/messages", headers=auth_headers)
        assert response.status_code == 404

    async def test_continue_existing_session(
        self, client: AsyncClient, auth_headers: dict, workspace_id: str,
        mock_embeddings, mock_qdrant_search
    ):
        """Send two messages with same session_id to verify session continuity."""
        # First message creates session — parse session_id from SSE stream
        with patch("backend.modules.chat.service.stream_llm_response") as mock_llm:
            mock_llm.return_value = _fake_stream("First response")
            resp1 = await client.post(
                f"/workspaces/{workspace_id}/chat",
                json={"message": "First question"},
                headers=auth_headers,
            )
            assert resp1.status_code == 200
            # Extract session_id from SSE stream (first event)
            body = resp1.text
            # session_id would be in first SSE data event

    async def test_no_relevant_chunks_returns_info_message(
        self, client: AsyncClient, auth_headers: dict, workspace_id: str,
        mock_embeddings
    ):
        """When all chunks are below score threshold 0.7, inform the user."""
        with patch("backend.modules.chat.service.retrieve_relevant_chunks") as mock_retrieve:
            mock_retrieve.return_value = []  # No chunks above threshold
            with patch("backend.modules.chat.service.stream_llm_response") as mock_llm:
                mock_llm.return_value = _fake_stream("No relevant info found")
                response = await client.post(
                    f"/workspaces/{workspace_id}/chat",
                    json={"message": "Something unrelated"},
                    headers=auth_headers,
                )
                assert response.status_code == 200
                mock_retrieve.assert_called_once()
```

Helper for mock streaming:
```python
async def _fake_stream(content: str):
    """Helper to create a fake SSE stream for LLM mock."""
    import json
    chunk = json.dumps({"choices": [{"delta": {"content": content}}]})
    yield chunk
```

- [ ] **Step 2: Write RAG pipeline integration tests**

```python
# tests/integration/test_rag_pipeline.py
from backend.modules.chat.service import build_context


class TestBuildContext:
    def test_build_context_with_chunks(self):
        chunks = [
            {
                "document_id": "doc-1",
                "chunk_text": "AI is transforming industries.",
                "chunk_index": 0,
                "score": 0.95,
                "metadata": {"source_file": "report.txt"},
            },
            {
                "document_id": "doc-1",
                "chunk_text": "Machine learning is a subset of AI.",
                "chunk_index": 1,
                "score": 0.88,
                "metadata": {"source_file": "report.txt"},
            },
        ]
        context = build_context(chunks)
        assert "AI is transforming industries." in context
        assert "Machine learning is a subset of AI." in context
        assert "[Source 1: report.txt]" in context
        assert "[Source 2: report.txt]" in context

    def test_build_context_empty_chunks(self):
        context = build_context([])
        assert "No relevant documents found" in context

    def test_build_context_missing_metadata(self):
        chunks = [
            {
                "document_id": "doc-1",
                "chunk_text": "Some content",
                "chunk_index": 0,
                "score": 0.9,
                "metadata": {},
            },
        ]
        context = build_context(chunks)
        assert "Some content" in context
        assert "Unknown" in context  # fallback for missing source_file
```

- [ ] **Step 3: Write Qdrant integration tests**

```python
# tests/integration/test_qdrant_integration.py
import pytest
from backend.core.qdrant import get_qdrant_client, ensure_collection_exists, COLLECTION_NAME


class TestQdrantIntegration:
    """These tests require a running Qdrant instance (via Docker Compose)."""

    def test_client_connects(self):
        client = get_qdrant_client()
        assert client is not None

    def test_ensure_collection_creates_collection(self):
        client = get_qdrant_client()
        ensure_collection_exists(client)
        collections = client.get_collections().collections
        names = [c.name for c in collections]
        assert COLLECTION_NAME in names

    def test_collection_vector_config(self):
        client = get_qdrant_client()
        ensure_collection_exists(client)
        info = client.get_collection(COLLECTION_NAME)
        assert info.config.params.vectors.size == 4096
        assert info.config.params.vectors.distance.name == "COSINE"

    def test_ensure_collection_idempotent(self):
        client = get_qdrant_client()
        ensure_collection_exists(client)
        ensure_collection_exists(client)  # should not raise
        collections = client.get_collections().collections
        count = sum(1 for c in collections if c.name == COLLECTION_NAME)
        assert count == 1
```

- [ ] **Step 4: Run chat & RAG tests**

```bash
pytest tests/integration/test_chat_api.py tests/integration/test_rag_pipeline.py tests/integration/test_qdrant_integration.py -v
# Expected: all tests pass
```

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_chat_api.py tests/integration/test_rag_pipeline.py tests/integration/test_qdrant_integration.py
git commit -m "test: chat API, RAG pipeline, and Qdrant integration tests"
```

---

## Task 7: Frontend Tests — Component Tests (Jest + RTL)

**Files:**
- Create: `frontend/__tests__/components/ChatMessage.test.tsx`
- Create: `frontend/__tests__/components/ChatInput.test.tsx`
- Create: `frontend/__tests__/components/DocumentList.test.tsx`
- Create: `frontend/__tests__/components/WorkspaceCard.test.tsx`
- Create: `frontend/__tests__/components/SourceCitation.test.tsx`
- Create: `frontend/__tests__/lib/api.test.ts`
- Create: `frontend/__tests__/lib/auth.test.ts`

- [ ] **Step 1: Install test dependencies**

```bash
cd frontend
npm install --save-dev jest @testing-library/react @testing-library/jest-dom @testing-library/user-event jest-environment-jsdom @types/jest ts-jest
```

- [ ] **Step 2: Configure Jest**

Create `frontend/jest.config.ts`:
```typescript
import type { Config } from "jest";
import nextJest from "next/jest.js";

const createJestConfig = nextJest({ dir: "./" });

const config: Config = {
  testEnvironment: "jsdom",
  setupFilesAfterSetup: ["<rootDir>/jest.setup.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
};

export default createJestConfig(config);
```

Create `frontend/jest.setup.ts`:
```typescript
import "@testing-library/jest-dom";
```

- [ ] **Step 3: Write ChatMessage component tests**

```tsx
// frontend/__tests__/components/ChatMessage.test.tsx
import { render, screen } from "@testing-library/react";
import ChatMessage from "@/components/ChatMessage";

describe("ChatMessage", () => {
  it("renders user message with correct styling", () => {
    render(<ChatMessage role="user" content="Hello AI" />);
    expect(screen.getByText("Hello AI")).toBeInTheDocument();
  });

  it("renders assistant message", () => {
    render(<ChatMessage role="assistant" content="Hi! How can I help?" />);
    expect(screen.getByText(/How can I help/)).toBeInTheDocument();
  });

  it("renders markdown as formatted HTML, not raw symbols", () => {
    render(<ChatMessage role="assistant" content="**bold text** and *italic*" />);
    // Should render as HTML bold, not raw **
    expect(screen.queryByText("**bold text**")).not.toBeInTheDocument();
    const boldEl = screen.getByText("bold text");
    expect(boldEl.tagName).toBe("STRONG");
  });

  it("renders code blocks properly", () => {
    render(<ChatMessage role="assistant" content="```python\nprint('hello')\n```" />);
    expect(screen.getByText("print('hello')")).toBeInTheDocument();
  });

  it("renders bullet lists as HTML lists", () => {
    render(<ChatMessage role="assistant" content="- Item one\n- Item two\n- Item three" />);
    const listItems = screen.getAllByRole("listitem");
    expect(listItems).toHaveLength(3);
  });
});
```

- [ ] **Step 4: Write ChatInput component tests**

```tsx
// frontend/__tests__/components/ChatInput.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ChatInput from "@/components/ChatInput";

describe("ChatInput", () => {
  it("renders input field and send button", () => {
    render(<ChatInput onSend={jest.fn()} isStreaming={false} />);
    expect(screen.getByPlaceholderText(/ask a question/i)).toBeInTheDocument();
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("calls onSend with message text", async () => {
    const onSend = jest.fn();
    render(<ChatInput onSend={onSend} isStreaming={false} />);
    const input = screen.getByPlaceholderText(/ask a question/i);
    await userEvent.type(input, "What is AI?");
    await userEvent.click(screen.getByRole("button"));
    expect(onSend).toHaveBeenCalledWith("What is AI?");
  });

  it("clears input after sending", async () => {
    render(<ChatInput onSend={jest.fn()} isStreaming={false} />);
    const input = screen.getByPlaceholderText(/ask a question/i) as HTMLInputElement;
    await userEvent.type(input, "Hello");
    await userEvent.click(screen.getByRole("button"));
    expect(input.value).toBe("");
  });

  it("disables input and button while streaming", () => {
    render(<ChatInput onSend={jest.fn()} isStreaming={true} />);
    expect(screen.getByPlaceholderText(/ask a question/i)).toBeDisabled();
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("does not send empty message", async () => {
    const onSend = jest.fn();
    render(<ChatInput onSend={onSend} isStreaming={false} />);
    await userEvent.click(screen.getByRole("button"));
    expect(onSend).not.toHaveBeenCalled();
  });

  it("enforces 4000 character limit", async () => {
    const onSend = jest.fn();
    render(<ChatInput onSend={onSend} isStreaming={false} />);
    const input = screen.getByPlaceholderText(/ask a question/i) as HTMLInputElement;
    expect(input.maxLength).toBe(4000);
  });
});
```

- [ ] **Step 5: Write DocumentList component tests**

```tsx
// frontend/__tests__/components/DocumentList.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DocumentList from "@/components/DocumentList";

const mockDocuments = [
  { id: "1", filename: "report.pdf", file_type: "pdf", status: "completed", file_size: 1024, chunk_count: 5 },
  { id: "2", filename: "notes.txt", file_type: "txt", status: "processing", file_size: 512, chunk_count: 0 },
  { id: "3", filename: "broken.pdf", file_type: "pdf", status: "failed", file_size: 2048, chunk_count: 0 },
];

describe("DocumentList", () => {
  it("renders all documents", () => {
    render(<DocumentList documents={mockDocuments} onUpload={jest.fn()} onDelete={jest.fn()} />);
    expect(screen.getByText("report.pdf")).toBeInTheDocument();
    expect(screen.getByText("notes.txt")).toBeInTheDocument();
    expect(screen.getByText("broken.pdf")).toBeInTheDocument();
  });

  it("shows correct status badges", () => {
    render(<DocumentList documents={mockDocuments} onUpload={jest.fn()} onDelete={jest.fn()} />);
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getByText("processing")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
  });

  it("shows empty state when no documents", () => {
    render(<DocumentList documents={[]} onUpload={jest.fn()} onDelete={jest.fn()} />);
    expect(screen.getByText(/no documents/i)).toBeInTheDocument();
  });

  it("has upload input", () => {
    render(<DocumentList documents={[]} onUpload={jest.fn()} onDelete={jest.fn()} />);
    const fileInput = document.querySelector('input[type="file"]');
    expect(fileInput).toBeInTheDocument();
    expect(fileInput).toHaveAttribute("accept", ".pdf,.txt");
  });

  it("calls onDelete when delete button clicked", async () => {
    const onDelete = jest.fn();
    render(<DocumentList documents={mockDocuments} onUpload={jest.fn()} onDelete={onDelete} />);
    const deleteButtons = screen.getAllByRole("button", { name: /delete/i });
    await userEvent.click(deleteButtons[0]);
    expect(onDelete).toHaveBeenCalledWith("1");
  });
});
```

- [ ] **Step 6: Write WorkspaceCard and SourceCitation tests**

```tsx
// frontend/__tests__/components/WorkspaceCard.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import WorkspaceCard from "@/components/WorkspaceCard";

describe("WorkspaceCard", () => {
  const workspace = { id: "ws-1", name: "Research", description: "My research workspace" };

  it("renders workspace name and description", () => {
    render(<WorkspaceCard workspace={workspace} onDelete={jest.fn()} onClick={jest.fn()} />);
    expect(screen.getByText("Research")).toBeInTheDocument();
    expect(screen.getByText("My research workspace")).toBeInTheDocument();
  });

  it("calls onClick when card is clicked", async () => {
    const onClick = jest.fn();
    render(<WorkspaceCard workspace={workspace} onDelete={jest.fn()} onClick={onClick} />);
    await userEvent.click(screen.getByText("Research"));
    expect(onClick).toHaveBeenCalledWith("ws-1");
  });

  it("calls onDelete when delete button clicked", async () => {
    const onDelete = jest.fn();
    render(<WorkspaceCard workspace={workspace} onDelete={onDelete} onClick={jest.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /delete/i }));
    expect(onDelete).toHaveBeenCalledWith("ws-1");
  });
});
```

```tsx
// frontend/__tests__/components/SourceCitation.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import SourceCitation from "@/components/SourceCitation";

const mockSources = [
  { document_id: "doc-1", chunk_text_preview: "AI is transforming...", similarity_score: 0.95 },
  { document_id: "doc-2", chunk_text_preview: "Machine learning is...", similarity_score: 0.88 },
];

describe("SourceCitation", () => {
  it("renders source count", () => {
    render(<SourceCitation sources={mockSources} />);
    expect(screen.getByText(/2 sources/i)).toBeInTheDocument();
  });

  it("expands to show source details on click", async () => {
    render(<SourceCitation sources={mockSources} />);
    await userEvent.click(screen.getByText(/2 sources/i));
    expect(screen.getByText(/AI is transforming/)).toBeInTheDocument();
    expect(screen.getByText(/Machine learning is/)).toBeInTheDocument();
  });

  it("shows similarity scores", async () => {
    render(<SourceCitation sources={mockSources} />);
    await userEvent.click(screen.getByText(/2 sources/i));
    expect(screen.getByText(/95%/)).toBeInTheDocument();
    expect(screen.getByText(/88%/)).toBeInTheDocument();
  });

  it("renders nothing when no sources", () => {
    const { container } = render(<SourceCitation sources={[]} />);
    expect(container.firstChild).toBeNull();
  });
});
```

- [ ] **Step 7: Write lib/api and lib/auth tests**

```typescript
// frontend/__tests__/lib/auth.test.ts
import { getToken, setToken, removeToken, isAuthenticated } from "@/lib/auth";

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
  };
})();
Object.defineProperty(window, "localStorage", { value: localStorageMock });

describe("auth helpers", () => {
  beforeEach(() => localStorageMock.clear());

  it("setToken stores token in localStorage", () => {
    setToken("my-jwt-token");
    expect(getToken()).toBe("my-jwt-token");
  });

  it("removeToken clears token", () => {
    setToken("my-jwt-token");
    removeToken();
    expect(getToken()).toBeNull();
  });

  it("isAuthenticated returns true when token exists", () => {
    setToken("my-jwt-token");
    expect(isAuthenticated()).toBe(true);
  });

  it("isAuthenticated returns false when no token", () => {
    expect(isAuthenticated()).toBe(false);
  });
});
```

```typescript
// frontend/__tests__/lib/api.test.ts
import { apiFetch } from "@/lib/api";

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe("apiFetch", () => {
  beforeEach(() => {
    mockFetch.mockClear();
    localStorage.clear();
  });

  it("attaches Authorization header when token exists", async () => {
    localStorage.setItem("token", "my-token");
    mockFetch.mockResolvedValue({ ok: true, json: () => Promise.resolve({}) });

    await apiFetch("/workspaces");

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/workspaces"),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer my-token",
        }),
      })
    );
  });

  it("throws on non-ok response", async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 401, json: () => Promise.resolve({ detail: "Unauthorized" }) });

    await expect(apiFetch("/workspaces")).rejects.toThrow();
  });
});
```

- [ ] **Step 8: Run all frontend tests**

```bash
cd frontend && npm test -- --watchAll=false
# Expected: all tests pass
```

- [ ] **Step 9: Commit**

```bash
git add frontend/__tests__/ frontend/jest.config.ts frontend/jest.setup.ts frontend/package.json
git commit -m "test: frontend component and lib tests (Jest + RTL)"
```

---

## Task 8: E2E Browser Tests (Playwright)

**Files:**
- Create: `frontend/e2e/playwright.config.ts`
- Create: `frontend/e2e/auth.spec.ts`
- Create: `frontend/e2e/workspace.spec.ts`
- Create: `frontend/e2e/document.spec.ts`
- Create: `frontend/e2e/chat.spec.ts`

**Prerequisites:** Backend (FastAPI + Celery) and frontend (Next.js) must be running locally. Docker Compose services (PostgreSQL, Qdrant, Redis) must be up.

- [ ] **Step 1: Install Playwright**

```bash
cd frontend
npm install --save-dev @playwright/test
npx playwright install chromium
```

- [ ] **Step 2: Configure Playwright**

```typescript
// frontend/e2e/playwright.config.ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  timeout: 30000,
  retries: 1,
  use: {
    baseURL: "http://localhost:3000",
    headless: true,
    screenshot: "only-on-failure",
    trace: "on-first-retry",
  },
  webServer: [
    {
      command: "cd .. && cd backend && uvicorn main:app --port 8000",
      port: 8000,
      reuseExistingServer: true,
    },
    {
      command: "npm run dev",
      port: 3000,
      reuseExistingServer: true,
    },
  ],
});
```

- [ ] **Step 3: Write auth E2E tests**

```typescript
// frontend/e2e/auth.spec.ts
import { test, expect } from "@playwright/test";

test.describe("Authentication", () => {
  const testEmail = `e2e-${Date.now()}@example.com`;
  const testPassword = "e2epassword123";

  test("register new user and redirect to dashboard", async ({ page }) => {
    await page.goto("/login");
    // Switch to register mode if needed
    await page.getByText(/register/i).click();
    await page.getByLabel(/email/i).fill(testEmail);
    await page.getByLabel(/password/i).fill(testPassword);
    await page.getByRole("button", { name: /register/i }).click();

    // Should redirect to dashboard
    await expect(page).toHaveURL(/dashboard/);
  });

  test("login with existing user", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel(/email/i).fill(testEmail);
    await page.getByLabel(/password/i).fill(testPassword);
    await page.getByRole("button", { name: /login/i }).click();

    await expect(page).toHaveURL(/dashboard/);
  });

  test("login with wrong password shows error", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel(/email/i).fill(testEmail);
    await page.getByLabel(/password/i).fill("wrongpassword");
    await page.getByRole("button", { name: /login/i }).click();

    await expect(page.getByText(/invalid|error|incorrect/i)).toBeVisible();
  });

  test("unauthenticated user redirected to login", async ({ page }) => {
    await page.goto("/dashboard");
    await expect(page).toHaveURL(/login/);
  });
});
```

- [ ] **Step 4: Write workspace E2E tests**

```typescript
// frontend/e2e/workspace.spec.ts
import { test, expect } from "@playwright/test";

test.describe("Workspaces", () => {
  test.beforeEach(async ({ page }) => {
    // Register & login
    const email = `ws-e2e-${Date.now()}@example.com`;
    await page.goto("/login");
    await page.getByText(/register/i).click();
    await page.getByLabel(/email/i).fill(email);
    await page.getByLabel(/password/i).fill("password123");
    await page.getByRole("button", { name: /register/i }).click();
    await expect(page).toHaveURL(/dashboard/);
  });

  test("create workspace and see it in list", async ({ page }) => {
    await page.getByRole("button", { name: /create|new/i }).click();
    await page.getByLabel(/name/i).fill("My Test Workspace");
    await page.getByRole("button", { name: /create|save/i }).click();

    await expect(page.getByText("My Test Workspace")).toBeVisible();
  });

  test("navigate into workspace", async ({ page }) => {
    // Create workspace first
    await page.getByRole("button", { name: /create|new/i }).click();
    await page.getByLabel(/name/i).fill("Navigate WS");
    await page.getByRole("button", { name: /create|save/i }).click();

    // Click into workspace
    await page.getByText("Navigate WS").click();
    await expect(page).toHaveURL(/workspace\//);
  });

  test("delete workspace", async ({ page }) => {
    // Create workspace
    await page.getByRole("button", { name: /create|new/i }).click();
    await page.getByLabel(/name/i).fill("Delete Me WS");
    await page.getByRole("button", { name: /create|save/i }).click();

    // Delete it
    await page.getByRole("button", { name: /delete/i }).click();
    await expect(page.getByText("Delete Me WS")).not.toBeVisible();
  });
});
```

- [ ] **Step 5: Write document E2E tests**

```typescript
// frontend/e2e/document.spec.ts
import { test, expect } from "@playwright/test";
import path from "path";

test.describe("Documents", () => {
  test.beforeEach(async ({ page }) => {
    const email = `doc-e2e-${Date.now()}@example.com`;
    await page.goto("/login");
    await page.getByText(/register/i).click();
    await page.getByLabel(/email/i).fill(email);
    await page.getByLabel(/password/i).fill("password123");
    await page.getByRole("button", { name: /register/i }).click();
    await expect(page).toHaveURL(/dashboard/);

    // Create workspace and navigate into it
    await page.getByRole("button", { name: /create|new/i }).click();
    await page.getByLabel(/name/i).fill("Doc Test WS");
    await page.getByRole("button", { name: /create|save/i }).click();
    await page.getByText("Doc Test WS").click();
    await expect(page).toHaveURL(/workspace\//);
  });

  test("upload text file and see pending status", async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: "test.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("This is test content for RAG."),
    });

    await expect(page.getByText("test.txt")).toBeVisible();
    await expect(page.getByText(/pending|processing/i)).toBeVisible();
  });

  test("reject non-supported file types", async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    // The file input should have accept=".pdf,.txt"
    await expect(fileInput).toHaveAttribute("accept", ".pdf,.txt");
  });

  test("delete document", async ({ page }) => {
    // Upload first
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: "to_delete.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("Delete me."),
    });
    await expect(page.getByText("to_delete.txt")).toBeVisible();

    // Delete
    await page.getByRole("button", { name: /delete/i }).first().click();
    await expect(page.getByText("to_delete.txt")).not.toBeVisible();
  });
});
```

- [ ] **Step 6: Write chat E2E tests**

```typescript
// frontend/e2e/chat.spec.ts
import { test, expect } from "@playwright/test";

test.describe("Chat", () => {
  test.beforeEach(async ({ page }) => {
    const email = `chat-e2e-${Date.now()}@example.com`;
    await page.goto("/login");
    await page.getByText(/register/i).click();
    await page.getByLabel(/email/i).fill(email);
    await page.getByLabel(/password/i).fill("password123");
    await page.getByRole("button", { name: /register/i }).click();
    await expect(page).toHaveURL(/dashboard/);

    await page.getByRole("button", { name: /create|new/i }).click();
    await page.getByLabel(/name/i).fill("Chat Test WS");
    await page.getByRole("button", { name: /create|save/i }).click();
    await page.getByText("Chat Test WS").click();
    await expect(page).toHaveURL(/workspace\//);
  });

  test("send message and see user message displayed", async ({ page }) => {
    const input = page.getByPlaceholder(/ask a question/i);
    await input.fill("What is artificial intelligence?");
    await page.getByRole("button", { name: /send/i }).click();

    // User message should appear in chat
    await expect(page.getByText("What is artificial intelligence?")).toBeVisible();
  });

  test("input is disabled while streaming response", async ({ page }) => {
    const input = page.getByPlaceholder(/ask a question/i);
    await input.fill("Tell me about RAG");
    await page.getByRole("button", { name: /send/i }).click();

    // Input should be disabled briefly while streaming
    await expect(input).toBeDisabled({ timeout: 1000 }).catch(() => {
      // If streaming is too fast, this may not catch it — acceptable
    });
  });

  test("assistant response renders without raw markdown", async ({ page }) => {
    const input = page.getByPlaceholder(/ask a question/i);
    await input.fill("What is AI?");
    await page.getByRole("button", { name: /send/i }).click();

    // Wait for assistant response
    await page.waitForSelector('[data-role="assistant"]', { timeout: 30000 });

    // Check no raw markdown symbols visible
    const assistantMessages = page.locator('[data-role="assistant"]');
    const text = await assistantMessages.first().textContent();
    expect(text).not.toContain("**");
    expect(text).not.toContain("###");
  });

  test("empty workspace shows info message", async ({ page }) => {
    const input = page.getByPlaceholder(/ask a question/i);
    await input.fill("Hello");
    await page.getByRole("button", { name: /send/i }).click();

    // Should indicate no documents available
    await expect(
      page.getByText(/no documents|upload.*document|belum ada dokumen/i)
    ).toBeVisible({ timeout: 30000 });
  });
});
```

- [ ] **Step 7: Run Playwright E2E tests**

```bash
cd frontend && npx playwright test --config=e2e/playwright.config.ts
# Expected: all tests pass (requires running services)
```

- [ ] **Step 8: Commit**

```bash
git add frontend/e2e/ frontend/package.json
git commit -m "test: E2E browser tests with Playwright (auth, workspace, document, chat)"
```

---

## Task 9: Manual End-to-End Test Script

This is a **human-executed** test script. No automation — the tester follows these steps manually to verify the full system works end-to-end including Celery processing, real API calls, and RAG quality.

**Prerequisite checklist before starting:**

- [ ] Docker Compose running (`docker compose up -d`)
- [ ] PostgreSQL, Qdrant, Redis all healthy (`docker compose ps`)
- [ ] `.env` has valid `OPENROUTER_API_KEY` and `OCTEN_API_KEY`
- [ ] Backend running (`cd backend && uvicorn main:app --reload --port 8000`)
- [ ] Celery worker running (`cd backend && celery -A core.celery_app worker --loglevel=info`)
- [ ] Frontend running (`cd frontend && npm run dev`)
- [ ] Test database created (`docker compose exec postgres psql -U omnibrain -c "CREATE DATABASE omnibrain_test;"`)

---

### Scenario 1: User Registration & Login

| # | Action | Expected Result | Pass? |
|---|--------|----------------|-------|
| 1.1 | Open `http://localhost:3000` | Redirects to login page | [ ] |
| 1.2 | Click "Register" toggle | Registration form shown | [ ] |
| 1.3 | Enter email: `test@omnibrain.local`, password: `testpass123` | Fields filled | [ ] |
| 1.4 | Click Register | Redirects to dashboard | [ ] |
| 1.5 | Logout (if available) or clear localStorage | Returned to login | [ ] |
| 1.6 | Login with same credentials | Redirects to dashboard | [ ] |
| 1.7 | Login with wrong password | Error message shown, stays on login | [ ] |

---

### Scenario 2: Workspace Management

| # | Action | Expected Result | Pass? |
|---|--------|----------------|-------|
| 2.1 | On dashboard, click "Create Workspace" | Create form/modal appears | [ ] |
| 2.2 | Enter name: "AI Research", description: "Testing RAG" | Fields filled | [ ] |
| 2.3 | Submit | Workspace card appears in list | [ ] |
| 2.4 | Create second workspace: "Meeting Notes" | Two workspace cards visible | [ ] |
| 2.5 | Click "AI Research" workspace | Navigates to workspace view with doc panel + chat panel | [ ] |
| 2.6 | Go back, delete "Meeting Notes" | Workspace removed from list | [ ] |

---

### Scenario 3: Document Upload & Processing

**Prepare test files:**
- `test_ai.txt` — 500+ words about artificial intelligence (copy from `tests/fixtures/sample.txt`)
- `test_rag.pdf` — any 2-page PDF about RAG or related topic

| # | Action | Expected Result | Pass? |
|---|--------|----------------|-------|
| 3.1 | In "AI Research" workspace, click upload | File picker opens | [ ] |
| 3.2 | Upload `test_ai.txt` | File appears in list with status "pending" | [ ] |
| 3.3 | Watch Celery worker terminal | Logs show: parsing → chunking → embedding → storing to Qdrant | [ ] |
| 3.4 | Refresh/poll document status | Status changes to "processing" then "completed" | [ ] |
| 3.5 | Check chunk_count | Should be > 0 (verify via `GET /documents/{id}`) | [ ] |
| 3.6 | Upload `test_rag.pdf` | PDF file appears, processes to "completed" | [ ] |
| 3.7 | Try uploading a `.jpg` file | Rejected with error message | [ ] |
| 3.8 | Re-upload `test_ai.txt` (same filename) | Old document replaced, new processing starts | [ ] |
| 3.9 | Verify only 2 documents in list (not 3) | Idempotency works correctly | [ ] |

---

### Scenario 4: RAG Chat — Basic Functionality

| # | Action | Expected Result | Pass? |
|---|--------|----------------|-------|
| 4.1 | In chat panel, type: "What is artificial intelligence?" | Message appears in chat | [ ] |
| 4.2 | Click Send | User message displayed, input disabled during streaming | [ ] |
| 4.3 | Observe response | Text streams in real-time (character by character animation) | [ ] |
| 4.4 | Check response content | Answer references information from uploaded documents | [ ] |
| 4.5 | Check for raw markdown | No `**`, `###`, `` ``` `` visible — properly formatted HTML | [ ] |
| 4.6 | Check source citations | Sources section shows which documents were used, with similarity scores | [ ] |

---

### Scenario 5: RAG Chat — Quality & Edge Cases

| # | Action | Expected Result | Pass? |
|---|--------|----------------|-------|
| 5.1 | Ask: "What color is the sky?" (not in documents) | Response indicates info not found in documents, does NOT hallucinate | [ ] |
| 5.2 | Ask: "Summarize the key points from my documents" | Provides accurate summary referencing actual document content | [ ] |
| 5.3 | Ask a follow-up: "Can you elaborate on the first point?" | Uses chat history context to understand "first point" | [ ] |
| 5.4 | Check source citations have reasonable scores | Scores should be > 0.7 for relevant answers | [ ] |
| 5.5 | Ask in Indonesian: "Apa itu kecerdasan buatan?" | Responds appropriately (documents are in English but model should handle) | [ ] |

---

### Scenario 6: Chat Session Management

| # | Action | Expected Result | Pass? |
|---|--------|----------------|-------|
| 6.1 | After chatting, check sessions list | At least 1 session visible | [ ] |
| 6.2 | Click on session | Shows full message history | [ ] |
| 6.3 | Start a new chat (if UI supports) | New session created, clean history | [ ] |
| 6.4 | Delete a session | Session removed from list | [ ] |

---

### Scenario 7: Memory & Summarization

| # | Action | Expected Result | Pass? |
|---|--------|----------------|-------|
| 7.1 | In a workspace with documents, start a new chat session | Fresh session, no summary | [ ] |
| 7.2 | Send 15-20 messages back and forth (long conversation) | All responses work normally | [ ] |
| 7.3 | Check Celery/backend logs for summarization trigger | Log shows summarization called after token threshold exceeded | [ ] |
| 7.4 | After summarization, send another message | Response still has context from earlier messages (via summary) | [ ] |
| 7.5 | Check `chat_sessions.context_summary` via API or DB query | Summary field is populated with a non-null value | [ ] |
| 7.6 | Ask "What did we discuss at the start of this conversation?" | AI can recall earlier topics via summary, not just last 10 messages | [ ] |

---

### Scenario 8: Error Handling

| # | Action | Expected Result | Pass? |
|---|--------|----------------|-------|
| 8.1 | Stop Celery worker, upload document | Document stays "pending" — UI shows pending status, no crash | [ ] |
| 8.2 | Start Celery worker again | Document processes normally | [ ] |
| 8.3 | Set invalid OPENROUTER_API_KEY, send chat message | Error message shown via SSE, backend doesn't crash | [ ] |
| 8.4 | Restore valid API key, retry | Chat works normally | [ ] |
| 8.5 | Chat in a workspace with no documents | Appropriate "no documents" message | [ ] |
| 8.6 | Try sending empty message via API (curl) | 422 validation error returned | [ ] |
| 8.7 | Try uploading file > 20MB via API (curl) | 422 error with size limit message | [ ] |

---

### Scenario 9: Security Checks

| # | Action | Expected Result | Pass? |
|---|--------|----------------|-------|
| 9.1 | Call `GET /workspaces` without Authorization header | 403 Forbidden | [ ] |
| 9.2 | Call with `Authorization: Bearer invalidtoken` | 401 Unauthorized | [ ] |
| 9.3 | As user A, try to access user B's workspace via API | 404 Not Found (not 403 — don't leak existence) | [ ] |
| 9.4 | As user A, try to delete user B's document via API | 404 Not Found | [ ] |
| 9.5 | Check `.env` file is in `.gitignore` | Confirmed — secrets not in git | [ ] |
| 9.6 | Check API responses don't contain `hashed_password` | Confirmed — no password leaks | [ ] |

---

### Scenario 10: API Smoke Test via Swagger

| # | Action | Expected Result | Pass? |
|---|--------|----------------|-------|
| 10.1 | Open `http://localhost:8000/docs` | Swagger UI loads | [ ] |
| 10.2 | `GET /health` | Returns `{"status": "ok", "version": "0.1.0"}` | [ ] |
| 10.3 | `POST /auth/register` with valid body | 201, returns user id and email | [ ] |
| 10.4 | `POST /auth/login` | 200, returns access_token | [ ] |
| 10.5 | Click "Authorize" in Swagger, paste Bearer token | Token set | [ ] |
| 10.6 | `POST /workspaces` | 201, workspace created | [ ] |
| 10.7 | `POST /workspaces/{id}/documents` (upload .txt) | 201, document created with pending status | [ ] |
| 10.8 | `GET /documents/{id}` (poll until completed) | Status eventually "completed" | [ ] |
| 10.9 | `POST /workspaces/{id}/chat` | 200, SSE stream with response | [ ] |
| 10.10 | All CRUD endpoints return expected status codes | Verified per spec section 8 | [ ] |

---

## Summary

| Task | Type | Scope | Test Count (approx) |
|------|------|-------|-------------------|
| 1 | Setup | Test infrastructure, fixtures, mocks | — |
| 2 | Unit | Security, chunking, memory, prompts, config | ~30 |
| 3 | Integration | Auth API endpoints | ~12 |
| 4 | Integration | Workspace API endpoints | ~14 |
| 5 | Integration | Document API + pipeline | ~15 |
| 6 | Integration | Chat API, RAG pipeline, Qdrant | ~12 |
| 7 | Component | Frontend components + lib (Jest + RTL) | ~25 |
| 8 | E2E Browser | Playwright: auth, workspace, document, chat | ~15 |
| 9 | Manual E2E | Full system verification (10 scenarios, 60+ checks) | 60+ |
| | | **Total** | **~180+** |

**Coverage targets:**
- Backend unit + integration: **80%+ line coverage**
- Frontend components: **80%+ branch coverage**
- E2E: All critical user flows covered
- Manual: Full system verification including error handling, security, and RAG quality

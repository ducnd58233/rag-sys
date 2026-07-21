import pytest
from scripts.evaluation.client import EvalHttpClient

_ASK_PAYLOAD = {
    "query": "what is self-attention",
    "answer": "Self-attention relates positions of a single sequence.",
    "citations": [
        {
            "chunk_id": "chunk-1",
            "document_id": "42",
            "content": "self-attention content",
            "score": 0.9,
        },
    ],
    "refused": False,
}

_PAYLOAD = {
    "query": "what is self-attention",
    "plan": {
        "router_kind": "auto",
        "reason": "hybrid query",
        "confidence": 0.9,
        "strategies": [
            {"name": "hybrid", "weight": 0.7, "top_k": 10},
            {"name": "graph", "weight": 0.3, "top_k": 5},
        ],
    },
    "items": [
        {
            "chunk_id": "chunk-1",
            "document_id": "42",
            "content": "...",
            "score": 0.88,
            "metadata": {"filename": "1706.03762-attention-is-all-you-need.pdf"},
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "43",
            "content": "...",
            "score": 0.5,
            "metadata": {},
        },
    ],
}


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.requested_url: str | None = None
        self.requested_json: dict | None = None

    def post(self, url: str, json: dict) -> _FakeResponse:
        self.requested_url = url
        self.requested_json = json
        return _FakeResponse(self._payload)


@pytest.mark.asyncio
async def test_retrieve_parses_items_router_kind_and_strategies() -> None:
    client = EvalHttpClient("http://localhost:8000")
    fake_session = _FakeSession(_PAYLOAD)
    client._session = fake_session  # type: ignore[assignment]

    response = await client.retrieve(
        org_id=999000, query="what is self-attention", top_k=10
    )

    assert response.router_kind == "auto"
    assert response.router_reason == "hybrid query"
    assert response.strategies == ("hybrid", "graph")
    assert len(response.items) == 2
    assert response.items[0].filename == "1706.03762-attention-is-all-you-need.pdf"
    assert response.items[1].filename is None
    assert fake_session.requested_url == "http://localhost:8000/api/v1/retrieval/search"
    assert fake_session.requested_json == {
        "org_id": 999000,
        "query": "what is self-attention",
        "top_k": 10,
    }


@pytest.mark.asyncio
async def test_ask_parses_answer_citations_and_refused() -> None:
    client = EvalHttpClient("http://localhost:8000")
    fake_session = _FakeSession(_ASK_PAYLOAD)
    client._session = fake_session  # type: ignore[assignment]

    response = await client.ask(org_id=999000, query="what is self-attention", top_k=10)

    assert response.answer == "Self-attention relates positions of a single sequence."
    assert response.refused is False
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "chunk-1"
    assert response.citations[0].document_id == "42"
    assert fake_session.requested_url == "http://localhost:8000/api/v1/generation/ask"
    assert fake_session.requested_json == {
        "org_id": 999000,
        "query": "what is self-attention",
        "top_k": 10,
    }


@pytest.mark.asyncio
async def test_ask_parses_refused_response_with_no_citations() -> None:
    client = EvalHttpClient("http://localhost:8000")
    refused_payload = {
        "query": "unrelated question",
        "answer": "I could not find relevant information to answer this question.",
        "citations": [],
        "refused": True,
    }
    client._session = _FakeSession(refused_payload)  # type: ignore[assignment]

    response = await client.ask(org_id=999000, query="unrelated question", top_k=10)

    assert response.refused is True
    assert response.citations == ()

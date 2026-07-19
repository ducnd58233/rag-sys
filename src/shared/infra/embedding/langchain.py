import time
from collections.abc import Sequence

from langchain.embeddings import Embeddings
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.shared.observability.metrics import gen_ai_client_operation_duration

_tracer = trace.get_tracer(__name__)


class LangChainEmbeddingModel:
    def __init__(
        self,
        client: Embeddings,
        *,
        dimensions: int,
        model_name: str,
        provider_name: str,
    ) -> None:
        self._client = client
        self._dimensions = dimensions
        self._model_name = model_name
        self._provider_name = provider_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        metric_attributes = {
            "gen_ai.operation.name": "embeddings",
            "gen_ai.provider.name": self._provider_name,
            "gen_ai.request.model": self._model_name,
        }
        started_at = time.perf_counter()
        with _tracer.start_as_current_span(
            f"embeddings {self._model_name}",
            attributes={
                **metric_attributes,
                "document_count": len(texts),
            },
        ) as span:
            try:
                vectors = await self._client.aembed_documents(list(texts))
            except Exception as error:
                error_type = error.__class__.__name__
                span.set_status(Status(StatusCode.ERROR, error_type))
                gen_ai_client_operation_duration.record(
                    time.perf_counter() - started_at,
                    {**metric_attributes, "error.type": error_type},
                )
                raise
            gen_ai_client_operation_duration.record(
                time.perf_counter() - started_at,
                metric_attributes,
            )
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise RuntimeError(
                    f"Expected dim={self._dimensions}, got dim={len(vector)}"
                )
        return vectors

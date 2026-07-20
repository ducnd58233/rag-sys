from __future__ import annotations

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from src.shared.observability.tracing import traced, traced_span


def _tracer_with_exporter() -> tuple[object, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def _histogram_with_reader() -> tuple[object, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    histogram = provider.get_meter("test").create_histogram("test.duration", unit="s")
    return histogram, reader


def _recorded_data_points(reader: InMemoryMetricReader) -> list[object]:
    data = reader.get_metrics_data()
    points: list[object] = []
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                points.extend(metric.data.data_points)
    return points


@pytest.mark.asyncio
async def test_traced_records_success_span_and_metric() -> None:
    tracer, exporter = _tracer_with_exporter()
    histogram, reader = _histogram_with_reader()

    async with traced(
        tracer,
        "unit.test.span",
        attributes={"rag.test.key": "value"},
        duration_metric=histogram,
        metric_attributes={"strategy": "lexical"},
    ) as span:
        span.set_attribute("rag.test.result_count", 3)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "unit.test.span"
    assert spans[0].attributes["rag.test.key"] == "value"
    assert spans[0].attributes["rag.test.result_count"] == 3
    assert spans[0].status.status_code == StatusCode.UNSET

    points = _recorded_data_points(reader)
    assert len(points) == 1
    assert points[0].sum >= 0
    assert dict(points[0].attributes) == {"strategy": "lexical", "outcome": "success"}


@pytest.mark.asyncio
async def test_traced_records_failure_span_and_metric_then_reraises() -> None:
    tracer, exporter = _tracer_with_exporter()
    histogram, reader = _histogram_with_reader()

    with pytest.raises(ValueError, match="boom"):
        async with traced(
            tracer,
            "unit.test.span",
            duration_metric=histogram,
            metric_attributes={"strategy": "lexical"},
        ):
            raise ValueError("boom")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].status.description == "ValueError"

    points = _recorded_data_points(reader)
    assert len(points) == 1
    assert dict(points[0].attributes) == {"strategy": "lexical", "outcome": "failure"}


@pytest.mark.asyncio
async def test_traced_without_duration_metric_does_not_record_anything() -> None:
    tracer, exporter = _tracer_with_exporter()

    async with traced(tracer, "unit.test.span"):
        pass

    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_traced_span_decorator_wraps_a_function_and_preserves_metadata() -> None:
    tracer, exporter = _tracer_with_exporter()
    histogram, reader = _histogram_with_reader()

    @traced_span(
        tracer,
        "unit.test.decorated",
        duration_metric=histogram,
        metric_attributes={"op": "example"},
    )
    async def do_work(x: int, *, y: int) -> int:
        """Docstring should survive wrapping."""
        return x + y

    assert do_work.__name__ == "do_work"
    assert do_work.__doc__ == "Docstring should survive wrapping."

    result = await do_work(1, y=2)

    assert result == 3
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "unit.test.decorated"
    points = _recorded_data_points(reader)
    assert dict(points[0].attributes) == {"op": "example", "outcome": "success"}


@pytest.mark.asyncio
async def test_traced_span_decorator_propagates_exceptions() -> None:
    tracer, exporter = _tracer_with_exporter()

    @traced_span(tracer, "unit.test.decorated")
    async def do_work() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await do_work()

    spans = exporter.get_finished_spans()
    assert spans[0].status.status_code == StatusCode.ERROR

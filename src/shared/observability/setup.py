from __future__ import annotations

import os
from dataclasses import dataclass

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.elasticsearch import ElasticsearchInstrumentor
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.shared.configs.settings import ObservabilitySettings

os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "http")


@dataclass(frozen=True, slots=True)
class Observability:
    tracer_provider: TracerProvider | None
    meter_provider: MeterProvider | None

    async def shutdown(self) -> None:
        if self.tracer_provider is not None:
            self.tracer_provider.shutdown()
        if self.meter_provider is not None:
            self.meter_provider.shutdown()


def setup_observability(
    settings: ObservabilitySettings,
    *,
    service_name: str,
) -> Observability:
    if settings.sdk_disabled:
        return Observability(tracer_provider=None, meter_provider=None)

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": settings.service_version,
            "deployment.environment": settings.environment,
        },
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.exporter_otlp_endpoint)),
    )
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=settings.exporter_otlp_endpoint),
                export_interval_millis=settings.metric_export_interval_ms,
            ),
        ],
        views=_histogram_views(),
    )
    metrics.set_meter_provider(meter_provider)

    ElasticsearchInstrumentor().instrument()
    SystemMetricsInstrumentor().instrument()

    return Observability(tracer_provider=tracer_provider, meter_provider=meter_provider)


def _histogram_views() -> list[View]:
    return [
        View(
            instrument_name="http.server.request.duration",
            aggregation=ExplicitBucketHistogramAggregation(
                (
                    0.005,
                    0.01,
                    0.025,
                    0.05,
                    0.075,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    2.5,
                    5.0,
                    7.5,
                    10.0,
                ),
            ),
        ),
        View(
            instrument_name="rag.ingestion.step.duration",
            aggregation=ExplicitBucketHistogramAggregation(
                (
                    0.5,
                    1.0,
                    2.5,
                    5.0,
                    10.0,
                    30.0,
                    60.0,
                    120.0,
                    300.0,
                    600.0,
                    900.0,
                    1500.0,
                    1800.0,
                ),
            ),
        ),
        View(
            instrument_name="rag.messaging.publish.duration",
            aggregation=ExplicitBucketHistogramAggregation(
                (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
            ),
        ),
    ]

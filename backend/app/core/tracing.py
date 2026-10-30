from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

resource = Resource(attributes={
    SERVICE_NAME: "docuflow-ai-backend"
})

provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("docuflow.tracer")

def get_current_trace_id() -> str:
    """Helper to get current active 32-character hex tracer ID."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        return format(ctx.trace_id, '032x')
    return "00000000000000000000000000000000"
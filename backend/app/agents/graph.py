from typing import Literal
from langgraph.graph import StateGraph, START, END

from app.agents.state import ExtractionGraphState
from app.agents.nodes.retriever_node import retrieve_context_node
from app.agents.nodes.extractor_node import extract_data_node
from app.agents.nodes.validator_node import validate_extraction_node

MAX_RETRIES = 3

def route_after_validations(state: ExtractionGraphState) -> Literal["extract", "end"]:
    """
    Decides the next node after validation:
    - If clean (no validation errors): finish successfully ("end").
    - If errors exist and retries remain (< 3): loop back to "extract".
    - If retry limit reached (>= 3): terminate to avoid infinite token burning ("end").
    """
    errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", 0)

    if not errors:
        return "end"
    
    if retry_count < MAX_RETRIES:
        return "extract"

    return "end"

# Instantiate the Graph Builder
builder = StateGraph(ExtractionGraphState)

# Register Nodes
builder.add_node("retrieve", retrieve_context_node)
builder.add_node("extract", extract_data_node)
builder.add_node("validate", validate_extraction_node)

# Add Linear Edges
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "extract")
builder.add_edge("extract", "validate")

# Add Conditional Edge for 'validate'
builder.add_conditional_edges("validate", route_after_validations,{
    "extract": "extract",
    "end": END
})

# Compile runnable graph
extraction_graph = builder.compile()
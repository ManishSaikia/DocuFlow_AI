import json
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.agents.state import ExtractionGraphState
from app.core.config import settings
from app.schemas.extraction import DocumentExtractionResult

import time
from app.services.cost_tracker import calculate_call_cost

llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model_name=getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0.0
)

structured_extractor = llm.with_structured_output(DocumentExtractionResult, include_raw=True)

SYSTEM_PROMPT = """You are an expert Document Extraction Agent.
Your job is to extract structured data from the provided document chunks according to the required schema.

Strict Rules:
1. Base all extracted values (names, dates, line items, monetary amounts) STRICTLY on evidence in the provided chunks.
2. In the "citations" list, record the exact integer "Chunk Index" values you used as evidence.
3. If a field is not present in the chunks, leave it as null/empty. Do NOT invent data.
4. Provide a self-evaluated confidence score between 0.0 and 1.0."""

async def extract_data_node(state: ExtractionGraphState) -> Dict[str, Any]:
    """
    Extractor Agent Node:
    - Pass 1 (retry_count == 0): Runs standard extraction from chunks.
    - Pass 2 (retry_count > 0): Injects previous validation errors and previous extracted data so the LLM self-corrects its mistakes.
    """
    
    start_time = time.perf_counter()

    retrieved_chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)
    validation_errors = state.get("validation_errors", [])
    previous_data = state.get("extracted_data")

    context_text = "\n\n---\n\n".join(chunk.get("formatted_text") or chunk.get("content", "") for chunk in retrieved_chunks)

    if retry_count == 0 or not validation_errors:
        user_prompt = f"""
        DOCUMENT CHUNKS:
        {context_text}

        USER REQUEST:
        {state.get("query", "Extract document metadata, financial figures, line items, and citations.")}

        INSTRUCTIONS:
        1. Extract the structured fields strictly based on the provided chunks.
        2. CRITICAL CONSTRAINT: CRITICAL CONSTRAINT: You must strictly align all extracted numbers, totals, and line items with the specific constraints, time periods, quarters, or criteria requested in the USER REQUEST.- For example: if the user asks for "Q1 2024", populate total_amount and line_items specifically with the Q1 2024 figures (convert abbreviations like $1.2M to 1200000.0), NOT other years or general document headlines.
        """
    else:
        errors_formatted = "\n".join(f"- {err}" for err in validation_errors)
        previous_json = json.dumps(previous_data, indent=2) if previous_data else "{}"

        user_prompt = f"""
        CRITIC FEEDBACK (PREVIOUS ATTEMPT FAILED):

        Your previous extraction attempt failed validation with the following errors: {errors_formatted}

        PREVIOUS EXTRACTED DATA: {previous_json}

        SOURCE DOCUMENT CHUNKS: {context_text}

        INSTRUCTIONS:
        Carefully review the errors above. Fix ONLY the failed fields using the source chunks. Ensure all citations and numerical calculations match the document text exactly.
        """
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    try:
        call_result = await structured_extractor.ainvoke(messages)
        extracted_result: DocumentExtractionResult = call_result["parsed"]
        raw_message = call_result["raw"]

        # Read Tokens from usage_metadata
        usage = getattr(raw_message, "usage_metadata", {}) or {}
        call_prompt_tokens = usage.get("input_tokens", 0)
        call_completion_tokens = usage.get("output_tokens", 0)

        # Calculate call cost
        model_name = getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
        call_cost = calculate_call_cost(model_name, call_prompt_tokens, call_completion_tokens)

        # Accumulate cumulative metrics
        node_durations_ms = round((time.perf_counter() - start_time) * 1000, 2)
        latencies = dict(state.get("node_latencies_ms") or {})
        latencies[f"extractor_attempt_{retry_count}"] = node_durations_ms

        return {
            "extracted_data": extracted_result.model_dump(),
            "prompt_tokens": state.get("prompt_tokens", 0) + call_prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + call_completion_tokens,
            "total_cost_usd": round(state.get("total_cost_usd", 0.0) + call_cost, 6),
            "node_latencies_ms": latencies,
            }
    except Exception as e:
        return {
            "extracted_data": previous_data,
            "validation_errors": validation_errors + [f"LLM Schema Generation Error: {str(e)}"]
            }    
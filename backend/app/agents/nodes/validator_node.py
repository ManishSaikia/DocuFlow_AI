from typing import Any, Dict, List
from app.agents.state import ExtractionGraphState

async def validate_extraction_node(state: ExtractionGraphState) -> Dict[str, Any]:
    extracted = state.get("extracted_data")
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)
    errors: List[str] = []

    if not extracted:
        return {
            "validation_errors": ["No extraction data produced by extractor node."],
            "retry_count": retry_count + 1,
            "status": "IN_PROGRESS"
        }
    
    valid_indices = {c.get("chunk_index") for c in chunks if "chunk_index" in c}
    citations = extracted.get("citations", [])

    if not citations:
        errors.append("No citations provided: you must cite at least one Chunk Index as evidence.")
    else:
        for idx in citations:
            if idx not in valid_indices:
                errors.append(f"Hallucinated Citation: Chunk index {idx} does not exists in retrieved chunks {sorted(list(valid_indices))}")
    
    line_items = extracted.get("line_items") or []
    subtotal = extracted.get("subtotal")
    tax_amount = extracted.get("tax_amount")
    total_amount = extracted.get("total_amount")

    # Check line item Math ( qty * unit_price == total_price )
    calculated_items_sum = 0.0
    for item in line_items:
        qty = item.get("quantity") or 0.0
        unit_price = item.get("unit_price") or 0.0
        item_total = item.get("total_price") or 0.0
        calculated_items_sum += item_total
        
        if abs(qty * unit_price - item_total) > 0.05:
            errors.append(f"Line item math error in '{item.get('description')}': {qty} * {unit_price} != {item_total}")
        
    if line_items and subtotal is not None:
        if abs(calculated_items_sum - subtotal) > 0.05:
            errors.append(f"Subtotal mismatch: sum of line items ({calculated_items_sum:.2f}) does not match extracted subtotal ({subtotal:.2f})")
        
    if line_items and subtotal is None and total_amount is not None:
        if abs(calculated_items_sum - total_amount) > 0.05:
            errors.append(f"Total amount mismatch: sum of line items ({calculated_items_sum:.2f}) does not match total amount ({total_amount:.2f})")
        
    if subtotal is not None and tax_amount is not None and total_amount is not None:
        if abs((subtotal + tax_amount) - total_amount) > 0.05:
            errors.append(f"Total amount mismatch: subtotal ({subtotal:.2f}) + tax ({tax_amount:.2f}) != total_amount ({total_amount:.2f})")
    
    FORBIDDEN_PLACEHOLDERS = {"n/a", "unknown", "none", "null", "not specified", "unspecified"}
    for field in ["document_title", "vendor_or_sender"]:
        val = extracted.get(field)
        if isinstance(val, str) and val.strip().lower() in FORBIDDEN_PLACEHOLDERS:
            errors.append(f"Invalid placeholder value '{val}' for field '{field}'. Leave null if absent.")
    
    cited_chunks_text = " ".join(c.get("content", "") for c in chunks if c.get("chunk_index") in citations).lower()

    vendor = extracted.get("vendor_or_sender")
    if vendor and len(vendor) > 3 and vendor.lower() not in cited_chunks_text:
        errors.append(f"Grounding failure: vendor '{vendor}' was not in the text of cited chunks {citations}")

    if not errors:
        return {
            "validation_errors": [],
            "status": "VALIDATED"
        }
    else:
        return {
            "validation_errors": errors,
            "retry_count": retry_count + 1,
            "status": "IN_PROGRESS"
        }
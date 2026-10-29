from typing import List, Optional
from pydantic import BaseModel, Field
import uuid

class LineItem(BaseModel):
    description: str = Field(..., description="Description of the item or service")
    quantity: float = Field(..., description="Quantity purchased or billed")
    unit_price: float = Field(..., description="Unit price per item")
    total_price: float = Field(..., description="Total price for this line item (qty * unit_price)")

class DocumentExtractionResult(BaseModel):
    document_title: str = Field(..., description="Title or header of the document")
    document_type: str = Field(..., description="Detected document type (e.g. INVOICE, RECEIPT, CONTRACT, REPORT)")
    document_date: Optional[str] = Field(None, description="Date on the document in YYYY-MM-DD or raw format")
    vendor_or_sender: Optional[str] = Field(None, description="Issuing entity, vendor, or sender")
    recipient: Optional[str] = Field(None, description="Recipient, customer or counterparty")

    # Financial breakdown
    line_items: List[LineItem] = Field(default_factory=list, description="Extracted line items")
    subtotal: Optional[float] = Field(None, description="Pre-tax amount")
    tax_amount: Optional[float] = Field(None, description="Tax or VAT amount")
    total_amount: float = Field(..., description="Grand total payable")

    # Anti-Hallucination & Provenance Tracking
    citations: List[int] = Field(..., description="List of chunk_index values used as direct evidence for this extraction")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model self-evaluated confidence score between 0.0 and 1.0")

class ExtractionRequest(BaseModel):
    job_id: uuid.UUID = Field(..., description="Ingestion Job ID to extract from")
    query: Optional[str] = Field(default="Extract document metadata, financial figures, line items, and citations.", description="Extraction instructions or focus query")

class ExtractionResponse(BaseModel):
    job_id: str
    status: str
    retry_count: int
    extracted_data: Optional[DocumentExtractionResult] = None
    citations: List[int] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
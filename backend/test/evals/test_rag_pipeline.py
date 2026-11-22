import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.db.session import AsyncSessionLocal
from app.services.embedder import embedding_service
from app.services.retriever import hybrid_search
from app.services.reranker import reranker_service
from app.agents.graph import extraction_graph

GOLDEN_DATASET_FILE = Path(__file__).parent / "golden_dataset.json"

# ---------------------------------------------------------------------------
# RAG Triad Metric Calculators
# ---------------------------------------------------------------------------

def calculate_context_recall(retrieved_indices: List[int], ground_truth_indices: List[int]) -> float:
    """Measures what fraction of the required ground truth chunks were retrieved."""
    if not ground_truth_indices:
        # Adversarial query with no required chunks
        return 1.0
    hits = set(retrieved_indices).intersection(set(ground_truth_indices))
    return len(hits) / len(ground_truth_indices)

def calculate_context_precision(retrieved_indices: List[int], ground_truth_indices: List[int]) -> float:
    """
    Measures rank-weighted precision (MRR):
    Did relevant chunks appear at the very top of the list?
    """
    if not ground_truth_indices:
        return 1.0
    
    first_hit_rank = None
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in ground_truth_indices:
            first_hit_rank = rank
            break
            
    if first_hit_rank is None:
        return 0.0
    return 1.0 / first_hit_rank

def calculate_faithfulness(extracted_data: Dict[str, Any], retrieved_text: str, is_adversarial: bool) -> float:
    """
    Checks if extracted numeric and string values actually appear in the source chunks.
    For adversarial queries, verifies that ungrounded fields remained null.
    """
    if not extracted_data:
        return 1.0 if is_adversarial else 0.0

    retrieved_text_lower = retrieved_text.lower()

    if is_adversarial:
        # Check that the model did NOT invent data for non-existent fields
        for key in ["vat_number", "iban", "swift", "penalty_clause", "tokyo_marketing", "shipping_address"]:
            if key in extracted_data and extracted_data[key] is not None:
                return 0.0  # Hallucinated a negative probe!
        return 1.0

    # For standard extraction: check numerical & core string grounding
    checked_items = 0
    grounded_items = 0

    total_amount = extracted_data.get("total_amount")
    if total_amount is not None:
        checked_items += 1
        # Search for integer or float format in text (e.g. 945 or 945.00)
        num_str = str(int(total_amount)) if float(total_amount).is_integer() else str(total_amount)
        if num_str in retrieved_text_lower:
            grounded_items += 1

    doc_date = extracted_data.get("document_date")
    if doc_date and isinstance(doc_date, str):
        checked_items += 1
        # Check year or month presence
        if "2021" in doc_date and "2021" in retrieved_text_lower:
            grounded_items += 1

    if checked_items == 0:
        return 1.0
    return grounded_items / checked_items

def calculate_citation_accuracy(citations: List[int], retrieved_indices: List[int]) -> float:
    """Checks that all cited chunk indices exist within the retrieved candidates."""
    if not citations:
        return 1.0
    valid_citations = [c for c in citations if c in retrieved_indices]
    return len(valid_citations) / len(citations)

# ---------------------------------------------------------------------------
# Single Test Case Evaluator
# ---------------------------------------------------------------------------

async def evaluate_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    query = test_case["input_query"]
    job_uuid = uuid.UUID(test_case["job_id"])
    ground_truth = test_case["ground_truth_chunks"]
    is_adversarial = test_case.get("is_adversarial", False)

    # 1. Evaluate Retriever (Hybrid Search + Cross-Encoder)
    query_vec = embedding_service.embed_query(query)
    async with AsyncSessionLocal() as session:
        candidates = await hybrid_search(
            session=session,
            query_text=query,
            query_embedding=query_vec,
            job_id=job_uuid,
            top_k=20,
            limit=20,
        )

    reranked = reranker_service.rerank(query=query, candidates=candidates, top_n=5)
    retrieved_indices = [c.get("chunk_index", 0) for c in reranked]
    retrieved_text = " ".join(c.get("content", "") for c in reranked)

    # Compute Retriever Metrics
    recall = calculate_context_recall(retrieved_indices, ground_truth)
    precision = calculate_context_precision(retrieved_indices, ground_truth)

    # 2. Evaluate Extractor Agent (LangGraph Workflow)
    state = {
        "job_id": str(job_uuid),
        "query": query,
        "retrieved_chunks": [],
        "extracted_data": None,
        "validation_errors": [],
        "validation_history": [],
        "retry_count": 0,
        "status": "IN_PROGRESS",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost_usd": 0.0,
        "node_latencies_ms": {},
    }

    final_state = await extraction_graph.ainvoke(state)
    extracted = final_state.get("extracted_data") or {}
    citations = extracted.get("citations", []) if extracted else []

    # Compute Agent Metrics
    faithfulness = calculate_faithfulness(extracted, retrieved_text, is_adversarial)
    citation_acc = calculate_citation_accuracy(citations, retrieved_indices)

    passed = (recall >= 0.5) and (faithfulness >= 0.8)

    return {
        "test_id": test_case["test_id"],
        "archetype": test_case["archetype"],
        "query": query[:40] + "..." if len(query) > 40 else query,
        "recall": recall,
        "precision": precision,
        "faithfulness": faithfulness,
        "citation_acc": citation_acc,
        "passed": passed,
    }

# ---------------------------------------------------------------------------
# Test Runner & Benchmark Aggregator
# ---------------------------------------------------------------------------

async def run_evaluation_suite(limit: int = 10) -> Dict[str, Any]:
    """
    Executes the golden suite and prints a tabular benchmark report.
    `limit=10` runs a quick subset by default; set to 25 to run the full suite.
    """
    if not GOLDEN_DATASET_FILE.exists():
        raise FileNotFoundError(f"Golden dataset not found at {GOLDEN_DATASET_FILE}. Run dataset.py first!")

    with open(GOLDEN_DATASET_FILE, "r", encoding="utf-8") as f:
        suite = json.load(f)

    test_slice = suite[:limit]
    print(f"\n🚀 Running DocuFlow AI RAG Triad Benchmark ({len(test_slice)} / {len(suite)} test cases)...\n")

    results = []
    print(f"{'Test ID':<10} | {'Archetype':<20} | {'Recall@5':<9} | {'Prec@5':<8} | {'Faithful':<9} | {'Citations':<10} | {'Status'}")
    print("-" * 85)

    for tc in test_slice:
        res = await evaluate_test_case(tc)
        results.append(res)
        status_tag = "✅ PASS" if res["passed"] else "❌ FAIL"
        print(
            f"{res['test_id']:<10} | {res['archetype']:<20} | "
            f"{res['recall']:<9.2f} | {res['precision']:<8.2f} | "
            f"{res['faithfulness']:<9.2f} | {res['citation_acc']:<10.2f} | {status_tag}"
        )

    # Summary Aggregates
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_citation = sum(r["citation_acc"] for r in results) / len(results)
    pass_rate = (sum(1 for r in results if r["passed"]) / len(results)) * 100

    print("\n" + "=" * 85)
    print(" 📊 SUMMARY RAG TRIAD METRICS")
    print("=" * 85)
    print(f" • Average Context Recall @ 5:    {avg_recall * 100:.1f}%")
    print(f" • Average Context Precision @ 5: {avg_precision * 100:.1f}%")
    print(f" • Average Faithfulness:          {avg_faithfulness * 100:.1f}%")
    print(f" • Average Citation Accuracy:     {avg_citation * 100:.1f}%")
    print(f" • Suite Pass Rate:               {pass_rate:.1f}% ({sum(1 for r in results if r['passed'])}/{len(results)} passed)")
    print("=" * 85 + "\n")

    # Output to GitHub Step Summary in CI/CD environments
    import os
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        try:
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write("## 📊 DocuFlow AI RAG Triad Regression Benchmark\n\n")
                f.write(f"**Evaluated {len(test_slice)} Test Cases** across Alphanumeric, Semantic, and Adversarial archetypes.\n\n")
                f.write("| Metric | Score | CI Threshold | Status |\n")
                f.write("| :--- | :--- | :--- | :--- |\n")
                f.write(f"| **Context Recall @ 5** | `{avg_recall * 100:.1f}%` | >= 70% | {'✅ PASS' if avg_recall >= 0.70 else '❌ FAIL'} |\n")
                f.write(f"| **Context Precision @ 5** | `{avg_precision * 100:.1f}%` | >= 60% | {'✅ PASS' if avg_precision >= 0.60 else '❌ FAIL'} |\n")
                f.write(f"| **Faithfulness / Groundedness** | `{avg_faithfulness * 100:.1f}%` | >= 80% | {'✅ PASS' if avg_faithfulness >= 0.80 else '❌ FAIL'} |\n")
                f.write(f"| **Citation Accuracy** | `{avg_citation * 100:.1f}%` | 100% | {'✅ PASS' if avg_citation >= 0.99 else '❌ FAIL'} |\n")
                f.write(f"| **Overall Suite Pass Rate** | `{pass_rate:.1f}%` | >= 70% | {'✅ PASS' if pass_rate >= 70.0 else '❌ FAIL'} |\n")
        except Exception as e:
            print(f"⚠️ Failed to write to GITHUB_STEP_SUMMARY: {e}")

    return {
        "avg_recall": avg_recall,
        "avg_precision": avg_precision,
        "avg_faithfulness": avg_faithfulness,
        "pass_rate": pass_rate,
    }

# Pytest Entrypoint
def test_rag_pipeline_golden_suite():
    metrics = asyncio.run(run_evaluation_suite(limit=5))
    assert metrics["avg_recall"] >= 0.70, "Context Recall fell below 70% threshold!"
    assert metrics["avg_faithfulness"] >= 0.80, "Faithfulness fell below 80% threshold!"

if __name__ == "__main__":
    asyncio.run(run_evaluation_suite(limit=10)) 
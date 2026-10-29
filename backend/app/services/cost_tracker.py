from typing import Dict

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "openai/gpt-oss-120b": {"input_per_million": 0.59, "output_per_million": 0.79},
    "llama-3.3-70b-versatile": {"input_per_million": 0.59, "output_per_million": 0.79},
    "llama-3.1-8b-instant": {"input_per_million": 0.05, "output_per_million": 0.08},
    "default": {"input_per_million": 0.50, "output_per_million": 0.75}
}

def calculate_call_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
    input_cost = (prompt_tokens / 1_000_000.0) * pricing["input_per_million"]
    output_cost = (completion_tokens / 1_000_000.0) * pricing["output_per_million"]
    return round(input_cost + output_cost, 6)
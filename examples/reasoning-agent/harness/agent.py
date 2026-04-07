"""Reasoning QA agent — answers questions using Claude."""

from __future__ import annotations

import os
import re
from pathlib import Path

from anthropic import Anthropic

_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.is_file():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# --- EVO:MUTABLE START ---

SYSTEM_PROMPT = """You are a helpful assistant. Answer the question. 
Give your final answer at the end."""

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 512
TEMPERATURE = 0.0

# --- EVO:MUTABLE END ---


def _extract_answer(text: str) -> str:
    """Pull the final answer from the model's response."""
    # --- EVO:MUTABLE START ---
    text = text.strip()
    lines = text.strip().split("\n")
    return lines[-1].strip()
    # --- EVO:MUTABLE END ---


def run(input_data, trace_callback=None):
    """Answer a reasoning question using Claude."""
    question = str(input_data) if input_data is not None else ""
    if not question:
        return ""

    if trace_callback:
        trace_callback({"type": "llm_call", "model": MODEL, "question": question})

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )

    raw_output = response.content[0].text
    if trace_callback:
        trace_callback({
            "type": "llm_response",
            "raw_output": raw_output,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

    answer = _extract_answer(raw_output)
    if trace_callback:
        trace_callback({"type": "extracted_answer", "answer": answer})

    return answer

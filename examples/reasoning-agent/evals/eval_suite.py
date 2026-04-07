"""Evaluation suite for the reasoning QA agent.

25 tasks across math, logic, and general knowledge with deterministic
expected answers. Scoring uses exact-match and a softer 'contains' check
so partial credit is possible.
"""

from __future__ import annotations

import re
from typing import Any, Callable

TASKS: list[dict[str, Any]] = [
    # --- Arithmetic ---
    {
        "task_id": "math-01",
        "input_data": "What is 17 * 24?",
        "expected": "408",
        "category": "arithmetic",
    },
    {
        "task_id": "math-02",
        "input_data": "What is 1234 + 5678?",
        "expected": "6912",
        "category": "arithmetic",
    },
    {
        "task_id": "math-03",
        "input_data": "What is 999 - 456?",
        "expected": "543",
        "category": "arithmetic",
    },
    {
        "task_id": "math-04",
        "input_data": "What is 144 / 12?",
        "expected": "12",
        "category": "arithmetic",
    },
    {
        "task_id": "math-05",
        "input_data": "What is 2 to the power of 10?",
        "expected": "1024",
        "category": "arithmetic",
    },
    # --- Word problems ---
    {
        "task_id": "word-01",
        "input_data": (
            "A store sells apples for $2 each and oranges for $3 each. "
            "If you buy 5 apples and 4 oranges, how much do you spend in total? "
            "Answer with just the dollar amount as a number."
        ),
        "expected": "22",
        "category": "word_problem",
    },
    {
        "task_id": "word-02",
        "input_data": (
            "A train travels at 60 mph. How many miles does it cover in 2.5 hours? "
            "Answer with just the number."
        ),
        "expected": "150",
        "category": "word_problem",
    },
    {
        "task_id": "word-03",
        "input_data": (
            "If 3 workers can paint a fence in 6 hours, how many hours would it "
            "take 6 workers to paint the same fence? Answer with just the number."
        ),
        "expected": "3",
        "category": "word_problem",
    },
    {
        "task_id": "word-04",
        "input_data": (
            "A rectangle has a length of 15 cm and a width of 8 cm. "
            "What is its area in square centimeters? Answer with just the number."
        ),
        "expected": "120",
        "category": "word_problem",
    },
    {
        "task_id": "word-05",
        "input_data": (
            "You have 3 shirts and 4 pairs of pants. How many different outfits "
            "can you make? Answer with just the number."
        ),
        "expected": "12",
        "category": "word_problem",
    },
    # --- Logic ---
    {
        "task_id": "logic-01",
        "input_data": (
            "All roses are flowers. Some flowers fade quickly. "
            "Can we conclude that some roses fade quickly? "
            "Answer only 'yes' or 'no'."
        ),
        "expected": "no",
        "category": "logic",
    },
    {
        "task_id": "logic-02",
        "input_data": (
            "If it rains, the ground is wet. The ground is wet. "
            "Did it rain? Answer only 'yes', 'no', or 'cannot determine'."
        ),
        "expected": "cannot determine",
        "category": "logic",
    },
    {
        "task_id": "logic-03",
        "input_data": (
            "What comes next in the sequence: 2, 6, 18, 54, ...? "
            "Answer with just the number."
        ),
        "expected": "162",
        "category": "logic",
    },
    {
        "task_id": "logic-04",
        "input_data": (
            "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. "
            "Each label is wrong. You pick one fruit from the 'Mixed' box and "
            "it's an apple. What does the 'Mixed' box actually contain? "
            "Answer with one word."
        ),
        "expected": "apples",
        "category": "logic",
    },
    {
        "task_id": "logic-05",
        "input_data": (
            "How many times does the letter 'e' appear in the word 'Tennessee'? "
            "Answer with just the number."
        ),
        "expected": "4",
        "category": "logic",
    },
    # --- General knowledge ---
    {
        "task_id": "gk-01",
        "input_data": "What is the chemical symbol for gold? Answer with just the symbol.",
        "expected": "Au",
        "category": "general",
    },
    {
        "task_id": "gk-02",
        "input_data": "How many sides does a hexagon have? Answer with just the number.",
        "expected": "6",
        "category": "general",
    },
    {
        "task_id": "gk-03",
        "input_data": "What is the smallest prime number? Answer with just the number.",
        "expected": "2",
        "category": "general",
    },
    {
        "task_id": "gk-04",
        "input_data": (
            "What is the boiling point of water in degrees Celsius? "
            "Answer with just the number."
        ),
        "expected": "100",
        "category": "general",
    },
    {
        "task_id": "gk-05",
        "input_data": (
            "In what year did World War II end? Answer with just the year."
        ),
        "expected": "1945",
        "category": "general",
    },
    # --- Harder reasoning ---
    {
        "task_id": "reason-01",
        "input_data": (
            "A farmer has 17 sheep. All but 9 die. How many sheep are left? "
            "Answer with just the number."
        ),
        "expected": "9",
        "category": "reasoning",
    },
    {
        "task_id": "reason-02",
        "input_data": (
            "If you rearrange the letters 'CIFAIPC', you get the name of a(n) "
            "ocean. What is it? Answer with one word."
        ),
        "expected": "Pacific",
        "category": "reasoning",
    },
    {
        "task_id": "reason-03",
        "input_data": (
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
            "than the ball. How much does the ball cost in cents? "
            "Answer with just the number."
        ),
        "expected": "5",
        "category": "reasoning",
    },
    {
        "task_id": "reason-04",
        "input_data": (
            "If you have a 3-gallon jug and a 5-gallon jug, what is the fewest "
            "number of steps to measure exactly 4 gallons? Answer with just the number."
        ),
        "expected": "6",
        "category": "reasoning",
    },
    {
        "task_id": "reason-05",
        "input_data": (
            "What is the sum of the first 10 positive integers? "
            "Answer with just the number."
        ),
        "expected": "55",
        "category": "reasoning",
    },
]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _score(output: str, expected: str) -> dict[str, float]:
    norm_out = _normalize(output)
    norm_exp = _normalize(expected)

    exact = 1.0 if norm_out == norm_exp else 0.0
    contains = 1.0 if norm_exp in norm_out else 0.0

    if exact == 1.0:
        accuracy = 1.0
    elif contains == 1.0:
        accuracy = 0.5
    else:
        accuracy = 0.0

    return {
        "accuracy": accuracy,
        "exact_match": exact,
        "contains_match": contains,
    }


def get_tasks(split: str) -> list[dict[str, Any]]:
    if split == "search":
        return TASKS[:20]
    if split == "test":
        return TASKS
    return []


def evaluate(
    harness_module: Any,
    task: dict[str, Any],
    trace_callback: Callable[[dict], None],
) -> dict[str, Any]:
    agent = __import__(f"{harness_module.__name__}.agent", fromlist=["agent"])
    question = task.get("input_data", "")
    expected = task.get("expected", "")

    trace_callback({"type": "task_start", "task_id": task["task_id"], "question": question})

    try:
        output = agent.run(question, trace_callback)
    except Exception as exc:
        trace_callback({"type": "error", "message": str(exc)})
        return {
            "task_id": task["task_id"],
            "scores": {"accuracy": 0.0, "exact_match": 0.0, "contains_match": 0.0},
            "output": f"ERROR: {exc}",
        }

    output_str = str(output) if output is not None else ""
    scores = _score(output_str, expected)

    trace_callback({
        "type": "evaluation",
        "expected": expected,
        "output": output_str,
        "scores": scores,
    })

    return {
        "task_id": task["task_id"],
        "scores": scores,
        "output": output_str,
    }

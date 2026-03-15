import re
from pydantic import validator, Field
from typing import Any


class SafeResponse(str):
    """
    Custom response type intended for validating generated output.

    Goal:
    - Add a guardrail layer before returning the response to the user.
    - Detect speculative wording, unsafe content, and missing source references.
    - Support the capstone requirement for reliability and safety controls. [file:24]
    """

    @validator('value')
    def check_safety(cls, v):
        # Check for speculative language that may indicate hallucination.
        # Words such as "guess" or "maybe" suggest weak grounding.
        if re.search(r'\b(guess|assume|probably|maybe)\b', v, re.I):
            raise ValueError("Potential hallucination detected.")

        # Check for sensitive or restricted terms that should not appear
        # in a normal enterprise document-answering response.
        unsafe = ['confidential', 'password', 'SSN']
        if any(word in v.lower() for word in unsafe):
            raise ValueError("Unsafe content detected.")

        # Require source attribution for meaningful answers.
        # This helps keep the output grounded in retrieved documents.
        if 'source' not in v.lower() and len(v) > 50:
            raise ValueError("Missing source attribution.")

        return v


def validate_output(response: str) -> str:
    """
    Apply guardrails before returning the final answer.

    Args:
        response: LLM-generated answer.

    Returns:
        Validated response if checks pass, otherwise an error message.
    """
    try:
        return SafeResponse(response)
    except ValueError as e:
        return f"Validation failed: {e}. Response rejected for safety."

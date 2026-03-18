import re


def validate_output(response: str) -> str:
    """
    Apply lightweight guardrails before returning the final answer.

    Checks:
    - speculative wording that may indicate hallucination
    - unsafe sensitive terms
    - missing source attribution for longer answers
    """
    if not response or not response.strip():
        return "Validation failed: Empty response."

    # Flag speculative wording that may suggest the model is guessing.
    if re.search(r'\b(guess|assume|probably|maybe)\b', response, re.I):
        return "Validation failed: Potential hallucination detected."

    # Block obviously sensitive content from being returned to the user.
    unsafe_terms = ["confidential", "password", "ssn"]
    if any(term in response.lower() for term in unsafe_terms):
        return "Validation failed: Unsafe content detected."

    # Require source attribution for longer answers to improve grounding.
    if len(response) > 50 and "source" not in response.lower():
        return "Validation failed: Missing source attribution."

    return response

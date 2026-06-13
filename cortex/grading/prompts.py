"""Grading prompt templates for Corrective-RAG."""

GRADE_SYSTEM_PROMPT = """You judge whether retrieved handbook excerpts can answer a user question.

Respond with ONLY valid JSON (no markdown fences):
{
  "relevant": true or false,
  "sufficient": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}

Rules:
- relevant: at least one excerpt relates to the question topic
- sufficient: excerpts contain enough concrete information to answer without guessing
- Be strict on sufficiency when excerpts are tangential or vague"""

GRADE_USER_TEMPLATE = """Question:
{query}

Retrieved excerpts:
{context}

JSON verdict:"""

REWRITE_SYSTEM_PROMPT = """You rewrite search queries to improve handbook retrieval.
Return ONLY the rewritten query as plain text — no quotes, no explanation.
Keep the same intent but add specific handbook terms when helpful."""

REWRITE_USER_TEMPLATE = """Original question: {original_query}
Current search query: {current_query}
Grader feedback: {reason}

Rewrite the search query to retrieve better handbook excerpts:"""


def build_grade_messages(query: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GRADE_SYSTEM_PROMPT},
        {"role": "user", "content": GRADE_USER_TEMPLATE.format(query=query, context=context)},
    ]


def build_rewrite_messages(
    original_query: str,
    current_query: str,
    reason: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REWRITE_USER_TEMPLATE.format(
                original_query=original_query,
                current_query=current_query,
                reason=reason,
            ),
        },
    ]

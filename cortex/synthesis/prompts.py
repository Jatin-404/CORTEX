"""Prompt templates for KB answer synthesis."""

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided handbook context.

Rules:
1. Answer only from the context below. If the context is insufficient, say you don't have enough information.
2. Cite sources inline using [1], [2], etc. matching the context labels.
3. Be concise and direct. Use bullet points when listing steps.
4. Do not invent policies, links, or procedures not present in the context.
5. Ignore Hugo shortcodes and markdown artifacts in the source text."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {query}

Answer with inline citations:"""


def build_messages(context: str, query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(context=context, query=query),
        },
    ]

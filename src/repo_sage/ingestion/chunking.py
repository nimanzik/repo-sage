from functools import lru_cache

from google import genai

from ..defaults import DEFAULT_GEMINI_MODEL_ID

SECTION_DELIMITER = "<<<SECTION_BREAK>>>"

PROMPT_TEMPLATE = """
Split the provided document into logical sections for a Q&A system.

Each section should be self-contained and focused on a specific topic or concept.

CRITICAL RULES:
1. ONLY use exact text from the document: copy text VERBATIM
2. DO NOT add explanations, introductions, summaries, or commentary
3. Do NOT add phrases like "This section covers..." or "The document explains..."

<DOCUMENT>
{document}
</DOCUMENT>

Output format should be:

## [Short descriptive title]

[Exact verbatim text from document]

<<<SECTION_BREAK>>>

## [Another short descriptive title]

[Another exact verbatim text from document]

<<<SECTION_BREAK>>>
... and so on.
"""


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    """Get or create a cached Google GenAI client."""
    return genai.Client()


def chunk_document(document: str, gemini_model_id: str | None = None) -> list[str]:
    """Chunk a document into logical sections using a powerful LLM."""
    client = get_genai_client()
    response = client.models.generate_content(
        model=gemini_model_id or DEFAULT_GEMINI_MODEL_ID,
        contents=PROMPT_TEMPLATE.format(document=document),
    )
    if not response.text:
        return []

    sections = response.text.split(SECTION_DELIMITER)
    return [section.strip() for section in sections if section.strip()]

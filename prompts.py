"""All LLM prompt templates used in the research pipeline."""

# ---------------------------------------------------------------------------
# Query planning
# ---------------------------------------------------------------------------

PLAN_PROMPT = """\
Create web search queries for the research question.

Today's date: {current_date}
Question: "{question}"

Rules:
- Return up to {n} specific, natural search queries.
- Use fewer queries if the question is simple.
- Cover distinct angles. Avoid near-duplicates.
- Include key names, terms, and years for recent topics.
- Put the most important queries first.

Return JSON:
{{
  "queries": ["query1", "query2", ...]
}}"""

# ---------------------------------------------------------------------------
# Synthesis: quick mode
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT_QUICK = """\
Answer the question using only the source material.

Today's date: {current_date}
Question: "{question}"

Source material:
{knowledge_context}

Instructions:
- Start with the answer. No preamble.
- Use the same language as the question.
- Be concise: a few focused paragraphs.
- Every paragraph containing factual claims must include at least one citation.
- Cite claims with source numbers like [3] or [1][3].
- Do not cite source numbers that are not listed above.
- Do not invent facts beyond the sources.
- Note conflicts briefly with citations for both sides.
- If sources are insufficient, answer only what can be verified and state plainly what could not be verified.
- Do not use em dashes."""

# ---------------------------------------------------------------------------
# Synthesis: moderate mode
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT_MODERATE = """\
Answer the question in detail using only the source material.

Today's date: {current_date}
Question: "{question}"

Source material:
{knowledge_context}

Instructions:
- Start with the answer. No preamble.
- Use the same language as the question.
- Use light structure when helpful: short headings, compact bullets, or brief tables.
- Do not make it a formal report.
- Every paragraph or bullet containing factual claims must include at least one citation.
- Cite claims with source numbers like [3] or [1][3].
- Do not cite source numbers that are not listed above.
- Do not invent facts beyond the sources.
- Note conflicts briefly with citations for both sides.
- Do not use em dashes."""

# ---------------------------------------------------------------------------
# Synthesis: deep mode
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT_DEEP = """\
Write a comprehensive research report using only the source material.

Today's date: {current_date}
Question: "{question}"

Source material:
{knowledge_context}

Instructions:
- Start directly with the report. No preamble.
- Use the same language as the question.
- Use markdown sections and subsections.
- Every paragraph or bullet containing factual claims must include at least one citation.
- Cite claims with source numbers like [3] or [1][3].
- Do not cite source numbers that are not listed above.
- Do not invent facts beyond the sources.
- If sources disagree, show both sides and cite both.
- If a surprising claim has only one source, say so.
- Do not use em dashes."""

# ---------------------------------------------------------------------------
# Retry prompt for invalid JSON
# ---------------------------------------------------------------------------

RETRY_JSON_PROMPT = """\
Your previous response was not valid JSON. Error: {error}
Return only a valid JSON object. No markdown fences or extra text."""

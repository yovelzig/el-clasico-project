"""
LLM module — Gemini powered answer generation.
Uses RAG context + conversation memory for accurate, concise responses.
"""
import os
from google import genai
from google.genai import types

MODEL = "gemini-3-flash-preview"

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are an expert football analyst specialized in El Clasico, the legendary rivalry between FC Barcelona and Real Madrid.

Rules:
1. Use the provided DOCUMENT CONTEXT as your highest priority source of truth.
2. If the retrieved context contains a direct answer or a relevant statistic, always follow it even if it contradicts your general football knowledge.
3. Do not invent answers from general knowledge when the documents provide a clear response.
4. If there are multiple relevant facts in the context, choose the one that directly answers the question.
5. If the context does NOT contain the information, then and only then use general football knowledge and say that the documents did not provide the answer.
6. NEVER invent fake statistics, scores, or dates.
7. Keep answers SHORT and DIRECT: 1-3 sentences unless more detail is genuinely needed.
8. Use conversation history only to resolve pronouns and follow-up questions.
9. Sound like a knowledgeable football analyst.

Example good answers:
- Messi holds the record with 26 goals in official El Clasico matches.
- Barcelona biggest win was 5-0 vs Real Madrid on December 18 2010.
- That match was in April 2017 when Messi scored the winner at the Bernabeu.
"""


def ask(question: str, context: str, history: list) -> str:
    client = get_client()

    # Build conversation history as Gemini-compatible contents
    contents = []
    for turn in history[-6:]:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))

    # Add current question with injected context
    if context:
        user_content = f"DOCUMENT CONTEXT:\n{context}\n\nQuestion: {question}"
    else:
        user_content = f"[No relevant documents found - use general knowledge]\n\nQuestion: {question}"

    contents.append(types.Content(role="user", parts=[types.Part(text=user_content)]))

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=300,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    return response.text.strip()

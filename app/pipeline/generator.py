"""LLM Generation using Groq.

Leverages Groq's API for ultra-fast generation of answers using Llama 3.1.
Enforces a JSON output structure to extract the answer and cited sources.
"""

import json
import logging
import time
from typing import List

from app.config import settings
from app.models import GeneratedAnswer, RetrievalCandidate

logger = logging.getLogger(__name__)


class GroqGenerator:
    """Singleton-like class for interacting with the Groq API."""
    
    def __init__(self):
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError(
                "The 'groq' package is required for passage retrieval mode. "
                "Install it with: pip install groq"
            ) from e
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
        
    def generate(self, query: str, candidates: List[RetrievalCandidate]) -> GeneratedAnswer:
        """Generate an answer using retrieved context.
        
        Returns a structured GeneratedAnswer object.
        """
        start_time = time.perf_counter()
        
        # Build context string
        context_parts = []
        for i, cand in enumerate(candidates, start=1):
            context_parts.append(f"[Source {i}]:\n{cand.context_text}")
            
        context_text = "\n\n".join(context_parts)
        
        system_prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY the provided context.
If the context does not contain the answer, set 'grounded' to false and state that you cannot answer.

CONTEXT:
{context_text}

You must return your response as a JSON object with the following schema:
{{
    "answer": "Your detailed answer text. Do NOT include source markers like [Source 1] directly in this text.",
    "grounded": true or false,
    "confidence": 0.0 to 1.0 (your confidence in the answer),
    "sources": [1, 2] (Output a JSON list of INDIVIDUAL integer source IDs you used, separated by commas. For example, if you used Source 1 and Source 2, output [1, 2]. Do NOT concatenate them into [12]. You MUST ONLY use source IDs provided above from 1 to {len(candidates)}.)
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )
            
            raw_json = response.choices[0].message.content
            # logger.debug(f"raw json from LLM: {raw_json}")
            parsed = json.loads(raw_json)
            
            sources = parsed.get("sources", [])
            fixed_sources = []
            for s in sources:
                if isinstance(s, int) and s > len(candidates):
                    # Fix LLM string concatenation quirk (e.g., outputs [12] instead of [1, 2])
                    fixed_sources.extend([int(d) for d in str(s) if int(d) <= len(candidates)])
                else:
                    fixed_sources.append(s)
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return GeneratedAnswer(
                answer=parsed.get("answer", "Error generating answer."),
                grounded=parsed.get("grounded", False),
                confidence=float(parsed.get("confidence", 0.0)),
                sources=list(set(fixed_sources)),
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return GeneratedAnswer(
                answer="Sorry, there was an error generating the response.",
                grounded=False,
                confidence=0.0,
                sources=[],
                latency_ms=(time.perf_counter() - start_time) * 1000
            )

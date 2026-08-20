"""Cheap pre-retrieval input validation.

Rejects empty queries, excessive lengths, or obvious gibberish.
Does NOT use an LLM or do complex domain classification. Domain filtering
is handled naturally by the retrieval confidence gate later.
"""

import re
from typing import Optional

from app.models import InputGuardResult


class InputGuard:
    @staticmethod
    def validate(query: str) -> InputGuardResult:
        """Run cheap heuristics on the transcribed text."""
        query = query.strip()
        
        # 1. Empty or too short
        if not query or len(query) < 3:
            return InputGuardResult(
                passed=False,
                rejection_reason="Query is empty or too short."
            )
            
        # 2. Too long (prevents DDoS or blowing up embedding context limits)
        if len(query) > 500:
            return InputGuardResult(
                passed=False,
                rejection_reason="Query exceeds maximum length of 500 characters."
            )
            
        # 3. Gibberish Check: Character repetition (e.g., 'fjkdlsafjkdslajfklsdajlk')
        # Check if it's mostly a single word that's impossibly long
        words = query.split()
        if any(len(word) > 40 for word in words):
            return InputGuardResult(
                passed=False,
                rejection_reason="Query contains unusually long unbroken character sequences."
            )
            
        # 4. Gibberish Check: Vowel density (must have at least *some* vowels to be natural English)
        # We are assuming transcribed_text is in English (since STT translated it)
        vowel_count = sum(1 for char in query.lower() if char in "aeiouy")
        if vowel_count == 0 and len(query) > 10:
            return InputGuardResult(
                passed=False,
                rejection_reason="Query lacks typical vowel structure for an English sentence."
            )
            
        # 5. Simple safety blocklist (expand as needed, kept minimal for hackathon)
        unsafe_patterns = [
            r"\b(ignore previous instructions|system prompt)\b"
        ]
        
        lower_query = query.lower()
        for pattern in unsafe_patterns:
            if re.search(pattern, lower_query):
                return InputGuardResult(
                    passed=False,
                    rejection_reason="Query triggered safety blocklist."
                )

        return InputGuardResult(passed=True, rejection_reason=None)

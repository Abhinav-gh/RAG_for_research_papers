import requests
import threading
import time
from typing import List, Optional


class GeminiAPIKeyRotator:
    def __init__(self, api_keys: List[str], cooldown: float = 1.0):
        self.api_keys = api_keys
        self.cooldown = cooldown
        self.lock = threading.Lock()
        self.index = 0
        self.last_used = [0.0] * len(api_keys)

    def get_key(self) -> str:
        with self.lock:
            now = time.time()

            # Try to find a key that is not cooling down
            for i in range(len(self.api_keys)):
                idx = (self.index + i) % len(self.api_keys)
                if now - self.last_used[idx] > self.cooldown:
                    self.last_used[idx] = now
                    self.index = (idx + 1) % len(self.api_keys)
                    return self.api_keys[idx]

            # Fallback: use next key even if cooling down
            idx = self.index
            self.last_used[idx] = now
            self.index = (idx + 1) % len(self.api_keys)
            return self.api_keys[idx]


class GeminiAnswerGenerator:
    def __init__(self, api_keys: List[str]):
        self.rotator = GeminiAPIKeyRotator(api_keys)
        self.model = "models/gemini-2.5-flash"
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:generateContent"

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """Generate answer using Gemini 2.5 Flash"""

        prompt = self._build_prompt(query, contexts)
        api_key = self.rotator.get_key()

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                f"{self.api_url}?key={api_key}",
                json=payload,
                headers=headers,
                timeout=40  # Gemini 2.5 is fast but allow room
            )

            if response.status_code != 200:
                return f"[Gemini API Error {response.status_code}]: {response.text}"

            data = response.json()

            return self._parse_response(data)

        except Exception as e:
            return f"[Gemini Request Failed: {str(e)}]"

    def _parse_response(self, data: dict) -> str:
        """Robust parser for Gemini 2.5 Flash responses"""

        try:
            # New structured format
            candidates = data.get("candidates", [])
            if not candidates:
                return "[Empty Gemini response]"

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            if not parts:
                return "[Gemini returned no text parts]"

            # Latest model always uses parts[*].text
            text = parts[0].get("text")
            if text:
                return text.strip()

            return "[Gemini text field missing]"

        except Exception as e:
            return f"[Response parsing failed: {str(e)}]"

    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        ctx = "\n\n---\n\n".join(contexts)

        return (
            "You are a hybrid RAG assistant.\n"
            "Primary goal: use and prioritize the information provided in the context.\n"
            "Secondary goal: if the context is incomplete, unclear, or missing key details, "
            "you may use your own general knowledge — BUT keep it factual and reasonable.\n\n"

            "GUIDELINES:\n"
            "1. Prefer context when relevant.\n"
            "2. If context is insufficient, answer using your own knowledge.\n"
            "3. Clearly separate what comes from context vs your own reasoning.\n"
            "4. Do NOT hallucinate details that contradict the context.\n"
            "5. Keep the answer concise and helpful.\n\n"

            "======== CONTEXT START ========\n"
            f"{ctx}\n"
            "======== CONTEXT END ==========\n\n"

            f"QUESTION:\n{query}\n\n"

            "FINAL ANSWER FORMAT:\n"
            "- Answer: <your answer>\n"
            "- Context usage: <explain briefly whether context was used or not>\n\n"

            "Now produce the final answer.\n"
            "ANSWER:"
        )



"""Gemini API backend. No token logprobs, so confidence falls back to self-report."""
import os

from .base import Model, Response


class GeminiModel(Model):
    name = "gemini-2.5-flash-lite"
    supports_logprobs = False

    def __init__(self, model_id="gemini-2.5-flash-lite", api_key=None):
        from google import genai
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY")
        self.name = model_id
        self._c = genai.Client(api_key=key)

    def generate(self, prompt, temperature=0.0, max_tokens=64):
        from google.genai import types
        r = self._c.models.generate_content(
            model=self.name, contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature,
                                               max_output_tokens=max_tokens))
        return Response(text=r.text or "", logprobs=None, meta={"backend": "gemini"})

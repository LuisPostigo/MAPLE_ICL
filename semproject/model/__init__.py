"""Backend registry.

Imports are deferred so a missing SDK breaks only its own backend, and nothing
outside this package needs to know which model is in use.
"""
from .base import Model, Response

_BACKENDS = {
    "mock": (".mock", "MockModel"),
    "hf": (".hf_local", "HFModel"),
    "gemini": (".gemini", "GeminiModel"),
}


def get_model(name="mock", **kwargs) -> Model:
    if name not in _BACKENDS:
        raise ValueError(f"unknown model {name!r}; have {sorted(_BACKENDS)}")
    import importlib
    module_name, class_name = _BACKENDS[name]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)(**kwargs)


def available():
    """Which backends could run, checked without constructing any of them.

    Constructing a backend downloads weights or opens a client, so this looks
    only at whether the dependency imports and any credential is present.
    """
    import importlib.util
    import os

    has_transformers = importlib.util.find_spec("transformers") is not None
    has_genai = importlib.util.find_spec("google.genai") is not None
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY")
                          or os.environ.get("GOOGLE_API_KEY"))

    return {
        "mock": "ready",
        "hf": "ready (downloads weights on first use)" if has_transformers
              else "needs transformers",
        "gemini": "ready" if has_genai and has_gemini_key
                  else "needs GEMINI_API_KEY" if has_genai else "needs google-genai",
    }


__all__ = ["Model", "Response", "get_model", "available"]

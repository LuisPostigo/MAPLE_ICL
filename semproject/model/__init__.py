"""Backend registry. Import cost is deferred so a missing SDK breaks only its own backend."""
from .base import Model, Response

_BACKENDS = {
    "mock": (".mock", "MockModel"),
    "hf": (".hf_local", "HFModel"),
    "gemini": (".gemini", "GeminiModel"),
}


def get_model(name="mock", **kw) -> Model:
    if name not in _BACKENDS:
        raise ValueError(f"unknown model {name!r}; have {sorted(_BACKENDS)}")
    import importlib
    mod, cls = _BACKENDS[name]
    return getattr(importlib.import_module(mod, __package__), cls)(**kw)


def available():
    """Report which backends could run, without constructing any of them.

    Constructing a backend downloads weights or opens a client, so this checks
    only that the dependency imports and any credential is present.
    """
    import importlib.util
    import os
    out = {}
    for name in _BACKENDS:
        if name == "mock":
            out[name] = "ready"
        elif name == "hf":
            out[name] = ("ready (downloads weights on first use)"
                         if importlib.util.find_spec("transformers") else "needs transformers")
        elif name == "gemini":
            has_sdk = importlib.util.find_spec("google.genai") is not None
            has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            out[name] = ("ready" if has_sdk and has_key else
                         "needs GEMINI_API_KEY" if has_sdk else "needs google-genai")
    return out


__all__ = ["Model", "Response", "get_model", "available"]

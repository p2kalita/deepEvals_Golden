"""
================================================================================
llm_wrappers.py
================================================================================
Two DeepEval base-class implementations:

1. LiteLLMDeepEvalLLM
   Replaces app1.py's GroqDeepEvalLLM. Same dual-mode behaviour (free-text
   generation AND Pydantic-schema-validated JSON generation for DeepEval's
   internal filtration/evolution/styling machinery), but routes every call
   through LiteLLM's unified completion()/acompletion() interface instead of
   a provider-specific SDK. Swapping Gemini <-> GPT-4o <-> Sonnet <-> Groq <->
   local Ollama is then a ModelConfig (i.e. YAML) change only -- this class
   never changes.

2. LocalEmbeddingModel
   Identical in behaviour to app1.py's version. Embeddings are ONLY used by
   DeepEval internally for document chunking and context-group similarity
   inside generate_goldens_from_docs(); kept 100% local via
   sentence-transformers so no OpenAI key is ever required (this was the
   original bug app1.py worked around, and that fix is preserved verbatim).
================================================================================
"""

from __future__ import annotations

import json
from typing import List, Optional, Type, TypeVar

import litellm
from pydantic import BaseModel

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models import DeepEvalBaseEmbeddingModel

from config_loader import ModelConfig

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LiteLLMDeepEvalLLM(DeepEvalBaseLLM):
    """
    Production-ready DeepEval wrapper around LiteLLM's unified API.

    Handles both branches DeepEval needs:
      - plain string generation (most synthesizer prompts)
      - Pydantic-schema structured JSON generation (filtration scoring,
        evolution rewrites, styling rewrites, etc.)

    Which provider this actually talks to (Gemini / OpenAI / Anthropic /
    Groq / local Ollama) is determined entirely by the ModelConfig passed
    in at construction time -- this class has zero provider-specific code.
    """

    def __init__(self, config: ModelConfig, max_tokens: int = 4096) -> None:
        self.config = config
        self.max_tokens = max_tokens

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_messages(self, prompt: str, use_json: bool) -> list[dict]:
        """Build the messages array, honouring the YAML system_message if set."""
        default_system = (
            "You are a helpful assistant. "
            "Respond ONLY with a valid JSON object that matches the schema. "
            "Do not include markdown code fences or any extra text."
            if use_json
            else "You are a helpful assistant."
        )
        # An explicit system_message in YAML overrides the default, but for
        # JSON mode we still need the "respond only with JSON" instruction,
        # so we append it rather than discard it.
        system_content = self.config.system_message or default_system
        if use_json and self.config.system_message:
            system_content = f"{self.config.system_message}\n\n{default_system}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

    def _completion_kwargs(self, prompt: str, use_json: bool) -> dict:
        kwargs: dict = dict(
            model=self.config.litellm_model,
            messages=self._build_messages(prompt, use_json),
            temperature=self.config.temperature,
            max_tokens=self.max_tokens,
        )
        # Local Ollama needs an explicit api_base; hosted providers don't.
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        # Ask LiteLLM for JSON-mode output where the underlying provider
        # supports it. LiteLLM silently ignores this for providers that
        # don't support response_format, so it's always safe to pass.
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    @staticmethod
    def _parse_schema(raw: str, schema: Type[SchemaT]) -> SchemaT:
        """
        Parse a raw JSON string into a Pydantic model instance.
        Falls back to schema defaults on any parse error so a single bad
        completion never crashes an entire generation run.
        """
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.lstrip("`")
            if clean.lower().startswith("json"):
                clean = clean[4:]
            clean = clean.rstrip("`").strip()
        try:
            return schema(**json.loads(clean))
        except Exception as exc:
            print(
                f"  [LiteLLMDeepEvalLLM] JSON parse error "
                f"({type(exc).__name__}: {exc}); using schema defaults."
            )
            try:
                return schema.model_construct()
            except Exception:
                return schema()

    # ── DeepEvalBaseLLM interface ─────────────────────────────────────────

    def load_model(self):
        """Return self; LiteLLM has no client object to lazily construct."""
        return self

    def generate(self, prompt: str, schema: Optional[Type[SchemaT]] = None):
        """Synchronous generation via LiteLLM's completion()."""
        use_json = schema is not None
        response = litellm.completion(**self._completion_kwargs(prompt, use_json))
        raw = response.choices[0].message.content or ""
        return self._parse_schema(raw, schema) if use_json else raw

    async def a_generate(self, prompt: str, schema: Optional[Type[SchemaT]] = None):
        """Asynchronous generation via LiteLLM's acompletion()."""
        use_json = schema is not None
        response = await litellm.acompletion(**self._completion_kwargs(prompt, use_json))
        raw = response.choices[0].message.content or ""
        return self._parse_schema(raw, schema) if use_json else raw

    def get_model_name(self) -> str:
        return self.config.litellm_model


class LocalEmbeddingModel(DeepEvalBaseEmbeddingModel):
    """
    Zero-cost local embedding model backed by sentence-transformers.
    Runs on CPU/GPU, no API key, no per-call cost. Model weights are
    downloaded once on first use and cached thereafter.

    Preserved as-is from app1.py: this is what prevents DeepEval's
    generate_goldens_from_docs() from defaulting to OpenAI's
    text-embedding-3-small for document chunking and context grouping.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name_str = model_name
        self._model = None  # lazy-loaded on first call

    def load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            print(
                f"  [Embedder] Loading '{self.model_name_str}' "
                f"(first-time download if needed) …"
            )
            self._model = SentenceTransformer(self.model_name_str)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        return self.load_model().encode(text, convert_to_numpy=True).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.load_model().encode(texts, convert_to_numpy=True).tolist()

    # sentence-transformers has no async API; synchronous fallback.
    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        return f"sentence-transformers/{self.model_name_str}"

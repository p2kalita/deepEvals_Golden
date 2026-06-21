"""
================================================================================
config_loader.py
================================================================================
Loads a provider/model definition from a YAML file and turns it into the
exact model-string format LiteLLM expects, so swapping providers is a
config-only change everywhere else in the codebase.

LiteLLM routes by string prefix, e.g.:
    "gemini/gemini-2.5-pro-preview-03-25"   -> Google Gemini API
    "gpt-4o"                                -> OpenAI (no prefix needed)
    "claude-sonnet-4-6"                     -> Anthropic (no prefix needed,
                                                but we add "anthropic/" to be
                                                explicit and avoid ambiguity)
    "groq/llama-3.3-70b-versatile"          -> Groq
    "ollama/gemma2:latest"                  -> local Ollama server

Only this module knows about that mapping. Every other module just asks for
"the LiteLLM model string" and "the parameters" and does not care which
provider is behind them.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


# Maps the `components.api` value in YAML -> LiteLLM provider prefix.
# Empty string means "no prefix needed, LiteLLM infers it from the model name".
_PROVIDER_PREFIX: Dict[str, str] = {
    "google": "gemini",
    "openai": "",
    "anthropic": "anthropic",
    "groq": "groq",
    "ollama": "ollama",
}


@dataclass
class BatchProcessingConfig:
    """Mirrors the `parameters.batch_processing` block in YAML."""

    enabled: bool = True
    batch_size: int = 32


@dataclass
class ModelConfig:
    """
    Fully-resolved configuration for one provider/model, parsed from a single
    YAML file. This is the only object the rest of the codebase needs to
    instantiate a working LLM wrapper.
    """

    name: str
    description: str
    version: str
    api: str                       # raw value from YAML, e.g. "google"
    model: str                     # raw model id from YAML
    temperature: float
    system_message: str
    batch_processing: BatchProcessingConfig
    api_base: str | None = None    # only set for local providers (Ollama)
    extra_parameters: Dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def litellm_model(self) -> str:
        """
        Build the model string LiteLLM's completion()/acompletion() expects.
        This is the single place that encodes provider-prefix knowledge.
        """
        prefix = _PROVIDER_PREFIX.get(self.api.lower())
        if prefix is None:
            raise ValueError(
                f"Unknown provider '{self.api}' in config '{self.name}'. "
                f"Supported providers: {sorted(_PROVIDER_PREFIX)}"
            )
        if not prefix:
            return self.model
        # Ollama already encodes the tag (e.g. "gemma2:latest") in `model`.
        return f"{prefix}/{self.model}"


def load_model_config(path: str | Path) -> ModelConfig:
    """
    Parse one YAML file (in the format shown in the project brief) into a
    ModelConfig. Raises a clear error if required fields are missing instead
    of failing deep inside LiteLLM with a confusing message.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    try:
        components = raw["components"]
        parameters = raw.get("parameters", {})
        batch_raw = parameters.get("batch_processing", {})
    except KeyError as exc:
        raise ValueError(f"Config '{path}' is missing required key: {exc}") from exc

    return ModelConfig(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        version=raw.get("version", "0.0.0"),
        api=components["api"],
        model=components["model"],
        temperature=float(parameters.get("temperature", 0.0)),
        system_message=parameters.get("system_message", ""),
        batch_processing=BatchProcessingConfig(
            enabled=bool(batch_raw.get("enabled", True)),
            batch_size=int(batch_raw.get("batch_size", 32)),
        ),
        api_base=parameters.get("api_base"),
        extra_parameters={
            k: v
            for k, v in parameters.items()
            if k not in {"batch_processing", "temperature", "system_message", "api_base"}
        },
        source_path=path,
    )


def list_available_configs(configs_dir: str | Path) -> List[Path]:
    """Return every *.yaml / *.yml file in the configs directory, sorted."""
    configs_dir = Path(configs_dir)
    return sorted(
        p for p in configs_dir.iterdir() if p.suffix.lower() in {".yaml", ".yml"}
    )

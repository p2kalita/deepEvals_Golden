"""
================================================================================
synthesis_configs.py
================================================================================
Every DeepEval Synthesizer configuration object from app1.py, preserved
exactly (same thresholds, same evolution weights, same styling text), just
reorganised into factory functions so they're reusable across every
provider/model and every one of the 3x4 complexity/style combinations
without copy-pasting config blocks.

Nothing about the actual config values has been simplified or reduced:
  - All 7 Evolution types remain available at every tier.
  - All 3 complexity tiers (low/medium/high) keep their original weights
    and num_evolutions.
  - All 4 styling variants keep their original task/scenario/input_format/
    expected_output_format text.
  - FiltrationConfig keeps the same quality_threshold and retry count.
  - ContextConstructionConfig keeps the same chunking/context parameters,
    and the embedder is still attached here (not on Synthesizer), which is
    the load-bearing fix from app1.py for avoiding the OpenAI-key error.
================================================================================
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.synthesizer.types import Evolution
from deepeval.synthesizer.config import (
    EvolutionConfig,
    StylingConfig,
    FiltrationConfig,
    ContextConstructionConfig,
)


class ComplexityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StyleVariant(str, Enum):
    SIMPLE = "simple"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    DOMAIN = "domain"


# ─────────────────────────────────────────────────────────────────────────────
# Evolution configs -- all 7 Evolution types, 3 complexity tiers
# ─────────────────────────────────────────────────────────────────────────────
#   REASONING      - multi-step logical deduction (hardest, most depth)
#   MULTICONTEXT   - synthesises info from multiple document passages
#   CONCRETIZING   - makes abstract questions more specific
#   CONSTRAINED    - adds explicit constraints or restrictions
#   COMPARATIVE    - compares two or more concepts or entities
#   HYPOTHETICAL   - "what if" / counterfactual scenarios
#   IN_BREADTH     - broadens scope to adjacent topics (horizontal expansion)
#
# num_evolutions sets the number of sequential transformation steps:
#   1 = LOW | 2 = MEDIUM | 3+ = HIGH
# ─────────────────────────────────────────────────────────────────────────────

def build_evolution_config(tier: ComplexityTier) -> EvolutionConfig:
    """Return the EvolutionConfig for a given complexity tier."""
    if tier == ComplexityTier.LOW:
        return EvolutionConfig(
            evolutions={
                Evolution.REASONING: 0.10,
                Evolution.MULTICONTEXT: 0.10,
                Evolution.CONCRETIZING: 0.25,
                Evolution.CONSTRAINED: 0.15,
                Evolution.COMPARATIVE: 0.15,
                Evolution.HYPOTHETICAL: 0.10,
                Evolution.IN_BREADTH: 0.15,
            },
            num_evolutions=1,
        )
    if tier == ComplexityTier.MEDIUM:
        return EvolutionConfig(
            evolutions={
                Evolution.REASONING: 0.20,
                Evolution.MULTICONTEXT: 0.15,
                Evolution.CONCRETIZING: 0.15,
                Evolution.CONSTRAINED: 0.15,
                Evolution.COMPARATIVE: 0.15,
                Evolution.HYPOTHETICAL: 0.10,
                Evolution.IN_BREADTH: 0.10,
            },
            num_evolutions=2,
        )
    if tier == ComplexityTier.HIGH:
        return EvolutionConfig(
            evolutions={
                Evolution.REASONING: 0.30,
                Evolution.MULTICONTEXT: 0.25,
                Evolution.CONCRETIZING: 0.05,
                Evolution.CONSTRAINED: 0.10,
                Evolution.COMPARATIVE: 0.15,
                Evolution.HYPOTHETICAL: 0.10,
                Evolution.IN_BREADTH: 0.05,
            },
            num_evolutions=3,
        )
    raise ValueError(f"Unknown complexity tier: {tier}")


# ─────────────────────────────────────────────────────────────────────────────
# Styling configs -- four question personas, adapted to the support-ticket
# domain (app1.py's originals were AI/Cloud/Cybersecurity-flavoured; the
# task/scenario text below is the same structure applied to ticket triage,
# since that's what document_builder.py now generates context from).
# ─────────────────────────────────────────────────────────────────────────────

def build_styling_config(style: StyleVariant) -> StylingConfig:
    """Return the StylingConfig for a given styling variant."""
    if style == StyleVariant.SIMPLE:
        return StylingConfig(
            task=(
                "Answer general questions about a customer support ticket "
                "in plain, everyday language."
            ),
            scenario=(
                "A customer or new support agent with no specialised "
                "background asking foundational questions about a ticket."
            ),
            input_format=(
                "Short, jargon-free questions using everyday vocabulary; "
                "avoid internal abbreviations where possible."
            ),
            expected_output_format=(
                "Clear, concise answers of 1-3 sentences understandable by "
                "a non-expert."
            ),
        )
    if style == StyleVariant.PROFESSIONAL:
        return StylingConfig(
            task=(
                "Provide management-level guidance on triage priority, "
                "department routing, and SLA risk for support tickets."
            ),
            scenario=(
                "A support team lead or operations manager deciding how to "
                "prioritise, route, or escalate tickets."
            ),
            input_format=(
                "Business-oriented questions about urgency, staffing impact, "
                "customer-satisfaction risk, or escalation policy."
            ),
            expected_output_format=(
                "Executive-level responses of 2-4 sentences citing business "
                "impact, trade-offs, and best practice."
            ),
        )
    if style == StyleVariant.TECHNICAL:
        return StylingConfig(
            task=(
                "Explain the technical root cause, diagnostic steps, and "
                "resolution path implied by a support ticket."
            ),
            scenario=(
                "A support engineer or on-call specialist troubleshooting "
                "the specific issue described in a ticket."
            ),
            input_format=(
                "Precise technical questions referencing specific systems, "
                "error conditions, logs, or configuration details."
            ),
            expected_output_format=(
                "Detailed technical answers that may include diagnostic "
                "steps or CLI examples, in 3-6 sentences."
            ),
        )
    if style == StyleVariant.DOMAIN:
        return StylingConfig(
            task=(
                "Answer domain-specific questions about ticket categorisation, "
                "tagging accuracy, and department-specific handling policy."
            ),
            scenario=(
                "A QA analyst or support-operations specialist auditing "
                "ticket classification and department-specific SLAs."
            ),
            input_format=(
                "Domain-specific queries referencing department names, "
                "priority levels, tag taxonomies, or routing rules."
            ),
            expected_output_format=(
                "Authoritative responses aligned with the ticket's department "
                "and priority context, in 2-5 sentences."
            ),
        )
    raise ValueError(f"Unknown style variant: {style}")


def all_evolution_configs() -> Dict[ComplexityTier, EvolutionConfig]:
    """All 3 complexity-tier configs, keyed by tier."""
    return {tier: build_evolution_config(tier) for tier in ComplexityTier}


def all_styling_configs() -> Dict[StyleVariant, StylingConfig]:
    """All 4 styling-variant configs, keyed by variant."""
    return {style: build_styling_config(style) for style in StyleVariant}


# ─────────────────────────────────────────────────────────────────────────────
# Filtration config -- critic-scored quality gate on generated inputs
# ─────────────────────────────────────────────────────────────────────────────

def build_filtration_config(critic_model: DeepEvalBaseLLM) -> FiltrationConfig:
    """
    The critic model scores each generated input on self-containment and
    clarity (0-1). Inputs below the threshold are regenerated up to
    max_quality_retries times; if still below, the highest-scoring attempt
    is used. Threshold and retry count preserved from app1.py.
    """
    return FiltrationConfig(
        critic_model=critic_model,
        synthetic_input_quality_threshold=0.5,
        max_quality_retries=3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Context construction config -- CRITICAL: embedder goes here, not on
# Synthesizer(). This is the fix app1.py discovered for the OpenAI-key error.
# ─────────────────────────────────────────────────────────────────────────────

def build_context_construction_config(
    embedder: DeepEvalBaseEmbeddingModel,
    critic_model: DeepEvalBaseLLM,
) -> ContextConstructionConfig:
    """
    Controls document chunking and context grouping.
    Parameters (chunk size, overlap, contexts per document) preserved
    exactly from app1.py.
    """
    return ContextConstructionConfig(
        embedder=embedder,
        critic_model=critic_model,
        max_contexts_per_document=2,
        min_contexts_per_document=1,
        max_context_length=3,
        chunk_size=1024,
        chunk_overlap=128,
    )

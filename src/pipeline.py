"""
================================================================================
pipeline.py
================================================================================
Orchestrates the end-to-end run:

  1. Load a ModelConfig from YAML (provider/model selection).
  2. Build the LLM + local embedder.
  3. Turn tickets.csv into one document per row.
  4. For every (complexity tier x styling variant) combination -- 12 total --
     build a Synthesizer, call generate_goldens_from_docs() exactly as in
     app1.py, and save all 4 output formats.
  5. Wrap every batch's goldens in an EvaluationDataset and return the full
     set, so callers can push to Confident AI or inspect them directly.

This is the only module that depends on all the others; everything else is
independently reusable.
================================================================================
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, List

from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer

from config_loader import ModelConfig, load_model_config
from llm_wrappers import LiteLLMDeepEvalLLM, LocalEmbeddingModel
from document_builder import build_documents_from_csv
from synthesis_configs import (
    ComplexityTier,
    StyleVariant,
    build_evolution_config,
    build_styling_config,
    build_filtration_config,
    build_context_construction_config,
)
from golden_writer import save_goldens


class GoldenGenerationPipeline:
    """
    Single entry point for the whole golden-dataset-generation workflow.
    Each public method does one job (SOLID: single responsibility), and the
    class composes them so callers only need to call `run()`.
    """

    def __init__(
        self,
        model_config_path: str | Path,
        tickets_csv_path: str | Path,
        output_dir: str | Path = "outputs",
        documents_dir: str | Path = "outputs/ticket_documents",
        embed_model_name: str = "all-MiniLM-L6-v2",
        async_mode: bool = True,
        max_concurrent: int = 5,
        max_goldens_per_context: int = 2,
    ) -> None:
        self.model_config: ModelConfig = load_model_config(model_config_path)
        self.tickets_csv_path = Path(tickets_csv_path)
        self.output_dir = Path(output_dir)
        self.documents_dir = Path(documents_dir)
        self.async_mode = async_mode
        self.max_concurrent = max_concurrent
        self.max_goldens_per_context = max_goldens_per_context

        self.llm = LiteLLMDeepEvalLLM(config=self.model_config)
        self.embedder = LocalEmbeddingModel(model_name=embed_model_name)

        self._document_paths: List[str] | None = None

    # ── Step 1: documents ────────────────────────────────────────────────────

    def prepare_documents(self) -> List[str]:
        """Build (or reuse) the per-ticket documents. Cached after first call."""
        if self._document_paths is None:
            self._document_paths = build_documents_from_csv(
                csv_path=self.tickets_csv_path,
                output_dir=self.documents_dir,
            )
        return self._document_paths

    # ── Step 2: one (tier, style) combination ──────────────────────────────

    def run_combination(
        self,
        tier: ComplexityTier,
        style: StyleVariant,
    ) -> EvaluationDataset:
        """
        Run golden generation for exactly one complexity/style combination
        and persist all 4 output formats. Returns the resulting
        EvaluationDataset.
        """
        document_paths = self.prepare_documents()

        filtration_config = build_filtration_config(critic_model=self.llm)
        evolution_config = build_evolution_config(tier)
        styling_config = build_styling_config(style)
        context_construction_config = build_context_construction_config(
            embedder=self.embedder,
            critic_model=self.llm,
        )

        synthesizer = Synthesizer(
            model=self.llm,
            async_mode=self.async_mode,
            max_concurrent=self.max_concurrent,
            filtration_config=filtration_config,
            evolution_config=evolution_config,
            styling_config=styling_config,
        )

        print(f"\n>>> Generating goldens: complexity={tier.value}, style={style.value}")
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=document_paths,
            max_goldens_per_context=self.max_goldens_per_context,
            include_expected_output=True,
            context_construction_config=context_construction_config,
        )
        print(f"    Generated {len(goldens)} goldens.")

        file_name = f"goldens_{tier.value}_{style.value}"
        save_goldens(
            synthesizer=synthesizer,
            goldens=goldens,
            output_dir=self.output_dir,
            file_name=file_name,
        )

        return EvaluationDataset(goldens=synthesizer.synthetic_goldens)

    # ── Step 3: the full 3x4 grid ────────────────────────────────────────────

    def run_all_combinations(self) -> Dict[str, EvaluationDataset]:
        """
        Run every (tier, style) combination -- 3 tiers x 4 styles = 12 runs --
        and return a dict keyed by "<tier>_<style>" mapping to each
        resulting EvaluationDataset.
        """
        results: Dict[str, EvaluationDataset] = {}
        combinations = list(product(ComplexityTier, StyleVariant))
        print(f"\nRunning {len(combinations)} complexity x style combinations...")

        for tier, style in combinations:
            key = f"{tier.value}_{style.value}"
            results[key] = self.run_combination(tier, style)

        total_goldens = sum(len(ds.goldens) for ds in results.values())
        print(
            f"\nDone -- {len(results)} combinations, "
            f"{total_goldens} total goldens, saved under {self.output_dir}/"
        )
        return results

    # ── Convenience: a single combination by string (CLI-friendly) ────────

    def run(
        self,
        tier: str | None = None,
        style: str | None = None,
    ) -> Dict[str, EvaluationDataset] | EvaluationDataset:
        """
        If both `tier` and `style` are given, run just that one combination.
        Otherwise run the full 3x4 grid. This is the method most callers
        (including main.py) should use.
        """
        if tier is not None and style is not None:
            return self.run_combination(ComplexityTier(tier), StyleVariant(style))
        return self.run_all_combinations()

"""Prompt context for the loaded graph, built in-process.

This is the same sequence ``scripts/build_bundle.py`` runs -- vocabulary from
the IFC scan (+ IDS), the loaded graph's labels / relationship signatures /
written property keys folded in, then graph-sampled known values and
few-shots -- so a bundle is just this object serialised. Building it next to
the database at run time is what removes the bundle shuffle: switching model is
restore + rebuild, a few seconds each.

Order matters: graph stats are merged *before* sampling, so values and
few-shots see the full label set and none of the unwritten properties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.constraints.context_builder import ContextBuilder
from src.constraints.ids_parser import IDSSchema, parse_ids_file
from src.constraints.value_sampler import generate_few_shot_examples, sample_value_enumerations
from src.constraints.vocabulary_merger import (
    CombinedVocabulary,
    build_combined_vocabulary,
    get_system_schema_string,
    merge_graph_stats_into_vocabulary,
)
from src.etl.graph_cache import fingerprint, graph_summary, read_meta
from src.eval.neo4j_exec import collect_graph_stats

logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    vocabulary: CombinedVocabulary
    ids_schema: Optional[IDSSchema]
    graph_stats: Dict[str, Any]
    value_enumerations: Dict[str, List[str]]
    few_shot_examples: List[Dict[str, str]]
    graph: Dict[str, Any] = field(default_factory=dict)  # model_id, enrichment, fingerprint

    def schema_prompt(self, include_examples: bool = True) -> str:
        """Schema, relationship signatures, known values and examples as one block."""
        builder = ContextBuilder(self.vocabulary, self.few_shot_examples, self.value_enumerations)
        parts = [
            "## Available Schema\n\n" + get_system_schema_string(self.vocabulary),
            builder._build_known_values_section(),
        ]
        if include_examples:
            parts.append(builder._build_examples_section(max_examples=5))
        return "\n".join(p for p in parts if p)


def graph_identity(driver) -> Dict[str, Any]:
    """Which graph this context describes; warns if it changed since restore."""
    meta = read_meta(driver) or {}
    live = fingerprint(graph_summary(driver))
    if meta.get("fingerprint") not in (None, live):
        logger.warning("Graph changed since it was exported/restored as %s", meta.get("model_id"))
    return {
        "model_id": meta.get("model_id"),
        "enrichment": meta.get("enrichment"),
        "loader_rev": meta.get("loader_rev"),
        "fingerprint": live,
    }


def build_graph_context(
    driver,
    ifc_path: Path,
    ids_path: Optional[Path] = None,
) -> GraphContext:
    if ids_path is not None:
        logger.info("Building vocabulary from %s + %s", Path(ids_path).name, Path(ifc_path).name)
    else:
        logger.info("Building vocabulary from %s (no IDS)", Path(ifc_path).name)
    vocab = build_combined_vocabulary(ids_path=ids_path, ifc_path=ifc_path)
    ids_schema = parse_ids_file(ids_path) if ids_path is not None else None

    identity = graph_identity(driver)
    graph_stats = collect_graph_stats(driver)
    merge_graph_stats_into_vocabulary(vocab, graph_stats)
    value_enumerations = sample_value_enumerations(driver, vocab)
    few_shot_examples = generate_few_shot_examples(driver, vocab, enumerations=value_enumerations)
    return GraphContext(
        vocabulary=vocab,
        ids_schema=ids_schema,
        graph_stats=graph_stats,
        value_enumerations=value_enumerations,
        few_shot_examples=few_shot_examples,
        graph=identity,
    )

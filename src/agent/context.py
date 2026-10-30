"""Prompt context for the loaded graph, built in-process.

This is the same sequence ``scripts/build_bundle.py`` runs -- vocabulary from
the IFC scan (+ IDS), the loaded graph's labels / relationship signatures /
written property keys folded in, then graph-sampled known values and
few-shots -- so a bundle is just this object serialised. Building it next to
the database at run time is what removes the bundle shuffle: switching model is
restore + rebuild, a few seconds each.

Order matters: graph stats are merged *before* sampling, so values and
few-shots see the full label set and none of the unwritten properties.

The generic instance graph (state ``generic``) gets :class:`GenericContext`
instead: nothing of the above applies to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.prompts import GENERIC_SCHEMA
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


INSTANCE_LABELS = ("PrimaryNode", "ConnectionNode", "SecondaryNode", "InlineNode")


@dataclass
class GenericContext:
    """Prompt context of the generic instance graph (state ``generic``).

    Our builder above cannot describe it: vocabulary, known values and
    few-shots all assume a label per IFC class and typed edges, and here every
    node is one of four labels and every edge is ``:rel``. So the schema block
    is ECPPM's own description of this graph (``prompts.GENERIC_SCHEMA``,
    verbatim) plus, ours, the live ``EntityType`` counts per label -- the
    stand-in for the typed schema's label list, so the single-shot arm (which
    cannot run a discovery query) knows which classes exist.
    """
    entity_types: Dict[str, Dict[str, int]]   # label -> {EntityType: count}, most common first
    graph: Dict[str, Any] = field(default_factory=dict)
    graph_stats: Optional[Dict[str, Any]] = None  # no grammar is built from this graph

    def schema_prompt(self) -> str:
        lines = ["## Entity types in this graph", "",
                 "`EntityType` values per node label, with live node counts:"]
        for label in INSTANCE_LABELS:
            types = self.entity_types.get(label) or {}
            listed = ", ".join(f"{t} {c}" for t, c in types.items())
            lines.append(f"- `{label}` ({sum(types.values())}): {listed or '-'}")
        return GENERIC_SCHEMA + "\n\n" + "\n".join(lines)


def collect_entity_types(driver) -> Dict[str, Dict[str, int]]:
    """``EntityType`` counts per instance label (one pass over ``:Node``)."""
    out: Dict[str, Dict[str, int]] = {label: {} for label in INSTANCE_LABELS}
    with driver.session() as session:
        for r in session.run(
            "MATCH (n:Node) "
            "WITH [l IN labels(n) WHERE l IN $labels][0] AS label, n.EntityType AS type "
            "RETURN label, type, count(*) AS c ORDER BY label, c DESC, type",
            labels=list(INSTANCE_LABELS),
        ):
            if r["label"] is not None:
                out[r["label"]][str(r["type"])] = r["c"]
    return out


def build_generic_context(driver) -> GenericContext:
    return GenericContext(entity_types=collect_entity_types(driver), graph=graph_identity(driver))


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

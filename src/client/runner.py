"""API-driven experiment runner for the Colab side.

This is the counterpart to ``src.eval.runner.ExperimentRunner``: instead of
opening a Bolt connection to Neo4j, it drives the full test loop through the
authenticated FastAPI service, while the LLM still lives in the Colab runtime
(needed for Outlines-constrained decoding on a GPU).

Flow per setting:

1. Load the server-built ``bundle_<model>.json`` (vocabulary + IDS + model dump).
2. Rehydrate ``CombinedVocabulary`` and ``IDSSchema`` via ``from_dict``.
3. Build the LLM engine in-process via ``create_llm_engine``.
4. ``GET /test-set`` once.
5. For each case: ``engine.generate(...)`` → ``POST /evaluate`` → row.
6. Write per-setting CSV + JSON summary to ``output_dir``.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.client.api_client import ApiClient, ApiClientError
from src.config import ExperimentSetting, LLMProvider, Settings, get_settings
from src.constraints.ids_parser import IDSSchema
from src.constraints.vocabulary_merger import CombinedVocabulary
from src.eval.scoring import OutputType
from src.llm_engine import BaseLLMEngine, create_llm_engine

logger = logging.getLogger(__name__)


def _configure_logging(level: int = logging.INFO) -> None:
    """Idempotently install a stdout handler so Colab shows runner + engine logs."""
    root = logging.getLogger()
    has_runner_handler = any(
        getattr(h, "_chatdt_runner_handler", False) for h in root.handlers
    )
    if not has_runner_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        handler._chatdt_runner_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _maybe_tqdm(iterable, **kwargs):
    """Wrap iterable in tqdm if available; otherwise return as-is."""
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


CSV_FIELDS = [
    "index",
    "question",
    "experiment_setting",
    "output_type",
    "category",
    "difficulty",
    "model",
    "generated_cypher",
    "gold_cypher",
    "raw_output",
    "svr",
    "scr",
    "ea",
    "precision",
    "recall",
    "f1",
    "is_valid_syntax",
    "invalid_labels",
    "invalid_properties",
    "num_generated_results",
    "num_gold_results",
    "error",
]


@dataclass
class ApiRunnerConfig:
    """Configuration for an API-driven experiment run."""

    bundle_path: Path
    output_dir: Path = Path("results")
    name: str = ""
    settings: List[ExperimentSetting] = field(default_factory=list)

    def __post_init__(self):
        self.bundle_path = Path(self.bundle_path)
        self.output_dir = Path(self.output_dir)
        if not self.name:
            self.name = f"api_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not self.settings:
            self.settings = list(ExperimentSetting)


class ApiExperimentRunner:
    """Runs the 2x2 NL2Cypher experiment against a remote Chat.DT API."""

    def __init__(
        self,
        client: ApiClient,
        config: ApiRunnerConfig,
        app_settings: Optional[Settings] = None,
    ):
        self.client = client
        self.config = config
        self.settings = app_settings or get_settings()

        self.vocabulary: Optional[CombinedVocabulary] = None
        self.ids_schema: Optional[IDSSchema] = None
        self.model_dump: Optional[List[Dict[str, Any]]] = None
        self.model_context: Optional[str] = None
        self.engine: Optional[BaseLLMEngine] = None
        self.test_cases: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load bundle, rehydrate schemas, build the LLM engine, fetch test set."""
        _configure_logging()
        logger.info("setup: checking API health")
        self._check_api_health()
        logger.info("setup: loading bundle from %s", self.config.bundle_path)
        self._load_bundle()
        logger.info(
            "setup: building LLM engine (provider=%s model=%s) — this may "
            "download/load weights",
            self.settings.llm_provider.value,
            self.settings.llm_model_name,
        )
        self._build_engine()
        logger.info("setup: fetching test set from API")
        self.test_cases = self.client.get_test_set()
        if not self.test_cases:
            raise RuntimeError(
                "API returned an empty test set. Confirm the server has a "
                "data/test_set.csv or data/extended_test_set.json file mounted "
                "and that gold queries pre-executed at startup (see api lifespan logs)."
            )
        logger.info("Fetched %d test cases from API", len(self.test_cases))

    def _check_api_health(self) -> None:
        try:
            ready = self.client.health_ready()
        except ApiClientError as exc:
            raise RuntimeError(
                f"API readiness check failed ({exc.status_code}). Inspect the "
                f"response body: {exc.body}"
            ) from exc
        if not ready.get("neo4j", False):
            raise RuntimeError(
                f"API reports Neo4j is not ready: {ready.get('detail', '<no detail>')}"
            )

    def _load_bundle(self) -> None:
        if not self.config.bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {self.config.bundle_path}")
        with open(self.config.bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        if "combined_vocabulary" not in bundle:
            raise KeyError(
                f"Bundle {self.config.bundle_path} is missing 'combined_vocabulary'. "
                "Rebuild it with scripts/build_bundle.py."
            )
        self.vocabulary = CombinedVocabulary.from_dict(bundle["combined_vocabulary"])

        if "ids_schema" in bundle and isinstance(bundle["ids_schema"], dict):
            self.ids_schema = IDSSchema.from_dict(bundle["ids_schema"])
        else:
            self.ids_schema = self.vocabulary.ids_schema

        self.model_dump = bundle.get("model_dump") or []
        # Direct QA prompt context: serialize once, reuse across questions.
        self.model_context = (
            json.dumps(self.model_dump, indent=2) if self.model_dump else None
        )
        logger.info(
            "Bundle loaded: %d entities, %d properties, model_dump=%d elements",
            len(self.vocabulary.entities),
            len(self.vocabulary.all_properties),
            len(self.model_dump or []),
        )

    def _build_engine(self) -> None:
        wants_strict = any(s.uses_strict_constraints for s in self.config.settings)
        if wants_strict and self.settings.llm_provider != LLMProvider.LOCAL:
            logger.warning(
                "CYPHER_STRICT was requested but LLM provider is %s. Strict regex "
                "decoding requires the LOCAL provider; the run will fall back to "
                "soft constraints for that setting.",
                self.settings.llm_provider.value,
            )

        self.engine = create_llm_engine(
            provider=self.settings.llm_provider,
            settings=self.settings,
            schema=self.ids_schema,
            vocabulary=self.vocabulary,
            model_dump=self.model_dump,
        )
        # Initialize eagerly so failures (missing API key, model download) surface
        # before the first test question.
        if hasattr(self.engine, "initialize"):
            self.engine.initialize()
        logger.info(
            "LLM engine ready: provider=%s, model=%s",
            self.settings.llm_provider.value,
            self.settings.llm_model_name,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run_setting(self, setting: ExperimentSetting) -> List[Dict[str, Any]]:
        """Run all test cases for a single setting. Returns flat row dicts."""
        if self.engine is None:
            raise RuntimeError("Call setup() before run_setting().")

        output_type = OutputType.DIRECT_QA if setting.is_direct_qa else OutputType.CYPHER
        rows: List[Dict[str, Any]] = []
        total = len(self.test_cases)

        iterator = _maybe_tqdm(
            self.test_cases,
            total=total,
            desc=f"{setting.value}",
            unit="q",
            dynamic_ncols=True,
        )
        for case in iterator:
            idx = case.get("index")
            question = case.get("question", "")
            category = case.get("category", "")
            difficulty = case.get("difficulty", "")
            gold_cypher = case.get("gold_cypher", "")

            logger.info("[%s][%s/%s] %s", setting.value, idx + 1 if isinstance(idx, int) else "?", total, question[:60])

            generated_cypher: Optional[str] = None
            raw_output = ""
            try:
                gen = self.engine.generate(
                    question,
                    experiment_setting=setting,
                    model_context=self.model_context,
                )
                raw_output = gen.raw_output or ""
                if setting.is_cypher_gen:
                    generated_cypher = gen.query or ""
                    output = generated_cypher
                else:
                    if gen.raw_output:
                        output = gen.raw_output
                    elif gen.direct_answer is not None:
                        output = json.dumps(gen.direct_answer)
                    else:
                        output = "[]"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Generation failed for case %s", idx)
                rows.append(
                    self._error_row(
                        idx, question, setting, output_type, category, difficulty,
                        gold_cypher, error=f"generation_failed: {exc}",
                    )
                )
                continue

            try:
                resp = self.client.evaluate(
                    {
                        "question": question,
                        "output": output,
                        "output_type": output_type.value,
                        "gold_index": idx,
                        "experiment_setting": setting.value,
                    }
                )
            except ApiClientError as exc:
                logger.error("evaluate() failed for case %s: %s", idx, exc)
                rows.append(
                    self._error_row(
                        idx, question, setting, output_type, category, difficulty,
                        gold_cypher, error=f"evaluate_failed: {exc.status_code} {exc.body[:200]}",
                        generated_cypher=generated_cypher, raw_output=raw_output,
                    )
                )
                continue

            row = self._response_row(
                idx, setting, category, difficulty,
                generated_cypher=generated_cypher, raw_output=raw_output,
                response=resp,
            )
            rows.append(row)
            logger.info(
                "  SVR=%.0f SCR=%.2f EA=%.2f F1=%.2f",
                row["svr"], row["scr"], row["ea"], row["f1"],
            )

        return rows

    def run_comparison(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run every configured setting, persist results + summaries."""
        if self.engine is None:
            self.setup()

        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        all_rows: Dict[str, List[Dict[str, Any]]] = {}
        all_summaries: Dict[str, Dict[str, Any]] = {}

        for setting in self.config.settings:
            logger.info("=== Running setting: %s ===", setting.value)
            rows = self.run_setting(setting)
            all_rows[setting.value] = rows

            csv_path = out_dir / f"{self.config.name}_{setting.value}_results.csv"
            self._write_csv(csv_path, rows)

            summary = self._summarize(rows)
            all_summaries[setting.value] = summary
            summary_path = out_dir / f"{self.config.name}_{setting.value}_summary.json"
            self._write_summary(summary_path, setting, summary)

            logger.info(
                "%s done: SVR=%.2f SCR=%.2f EA=%.2f F1=%.2f (n=%d)",
                setting.value,
                summary["svr_mean"], summary["scr_mean"],
                summary["ea_mean"], summary["f1_mean"], summary["count"],
            )

        comparison_path = out_dir / f"{self.config.name}_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.settings.llm_model_name,
                    "provider": self.settings.llm_provider.value,
                    "test_cases": len(self.test_cases),
                    "settings": list(all_summaries.keys()),
                    "summaries": all_summaries,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info("Comparison report: %s", comparison_path)
        return all_rows

    # ------------------------------------------------------------------
    # Row builders
    # ------------------------------------------------------------------

    def _response_row(
        self,
        idx: Optional[int],
        setting: ExperimentSetting,
        category: str,
        difficulty: str,
        *,
        generated_cypher: Optional[str],
        raw_output: str,
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "index": idx,
            "question": response.get("question", ""),
            "experiment_setting": setting.value,
            "output_type": response.get("output_type", ""),
            "category": category,
            "difficulty": difficulty,
            "model": self.settings.llm_model_name,
            "generated_cypher": generated_cypher or response.get("generated_cypher") or "",
            "gold_cypher": response.get("gold_cypher") or "",
            "raw_output": (raw_output or "")[:500],
            "svr": float(response.get("svr", 0.0)),
            "scr": float(response.get("scr", 0.0)),
            "ea": float(response.get("ea", 0.0)),
            "precision": float(response.get("precision", 0.0)),
            "recall": float(response.get("recall", 0.0)),
            "f1": float(response.get("f1", 0.0)),
            "is_valid_syntax": bool(response.get("is_valid_syntax", False)),
            "invalid_labels": ",".join(response.get("invalid_labels") or []),
            "invalid_properties": ",".join(response.get("invalid_properties") or []),
            "num_generated_results": len(response.get("generated_ids") or []),
            "num_gold_results": len(response.get("gold_ids") or []),
            "error": response.get("error") or "",
        }

    def _error_row(
        self,
        idx: Optional[int],
        question: str,
        setting: ExperimentSetting,
        output_type: OutputType,
        category: str,
        difficulty: str,
        gold_cypher: str,
        *,
        error: str,
        generated_cypher: Optional[str] = None,
        raw_output: str = "",
    ) -> Dict[str, Any]:
        return {
            "index": idx,
            "question": question,
            "experiment_setting": setting.value,
            "output_type": output_type.value,
            "category": category,
            "difficulty": difficulty,
            "model": self.settings.llm_model_name,
            "generated_cypher": generated_cypher or "",
            "gold_cypher": gold_cypher or "",
            "raw_output": (raw_output or "")[:500],
            "svr": 0.0, "scr": 0.0, "ea": 0.0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "is_valid_syntax": False,
            "invalid_labels": "", "invalid_properties": "",
            "num_generated_results": 0, "num_gold_results": 0,
            "error": error,
        }

    # ------------------------------------------------------------------
    # Persistence + aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        logger.info("CSV written: %s", path)

    def _write_summary(
        self, path: Path, setting: ExperimentSetting, summary: Dict[str, Any]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "setting": setting.value,
                        "model": self.settings.llm_model_name,
                        "provider": self.settings.llm_provider.value,
                        "test_cases": len(self.test_cases),
                    },
                    "metrics": summary,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info("Summary written: %s", path)

    @staticmethod
    def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"count": 0}

        def mean(key: str) -> float:
            return sum(float(r.get(key, 0.0) or 0.0) for r in rows) / n

        valid_syntax = sum(1 for r in rows if r.get("is_valid_syntax"))
        errors = sum(1 for r in rows if r.get("error"))

        # Non-trivial subset: gold returned at least one row. Excludes cases
        # where empty gold + empty pred trivially scores EA=1.0.
        nt_rows = [r for r in rows if int(r.get("num_gold_results", 0) or 0) > 0]
        nt_n = len(nt_rows)

        def nt_mean(key: str) -> float:
            if nt_n == 0:
                return 0.0
            return sum(float(r.get(key, 0.0) or 0.0) for r in nt_rows) / nt_n

        by_difficulty: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            diff = r.get("difficulty") or "unknown"
            bucket = by_difficulty.setdefault(diff, {"count": 0, "ea_sum": 0.0, "f1_sum": 0.0})
            bucket["count"] += 1
            bucket["ea_sum"] += float(r.get("ea", 0.0) or 0.0)
            bucket["f1_sum"] += float(r.get("f1", 0.0) or 0.0)
        metrics_by_difficulty = {
            diff: {
                "count": b["count"],
                "ea_mean": b["ea_sum"] / b["count"] if b["count"] else 0.0,
                "f1_mean": b["f1_sum"] / b["count"] if b["count"] else 0.0,
            }
            for diff, b in by_difficulty.items()
        }

        return {
            "count": n,
            "svr_mean": mean("svr"),
            "scr_mean": mean("scr"),
            "ea_mean": mean("ea"),
            "precision_mean": mean("precision"),
            "recall_mean": mean("recall"),
            "f1_mean": mean("f1"),
            "syntax_valid_count": valid_syntax,
            "error_count": errors,
            "non_trivial_count": nt_n,
            "ea_mean_non_trivial": nt_mean("ea"),
            "f1_mean_non_trivial": nt_mean("f1"),
            "precision_mean_non_trivial": nt_mean("precision"),
            "recall_mean_non_trivial": nt_mean("recall"),
            "metrics_by_difficulty": metrics_by_difficulty,
        }

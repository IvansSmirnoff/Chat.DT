"""
Experiment runner: orchestrates a single NL2Cypher experimental setting
or a full comparison across all four settings.

The argparse CLI that drives this lives in ``src.eval.cli``.
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from neo4j import GraphDatabase

from src.config import get_settings, Settings, ExperimentSetting, LLMProvider
from src.llm_engine import (
    create_llm_engine,
    BaseLLMEngine,
    GeminiLLMEngine,
)
from src.constraints.ids_parser import IDSParser, IDSSchema
from src.constraints.value_sampler import (
    generate_few_shot_examples,
    sample_value_enumerations,
)
from src.constraints.vocabulary_merger import CombinedVocabulary, build_combined_vocabulary
from src.eval.scoring import EvaluationResult, OutputType
from src.eval.neo4j_exec import execute_gold_queries
from src.eval.aggregate import (
    evaluate_output,
    aggregate_results,
    aggregate_by_category,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    # Experiment identity
    name: str = ""
    description: str = ""
    
    # Settings to run
    settings: List[ExperimentSetting] = field(default_factory=list)
    
    # Paths
    test_set_path: Path = Path("data/test_set.csv")
    output_dir: Path = Path("results")
    model_dump_path: Optional[Path] = None
    
    # Neo4j connection (will use settings if not provided)
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    
    # Direct QA mode settings
    model_context: Optional[str] = None  # Pre-loaded context string

    # Cloud override
    cloud_direct: bool = False  # Force Settings 1-3 to Gemini 1.5 Flash

    # Sanity passthrough — skip the LLM entirely and feed the gold Cypher (or
    # serialised gold rows for Direct QA) as the "predicted" output. Used to
    # prove the scoring pipeline gives EA=1 on gold; any deviation is a bug in
    # scoring, not in the model.
    gold_as_prediction: bool = False

    def __post_init__(self):
        if not self.name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.name = f"experiment_{timestamp}"
        
        if not self.settings:
            # Default to running all settings
            self.settings = list(ExperimentSetting)


# =============================================================================
# Test set loading / result persistence helpers
# =============================================================================

def load_test_set(file_path: Path) -> List[Dict[str, str]]:
    """
    Load a test set from CSV or JSON.

    CSV expects columns ``Question``, ``Gold_Cypher`` (case-insensitive),
    optional ``Category`` and ``Difficulty``. JSON expects a list of objects
    with keys ``question``, ``gold_cypher`` (or any of the legacy aliases),
    optional ``category``, ``difficulty``.

    Args:
        file_path: Path to ``.csv`` or ``.json`` test-set file.

    Returns:
        List of test case dictionaries with the keys
        ``question, gold_cypher, category, difficulty``.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    raw_rows: List[Dict[str, Any]] = []

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "cases" in payload:
            payload = payload["cases"]
        if not isinstance(payload, list):
            raise ValueError(
                f"Test set JSON at {path} must be a list of objects (or "
                f"{{'cases': [...]}}); got {type(payload).__name__}."
            )
        raw_rows = list(payload)
    else:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_rows = list(reader)

    test_cases: List[Dict[str, str]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        case: Dict[str, str] = {}
        for key, value in row.items():
            normalized_key = str(key).strip().lower().replace(" ", "_")
            case[normalized_key] = str(value).strip() if value is not None else ""

        test_case = {
            "question": case.get("question", case.get("nl_query", "")),
            "gold_cypher": case.get(
                "gold_cypher", case.get("cypher", case.get("expected_cypher", ""))
            ),
            "category": case.get("category", ""),
            "difficulty": case.get("difficulty", ""),
        }

        if test_case["question"]:
            test_cases.append(test_case)

    logger.info(f"Loaded {len(test_cases)} test cases from {path}")
    return test_cases


def save_results_csv(
    results: List[EvaluationResult],
    output_path: Path,
) -> None:
    """Save evaluation results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all possible fields from to_dict
    if results:
        fieldnames = list(results[0].to_dict().keys())
    else:
        fieldnames = ["question", "experiment_setting", "output_type", "svr", "scr", "ea", "error"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow(result.to_dict())
    
    logger.info(f"Results saved to {output_path}")


def save_summary(
    summary: Dict[str, Any],
    output_path: Path,
    experiment_config: Dict[str, Any],
) -> None:
    """Save experiment summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": experiment_config,
        "metrics": summary,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Summary saved to {output_path}")



def save_comparison_report(
    all_results: Dict[ExperimentSetting, List[EvaluationResult]],
    output_path: Path,
    experiment_config: Dict[str, Any],
    model_name: Optional[str] = None,
) -> None:
    """
    Save comparison report across all settings.
    
    The report format is designed for multi-model comparison,
    allowing easy merging of results from different model runs.
    
    Args:
        all_results: Dict mapping setting to list of results
        output_path: Path for output JSON file
        experiment_config: Configuration metadata
        model_name: Optional model name override
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Aggregate by setting with difficulty breakdown
    comparison = {}
    complexity_results = {}
    category_results = {}
    
    for setting, results in all_results.items():
        summary = aggregate_results(results)
        comparison[setting.value] = summary
        
        # Extract difficulty metrics for this setting
        if "metrics_by_difficulty" in summary:
            complexity_results[setting.value] = summary["metrics_by_difficulty"]
        
        # Calculate category breakdown
        category_results[setting.value] = aggregate_by_category(results)
    
    # Build 2x2 matrix representation
    setting_matrix = {
        "direct_qa": {
            "baseline": comparison.get("direct_qa_baseline", {}),
            "grounded": comparison.get("direct_qa_grounded", {}),
        },
        "cypher": {
            "soft": comparison.get("cypher_soft", {}),
            "strict": comparison.get("cypher_strict", {}),
        }
    }
    
    # Determine model name
    resolved_model_name = model_name or experiment_config.get("model", "unknown")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": resolved_model_name,
        "config": experiment_config,
        "setting_results": comparison,
        "setting_matrix": setting_matrix,
        "complexity_results": complexity_results,
        "category_results": category_results,
        "ranking": _rank_settings(comparison),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Comparison report saved to {output_path}")


def _rank_settings(comparison: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Rank settings by each metric."""
    rankings = {}
    
    for metric in ["ea_mean", "svr_mean", "scr_mean", "f1_mean"]:
        sorted_settings = sorted(
            comparison.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )
        rankings[metric] = [s[0] for s in sorted_settings]
    
    return rankings


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """
    Runs NL2Cypher comparative evaluation experiments.
    
    Supports running individual settings or full 2x2 comparison experiments.
    
    Example:
        ```python
        # Run single setting
        runner = ExperimentRunner()
        results = runner.run_setting(
            ExperimentSetting.CYPHER_SOFT,
            test_set_path="data/test_set.csv"
        )
        
        # Run full comparison
        runner = ExperimentRunner()
        all_results = runner.run_comparison(
            test_set_path="data/test_set.csv",
            output_dir="results/"
        )
        ```
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[ExperimentConfig] = None,
    ):
        """
        Initialize the experiment runner.
        
        Args:
            settings: Application settings
            config: Experiment configuration
        """
        self.settings = settings or get_settings()
        self.config = config or ExperimentConfig()
        
        self.driver = None
        self.engine: Optional[BaseLLMEngine] = None
        self.schema: Optional[IDSSchema] = None
        self.vocabulary: Optional[CombinedVocabulary] = None
        
        # Pre-computed gold results
        self.gold_id_sets: Dict[int, Set[str]] = {}
        self.test_cases: List[Dict[str, str]] = []
        
        # Valid labels/properties for SCR
        self.valid_labels: Set[str] = set()
        self.valid_properties: Set[str] = set()
        
        # Model context for Direct QA
        self.model_context: Optional[str] = None
        self.model_dump: Optional[Dict[str, Any]] = None

        # Graph-sourced prompt context (sampled in setup()).
        self.value_enumerations: Dict[str, List[str]] = {}
        self.few_shot_examples: List[Dict[str, str]] = []
    
    def setup(self) -> None:
        """Initialize database connection and load schemas."""
        logger.info("Setting up experiment runner...")

        # Cloud override: ensure provider is Gemini, use model from .env
        if self.config.cloud_direct:
            self.settings = self.settings.model_copy(update={
                "llm_provider": LLMProvider.GEMINI,
            })
            logger.info(
                f"Cloud override active: provider={self.settings.llm_provider.value}, "
                f"model={self.settings.llm_model_name}"
            )

        # Connect to Neo4j
        neo4j_uri = self.config.neo4j_uri or self.settings.neo4j_uri
        neo4j_user = self.config.neo4j_user or self.settings.neo4j_user
        neo4j_password = self.config.neo4j_password or self.settings.neo4j_password_value
        
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        self.driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
        
        # Load IDS schema
        ids_path = self.settings.ids_file_path
        if Path(ids_path).exists():
            parser = IDSParser()
            self.schema = parser.parse(ids_path)
            logger.info(
                f"Loaded IDS schema: {len(self.schema.entities)} entities, "
                f"{len(self.schema.properties)} properties"
            )
        else:
            logger.warning(f"IDS file not found: {ids_path}")
            self.schema = IDSSchema()
        
        # Load combined vocabulary
        ifc_path = self.settings.ifc_file_path
        if Path(ifc_path).exists():
            try:
                ids_arg = ids_path if Path(ids_path).exists() else None
                self.vocabulary = build_combined_vocabulary(ids_arg, ifc_path)
                logger.info(
                    f"Loaded vocabulary: {len(self.vocabulary.get_entity_names())} entities, "
                    f"{len(self.vocabulary.get_all_property_names())} properties"
                )
            except Exception as e:
                logger.warning(f"Could not load vocabulary: {e}")
        
        # Build valid labels/properties for SCR
        if self.vocabulary:
            self.valid_labels = self.vocabulary.get_entity_names()
            self.valid_properties = self.vocabulary.get_all_property_names()
        elif self.schema:
            self.valid_labels = self.schema.get_neo4j_labels()
            self.valid_properties = self.schema.properties

        # Graph-sourced prompt context: sample categorical values + few-shots
        # straight from the live driver so the local CLI matches the bundle
        # path used in Colab.
        if self.vocabulary:
            try:
                self.value_enumerations = sample_value_enumerations(
                    self.driver, self.vocabulary
                )
                self.few_shot_examples = generate_few_shot_examples(
                    self.driver,
                    self.vocabulary,
                    enumerations=self.value_enumerations,
                )
                logger.info(
                    f"Graph-sourced prompt context: "
                    f"{len(self.value_enumerations)} enumerations, "
                    f"{len(self.few_shot_examples)} few-shot examples"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Failed to sample graph-sourced prompt context: {exc}. "
                    "Continuing with empty enumerations / few-shots."
                )
                self.value_enumerations = {}
                self.few_shot_examples = []
        
        # Add common properties
        self.valid_properties.update({
            "GlobalId", "Name", "Description", "ObjectType"
        })
        
        # Model dump loading
        model_dump_path = self.config.model_dump_path or self.settings.model_dump_path
        if model_dump_path and Path(model_dump_path).exists():
            with open(model_dump_path, 'r') as f:
                self.model_dump = json.load(f)
            self.model_context = json.dumps(self.model_dump, indent=2)
            logger.info(
                f"Model dump loaded from {model_dump_path} "
                f"({len(self.model_context):,} chars)"
            )
        elif self.config.cloud_direct:
            raise ValueError(
                "--model-dump is required for --cloud-direct mode. "
                "Provide the path to model_dump.json containing the full BIM model data."
            )
        else:
            self.model_context = None
            logger.info("No model dump provided")
    
    def teardown(self) -> None:
        """Clean up resources."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def load_test_set(self, test_set_path: Path) -> None:
        """Load test cases and pre-execute gold queries."""
        self.test_cases = load_test_set(test_set_path)
        
        # Pre-execute gold queries to get gold ID sets
        logger.info("Pre-executing gold queries to establish gold standard...")
        self.gold_id_sets = execute_gold_queries(
            self.test_cases,
            self.driver,
            id_property="GlobalId"
        )
        
        # Log statistics
        total_gold_ids = sum(len(ids) for ids in self.gold_id_sets.values())
        non_empty = sum(1 for ids in self.gold_id_sets.values() if ids)
        logger.info(
            f"Gold standard: {total_gold_ids} total IDs across {non_empty}/{len(self.test_cases)} test cases"
        )
    
    def _create_engine_for_setting(
        self,
        setting: ExperimentSetting,
    ) -> BaseLLMEngine:
        """Create and configure an LLM engine for the given setting."""
        # Cloud override: use GeminiLLMEngine with context caching for Direct QA
        if self.config.cloud_direct and setting.is_direct_qa:
            logger.info(
                f"Cloud override: creating GeminiLLMEngine(use_context_cache=True) "
                f"for {setting.value}"
            )
            engine = GeminiLLMEngine(
                settings=self.settings,
                schema=self.schema,
                vocabulary=self.vocabulary,
                model_dump=self.model_dump,
                use_context_cache=True,
                few_shot_examples=self.few_shot_examples or None,
                value_enumerations=self.value_enumerations or None,
            )
            engine.initialize()
            return engine

        engine = create_llm_engine(
            settings=self.settings,
            schema=self.schema,
            vocabulary=self.vocabulary,
            model_dump=self.model_dump,
            few_shot_examples=self.few_shot_examples or None,
            value_enumerations=self.value_enumerations or None,
        )

        # For strict mode, prefer local engine if available
        if setting.uses_strict_constraints:
            if self.settings.llm_provider != LLMProvider.LOCAL:
                logger.warning(
                    f"Setting {setting.value} requires LOCAL provider for strict constraints. "
                    f"Using {self.settings.llm_provider.value} with soft constraints instead."
                )

        engine.initialize()
        return engine
    
    def run_setting(
        self,
        setting: ExperimentSetting,
        output_dir: Optional[Path] = None,
    ) -> List[EvaluationResult]:
        """
        Run experiment for a single setting.
        
        Args:
            setting: Experimental setting to run
            output_dir: Directory for output files
            
        Returns:
            List of EvaluationResult
        """
        logger.info(f"Running setting: {setting.value}")
        
        # Determine output type
        output_type = OutputType.DIRECT_QA if setting.is_direct_qa else OutputType.CYPHER

        # Create engine for this setting — skipped entirely in gold-passthrough
        # mode so we don't pay for model init when the LLM output is never used.
        engine = (
            None
            if self.config.gold_as_prediction
            else self._create_engine_for_setting(setting)
        )

        results = []
        total = len(self.test_cases)

        for i, case in enumerate(self.test_cases):
            question = case["question"]
            gold_cypher = case.get("gold_cypher")
            gold_ids = self.gold_id_sets.get(i, set())

            logger.info(f"[{i+1}/{total}] Processing: {question[:50]}...")

            # Generate output
            if self.config.gold_as_prediction:
                if setting.is_cypher_gen:
                    output = gold_cypher or ""
                else:
                    # Direct QA: emit the pre-tokenized gold IDs as a JSON list;
                    # the scorer will coerce each element through
                    # ``_coerce_direct_qa_token`` and recover the gold tokens.
                    output = json.dumps(sorted(gold_ids))
            else:
                try:
                    gen_result = engine.generate(
                        question,
                        experiment_setting=setting,
                        model_context=self.model_context,
                    )

                    if setting.is_cypher_gen:
                        output = gen_result.query or ""
                    else:
                        # For Direct QA, use raw output or serialize the answer
                        if gen_result.raw_output:
                            output = gen_result.raw_output
                        elif gen_result.direct_answer is not None:
                            output = json.dumps(gen_result.direct_answer)
                        else:
                            output = "[]"

                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    result = EvaluationResult(
                        question=question,
                        experiment_setting=setting,
                        output_type=output_type,
                        gold_ids=gold_ids,
                        gold_cypher=gold_cypher,
                        error=f"Generation failed: {e}",
                    )
                    results.append(result)
                    continue
            
            # Evaluate output
            result = evaluate_output(
                question=question,
                output=output,
                output_type=output_type,
                gold_ids=gold_ids,
                driver=self.driver,
                valid_labels=self.valid_labels,
                valid_properties=self.valid_properties,
                experiment_setting=setting,
                gold_cypher=gold_cypher,
            )
            
            # Add metadata
            result.metadata["category"] = case.get("category", "")
            result.metadata["difficulty"] = case.get("difficulty", "")
            result.metadata["model"] = self.settings.llm_model_name
            
            results.append(result)
            
            logger.info(
                f"  SVR={result.svr:.0f}, SCR={result.scr:.2f}, EA={result.ea:.2f}"
            )
        
        # Save results if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            results_path = output_dir / f"{self.config.name}_{setting.value}_results.csv"
            save_results_csv(results, results_path)
            
            summary = aggregate_results(results)
            summary_path = output_dir / f"{self.config.name}_{setting.value}_summary.json"
            save_summary(summary, summary_path, {
                "setting": setting.value,
                "model": self.settings.llm_model_name,
                "test_cases": len(self.test_cases),
            })
        
        return results
    
    def _get_sanitized_model_name(self) -> str:
        """Get a filesystem-safe model name for use in file paths."""
        model_name = self.settings.llm_model_name
        # Remove path separators and special characters
        safe_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        # Truncate if too long
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        return safe_name
    
    def run_comparison(
        self,
        test_set_path: Path,
        output_dir: Path,
        settings: Optional[List[ExperimentSetting]] = None,
    ) -> Dict[ExperimentSetting, List[EvaluationResult]]:
        """
        Run full comparison experiment across multiple settings.
        
        Args:
            test_set_path: Path to test set CSV
            output_dir: Directory for output files
            settings: List of settings to run (default: all 4)
            
        Returns:
            Dict mapping setting to list of results
        """
        settings = settings or list(ExperimentSetting)
        
        logger.info("=" * 60)
        logger.info("STARTING COMPARISON EXPERIMENT")
        logger.info("=" * 60)
        logger.info(f"Settings to run: {[s.value for s in settings]}")
        logger.info(f"Test set: {test_set_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model: {self.settings.llm_model_name}")
        
        # Setup
        self.setup()
        
        try:
            # Load test set and gold standards
            self.load_test_set(test_set_path)
            
            # Run each setting
            all_results: Dict[ExperimentSetting, List[EvaluationResult]] = {}
            
            for setting in settings:
                logger.info("\n" + "-" * 60)
                logger.info(f"SETTING: {setting.value}")
                logger.info("-" * 60)
                
                results = self.run_setting(setting, output_dir)
                all_results[setting] = results
                
                # Print setting summary
                summary = aggregate_results(results)
                logger.info(f"\n{setting.value} Results:")
                logger.info(f"  SVR: {summary['svr_mean']:.2%}")
                logger.info(f"  SCR: {summary['scr_mean']:.2%}")
                logger.info(f"  EA: {summary['ea_mean']:.2%}")
                logger.info(f"  F1: {summary['f1_mean']:.2%}")
                
                # Print difficulty breakdown if available
                if "metrics_by_difficulty" in summary:
                    logger.info("  Complexity Analysis (EA by Difficulty):")
                    for diff_level, diff_metrics in summary["metrics_by_difficulty"].items():
                        logger.info(f"    {diff_level}: EA={diff_metrics['ea_mean']:.2%} (n={diff_metrics['count']})")
            
            # Save comparison report with unique model-based naming
            safe_model_name = self._get_sanitized_model_name()
            timestamp = self.config.name.split("_")[-1] if "_" in self.config.name else datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_filename = f"results_{safe_model_name}_{timestamp}.json"
            comparison_path = output_dir / comparison_filename
            
            save_comparison_report(
                all_results, 
                comparison_path, 
                {
                    "settings": [s.value for s in settings],
                    "model": self.settings.llm_model_name,
                    "test_cases": len(self.test_cases),
                    "provider": self.settings.llm_provider.value,
                },
                model_name=self.settings.llm_model_name,
            )
            
            # Also save with the legacy filename for backward compatibility
            legacy_path = output_dir / f"{self.config.name}_comparison.json"
            save_comparison_report(
                all_results, 
                legacy_path, 
                {
                    "settings": [s.value for s in settings],
                    "model": self.settings.llm_model_name,
                    "test_cases": len(self.test_cases),
                },
                model_name=self.settings.llm_model_name,
            )
            
            # Print final comparison
            self._print_comparison_summary(all_results)
            
            return all_results
            
        finally:
            self.teardown()
    
    def _print_comparison_summary(
        self,
        all_results: Dict[ExperimentSetting, List[EvaluationResult]]
    ) -> None:
        """Print comparison summary table."""
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 80)
        
        # Header
        header = f"{'Setting':<25} {'SVR':>8} {'SCR':>8} {'EA':>8} {'F1':>8} {'Valid':>8}"
        logger.info(header)
        logger.info("-" * 80)
        
        # Rows
        for setting, results in all_results.items():
            summary = aggregate_results(results)
            row = (
                f"{setting.value:<25} "
                f"{summary['svr_mean']:>7.1%} "
                f"{summary['scr_mean']:>7.1%} "
                f"{summary['ea_mean']:>7.1%} "
                f"{summary['f1_mean']:>7.1%} "
                f"{summary['syntax_valid_count']:>4}/{summary['count']:<3}"
            )
            logger.info(row)
        
        logger.info("=" * 80)
        
        # Best performers
        ea_best = max(all_results.items(), key=lambda x: aggregate_results(x[1])['ea_mean'])
        logger.info(f"\nBest EA: {ea_best[0].value} ({aggregate_results(ea_best[1])['ea_mean']:.1%})")


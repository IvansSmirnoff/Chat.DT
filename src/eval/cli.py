"""
CLI entry point for running NL2Cypher evaluation experiments.

Usage:
    python -m src.eval.cli --test-set data/test_set.csv --setting cypher_soft
    python -m src.eval.cli --test-set data/test_set.csv --all-settings
    python -m src.eval.cli --test-set data/test_set.csv --comparison
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import ExperimentSetting, get_settings
from src.eval.aggregate import aggregate_results
from src.eval.runner import ExperimentConfig, ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run NL2Cypher comparative evaluation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single setting
  python -m src.eval.cli --test-set data/test_set.csv --setting cypher_soft
  
  # Run all 4 settings
  python -m src.eval.cli --test-set data/test_set.csv --all-settings
  
  # Run comparison (all settings)
  python -m src.eval.cli --test-set data/test_set.csv --comparison
  
Settings:
  direct_qa_baseline  - Direct QA without IDS grounding
  direct_qa_grounded  - Direct QA with IDS grounding
  cypher_soft         - Cypher generation with soft constraints
  cypher_strict       - Cypher generation with strict (grammar) constraints
        """
    )
    
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("data/test_set.csv"),
        help="Path to test set CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=[s.value for s in ExperimentSetting],
        help="Single setting to run",
    )
    parser.add_argument(
        "--all-settings",
        action="store_true",
        help="Run all 4 experimental settings",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run full comparison experiment across all settings",
    )
    parser.add_argument(
        "--model-dump",
        type=Path,
        default=None,
        help="Path to model dump JSON for Direct QA mode",
    )
    parser.add_argument(
        "--cloud-direct",
        action="store_true",
        help=(
            "Force Settings 1-3 to use Gemini 1.5 Flash with context caching. "
            "Requires --model-dump for Direct QA settings. "
            "CYPHER_STRICT (Setting 4) is auto-skipped (incompatible with remote API)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verify test set exists
    if not args.test_set.exists():
        logger.error(f"Test set not found: {args.test_set}")
        sys.exit(1)
    
    # Create config
    config = ExperimentConfig(
        name=args.name or "",
        test_set_path=args.test_set,
        output_dir=args.output,
        model_dump_path=args.model_dump,
    )
    
    # Determine which settings to run
    if args.comparison or args.all_settings:
        settings = list(ExperimentSetting)
    elif args.setting:
        settings = [ExperimentSetting(args.setting)]
    else:
        # Default to CYPHER_SOFT for backward compatibility
        settings = [ExperimentSetting.CYPHER_SOFT]

    # Cloud override validation and setting filtering
    if args.cloud_direct:
        config.cloud_direct = True

        # Validate: --model-dump is required if any Direct QA setting is included
        has_direct_qa = any(s.is_direct_qa for s in settings)
        if has_direct_qa and not args.model_dump:
            logger.error(
                "--model-dump is required when --cloud-direct includes "
                "Direct QA settings (direct_qa_baseline, direct_qa_grounded)"
            )
            sys.exit(1)

        # Validate: CYPHER_STRICT is incompatible with cloud mode
        if ExperimentSetting.CYPHER_STRICT in settings:
            if args.setting == ExperimentSetting.CYPHER_STRICT.value:
                # User explicitly requested only CYPHER_STRICT — that's an error
                logger.error(
                    "CYPHER_STRICT is incompatible with --cloud-direct. "
                    "Strict constrained decoding requires a local model with Outlines."
                )
                sys.exit(1)
            else:
                # Auto-skip CYPHER_STRICT from multi-setting runs
                settings = [
                    s for s in settings
                    if s != ExperimentSetting.CYPHER_STRICT
                ]
                logger.warning(
                    "CYPHER_STRICT auto-skipped in --cloud-direct mode "
                    "(incompatible with remote API, requires Outlines constrained decoding)"
                )

        logger.info(
            f"Cloud override enabled: using {get_settings().llm_model_name} for settings "
            f"{[s.value for s in settings]}"
        )

    config.settings = settings
    
    # Create runner
    runner = ExperimentRunner(config=config)
    
    try:
        if args.comparison or args.all_settings:
            runner.run_comparison(
                test_set_path=args.test_set,
                output_dir=args.output,
                settings=settings,
            )
        else:
            # Single setting run
            runner.setup()
            try:
                runner.load_test_set(args.test_set)
                results = runner.run_setting(settings[0], args.output)
                
                # Print summary
                summary = aggregate_results(results)
                logger.info("\n" + "=" * 60)
                logger.info("EXPERIMENT SUMMARY")
                logger.info("=" * 60)
                logger.info(f"Setting: {settings[0].value}")
                logger.info(f"Total test cases: {summary['count']}")
                logger.info(f"SVR (Syntactic Validity): {summary['svr_mean']:.2%}")
                logger.info(f"SCR (Schema Compliance): {summary['scr_mean']:.2%}")
                logger.info(f"EA (Execution Accuracy): {summary['ea_mean']:.2%}")
                logger.info(f"F1 Score: {summary['f1_mean']:.2%}")
                logger.info(f"Syntax valid: {summary['syntax_valid_count']}/{summary['count']}")
                logger.info("=" * 60)
            finally:
                runner.teardown()
        
        logger.info(f"\nExperiment complete. Results saved to {args.output}/")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    main()

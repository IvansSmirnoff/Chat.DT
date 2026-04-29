"""Tests for the Colab-side ApiClient + ApiExperimentRunner.

Uses ``httpx.MockTransport`` to drive the API surface without a live server,
and a fake LLM engine to skip torch/outlines.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from src.client.api_client import ApiClient, ApiClientError
from src.client.runner import ApiExperimentRunner, ApiRunnerConfig
from src.config import ExperimentSetting, LLMProvider, Settings
from src.llm_engine import GenerationResult


# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------

class FakeEngine:
    """Stand-in for BaseLLMEngine — returns a canned GenerationResult per call."""

    def __init__(self, scripted: List[GenerationResult]):
        self._scripted = list(scripted)
        self.calls: List[Dict[str, Any]] = []

    def initialize(self):
        pass

    def generate(self, user_query: str, experiment_setting, model_context=None):
        self.calls.append(
            {
                "query": user_query,
                "setting": experiment_setting,
                "has_context": model_context is not None,
            }
        )
        return self._scripted.pop(0)


def _build_client(handler) -> ApiClient:
    transport = httpx.MockTransport(handler)
    client = ApiClient("http://test.invalid", "test-token")
    client._client.close()
    client._client = httpx.Client(
        base_url=client.base_url,
        timeout=10.0,
        headers={"Authorization": "Bearer test-token"},
        transport=transport,
    )
    return client


# -----------------------------------------------------------------------------
# ApiClient
# -----------------------------------------------------------------------------

def test_api_client_sends_bearer_header_and_parses_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["path"] = request.url.path
        return httpx.Response(200, json={"count": 1, "cases": [{"index": 0, "question": "Q"}]})

    client = _build_client(handler)
    cases = client.get_test_set()
    assert seen["auth"] == "Bearer test-token"
    assert seen["path"] == "/test-set"
    assert cases == [{"index": 0, "question": "Q"}]


def test_api_client_raises_on_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="missing auth header")

    client = _build_client(handler)
    with pytest.raises(ApiClientError) as exc:
        client.get_test_set()
    assert exc.value.status_code == 401
    assert "missing auth header" in str(exc.value)


# -----------------------------------------------------------------------------
# ApiExperimentRunner
# -----------------------------------------------------------------------------

def _evaluate_response(question: str, setting: ExperimentSetting, **overrides) -> Dict[str, Any]:
    base = {
        "question": question,
        "output_type": "cypher" if setting.is_cypher_gen else "direct_qa",
        "experiment_setting": setting.value,
        "generated_cypher": overrides.get("generated_cypher"),
        "gold_cypher": "MATCH (n:IfcWall) RETURN n.GlobalId",
        "generated_ids": ["abc-123"],
        "gold_ids": ["abc-123"],
        "svr": 1.0,
        "scr": 1.0,
        "ea": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "is_valid_syntax": True,
        "invalid_labels": [],
        "invalid_properties": [],
        "error": None,
        "metadata": {},
    }
    base.update(overrides)
    return base


def test_run_setting_posts_evaluate_with_correct_payload(tmp_path):
    posted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/test-set":
            return httpx.Response(
                200,
                json={
                    "count": 1,
                    "cases": [
                        {
                            "index": 0,
                            "question": "How many walls?",
                            "gold_cypher": "MATCH (n:IfcWall) RETURN n.GlobalId",
                            "category": "count",
                            "difficulty": "easy",
                        }
                    ],
                },
            )
        if request.url.path == "/evaluate":
            payload = json.loads(request.content.decode())
            posted.append(payload)
            return httpx.Response(
                200,
                json=_evaluate_response(
                    payload["question"],
                    ExperimentSetting(payload["experiment_setting"]),
                    generated_cypher=payload["output"],
                ),
            )
        return httpx.Response(404)

    client = _build_client(handler)
    runner = ApiExperimentRunner(
        client=client,
        config=ApiRunnerConfig(bundle_path=tmp_path / "unused.json", output_dir=tmp_path / "out"),
        app_settings=Settings(llm_provider=LLMProvider.GEMINI, llm_model_name="m"),
    )
    runner.engine = FakeEngine(
        [
            GenerationResult(
                query="MATCH (n:IfcWall) RETURN n.GlobalId",
                is_valid=True,
                model_name="fake",
                raw_output="MATCH (n:IfcWall) RETURN n.GlobalId",
            )
        ]
    )
    runner.test_cases = client.get_test_set()
    runner.model_context = None

    rows = runner.run_setting(ExperimentSetting.CYPHER_SOFT)

    assert len(rows) == 1
    row = rows[0]
    assert row["svr"] == 1.0
    assert row["ea"] == 1.0
    assert row["category"] == "count"
    assert row["difficulty"] == "easy"
    assert row["generated_cypher"] == "MATCH (n:IfcWall) RETURN n.GlobalId"

    assert len(posted) == 1
    sent = posted[0]
    assert sent["question"] == "How many walls?"
    assert sent["gold_index"] == 0
    assert sent["output_type"] == "cypher"
    assert sent["experiment_setting"] == "cypher_soft"
    assert sent["output"] == "MATCH (n:IfcWall) RETURN n.GlobalId"


def test_run_setting_records_evaluate_failure(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/test-set":
            return httpx.Response(
                200,
                json={
                    "count": 1,
                    "cases": [
                        {
                            "index": 0,
                            "question": "boom",
                            "gold_cypher": "X",
                            "category": "",
                            "difficulty": "",
                        }
                    ],
                },
            )
        if request.url.path == "/evaluate":
            return httpx.Response(500, text="server exploded")
        return httpx.Response(404)

    client = _build_client(handler)
    runner = ApiExperimentRunner(
        client=client,
        config=ApiRunnerConfig(bundle_path=tmp_path / "u.json", output_dir=tmp_path / "out"),
        app_settings=Settings(llm_provider=LLMProvider.GEMINI, llm_model_name="m"),
    )
    runner.engine = FakeEngine(
        [GenerationResult(query="MATCH (n) RETURN n", is_valid=True, raw_output="MATCH (n) RETURN n")]
    )
    runner.test_cases = client.get_test_set()
    runner.model_context = None

    rows = runner.run_setting(ExperimentSetting.CYPHER_SOFT)

    assert len(rows) == 1
    assert rows[0]["error"].startswith("evaluate_failed: 500")
    assert rows[0]["svr"] == 0.0


def test_run_comparison_writes_csv_and_summary(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/test-set":
            return httpx.Response(
                200,
                json={
                    "count": 2,
                    "cases": [
                        {
                            "index": 0,
                            "question": "Q1",
                            "gold_cypher": "G1",
                            "category": "c",
                            "difficulty": "easy",
                        },
                        {
                            "index": 1,
                            "question": "Q2",
                            "gold_cypher": "G2",
                            "category": "c",
                            "difficulty": "hard",
                        },
                    ],
                },
            )
        if request.url.path == "/evaluate":
            payload = json.loads(request.content.decode())
            return httpx.Response(
                200,
                json=_evaluate_response(
                    payload["question"],
                    ExperimentSetting(payload["experiment_setting"]),
                    generated_cypher=payload["output"],
                ),
            )
        return httpx.Response(404)

    client = _build_client(handler)
    out_dir = tmp_path / "results"
    runner = ApiExperimentRunner(
        client=client,
        config=ApiRunnerConfig(
            bundle_path=tmp_path / "u.json",
            output_dir=out_dir,
            name="t",
            settings=[ExperimentSetting.CYPHER_SOFT],
        ),
        app_settings=Settings(llm_provider=LLMProvider.GEMINI, llm_model_name="m"),
    )
    runner.engine = FakeEngine(
        [
            GenerationResult(query="C1", is_valid=True, raw_output="C1"),
            GenerationResult(query="C2", is_valid=True, raw_output="C2"),
        ]
    )
    runner.test_cases = client.get_test_set()
    runner.model_context = None

    runner.run_comparison()

    csv_path = out_dir / "t_cypher_soft_results.csv"
    summary_path = out_dir / "t_cypher_soft_summary.json"
    comparison_path = out_dir / "t_comparison.json"
    assert csv_path.exists()
    assert summary_path.exists()
    assert comparison_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["metrics"]["count"] == 2
    assert summary["metrics"]["ea_mean"] == 1.0
    assert "easy" in summary["metrics"]["metrics_by_difficulty"]
    assert "hard" in summary["metrics"]["metrics_by_difficulty"]


def test_runner_load_bundle_rehydrates_vocabulary(tmp_path):
    """End-to-end bundle load: write asdict-shaped JSON, confirm rehydration."""
    from dataclasses import asdict

    from src.constraints.ids_parser import IDSSchema
    from src.constraints.vocabulary_merger import (
        CombinedVocabulary,
        EntityVocabulary,
        PropertyType,
        PropertyVocabulary,
    )

    vocab = CombinedVocabulary(
        entities={"IfcWall": EntityVocabulary(name="IfcWall", from_ifc=True)},
        all_properties={
            "FireRating": PropertyVocabulary(
                name="FireRating",
                property_type=PropertyType.STRICT,
                allowed_values={"EI30"},
            )
        },
        strict_properties={"FireRating"},
    )
    bundle = {
        "schema_version": 1,
        "combined_vocabulary": asdict(vocab),
        "ids_schema": asdict(IDSSchema(entities={"IFCWALL"})),
        "model_dump": [{"GlobalId": "abc"}],
    }
    bundle_path = tmp_path / "bundle.json"
    # Sets won't survive plain json.dumps — mimic the build_bundle encoder by sorting.
    bundle_path.write_text(
        json.dumps(bundle, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    )

    client = _build_client(lambda r: httpx.Response(404))
    runner = ApiExperimentRunner(
        client=client,
        config=ApiRunnerConfig(bundle_path=bundle_path, output_dir=tmp_path / "out"),
        app_settings=Settings(llm_provider=LLMProvider.GEMINI, llm_model_name="m"),
    )
    runner._load_bundle()
    assert runner.vocabulary.get_entity_names() == {"IfcWall"}
    assert runner.vocabulary.strict_properties == {"FireRating"}
    assert runner.ids_schema.entities == {"IFCWALL"}
    assert runner.model_context is not None

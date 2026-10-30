"""The run driver: resume, judge failures, question files, snapshot prep.

No Neo4j and no network: ``run.main`` is driven against a two-model manifest in
a temp dir, with the graph, the paradigms, the LLM clients and the judge faked.
"""

import json
from types import SimpleNamespace

import pytest
import yaml

from src.agent import loop, run
from src.agent.llm import LLMError
from src.eval.judge import DEFAULT_JUDGE, JudgeResult

MODELS = ("clinic__arc", "clinic__str")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _question(qid, model_id, split="train", **extra):
    project, ifc_model = model_id.split("__")
    return {"id": qid, "question": f"question {qid}?", "ground_truth": f"answer {qid}",
            "category": 1, "split": split, "project": project, "ifc_model": ifc_model, **extra}


class Harness:
    """Fakes for everything ``run.main`` reaches outside the process."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path
        self.out = tmp_path / "runs"
        self.manifest = tmp_path / "data" / "ifcbench" / "manifest.yaml"
        self.manifest.parent.mkdir(parents=True)
        models = []
        for mid in MODELS:
            project, ifc_model = mid.split("__")
            models.append({"model_id": mid, "project": project, "ifc_model": ifc_model,
                           "questions": {"test": 1, "train": 1},
                           "paths": {"ifc": f"data/ifcbench/raw/{mid}.ifc",
                                     "testset": f"data/ifcbench/testsets/{mid}.jsonl",
                                     "graph_cache": f"data/ifcbench/cache/{mid}.base.jsonl.gz"}})
            _write_jsonl(tmp_path / "data" / "ifcbench" / "testsets" / f"{mid}.jsonl",
                         [_question(10 + len(models), mid, "train"),
                          _question(20 + len(models), mid, "test")])
        self.manifest.write_text(yaml.safe_dump({"dataset": {"commit": "c"}, "models": models}))

        self.episodes_run, self.graphs, self.judged, self.variants = [], [], [], []
        self.judge_fails = False
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        monkeypatch.setattr(run, "GraphDatabase", SimpleNamespace(
            driver=lambda *a, **k: SimpleNamespace(close=lambda: None)))
        monkeypatch.setattr(run, "client_from_spec", lambda spec, **k: SimpleNamespace(
            spec=spec, temperature=0.1, max_tokens=8192))
        monkeypatch.setattr(run, "make_judge", lambda spec, providers=None: SimpleNamespace(spec=spec))
        monkeypatch.setattr(run, "ensure_graph", self._ensure_graph)
        monkeypatch.setattr(run, "build_graph_context", lambda driver, ifc: SimpleNamespace(
            schema_prompt=lambda: "SCHEMA", graph={"model_id": "m"}, graph_stats={}))
        monkeypatch.setattr(run, "judge_answer", self._judge)
        for name in run.PARADIGMS:
            monkeypatch.setitem(run.PARADIGMS, name, self._paradigm(name))

    def _ensure_graph(self, driver, manifest, entry, state, policy, etl_report_dir):
        self.graphs.append((entry["model_id"], state, policy))
        return {"fingerprint": "f" * 64, "restored": False}

    def _paradigm(self, name):
        def episode(qid, question, schema, llm, tools, **kwargs):
            self.episodes_run.append((qid, name))
            self.variants.append(kwargs["prompt_variant"])
            return loop.Episode(qid, question, name, llm.spec, answer=f"answer {qid} (from a query)",
                                cost_usd=0.01, prompt_variant=kwargs["prompt_variant"])
        return episode

    def _judge(self, client, *, question, category, ground_truth, answer):
        self.judged.append((question, client.spec))
        if self.judge_fails:
            raise LLMError("openrouter:z-ai/glm-4.7: gave up after 5 attempts: HTTP 503")
        return JudgeResult(abstention=False, faithfulness="Yes", completeness="Yes",
                           transparency="Yes", relevance="Yes", justification="ok",
                           verdict="correct", judge=client.spec, cost_usd=0.001)

    def main(self, *args, run_name="r", llm=True):
        argv = [*args, "--run", run_name, "--out-root", str(self.out), "--manifest", str(self.manifest)]
        if llm:
            argv += ["--llm", "openrouter:z-ai/glm-4.5-air", "--provider", "z-ai"]
        return run.main(argv)

    def records(self, run_name="r"):
        return [json.loads(line) for line in open(self.out / run_name / "episodes.jsonl")]

    def config(self, run_name="r"):
        return json.loads((self.out / run_name / "config.json").read_text())


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)


# -----------------------------------------------------------------------------
# Judge failures and judging later
# -----------------------------------------------------------------------------

def test_judge_failure_keeps_the_paid_episode_and_the_resume_only_judges(harness):
    harness.judge_fails = True
    harness.main("--model-id", "clinic__arc", "--split", "all", "--paradigm", "agentic")
    first = harness.records()
    assert len(first) == 2 and len(harness.episodes_run) == 2
    assert all(r["judgement"] is None and "HTTP 503" in r["judge_error"] for r in first)
    assert all(r["episode"]["answer"] for r in first)

    harness.judge_fails = False
    harness.main("--model-id", "clinic__arc", "--split", "all", "--paradigm", "agentic")
    assert len(harness.episodes_run) == 2           # no episode ran again
    assert len(harness.graphs) == 1                 # nor was the graph needed to judge
    latest = run.latest_records(harness.out / "r" / "episodes.jsonl")
    assert all(r["judgement"]["verdict"] == "correct" and "judge_error" not in r
               for r in latest.values())
    assert [r["episode"] for r in latest.values()] == [r["episode"] for r in first]
    summary = json.loads((harness.out / "r" / "summary.json").read_text())
    assert summary["agentic"]["n"] == 2 and summary["agentic"]["cost_usd"] == pytest.approx(0.02)


def test_infrastructure_errors_are_rerun_not_judged(harness, monkeypatch):
    def broken(qid, question, schema, llm, tools, **kwargs):
        harness.episodes_run.append((qid, "single_shot"))
        return loop.Episode(qid, question, "single_shot", llm.spec, status="error", error="LLMError: x")

    monkeypatch.setitem(run.PARADIGMS, "single_shot", broken)
    harness.main("--model-id", "clinic__arc", "--paradigm", "single_shot")
    assert harness.judged == [] and harness.records()[0]["judgement"] is None
    assert "judge_error" not in harness.records()[0]
    harness.main("--model-id", "clinic__arc", "--paradigm", "single_shot")
    assert len(harness.episodes_run) == 2


def test_a_no_judge_run_is_judged_by_a_later_resume(harness):
    harness.main("--model-id", "clinic__arc", "--no-judge")
    assert harness.judged == [] and len(harness.episodes_run) == 2
    assert harness.config()["settings"]["judge"] is None

    harness.main("--model-id", "clinic__arc", "--judge-provider", "z-ai")
    assert len(harness.episodes_run) == 2 and len(harness.judged) == 2
    stamp = harness.config()["settings"]
    assert (stamp["judge"], stamp["judge_provider"]) == (DEFAULT_JUDGE, ["z-ai"])

    # --no-judge on a judged run keeps its judge; another judge is refused.
    harness.main("--model-id", "clinic__arc", "--no-judge")
    assert harness.config()["settings"]["judge"] == DEFAULT_JUDGE
    with pytest.raises(SystemExit):
        harness.main("--model-id", "clinic__arc", "--judge", "openrouter:z-ai/glm-4.5-air",
                     "--judge-provider", "z-ai")
    with pytest.raises(SystemExit):  # the episode settings still bind
        harness.main("--model-id", "clinic__arc", "--reasoning", "on", "--judge-provider", "z-ai")


def test_prompt_variant_is_stamped_only_when_not_hellin(harness):
    # Default: no stamp key, so run directories from before the flag still resume.
    harness.main("--model-id", "clinic__arc", "--paradigm", "agentic")
    assert harness.variants == ["hellin"]
    assert "prompt_variant" not in harness.config()["settings"]
    assert harness.config()["prompt_variant"] == "hellin"
    assert harness.records()[0]["episode"]["prompt_variant"] == "hellin"
    harness.main("--model-id", "clinic__arc", "--paradigm", "agentic", "--prompt-variant", "hellin")
    assert len(harness.episodes_run) == 1             # resumed, nothing to run
    with pytest.raises(SystemExit):                   # but ours is another cell
        harness.main("--model-id", "clinic__arc", "--paradigm", "agentic", "--prompt-variant", "ours")

    harness.main("--model-id", "clinic__arc", "--paradigm", "agentic", "--prompt-variant", "ours",
                 run_name="r2")
    assert harness.variants == ["hellin", "ours"]
    assert harness.config("r2")["settings"]["prompt_variant"] == "ours"
    assert harness.records("r2")[0]["episode"]["prompt_variant"] == "ours"
    harness.main("--model-id", "clinic__arc", "--paradigm", "agentic", "--prompt-variant", "ours",
                 run_name="r2")
    assert len(harness.episodes_run) == 2
    with pytest.raises(SystemExit):
        harness.main("--model-id", "clinic__arc", "--paradigm", "agentic", run_name="r2")


def test_resume_settings_lets_only_the_judge_arrive_late():
    unjudged = {"llm": "a", "judge": None, "judge_provider": None}
    judged = {"llm": "a", "judge": "j", "judge_provider": ["z-ai"]}
    assert run.resume_settings(unjudged, judged) == judged
    assert run.resume_settings(judged, unjudged) == judged
    assert run.resume_settings(judged, judged) == judged
    assert run.resume_settings(judged, {**judged, "judge": "k"}) is None
    assert run.resume_settings(judged, {**judged, "judge_provider": None}) is None
    assert run.resume_settings(unjudged, {**judged, "llm": "b"}) is None
    assert run.resume_settings({}, unjudged) is None


# -----------------------------------------------------------------------------
# --questions
# -----------------------------------------------------------------------------

def test_questions_file_replaces_the_testsets(harness):
    authored = harness.root / "data" / "ifcbench" / "authored" / "clinic.jsonl"
    _write_jsonl(authored, [
        _question(501, "clinic__arc", kind="adjacency", hops=2, source="authored",
                  answer="1BC2", gold_cypher="MATCH ..."),
        _question(502, "clinic__arc", kind="path", hops=3, source="authored"),
        _question(601, "clinic__str", kind="adjacency", hops=1),
    ])
    harness.main("--questions", str(authored), "--split", "train", "--paradigm", "agentic")
    assert sorted(harness.episodes_run) == [(501, "agentic"), (502, "agentic"), (601, "agentic")]
    assert sorted(m for m, _, _ in harness.graphs) == list(MODELS)
    rec = next(r for r in harness.records() if r["question_id"] == 501)
    assert (rec["model_id"], rec["kind"], rec["hops"], rec["source"]) == ("clinic__arc", "adjacency", 2, "authored")
    assert "answer" not in rec and "gold_cypher" not in rec

    config = harness.config()
    assert config["questions"]["path"] == str(authored)
    assert config["settings"]["questions"] == config["questions"]["sha256"]
    by_kind = json.loads((harness.out / "r" / "summary.json").read_text())["agentic"]["by_kind"]
    assert by_kind == {"adjacency": {"n": 2, "accuracy": 1.0}, "path": {"n": 1, "accuracy": 1.0}}

    # Another question set is another cell (ids may collide); --ids and --model-id still select.
    other = authored.with_name("other.jsonl")
    _write_jsonl(other, [_question(501, "clinic__arc")])
    with pytest.raises(SystemExit):
        harness.main("--questions", str(other), "--split", "train", "--paradigm", "agentic")
    harness.main("--questions", str(authored), "--model-id", "clinic__arc", "--ids", "502",
                 "--split", "train", "--paradigm", "single_shot", run_name="r2")
    assert harness.episodes_run[3:] == [(502, "single_shot")]


def test_testset_runs_carry_no_question_file_and_no_kind(harness):
    harness.main("--model-id", "clinic__arc", "--split", "all", "--paradigm", "single_shot")
    assert "questions" not in harness.config() and "questions" not in harness.config()["settings"]
    assert "by_kind" not in json.loads((harness.out / "r" / "summary.json").read_text())["single_shot"]


@pytest.mark.parametrize("rows, why", [
    ([_question(1, "clinic__mep")], "not in the manifest"),
    ([_question(1, "clinic__arc"), _question(1, "clinic__arc")], "repeats"),
    ([{"id": 1, "question": "q"}], "missing"),
])
def test_questions_file_is_validated(harness, capsys, rows, why):
    bad = harness.root / "bad.jsonl"
    _write_jsonl(bad, rows)
    with pytest.raises(SystemExit):
        harness.main("--questions", str(bad))
    assert why in capsys.readouterr().err


# -----------------------------------------------------------------------------
# --snapshots-only
# -----------------------------------------------------------------------------

def test_snapshots_only_builds_what_is_missing_without_an_llm(harness, monkeypatch):
    def no_llm(*a, **k):
        raise AssertionError("snapshot prep must not build an LLM client")

    monkeypatch.setattr(run, "client_from_spec", no_llm)
    present = harness.root / "data" / "ifcbench" / "cache" / "clinic__arc.base.jsonl.gz"
    present.parent.mkdir(parents=True)
    present.write_bytes(b"")
    assert harness.main("--manifest-split", "test", "--snapshots-only", llm=False) == 0
    assert harness.graphs == [("clinic__str", "base", "auto")]
    assert harness.episodes_run == [] and not (harness.out / "r" / "episodes.jsonl").exists()


def test_snapshots_only_goes_on_past_a_failure_and_reports_it(harness, monkeypatch):
    def flaky(driver, manifest, entry, state, policy, etl_report_dir):
        harness.graphs.append(entry["model_id"])
        if entry["model_id"] == "clinic__arc":
            raise RuntimeError("clinic__arc: arc.ifc missing")
        return {"fingerprint": "f" * 64, "restored": False, "built": True}

    monkeypatch.setattr(run, "ensure_graph", flaky)
    assert harness.main("--manifest-split", "test", "--snapshots-only", "--state", "topology",
                        llm=False) == 1
    assert harness.graphs == list(MODELS)


def test_llm_is_required_unless_snapshots_only(harness):
    with pytest.raises(SystemExit):
        harness.main("--model-id", "clinic__arc", llm=False)
    with pytest.raises(SystemExit):
        harness.main("--model-id", "clinic__arc", "--snapshots-only", "--restore", "never", llm=False)
    with pytest.raises(SystemExit):  # some model selection is still needed
        harness.main()

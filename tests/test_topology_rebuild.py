"""Snapshot revisions: a topology snapshot built by older rules is kept aside and rebuilt.

``src.agent.run`` builds missing snapshots; a topology snapshot whose header
says ``topology_rev`` < ``TOPOLOGY_REV`` counts as missing, so the laptop and
the Colab notebook rebuild stale ones without anyone deleting files. No Neo4j:
``build_snapshot`` and the graph calls are replaced by fakes.
"""

import pytest

from src.etl import topology as tp


def _fake_snapshot(path, rev):
    import gzip
    import json

    from src.etl.graph_cache import FORMAT

    meta = {"model_id": "m", "enrichment": "topology"}
    if rev is not None:
        meta["topology_rev"] = rev
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"format": FORMAT, "fingerprint": f"fp-rev{rev}", "meta": meta, "summary": {}}) + "\n")


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """A manifest with one model and a build_snapshot that writes a current-rev snapshot."""
    from src.agent import run

    entry = {"model_id": "m", "paths": {"graph_cache": "data/ifcbench/cache/m.base.jsonl.gz",
                                        "ifc": "data/m.ifc"}}
    manifest = {"_root": tmp_path, "models": [entry]}
    built = []

    def build(driver, manifest, entry, snapshot, etl_report_dir, state="base"):
        built.append(state)
        _fake_snapshot(snapshot, tp.TOPOLOGY_REV)
        return f"fp-rev{tp.TOPOLOGY_REV}"

    monkeypatch.setattr(run, "build_snapshot", build)
    return manifest, entry, tmp_path / "data/ifcbench/cache", built


@pytest.mark.parametrize("old_rev", [None, tp.TOPOLOGY_REV - 1])
def test_a_topology_snapshot_of_an_older_revision_is_kept_aside_and_rebuilt(fake_cache, old_rev):
    from src.agent import run

    manifest, entry, cache, built = fake_cache
    _fake_snapshot(cache / "m.topology.jsonl.gz", old_rev)        # None: a v1 snapshot, never stamped
    identity = run.ensure_graph(None, manifest, entry, "topology", "auto", cache)
    assert identity == {"fingerprint": f"fp-rev{tp.TOPOLOGY_REV}", "restored": False, "built": True}
    assert built == ["topology"]
    kept = cache / f"m.topology-v{old_rev or 1}.jsonl.gz"
    assert kept.is_file() and run.stale_topology_rev(kept) == (old_rev or 1)
    assert run.stale_topology_rev(cache / "m.topology.jsonl.gz") is None


def test_a_current_topology_snapshot_is_restored_not_rebuilt(fake_cache, monkeypatch):
    from src.agent import run

    manifest, entry, cache, built = fake_cache
    _fake_snapshot(cache / "m.topology.jsonl.gz", tp.TOPOLOGY_REV)
    restored = []
    monkeypatch.setattr(run, "graph_summary", lambda driver: {})
    monkeypatch.setattr(run, "fingerprint", lambda summary: "something else")
    monkeypatch.setattr(run, "restore_graph", lambda driver, path: restored.append(path.name))
    identity = run.ensure_graph(None, manifest, entry, "topology", "auto", cache)
    assert identity["restored"] and restored == ["m.topology.jsonl.gz"] and built == []


def test_never_refuses_a_stale_topology_snapshot_and_leaves_it(fake_cache):
    from src.agent import run

    manifest, entry, cache, built = fake_cache
    _fake_snapshot(cache / "m.topology.jsonl.gz", tp.TOPOLOGY_REV - 1)
    with pytest.raises(RuntimeError, match="rebuild it"):
        run.ensure_graph(None, manifest, entry, "topology", "never", cache)
    assert (cache / "m.topology.jsonl.gz").is_file() and built == []


def test_snapshot_prep_rebuilds_stale_topology_only(fake_cache):
    from src.agent import run

    manifest, entry, cache, built = fake_cache
    _fake_snapshot(cache / "m.base.jsonl.gz", None)                # base snapshots carry no topology_rev
    assert run.prepare_snapshots(None, manifest, [entry], "base", cache) == 0 and built == []
    _fake_snapshot(cache / "m.topology.jsonl.gz", tp.TOPOLOGY_REV)
    assert run.prepare_snapshots(None, manifest, [entry], "topology", cache) == 0 and built == []
    _fake_snapshot(cache / "m.topology.jsonl.gz", tp.TOPOLOGY_REV - 1)
    assert run.prepare_snapshots(None, manifest, [entry], "topology", cache) == 0 and built == ["topology"]
    assert (cache / f"m.topology-v{tp.TOPOLOGY_REV - 1}.jsonl.gz").is_file()

"""IFC-Bench fetch / manifest helpers that need no network and no Neo4j."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ifcbench"))
import build_testset as bt  # noqa: E402
import fetch  # noqa: E402


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_split_ifc_paths_join_on_question_id(tmp_path):
    questions = tmp_path / "q.csv"
    split = tmp_path / "s.csv"
    _write_csv(questions, ["id", "question", "ground_truth", "ifc_model", "project", "category"], [
        ["1", "q1", "a1", "arc", "dental_clinic", "1"],
        ["2", "q2", "a2", "str", "dental_clinic", "2"],
        ["3", "q3", "a3", "heating", "digital_hub", "2"],
    ])
    _write_csv(split, ["id", "split"], [["1", "test"], ["2", "train"], ["3", "test"]])

    assert fetch.ifc_paths_for_split(questions, split, "test") == {
        "projects/dental_clinic/arc.ifc", "projects/digital_hub/heating.ifc",
    }
    assert len(fetch.ifc_paths_for_split(questions, split, "all")) == 3


def test_select_takes_question_csvs_and_only_the_wanted_ifc_files():
    entries = [{"path": p} for p in (
        "questions/ifc-bench-v2.csv",
        "projects/dental_clinic/arc.ifc",
        "projects/dental_clinic/mep.ifc",
        "projects/duplex/arc.ifc",
        "projects/duplex/document/Pump.pdf",
        "README.md",
    )]
    chosen = {e["path"] for e in fetch.select(entries, {"projects/duplex/arc.ifc"}, {"dental_clinic"})}
    assert chosen == {
        "questions/ifc-bench-v2.csv",
        "projects/dental_clinic/arc.ifc",
        "projects/dental_clinic/mep.ifc",
        "projects/duplex/arc.ifc",
    }


def test_step_args_respect_quotes_and_nested_lists():
    text = "'Project, Number','2011-09-06',(''),('A','B'),'Revit','it''s Solibri','');"
    args = bt.split_step_args(text)
    assert args == ["'Project, Number'", "'2011-09-06'", "('')", "('A','B')",
                    "'Revit'", "'it''s Solibri'", "''"]
    assert bt._unquote(args[5]) == "it's Solibri"
    assert bt._unquote("''") is None


def test_header_reads_both_tool_fields(tmp_path):
    ifc = tmp_path / "arc.ifc"
    ifc.write_text(
        "ISO-10303-21;\nHEADER;\n"
        "FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');\n"
        "FILE_NAME('Project Number','2011-09-06T11:22:43',(''),(''),"
        "'Autodesk Revit Architecture 2011 - 1.0','20100326_1700 (Solibri IFC Optimizer)','');\n"
        "FILE_SCHEMA(('IFC2X3'));\nENDSEC;\n"
    )
    assert bt.read_header(ifc) == {
        "schema": "IFC2X3",
        "preprocessor_version": "Autodesk Revit Architecture 2011 - 1.0",
        "originating_system": "20100326_1700 (Solibri IFC Optimizer)",
    }

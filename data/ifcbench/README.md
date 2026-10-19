# IFC-Bench data

Questions and ground truth in `testsets/` are derived from **IFC-Bench**
(https://huggingface.co/datasets/sylvainHellin/ifc-bench), licensed CC-BY-4.0,
at the dataset commit recorded in `manifest.yaml`:

> S. Hellin, M. Jang, S. Fuchs, S. Nousias, A. Borrmann, "Agentic search for BIM
> information extraction: Systematic evaluation", *Automation in Construction*
> 192 (2026) 107260. doi:10.1016/j.autcon.2026.107260

They are re-sliced by target IFC file and otherwise unchanged. IFC models are not
redistributed here. Their licenses differ per project (some GPL v3), so fetch them
with `scripts/ifcbench/fetch.py`.

| Path | Tracked | Produced by |
|---|---|---|
| `raw/` | no | `scripts/ifcbench/fetch.py`: question CSVs and IFC files, sha256-verified, pinned commit |
| `manifest.yaml` | yes | `scripts/ifcbench/build_testset.py --split test`: one entry per IFC file |
| `testsets/<model_id>.jsonl` | yes | same, one record per question |
| `cache/<model_id>.<state>.jsonl.gz` | no | `python -m src.etl.graph_cache export`: graph snapshots, restored in seconds |

`model_id` is `<project>__<ifc_model>`, for example `dental_clinic__arc`.

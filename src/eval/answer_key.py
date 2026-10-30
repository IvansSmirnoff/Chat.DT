"""Judge-independent answer-key scoring of the authored questions.

Why this exists
---------------
The authored questions (``data/ifcbench/authored/<model_id>.jsonl``) carry an
exact structured ``answer`` (room lists, door counts, routes, ``[room,
corridor]`` pairs, booleans, areas), but they are IFC-Bench category 3, and
the LLM judge (``src/eval/judge.py``, Hellin's rubric verbatim) lets a
category-3 answer differ numerically as long as its method is disclosed: q9021
answered 8 doors where the truth is 6 and was judged correct. This module gives
those runs a second score that no judge decides.

Two passes per cell (the latest record per ``(model_id, question_id,
paradigm)`` of ``RUN_DIR/episodes.jsonl``):

1. **Extraction (an LLM, not a judgement).** The extractor sees the question,
   the answer's *schema* (keys and value types derived from ``answer``, never
   its values) and the system's final answer text, and returns JSON with the
   same keys holding what the answer *claims*, ``null`` where it states
   nothing. Its prompt forbids inferring, computing or correcting. The reply is
   validated against the schema and re-asked once if it does not fit; a second
   misfit is recorded as an extraction failure.
2. **Deterministic comparison** with the answer key, in the semantics of
   ``scripts/ifcbench/verify_authored.py`` (whose ``same`` lives here now):
   top-level lists are sets, sequences one level down (a route's spaces, a
   wall's layer build-up) keep their order, integers are exact and other
   numbers match within a relative tolerance (1%: answers round areas and
   thicknesses). Two additions, because the extractor is not shown values:

   * a fixed-length list (a ``[room, corridor]`` pair) is compared as a
     multiset -- the extractor cannot be told which slot holds what without
     the values, and in these answers each part's role shows in the value
     itself (room vs corridor number, floor name, number);
   * names are compared case- and whitespace-insensitively, and a Revit
     ``Family:Type[:Id]`` name is reduced to its type (``Basic Wall: Interior
     - Rated 1-HR (92mm Stud)`` is the wall type ``Interior - Rated 1-HR
     (92mm Stud)``). A gold space number also matches a claim that gives it
     with its long name and no other number (``2C22 LOUNGE``): the prompt asks
     for the number alone, but the extractor does not always comply.

   A route (``ROUTE_KEYS``) matches if it equals any listed gold route; the
   claim must give at least one and every route it gives must match. A route
   that leaves out the start or destination named by the question is
   completed with them.

The score is the **exact match**: every key stated and matching. Some keys
hold more than the question strictly asks for (9003 lists the rooms whose area
it sums, 9008 counts the doors of the route it asks about), so each cell also
gets a secondary **stated match**: at least one key stated and nothing stated
wrong. The judge-correct cells that miss the exact match only on unstated keys
are listed separately.

Abstentions -- the judge's ``abstained``, an episode that ended without an
answer (step or token limit, parse error), an empty answer, or an extraction
in which every key is ``null`` -- score as not correct and are flagged
separately. The extractor still reads an answer the judge called an
abstention, so the judge-vs-key confusion matrix shows one that does state the
right answer.

Outputs, per run directory:

    answer_key_cache.jsonl    every extraction (append-only); a cell already
                              extracted with the same extractor, prompt, template,
                              question and answer text is not sent again
    answer_key.jsonl          per cell: claim, match per key, exact match,
                              abstention flags, judge verdict
    answer_key_summary.json   exact-match accuracy per paradigm, by ``needs``
                              and ``kind``, judge-vs-key confusion matrix and
                              the disagreeing question ids

Usage
-----
    python -m src.eval.answer_key results/ifcbench/runs/p7_authored_glm52_think_base \\
        --questions data/ifcbench/authored/dental_clinic__arc.jsonl
    python -m src.eval.answer_key RUN_A RUN_B --questions ... --cache-only   # no LLM calls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.agent.actions import first_json_object
from src.agent.llm import REASONING_MODES, ChatClient, client_from_spec

logger = logging.getLogger("answer_key")

# A non-reasoning model of another family than the backbones (GLM) and the
# judge (GLM-4.7), served by its first party: temperature 0 and JSON mode are
# honoured, and there is no thinking to switch off.
DEFAULT_EXTRACTOR = "openrouter:openai/gpt-4.1-mini"
DEFAULT_EXTRACTOR_PROVIDER = "openai"
NUMBER_TOL = 0.01          # relative; answers round areas and thicknesses
PARSE_RETRIES = 1          # a reply that does not fit the schema is re-asked once
EXTRACTOR_MAX_TOKENS = 2048

# Keys under which the lists one level down are ordered sequences of variable
# length (a route's spaces, a wall's layers from one face to the other).
# Everywhere else a list inside a list is a fixed-length tuple.
SEQUENCE_KEYS = {"routes", "layer_sets"}
# Keys whose list holds acceptable alternatives rather than a set to reproduce.
ROUTE_KEYS = {"routes"}

Key = Tuple[str, int, str]
Shape = Dict[str, Any]


# -----------------------------------------------------------------------------
# Comparison (moved from scripts/ifcbench/verify_authored.py, unchanged)
# -----------------------------------------------------------------------------

def _sort_key(value: Any) -> str:
    """A stable key for set comparison; floats rounded so near-equal values sort together."""
    def norm(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, 3)
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        return v
    return json.dumps(norm(value), sort_keys=True)


def same(got: Any, want: Any, tol: float, ordered: bool = True) -> bool:
    """Structural equality; numbers within ``tol`` (relative), lists as sets unless ``ordered``."""
    if isinstance(want, bool) or isinstance(got, bool):
        return got is want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(got, want, rel_tol=tol, abs_tol=1e-9)
    if isinstance(want, list) and isinstance(got, list):
        if len(got) != len(want):
            return False
        if not ordered:
            got, want = sorted(got, key=_sort_key), sorted(want, key=_sort_key)
        return all(same(g, w, tol) for g, w in zip(got, want))
    if isinstance(want, dict) and isinstance(got, dict):
        return got.keys() == want.keys() and all(same(got[k], want[k], tol) for k in want)
    return got == want


# -----------------------------------------------------------------------------
# Schema: the answer's shape without its values
# -----------------------------------------------------------------------------

def _scalar_shape(value: Any) -> Shape:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    return {"type": "any"}


def _merge(a: Shape, b: Shape) -> Shape:
    """One shape for two list items (an integer and a float merge to a number)."""
    if a == b:
        return a
    if {a["type"], b["type"]} == {"integer", "number"}:
        return {"type": "number"}
    if a["type"] == b["type"] == "tuple" and len(a["slots"]) == len(b["slots"]):
        return {"type": "tuple", "slots": [_merge(x, y) for x, y in zip(a["slots"], b["slots"])]}
    if a["type"] == b["type"] and a["type"] in ("set", "sequence"):
        return {"type": a["type"], "items": _merge(a["items"], b["items"])}
    raise ValueError(f"list items of different shapes: {a} vs {b}")


def _items_shape(items: Sequence[Any], key: str, depth: int) -> Shape:
    if not items:
        return {"type": "any"}
    shape = derive_shape(items[0], key, depth)
    for item in items[1:]:
        shape = _merge(shape, derive_shape(item, key, depth))
    return shape


def derive_shape(value: Any, key: str, depth: int = 0) -> Shape:
    """The type structure of an answer value. It records types and the length
    of fixed-length tuples, never a value or the length of a variable list."""
    if not isinstance(value, list):
        return _scalar_shape(value)
    if depth == 0:
        return {"type": "set", "items": _items_shape(value, key, 1)}
    if depth == 1 and key in SEQUENCE_KEYS:
        return {"type": "sequence", "items": _items_shape(value, key, 2)}
    return {"type": "tuple", "slots": [derive_shape(v, key, depth + 1) for v in value]}


def derive_schema(answer: Dict[str, Any]) -> Dict[str, Shape]:
    return {k: derive_shape(v, k) for k, v in answer.items()}


def describe(shape: Shape) -> str:
    t: str = shape["type"]
    if t == "set":
        return f"list of {_describe_item(shape['items'])}, any order"
    if t == "sequence":
        return f"ordered list of {_describe_item(shape['items'])}"
    if t == "tuple":
        return "[" + ", ".join(describe(s) for s in shape["slots"]) + "]"
    return {"boolean": "true or false", "any": "value"}.get(t, t)


def _describe_item(shape: Shape) -> str:
    text = describe(shape)
    return f"({text})" if shape["type"] in ("set", "sequence") else text


_PLACEHOLDERS = {"string": '"<string>"', "integer": "<integer>", "number": "<number>",
                 "boolean": "<true or false>", "any": "<value>"}


def skeleton(shape: Shape) -> str:
    """The JSON nesting of a value, with placeholders: ``[["<string>", <number>], ...]``."""
    t = shape["type"]
    if t in ("set", "sequence"):
        return f"[{skeleton(shape['items'])}, ...]"
    if t == "tuple":
        return "[" + ", ".join(skeleton(s) for s in shape["slots"]) + "]"
    return _PLACEHOLDERS[t]


def render_template(schema: Dict[str, Shape]) -> str:
    lines = [f'  "{k}": {describe(s)}' + (f", written {skeleton(s)}" if s["type"] == "set" else "")
             for k, s in schema.items()]
    return "{\n" + ",\n".join(lines) + "\n}"


# -----------------------------------------------------------------------------
# Validation of the extractor's reply
# -----------------------------------------------------------------------------

def _conform(value: Any, shape: Shape, path: str, problems: List[str]) -> Any:
    t = shape["type"]
    if t == "boolean":
        if not isinstance(value, bool):
            problems.append(f"{path}: expected true or false, got {value!r}")
        return value
    if t == "integer":
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{path}: expected an integer, got {value!r}")
        return value
    if t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{path}: expected a number, got {value!r}")
        return value
    if t == "string":
        if not isinstance(value, str):
            problems.append(f"{path}: expected a string, got {value!r}")
        return value
    if t == "any":
        if isinstance(value, (list, dict)):
            problems.append(f"{path}: expected a single value, got {value!r}")
        return value
    if not isinstance(value, list):
        problems.append(f"{path}: expected a list, got {value!r}")
        return value
    if t == "tuple":
        if len(value) != len(shape["slots"]):
            problems.append(f"{path}: expected {len(shape['slots'])} parts "
                            f"{describe(shape)}, got {len(value)}")
            return value
        return [None if v is None else _conform(v, s, f"{path}[{i}]", problems)
                for i, (v, s) in enumerate(zip(value, shape["slots"]))]
    out = []
    for i, v in enumerate(value):
        if v is None:
            problems.append(f"{path}[{i}]: null inside a list; leave the item out instead")
            continue
        out.append(_conform(v, shape["items"], f"{path}[{i}]", problems))
    return out


def check_claim(obj: Any, schema: Dict[str, Shape]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """The claim cleaned to the schema (integral floats to int), or the problems."""
    if not isinstance(obj, dict):
        return None, ["the reply holds no JSON object"]
    problems: List[str] = []
    missing = [k for k in schema if k not in obj]
    extra = [k for k in obj if k not in schema]
    if missing:
        problems.append(f"missing keys {missing}")
    if extra:
        problems.append(f"keys not in the template {extra}")
    claim = {k: None if obj.get(k) is None else _conform(obj[k], s, k, problems)
             for k, s in schema.items()}
    return (None, problems) if problems else (claim, [])


# -----------------------------------------------------------------------------
# Comparison of a claim with the answer key
# -----------------------------------------------------------------------------

_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"})


def norm_text(text: str) -> str:
    """Case, dashes and whitespace folded; a Revit ``Family:Type[:Id]`` name reduced to its type."""
    s = " ".join(text.translate(_DASHES).split()).casefold()
    parts = [p.strip() for p in s.split(":")]
    if len(parts) >= 2 and parts[0] and parts[1]:
        s = parts[1]
    return s


_TOKEN_RE = re.compile(r"[0-9a-z]+(?:-[0-9a-z]+)*")


def _is_number_token(token: str) -> bool:
    """A space number such as ``2c22``, ``1e15-a``, ``2b3``: letters and digits in one token."""
    return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


def space_number(text: str) -> Optional[str]:
    """The one space number in ``LOUNGE 2C22`` / ``2C22 - LOUNGE`` (normalised), else None."""
    found = {t for t in _TOKEN_RE.findall(norm_text(text)) if _is_number_token(t)}
    return found.pop() if len(found) == 1 else None


def names_match(claim: str, gold: str) -> bool:
    """Normalised equality; a gold space number also matches a claim that gives it
    with its long name (and no other number), which the extractor sometimes keeps."""
    want = norm_text(gold)
    if norm_text(claim) == want:
        return True
    return bool(_TOKEN_RE.fullmatch(want)) and _is_number_token(want) and space_number(claim) == want


def _canon(value: Any) -> Any:
    if isinstance(value, str):
        return norm_text(value)
    if isinstance(value, list):
        return [_canon(v) for v in value]
    return value


def _bag_match(claims: List[Any], golds: List[Any], eq: Callable[[Any, int], bool]) -> bool:
    """A perfect matching of claims to gold items (``eq(claim, gold_index)``), order ignored."""
    if len(claims) != len(golds):
        return False
    owner: Dict[int, int] = {}          # gold index -> claim index

    def place(ci: int, seen: set) -> bool:
        for gi in range(len(golds)):
            if gi in seen or not eq(claims[ci], gi):
                continue
            seen.add(gi)
            if gi not in owner or place(owner[gi], seen):
                owner[gi] = ci
                return True
        return False

    return all(place(ci, set()) for ci in range(len(claims)))


def _dedup(items: List[Any]) -> List[Any]:
    seen, out = set(), []
    for item in items:
        k = json.dumps(_canon(item), sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def values_match(claim: Any, gold: Any, shape: Shape, tol: float = NUMBER_TOL) -> bool:
    if claim is None:
        return False
    t = shape["type"]
    if t == "integer":
        return not isinstance(claim, bool) and claim == gold
    if t == "string" or (t == "any" and isinstance(claim, str) and isinstance(gold, str)):
        return isinstance(claim, str) and names_match(claim, gold)
    if t in ("number", "boolean", "any"):
        return same(claim, gold, tol)
    if not isinstance(claim, list):
        return False
    if t == "sequence":
        return len(claim) == len(gold) and all(
            values_match(c, g, shape["items"], tol) for c, g in zip(claim, gold))
    if t == "set":
        return _bag_match(_dedup(claim), gold, lambda c, gi: values_match(c, gold[gi], shape["items"], tol))
    if t == "tuple":
        slots = shape["slots"]
        return _bag_match(claim, gold, lambda c, gi: values_match(c, gold[gi], slots[gi], tol))
    return False


def _complete_route(route: List[Any], gold: List[Any]) -> List[Any]:
    """Add the start and destination (named by the question) if the claimed route leaves them out."""
    if not gold or not route or not all(isinstance(x, str) for x in route):
        return route
    if not names_match(route[0], gold[0]):
        route = [gold[0]] + route
    if not names_match(route[-1], gold[-1]):
        route = route + [gold[-1]]
    return route


def routes_match(claim: Any, gold: List[List[Any]], shape: Shape, tol: float = NUMBER_TOL) -> bool:
    """At least one route claimed, and every claimed route is one of the gold routes."""
    if not isinstance(claim, list) or not claim:
        return False
    item = shape["items"]
    return all(any(values_match(c, g, item, tol) or values_match(_complete_route(c, g), g, item, tol)
                   for g in gold)
               for c in _dedup(claim) if isinstance(c, list))


def compare_claim(claim: Optional[Dict[str, Any]], answer: Dict[str, Any],
                  schema: Dict[str, Shape], tol: float = NUMBER_TOL) -> Dict[str, bool]:
    """Per key: does the claim match the answer key?"""
    out = {}
    for k, shape in schema.items():
        value = None if claim is None else claim.get(k)
        if k in ROUTE_KEYS and shape["type"] == "set":
            out[k] = routes_match(value, answer[k], shape, tol)
        else:
            out[k] = values_match(value, answer[k], shape, tol)
    return out


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------

EXTRACTOR_SYSTEM = """You turn a written answer into JSON. You record what the answer claims; you do not judge, correct or complete it.

You get a QUESTION about a building model, a TEMPLATE (the JSON keys to fill and the type of each value) and the ANSWER another system gave to the question. Return one JSON object with exactly the template's keys, filled from the ANSWER only:

1. Copy, do not fix. Record what the answer states as its answer, even if it looks wrong, contradicts the question or contradicts itself. Never use your own knowledge, never correct a value, never fill a gap with what the answer probably meant.
2. Do not compute. Do not count, add up, round or otherwise derive a value the answer does not state itself. The only conversion allowed is of units: a number goes in the unit its key names (`_m` metres, `_m2` square metres, `_mm` millimetres); a length or thickness whose key names no unit goes in metres.
3. null for what is not stated. If the answer gives no value for a key, use null for that key. If the answer declines, says it cannot determine the answer, or reports that nothing was found because its query failed, use null, not an empty list; an empty list only where the answer asserts that there are none. Inside a fixed-length list [...], use null for a part the answer does not state.
4. The answer's conclusion only: not alternatives it mentions and rejects, not candidates it discards, not intermediate results.
5. Spaces by number alone. Wherever the answer gives a room or space number, use just that number: "CONF. ROOM 3Z07", "3Z07 - CONF. ROOM" and "CONF. ROOM (3Z07)" all become "3Z07". Only if the answer gives no number for a space, copy the name it uses.
6. Lists. "list of ..., any order": every item the answer gives, each once. "ordered list": the items in the answer's order; a route lists the spaces from the start to the destination, both included. "[..., ...]": one fixed-length list per item the answer gives, its parts in the order the question asks for them.
7. true or false only where the answer says yes or no to that point.
8. Names of types, materials and floors: copy them as the answer writes them.

Reply with the JSON object only."""

EXTRACTOR_USER = """QUESTION:
{question}

TEMPLATE:
{template}

ANSWER:
{answer}"""

RETRY_USER = ("That reply does not fit the template: {problems}. Reply with only the JSON object, "
              "with exactly the template's keys and value types; null where the answer states nothing.")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PROMPT_SHA256 = _sha(EXTRACTOR_SYSTEM + "\n" + EXTRACTOR_USER + "\n" + RETRY_USER)


@dataclass
class Extraction:
    claim: Optional[Dict[str, Any]]
    error: Optional[str] = None
    attempts: int = 0
    raw: List[str] = field(default_factory=list)
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    served_by: List[str] = field(default_factory=list)


def make_extractor(spec: str = DEFAULT_EXTRACTOR, provider_order: Optional[List[str]] = None,
                   reasoning: str = "default") -> ChatClient:
    return client_from_spec(spec, provider_order=provider_order, reasoning=reasoning,
                            temperature=0.0, max_tokens=EXTRACTOR_MAX_TOKENS)


def extract_claim(client: ChatClient, question: str, schema: Dict[str, Shape], answer: str) -> Extraction:
    """What ``answer`` claims, in ``schema``'s keys. Re-asks once if the reply does not fit."""
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM},
        {"role": "user", "content": EXTRACTOR_USER.format(
            question=question, template=render_template(schema), answer=answer)},
    ]
    ex = Extraction(claim=None)
    for attempt in range(PARSE_RETRIES + 1):
        completion = client.complete(messages, response_format={"type": "json_object"})
        ex.attempts += 1
        ex.raw.append(completion.text)
        ex.cost_usd += completion.cost_usd or 0.0
        ex.prompt_tokens += completion.prompt_tokens
        ex.completion_tokens += completion.completion_tokens
        if completion.provider and completion.provider not in ex.served_by:
            ex.served_by.append(completion.provider)
        claim, problems = check_claim(first_json_object(completion.text or ""), schema)
        if claim is not None:
            ex.claim, ex.error = claim, None
            return ex
        ex.error = "; ".join(problems)
        if completion.truncated:
            ex.error = f"truncated reply: {ex.error}"
            break
        if attempt < PARSE_RETRIES:
            messages = messages + [
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": RETRY_USER.format(problems="; ".join(problems))},
            ]
    return ex


# -----------------------------------------------------------------------------
# Runs, cache and scoring
# -----------------------------------------------------------------------------

def latest_records(path: Path) -> Dict[Key, Dict[str, Any]]:
    """The latest record per cell (as ``src.agent.run.latest_records``, without its Neo4j imports)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {(r["model_id"], r["question_id"], r["paradigm"]): r
                for r in (json.loads(line) for line in f if line.strip())}


def load_questions(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("answer"), dict):
                raise ValueError(f"{path}:{n} (id {row.get('id')}): no structured 'answer'")
            out[(f"{row['project']}__{row['ifc_model']}", row["id"])] = row
    return out


def cache_key(extractor: Dict[str, Any], schema: Dict[str, Shape], question: str, answer: str) -> str:
    """Everything the extractor is sent: a change to the prompt, the template's
    wording, the question or the answer text makes a new extraction."""
    return _sha(json.dumps([extractor, PROMPT_SHA256, render_template(schema), question, _sha(answer)],
                           sort_keys=True))


def read_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {e["cache_key"]: e for e in (json.loads(line) for line in f if line.strip())}


def judge_verdict(rec: Dict[str, Any]) -> str:
    return (rec.get("judgement") or {}).get("verdict") or "unjudged"


def needs_extraction(rec: Dict[str, Any]) -> bool:
    """An episode that ended with an answer of its own (not a canned limit message)."""
    return rec["episode"]["status"] == "ok" and bool((rec["episode"].get("answer") or "").strip())


def score_cell(rec: Dict[str, Any], question: Dict[str, Any], schema: Dict[str, Shape],
               extraction: Optional[Dict[str, Any]], tol: float) -> Dict[str, Any]:
    episode = rec["episode"]
    answer = episode.get("answer") or ""
    verdict = judge_verdict(rec)
    reasons = []
    if verdict == "abstained":
        reasons.append("judge")
    if episode["status"] != "ok":
        reasons.append(f"episode_{episode['status']}")
    elif not answer.strip():
        reasons.append("empty_answer")
    claim = (extraction or {}).get("claim")
    if extraction is not None and claim is not None and all(v is None for v in claim.values()):
        reasons.append("no_claim")
    match = compare_claim(claim, question["answer"], schema, tol)
    key_match = claim is not None and all(match.values())
    unstated = [k for k in schema if claim is None or claim.get(k) is None]
    # Secondary: nothing the answer states is wrong. Some keys hold more than
    # the question strictly asks for (9003's room list, 9008's door count), so
    # a terse correct answer misses the exact match on an unstated key.
    stated_match = (claim is not None and len(unstated) < len(schema)
                    and all(match[k] for k in schema if k not in unstated))
    status = ("not_extracted" if not needs_extraction(rec)
              else "missing" if extraction is None
              else "failed" if extraction.get("error") else "ok")
    return {
        "model_id": rec["model_id"], "question_id": rec["question_id"], "paradigm": rec["paradigm"],
        "kind": question.get("kind"), "needs": question.get("needs"), "category": question.get("category"),
        "episode_status": episode["status"], "judge_verdict": verdict,
        "answer_sha256": _sha(answer),
        "extraction_status": status,
        "extraction_error": (extraction or {}).get("error"),
        "claim": claim, "expected": question["answer"], "match": match, "unstated": unstated,
        "key_match": key_match, "stated_match": stated_match,
        "abstained": bool(reasons), "abstained_by": reasons,
        "correct": key_match and not reasons,
        "stated_correct": stated_match and not reasons,
    }


def _acc(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    key = sum(r["correct"] for r in rows)
    stated = sum(r["stated_correct"] for r in rows)
    judge = sum(r["judge_verdict"] == "correct" for r in rows)
    return {"n": n, "key_correct": key, "key_accuracy": round(key / n, 4) if n else None,
            "stated_correct": stated, "stated_accuracy": round(stated / n, 4) if n else None,
            "judge_correct": judge, "judge_accuracy": round(judge / n, 4) if n else None}


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    by_paradigm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_paradigm[r["paradigm"]].append(r)
    for paradigm, rs in sorted(by_paradigm.items()):
        rs = sorted(rs, key=lambda r: r["question_id"])
        confusion: Dict[str, Counter] = defaultdict(Counter)
        for r in rs:
            confusion[r["judge_verdict"]]["match" if r["key_match"] else "mismatch"] += 1

        def ids(pred: Callable[[Dict[str, Any]], bool], rs: List[Dict[str, Any]] = rs) -> List[int]:
            return [r["question_id"] for r in rs if pred(r)]

        groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            name: defaultdict(list) for name in ("needs", "kind")}
        for r in rs:
            for name in groups:
                groups[name][str(r[name])].append(r)
        out[paradigm] = {
            **_acc(rs),
            "key_abstained": sum(r["abstained"] for r in rs),
            "judge_verdicts": dict(Counter(r["judge_verdict"] for r in rs)),
            "extraction_failed": ids(lambda r: r["extraction_status"] in ("failed", "missing")),
            "by_needs": {k: _acc(v) for k, v in sorted(groups["needs"].items())},
            "by_kind": {k: _acc(v) for k, v in sorted(groups["kind"].items())},
            # rows: judge verdict; columns: does the extracted claim match the key
            # (raw, before the abstention rule)
            "confusion": {v: {"match": c["match"], "mismatch": c["mismatch"]}
                          for v, c in sorted(confusion.items())},
            "disagreements": {
                "judge_correct_key_mismatch": ids(lambda r: r["judge_verdict"] == "correct" and not r["key_match"]),
                "judge_wrong_key_match": ids(lambda r: r["judge_verdict"] == "wrong" and r["key_match"]),
                "judge_abstained_key_match": ids(lambda r: r["judge_verdict"] == "abstained" and r["key_match"]),
                # of the judge-correct mismatches, those where only unstated keys fail
                "judge_correct_only_unstated": ids(lambda r: r["judge_verdict"] == "correct"
                                                   and not r["key_match"] and r["stated_match"]),
            },
        }
    return out


def score_run(run_dir: Path, questions: Dict[Tuple[str, int], Dict[str, Any]], questions_path: Path,
              client: Optional[ChatClient], extractor: Dict[str, Any], tol: float = NUMBER_TOL,
              workers: int = 4, retry_failed: bool = False) -> Dict[str, Any]:
    """Extract (through the cache), compare, and write ``answer_key.jsonl`` + ``answer_key_summary.json``.

    ``client=None`` scores from the cache alone; a cell without a cached
    extraction is then reported as ``missing``.
    """
    records = latest_records(run_dir / "episodes.jsonl")
    if not records:
        raise FileNotFoundError(f"{run_dir}: no episodes.jsonl records")
    cache_path = run_dir / "answer_key_cache.jsonl"
    cache = read_cache(cache_path)
    write_lock = threading.Lock()

    cells: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Shape]]] = []
    skipped: Counter = Counter()
    for _, rec in sorted(records.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        q = questions.get((rec["model_id"], rec["question_id"]))
        if q is None:
            skipped["not_in_questions"] += 1
        elif rec["episode"]["status"] == "error":
            skipped["episode_error"] += 1     # to be re-run by src.agent.run, not scored
        else:
            cells.append((rec, q, derive_schema(q["answer"])))

    def cell_key(rec, q, schema) -> str:
        return cache_key(extractor, schema, q["question"], rec["episode"]["answer"])

    todo = []
    for rec, q, schema in cells:
        if not needs_extraction(rec):
            continue
        hit = cache.get(cell_key(rec, q, schema))
        if hit is None or (retry_failed and hit.get("error")):
            todo.append((rec, q, schema))

    new_cost = 0.0
    if todo and client is not None:
        logger.info("%s: extracting %d cells (%d cached)", run_dir.name, len(todo),
                    sum(needs_extraction(r) for r, _, _ in cells) - len(todo))

        def work(rec, q, schema) -> Dict[str, Any]:
            ex = extract_claim(client, q["question"], schema, rec["episode"]["answer"])
            return {"cache_key": cell_key(rec, q, schema), "model_id": rec["model_id"],
                    "question_id": rec["question_id"], "paradigm": rec["paradigm"],
                    "extractor": extractor, "prompt_sha256": PROMPT_SHA256,
                    "answer_sha256": _sha(rec["episode"]["answer"]),
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **asdict(ex)}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, *t): t for t in todo}
            for fut in as_completed(futures):
                rec = futures[fut][0]
                try:
                    entry = fut.result()
                except Exception as exc:  # noqa: BLE001 - an API failure is not cached; a re-run retries it
                    logger.error("q%s %s: extraction failed: %s", rec["question_id"], rec["paradigm"], exc)
                    continue
                new_cost += entry["cost_usd"]
                with write_lock, open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                cache[entry["cache_key"]] = entry
                if entry["error"]:
                    logger.warning("q%s %s: extraction did not fit the schema: %s",
                                   entry["question_id"], entry["paradigm"], entry["error"][:200])
    elif todo:
        logger.warning("%s: %d cells have no cached extraction (--cache-only)", run_dir.name, len(todo))

    rows = []
    for rec, q, schema in cells:
        cached = cache.get(cell_key(rec, q, schema)) if needs_extraction(rec) else None
        rows.append(score_cell(rec, q, schema, cached, tol))
    with open(run_dir / "answer_key.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    used = {cell_key(rec, q, schema) for rec, q, schema in cells if needs_extraction(rec)}
    config_path = run_dir / "config.json"
    run_questions = (json.loads(config_path.read_text()).get("questions") or {}) if config_path.exists() else {}
    questions_sha = hashlib.sha256(questions_path.read_bytes()).hexdigest()
    if run_questions.get("sha256") and run_questions["sha256"] != questions_sha:
        logger.warning("%s was run on another version of %s (sha256 %s, now %s)", run_dir.name,
                       questions_path, run_questions["sha256"][:12], questions_sha[:12])
    summary = {
        "run": str(run_dir),
        "questions": {"path": str(questions_path), "sha256": questions_sha,
                      "run_sha256": run_questions.get("sha256")},
        "extractor": {**extractor, "prompt_sha256": PROMPT_SHA256, "temperature": 0.0},
        "number_tolerance": tol,
        "cells": len(rows), "skipped": dict(skipped),
        "extraction_cost_usd": {"this_invocation": round(new_cost, 6),
                                "all_used": round(sum((cache[k].get("cost_usd") or 0) for k in used if k in cache), 6)},
        "paradigms": summarise(rows),
    }
    (run_dir / "answer_key_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"\n{summary['run']}")
    print(f"{'paradigm':12s} {'n':>3s} {'key':>6s} {'stated':>6s} {'judge':>6s}   "
          f"by needs (key / stated / judge correct)")
    for par, s in summary["paradigms"].items():
        needs = "  ".join(f"{k} {v['key_correct']}/{v['stated_correct']}/{v['judge_correct']} of {v['n']}"
                          for k, v in s["by_needs"].items())
        print(f"{par:12s} {s['n']:3d} {s['key_accuracy']:6.3f} {s['stated_accuracy']:6.3f} "
              f"{s['judge_accuracy']:6.3f}   {needs}")
        d = s["disagreements"]
        print(f"{'':12s} judge correct / key mismatch: {d['judge_correct_key_mismatch']}; "
              f"judge wrong / key match: {d['judge_wrong_key_match']}; "
              f"judge abstained / key match: {d['judge_abstained_key_match']}")
        if s["extraction_failed"]:
            print(f"{'':12s} extraction failed: {s['extraction_failed']}")
    cost = summary["extraction_cost_usd"]
    print(f"extraction cost: ${cost['this_invocation']:.4f} now, ${cost['all_used']:.4f} for all cells used")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", type=Path, nargs="+", metavar="RUN_DIR", help="results/ifcbench/runs/<run>")
    p.add_argument("--questions", type=Path, required=True,
                   help="The authored question file the runs asked (structured 'answer' per row)")
    p.add_argument("--extractor", default=DEFAULT_EXTRACTOR, help=f"Default: {DEFAULT_EXTRACTOR}")
    p.add_argument("--extractor-provider", default=DEFAULT_EXTRACTOR_PROVIDER,
                   help="Pin the extractor's OpenRouter provider(s), comma-separated "
                        f"(default: {DEFAULT_EXTRACTOR_PROVIDER}; '' for none)")
    p.add_argument("--extractor-reasoning", choices=REASONING_MODES, default="default",
                   help="default sends nothing (the default extractor has no thinking)")
    p.add_argument("--tol", type=float, default=NUMBER_TOL, help="Relative tolerance for non-integer numbers")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-only", action="store_true", help="Score from cached extractions; no LLM calls")
    p.add_argument("--retry-failed", action="store_true",
                   help="Extract again the cells whose cached extraction did not fit the schema")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    providers = [v.strip() for v in args.extractor_provider.split(",") if v.strip()] or None
    extractor = {"spec": args.extractor, "provider": providers, "reasoning": args.extractor_reasoning}
    client = None if args.cache_only else make_extractor(args.extractor, providers, args.extractor_reasoning)
    questions = load_questions(args.questions)
    for run_dir in args.runs:
        print_summary(score_run(run_dir, questions, args.questions, client, extractor, tol=args.tol,
                                workers=args.workers, retry_failed=args.retry_failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())

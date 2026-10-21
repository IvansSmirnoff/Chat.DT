"""Parse the model's JSON action out of free text.

Not every OpenRouter backbone supports ``response_format`` (glm-4.5-air does
not), so the protocol is "a JSON object in the reply" and the parser is
tolerant of what models actually emit: markdown fences, a sentence before the
object, a trailing comment. It is strict about the content: an action without
its required field is a parse error, reported back to the model as such.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

_FENCE_RE = re.compile(r"```(?:json|cypher)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class ActionParseError(ValueError):
    pass


@dataclass
class Action:
    kind: str  # "cypher" | "final_answer" | "decline"
    thoughts: str = ""
    cypher: str = ""
    answer: str = ""


def first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """The first decodable top-level JSON object in ``text``.

    Fenced blocks are tried first, then the raw text; if nothing decodes, once
    more with trailing commas removed (``{"a": 1,}``), which several backbones
    emit and which BAML's own parser also accepts.
    """
    decoder = json.JSONDecoder()
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for clean in (lambda s: s, lambda s: _TRAILING_COMMA_RE.sub(r"\1", s)):
        for chunk in map(clean, candidates):
            for start in [i for i, c in enumerate(chunk) if c == "{"]:
                try:
                    obj, _ = decoder.raw_decode(chunk[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    return obj
    return None


def parse_action(text: str, allowed: tuple = ("cypher", "final_answer")) -> Action:
    """Parse an agent step. A bare ``cypher``/``answer`` object implies its kind.

    ``decline`` is only for callers that allow it (the single-shot generator): a
    well-formed reply with neither a query nor an answer, which is how a model
    says it finds nothing to query (``{"thoughts": "...", "cypher": ""}``).
    """
    obj = first_json_object(text or "")
    if obj is None:
        raise ActionParseError("no JSON object found in the reply")
    if ("decline" in allowed and not str(obj.get("cypher") or "").strip()
            and not str(obj.get("answer") or "").strip()):
        return Action(kind="decline", thoughts=str(obj.get("thoughts") or ""))
    kind = str(obj.get("action") or "").strip().lower()
    if not kind:
        kind = "cypher" if obj.get("cypher") else "final_answer" if obj.get("answer") else ""
    kind = {"cypheraction": "cypher", "finalanswer": "final_answer", "final": "final_answer",
            "answer": "final_answer"}.get(kind.replace("_", "").replace(" ", ""), kind)
    if kind not in allowed:
        raise ActionParseError(f"'action' must be one of {allowed}, got {obj.get('action')!r}")
    action = Action(kind=kind, thoughts=str(obj.get("thoughts") or ""))
    if kind == "cypher":
        action.cypher = str(obj.get("cypher") or "").strip()
        if not action.cypher:
            raise ActionParseError("a cypher action needs a non-empty 'cypher' field")
    else:
        action.answer = str(obj.get("answer") or "").strip()
        if not action.answer:
            raise ActionParseError("a final_answer action needs a non-empty 'answer' field")
    return action

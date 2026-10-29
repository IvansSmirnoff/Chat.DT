"""Context-free grammar for the agent's replies (llguidance Lark dialect).

Sent with vLLM's ``structured_outputs: {"grammar": ...}`` (server flag
``--structured-outputs-config.backend guidance``), it constrains a local model's
whole reply: the JSON envelope the agent prompts ask for (``prompts.py``), and
inside it a read-only Cypher query whose labels, relationship types and property
keys can only be ones the loaded graph carries. It replaces nothing on the API
backbones; it is the decoding-side factor of the design ladder.

The vocabulary is a graph summary (``labels``, ``relationships``,
``property_keys``): ``collect_graph_stats`` on the live graph, or the header of
a snapshot, which holds the same counts. So a model's grammar can be built in
the run or offline (``scripts/grammar_coverage.py``).

Two rules come from the decoding side, not from Cypher:

* Whitespace is spaces only, everywhere. The query sits inside a JSON string,
  which cannot hold a raw newline, so the reply is one line.
* Identifiers, aliases and string literals are length-capped. An unbounded
  identifier is the FSM trap ``grammar.py`` documents: a small model can pack
  its whole token budget into one name that stays grammar-valid.

Not yet here (P4 proper): per-label property sets, relationship signatures,
IDS value enumerations.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

REPLIES = ("agent", "generator")

_PLAIN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Read-only Cypher functions seen in the API backbones' queries, plus their
# obvious neighbours. Matched case-insensitively, as Neo4j does.
FUNCTIONS = (
    "abs", "avg", "ceil", "coalesce", "collect", "count", "elementId", "endNode",
    "floor", "head", "keys", "labels", "last", "left", "length", "max", "min",
    "nodes", "properties", "range", "relationships", "replace", "reverse", "right",
    "round", "size", "split", "sqrt", "startNode", "substring", "sum", "tail",
    "toBoolean", "toFloat", "toInteger", "toLower", "toString", "toUpper", "trim",
    "type",
)

_ENVELOPES = {
    # {"action": "cypher", "thoughts": ..., "cypher": ...} |
    # {"action": "final_answer", "thoughts": ..., "answer": ...}   (agent_messages)
    "agent": r'''
start: "{" "\"action\"" ":" (cypher_action | answer_action) "}"
cypher_action: "\"cypher\"" "," thoughts "," "\"cypher\"" ":" cypher_string
answer_action: "\"final_answer\"" "," thoughts "," "\"answer\"" ":" JSON_STRING
''',
    # {"thoughts": ..., "cypher": ...}; an empty query declines and an answer
    # answers outright, as loop.single_shot accepts (static_generator_messages)
    "generator": r'''
start: "{" thoughts "," (query_field | answer_field) "}"
query_field: "\"cypher\"" ":" (cypher_string | EMPTY_STRING)
answer_field: "\"answer\"" ":" JSON_STRING
EMPTY_STRING: "\"\""
''',
}

_COMMON = r'''
thoughts: "\"thoughts\"" ":" JSON_STRING
cypher_string: "\"" query "\""

JSON_STRING: /"(?:[^"\\\x00-\x1F]|\\["\\\/bfnrt]|\\u[0-9a-fA-F]{4})*"/

query: single_query (union single_query)*
union: UNION ALL?
single_query: (reading | with_clause | unwind)* return_clause
reading: OPTIONAL? MATCH pattern ("," pattern)* where?
with_clause: WITH DISTINCT? projection where? order? skip? limit?
unwind: UNWIND expr AS VAR
return_clause: RETURN DISTINCT? projection order? skip? limit?
projection: "*" | proj_item ("," proj_item)*
proj_item: expr (AS VAR)?
where: WHERE expr
order: ORDER BY sort_item ("," sort_item)*
sort_item: expr (ASC | DESC)?
skip: SKIP expr
limit: LIMIT expr

pattern: (VAR "=")? node (rel node)*
node: "(" VAR? node_labels? node_props? ")"
node_labels: (":" LABEL)+ | ":" LABEL ("|" LABEL)+
node_props: "{" prop_pair ("," prop_pair)* "}"
prop_pair: PROP ":" expr
rel: "-" rel_detail? "-" ">"? | "<" "-" rel_detail? "-"
rel_detail: "[" VAR? rel_types? hops? "]"
hops: "*" (INT? (".." INT?)?)?

expr: and_expr ((OR | XOR) and_expr)*
and_expr: not_expr (AND not_expr)*
not_expr: NOT* comparison
comparison: add_expr comp_tail*
comp_tail: COMP_OP add_expr
         | STARTS WITH add_expr
         | ENDS WITH add_expr
         | CONTAINS add_expr
         | IS NOT? NULL
         | IN add_expr
         | "=~" add_expr
add_expr: mul_expr (("+" | "-") mul_expr)*
mul_expr: pow_expr (("*" | "/" | "%") pow_expr)*
pow_expr: unary ("^" unary)*
unary: ("+" | "-")? postfix
postfix: atom postfix_op*
postfix_op: "." PROP | "[" expr "]" | "[" expr? ".." expr? "]"
atom: literal
    | VAR
    | VAR (":" LABEL)+
    | func_call
    | count_star
    | quantifier
    | list_literal
    | list_comprehension
    | map_literal
    | case_expr
    | exists_subquery
    | pattern_predicate
    | "(" expr ")"
func_call: FUNC "(" DISTINCT? (expr ("," expr)*)? ")"
count_star: COUNT "(" "*" ")"
quantifier: (ANY | ALL | NONE | SINGLE) "(" VAR IN expr WHERE expr ")"
list_literal: "[" (expr ("," expr)*)? "]"
list_comprehension: "[" VAR IN expr (WHERE expr)? ("|" expr)? "]"
map_literal: "{" (VAR ":" expr ("," VAR ":" expr)*)? "}"
case_expr: CASE expr? (WHEN expr THEN expr)+ (ELSE expr)? END
exists_subquery: EXISTS "{" MATCH? pattern ("," pattern)* where? "}"
pattern_predicate: node (rel node)+
literal: NUMBER | STRING | TRUE | FALSE | NULL

COMP_OP: "=" | "<>" | "<" | ">" | "<=" | ">="
NUMBER: /[0-9]{1,15}(\.[0-9]{1,10})?([eE][+-]?[0-9]{1,3})?/
INT: /[0-9]{1,6}/
STRING: /'[^'"\\]{0,200}'/ | /\\"[^'"\\]{0,200}\\"/
VAR: /[A-Za-z_][A-Za-z0-9_]{0,30}/

MATCH: "MATCH"i
OPTIONAL: "OPTIONAL"i
WHERE: "WHERE"i
WITH: "WITH"i
UNWIND: "UNWIND"i
RETURN: "RETURN"i
UNION: "UNION"i
ALL: "ALL"i
DISTINCT: "DISTINCT"i
AS: "AS"i
ORDER: "ORDER"i
BY: "BY"i
ASC: "ASC"i | "ASCENDING"i
DESC: "DESC"i | "DESCENDING"i
SKIP: "SKIP"i
LIMIT: "LIMIT"i
AND: "AND"i
OR: "OR"i
XOR: "XOR"i
NOT: "NOT"i
IS: "IS"i
NULL: "NULL"i
TRUE: "TRUE"i
FALSE: "FALSE"i
IN: "IN"i
STARTS: "STARTS"i
ENDS: "ENDS"i
CONTAINS: "CONTAINS"i
CASE: "CASE"i
WHEN: "WHEN"i
THEN: "THEN"i
ELSE: "ELSE"i
END: "END"i
EXISTS: "EXISTS"i
ANY: "ANY"i
NONE: "NONE"i
SINGLE: "SINGLE"i
COUNT: "count"i

%ignore / +/
'''


def _names(values: Iterable[str], kind: str) -> list[str]:
    """Sorted Lark literals for a vocabulary; odd names go in backticks.

    A name holding a quote or backslash could not sit inside the JSON string
    unescaped, so it is left out (logged): the model cannot write it either way.
    """
    out = []
    for name in sorted(set(values)):
        if _PLAIN_NAME.match(name):
            out.append(f'"{name}"')
        elif '"' in name or "\\" in name or "`" in name:
            logger.warning("%s %r cannot appear in the grammar; left out", kind, name)
        else:
            out.append(f'"`{name}`"')
    return out


def build_reply_grammar(
    labels: Iterable[str],
    rel_types: Iterable[str],
    property_keys: Iterable[str],
    reply: str = "agent",
) -> str:
    """The Lark grammar for one reply type over one graph's vocabulary.

    Deterministic for a given vocabulary (names are sorted), so its sha256
    identifies it in a run's config and vLLM's grammar cache keeps hitting.
    """
    if reply not in REPLIES:
        raise ValueError(f"reply must be one of {REPLIES}: {reply!r}")
    label_lits = _names(labels, "label")
    prop_lits = _names(property_keys, "property key")
    rel_lits = _names(rel_types, "relationship type")
    if not label_lits or not prop_lits:
        raise ValueError("a grammar needs at least one label and one property key")
    func_lits = [f'"{f}"i' for f in FUNCTIONS]
    vocab = [
        f"LABEL: {' | '.join(label_lits)}",
        f"PROP: {' | '.join(prop_lits)}",
        f"FUNC: {' | '.join(func_lits)}",
    ]
    if rel_lits:
        vocab.append(f"REL: {' | '.join(rel_lits)}")
        # `[:A|B]`, not the `[:A|:B]` form Neo4j 5 deprecates (and refuses for labels)
        vocab.append('rel_types: ":" REL ("|" REL)*')
    else:
        # No edges in this graph: a relationship pattern can still be written
        # (and matches nothing), but it cannot name a type.
        vocab.append('rel_types: ":" VAR')
    return (_ENVELOPES[reply].strip() + "\n" + _COMMON.strip() + "\n\n"
            + "\n".join(vocab) + "\n")


def grammar_from_summary(summary: Mapping[str, Any], reply: str = "agent") -> str:
    """Grammar from a graph summary: ``collect_graph_stats`` or a snapshot header."""
    return build_reply_grammar(
        summary.get("labels") or {},
        summary.get("relationships") or {},
        summary.get("property_keys") or [],
        reply=reply,
    )


def check_reply(grammar: str, reply: str) -> Optional[str]:
    """``None`` if ``reply`` is a complete sentence of ``grammar``, else why not.

    Byte-level, so acceptance does not depend on any model's tokenizer (the
    tokenizer only changes how the mask is computed, not the language). Needs
    ``llguidance`` (requirements-dev.txt); vLLM brings its own in Colab.
    """
    import llguidance

    matcher = llguidance.LLMatcher(_byte_tokenizer(), llguidance.LLMatcher.grammar_from_lark(grammar),
                                   log_level=0)
    if matcher.is_error():
        return f"grammar error: {matcher.get_error()}"
    data = reply.encode("utf-8")
    for i, byte in enumerate(data):
        if not matcher.consume_tokens([byte]):
            return f"rejected at byte {i}: ...{data[max(0, i - 40):i].decode(errors='replace')}[HERE]{data[i:i + 20].decode(errors='replace')}"
    if not matcher.is_accepting():
        return "incomplete: the reply stops before the grammar can end"
    return None


_BYTE_TOKENIZER = None


def _byte_tokenizer():
    """An LLTokenizer whose tokens are the 256 bytes, plus an EOS."""
    global _BYTE_TOKENIZER
    if _BYTE_TOKENIZER is None:
        import llguidance

        class _Bytes:
            eos_token_id = 256
            bos_token_id = None
            tokens = tuple(bytes([i]) for i in range(256)) + (b"<eos>",)
            special_token_ids = (256,)

            def __call__(self, text):
                return list(text if isinstance(text, bytes) else text.encode("utf-8"))

        _BYTE_TOKENIZER = llguidance.LLTokenizer(llguidance.TokenizerWrapper(_Bytes()))
    return _BYTE_TOKENIZER

"""The two paradigms, with identical accounting.

``single_shot`` mirrors Hellin's *static* arm: one generator call writes one
Cypher query, it is executed once, and a synthesiser call writes the answer
from the query and its result. No feedback reaches the generator.

``agentic`` mirrors his CodeAct agent: each step the model sees the question
and every previous attempt (query + result or error) and picks CypherAction or
FinalAnswer. Running out of steps is an abstention, as in his agent (25 steps).

Both return an :class:`Episode` holding every LLM call's tokens, cost and
latency and every query's result, so cost and accuracy come from one record.
The stored record keeps no more of a result than the model was shown
(:meth:`Step.to_dict`). An infrastructure failure (the LLM API, not the model)
is ``status="error"`` and is meant to be re-run, not scored.

A reply cut off at max_tokens is the model's doing, not the API's, and asking
again with the same budget only repeats it (GLM-4.5-air with thinking on spends
any budget, 32k measured, imagining query results). So it is never re-asked: in
single-shot it ends the episode as a ``token_limit`` abstention, the analogue of
the agent's ``step_limit``; in the agentic loop it costs a step.

``grammar`` (vLLM only, ``src/constraints/cfg_grammar.py``) constrains the calls
that write a query: the single-shot generator and every agent step. The
synthesiser is never constrained: it writes prose.

``prompt_variant`` (``prompts.PROMPT_VARIANTS``) picks the answer-writing
instructions: ``ours`` adds ``prompts.ANSWER_ADDENDUM`` to the synthesiser and
the agent prompts, never to the generator. It changes prompt text only, not the
reply envelope, so the grammar is the same under both. The episode records it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.agent import prompts
from src.agent.actions import Action, ActionParseError, parse_action
from src.agent.llm import ChatClient, Completion, LLMError
from src.agent.tools import GraphTools

MAX_STEPS = 25
PARSE_RETRIES = 2
RESULT_CHARS = 4000        # observation shown for recent steps
OLD_RESULT_CHARS = 600     # ... and for steps older than KEEP_FULL
KEEP_FULL = 5
STEP_LIMIT_ANSWER = "I cannot determine the answer: the step limit was reached before sufficient information was gathered."
TOKEN_LIMIT_ANSWER = "I cannot determine the answer: the reply was cut off at the token limit before it was complete."
PARSE_ERROR_ANSWER = "I cannot determine the answer: no usable reply was produced."
NO_QUERY_RESULT = "(no query was run: the query writer returned no query)"
REPLY_CHARS = 2000         # of an unusable reply, kept in the record for diagnosis
TRUNCATED_STEP = "it was cut off at the token limit before any JSON action; keep your reasoning short and reply with the action"


class TokenLimit(Exception):
    """A reply hit max_tokens before it contained a usable action."""


@dataclass
class Step:
    index: int
    thoughts: str = ""
    cypher: str = ""
    result: Optional[Dict[str, Any]] = None
    observation: str = ""
    parse_error: Optional[str] = None
    reply: str = ""               # the raw reply, only when it could not be parsed

    def to_dict(self) -> Dict[str, Any]:
        """The stored step: no more of the result than the model was ever shown.

        The model sees at most RESULT_CHARS of an observation (the synthesiser
        and recent agent steps; older steps less), so that clip is kept, with
        the full length and whether it was cut. The result keeps its columns,
        true row count and error but not its rows: the observation holds as many
        as the model saw. Full results made one Qwen3-32B episode 3M chars.
        """
        d = asdict(self)
        d["observation"] = _clip(self.observation, RESULT_CHARS)
        d["observation_chars"] = len(self.observation)
        d["observation_clipped"] = len(self.observation) > RESULT_CHARS
        if self.result is not None:
            d["result"] = {k: v for k, v in self.result.items() if k != "rows"}
        return d


@dataclass
class Episode:
    question_id: int
    question: str
    paradigm: str
    model: str
    answer: str = ""
    status: str = "ok"            # ok | step_limit | token_limit | parse_error | error
    error: Optional[str] = None
    steps: List[Step] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    providers: List[str] = field(default_factory=list)
    calls: List[Dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    prompt_variant: str = "hellin"

    def account(self, c: Completion, role: str) -> None:
        self.llm_calls += 1
        self.prompt_tokens += c.prompt_tokens
        self.completion_tokens += c.completion_tokens
        self.reasoning_tokens += c.reasoning_tokens
        self.cost_usd += c.cost_usd or 0.0
        if c.provider and c.provider not in self.providers:
            self.providers.append(c.provider)
        self.calls.append({"role": role, "prompt_tokens": c.prompt_tokens,
                           "completion_tokens": c.completion_tokens,
                           "reasoning_tokens": c.reasoning_tokens, "finish": c.finish_reason,
                           "provider": c.provider, "latency_s": round(c.latency_s, 2)})

    @property
    def final_cypher(self) -> str:
        for step in reversed(self.steps):
            if step.cypher and step.result and not step.result.get("error"):
                return step.cypher
        return ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        d["final_cypher"] = self.final_cypher
        d["n_queries"] = sum(1 for s in self.steps if s.cypher)
        return d


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} chars clipped]"


def _constrained(grammar: Optional[str]) -> Dict[str, Any]:
    """The request fields that hold one reply to ``grammar`` (none without one)."""
    return {"structured_outputs": {"grammar": grammar}} if grammar else {}


def _ask(llm: ChatClient, messages: List[Dict[str, str]], episode: Episode,
         allowed: tuple, role: str, grammar: Optional[str] = None) -> Action:
    """One LLM call, re-asked up to PARSE_RETRIES times if the JSON is unusable.

    A truncated reply is not re-asked (see the module docstring).
    """
    messages = list(messages)
    for attempt in range(PARSE_RETRIES + 1):
        completion = llm.complete(messages, **_constrained(grammar))
        episode.account(completion, role)
        try:
            return parse_action(completion.text, allowed)
        except ActionParseError as exc:
            if completion.truncated:
                raise TokenLimit(f"{role} reply truncated at the token limit "
                                 f"({completion.completion_tokens} completion tokens)") from exc
            if attempt == PARSE_RETRIES:
                raise ActionParseError(f"{exc}; last reply: {completion.text[:REPLY_CHARS]!r}") from exc
            messages += [
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": f"Your reply could not be parsed ({exc}). "
                                            f"{prompts.JSON_RULES}"},
            ]
    raise AssertionError("unreachable")


def single_shot(question_id: int, question: str, schema: str,
                llm: ChatClient, tools: GraphTools, grammar: Optional[str] = None,
                prompt_variant: str = "hellin") -> Episode:
    episode = Episode(question_id, question, "single_shot", llm.spec or llm.model,
                      prompt_variant=prompt_variant)
    start = time.monotonic()
    try:
        # The prompt asks for a query. A generator that answers outright is taken
        # at its word; one that declines (no query, no answer) leaves the
        # synthesiser to answer from nothing, as when Hellin's code fails to run.
        gen = _ask(llm, prompts.static_generator_messages(question, schema), episode,
                   ("cypher", "final_answer", "decline"), "generator", grammar)
        if gen.kind == "final_answer":
            episode.answer = gen.answer
        else:
            if gen.kind == "cypher":
                result = tools.run_cypher(gen.cypher)
                observation = result.render()
                episode.steps.append(Step(1, gen.thoughts, gen.cypher, result.to_dict(), observation))
            else:
                observation = NO_QUERY_RESULT
            final = _ask(llm, prompts.synthesizer_messages(
                question, gen.cypher, _clip(observation, RESULT_CHARS), prompt_variant), episode,
                ("final_answer",), "synthesizer")
            episode.answer = final.answer
    except TokenLimit as exc:
        episode.status, episode.answer, episode.error = "token_limit", TOKEN_LIMIT_ANSWER, str(exc)
    except ActionParseError as exc:
        # The model's failure after PARSE_RETRIES re-asks: scored, not re-run.
        episode.status, episode.answer, episode.error = "parse_error", PARSE_ERROR_ANSWER, str(exc)
    except LLMError as exc:
        episode.status, episode.error = "error", f"{type(exc).__name__}: {exc}"
    episode.wall_s = time.monotonic() - start
    return episode


def _previous_attempts(steps: List[Step]) -> str:
    if not steps:
        return ""
    parts = ["", "## PREVIOUS ATTEMPTS"]
    recent_from = len(steps) - KEEP_FULL
    for s in steps:
        limit = RESULT_CHARS if s.index > recent_from else OLD_RESULT_CHARS
        parts.append(f"\n### Step {s.index}")
        if s.parse_error:
            parts.append(f"Your reply could not be parsed: {s.parse_error}")
            continue
        if s.thoughts:
            parts.append(f"Thoughts: {s.thoughts}")
        parts.append(f"Cypher: {s.cypher}")
        parts.append("Result:\n" + _clip(s.observation, limit))
    return "\n".join(parts)


def agentic(question_id: int, question: str, schema: str,
            llm: ChatClient, tools: GraphTools, max_steps: int = MAX_STEPS,
            grammar: Optional[str] = None, prompt_variant: str = "hellin") -> Episode:
    episode = Episode(question_id, question, "agentic", llm.spec or llm.model,
                      prompt_variant=prompt_variant)
    start = time.monotonic()
    try:
        for index in range(1, max_steps + 1):
            messages = prompts.agent_messages(
                question, schema, _previous_attempts(episode.steps),
                max_steps=max_steps, max_rows=tools.max_rows, timeout=tools.timeout_s,
                variant=prompt_variant,
            )
            completion = llm.complete(messages, **_constrained(grammar))
            episode.account(completion, "agent")
            try:
                action = parse_action(completion.text)
            except ActionParseError as exc:
                # Costs a step, like any other wasted action; the next prompt says why.
                reason = TRUNCATED_STEP if completion.truncated else str(exc)
                episode.steps.append(Step(index, parse_error=reason, reply=completion.text[:REPLY_CHARS]))
                continue
            if action.kind == "final_answer":
                episode.answer = action.answer
                break
            result = tools.run_cypher(action.cypher)
            episode.steps.append(Step(index, action.thoughts, action.cypher,
                                      result.to_dict(), result.render()))
        else:
            episode.status, episode.answer = "step_limit", STEP_LIMIT_ANSWER
    except LLMError as exc:
        episode.status, episode.error = "error", f"LLMError: {exc}"
    episode.wall_s = time.monotonic() - start
    return episode

"""The two paradigms, with identical accounting.

``single_shot`` mirrors Hellin's *static* arm: one generator call writes one
Cypher query, it is executed once, and a synthesiser call writes the answer
from the query and its result. No feedback reaches the generator.

``agentic`` mirrors his CodeAct agent: each step the model sees the question
and every previous attempt (query + result or error) and picks CypherAction or
FinalAnswer. Running out of steps is an abstention, as in his agent (25 steps).

Both return an :class:`Episode` holding every LLM call's tokens, cost and
latency and every query's result, so cost and accuracy come from one record.
An infrastructure failure (the LLM API, not the model) is ``status="error"``
and is meant to be re-run, not scored.
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


@dataclass
class Step:
    index: int
    thoughts: str = ""
    cypher: str = ""
    result: Optional[Dict[str, Any]] = None
    observation: str = ""
    parse_error: Optional[str] = None


@dataclass
class Episode:
    question_id: int
    question: str
    paradigm: str
    model: str
    answer: str = ""
    status: str = "ok"            # ok | step_limit | error
    error: Optional[str] = None
    steps: List[Step] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    providers: List[str] = field(default_factory=list)
    wall_s: float = 0.0

    def account(self, c: Completion) -> None:
        self.llm_calls += 1
        self.prompt_tokens += c.prompt_tokens
        self.completion_tokens += c.completion_tokens
        self.cost_usd += c.cost_usd or 0.0
        if c.provider and c.provider not in self.providers:
            self.providers.append(c.provider)

    @property
    def final_cypher(self) -> str:
        for step in reversed(self.steps):
            if step.cypher and step.result and not step.result.get("error"):
                return step.cypher
        return ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["final_cypher"] = self.final_cypher
        d["n_queries"] = sum(1 for s in self.steps if s.cypher)
        return d


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} chars clipped]"


def _ask(llm: ChatClient, messages: List[Dict[str, str]], episode: Episode,
         allowed: tuple) -> Action:
    """One LLM call, re-asked up to PARSE_RETRIES times if the JSON is unusable."""
    messages = list(messages)
    for attempt in range(PARSE_RETRIES + 1):
        completion = llm.complete(messages)
        episode.account(completion)
        try:
            return parse_action(completion.text, allowed)
        except ActionParseError as exc:
            if attempt == PARSE_RETRIES:
                raise
            messages += [
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": f"Your reply could not be parsed ({exc}). "
                                            f"{prompts.JSON_RULES}"},
            ]
    raise AssertionError("unreachable")


def single_shot(question_id: int, question: str, schema: str,
                llm: ChatClient, tools: GraphTools) -> Episode:
    episode = Episode(question_id, question, "single_shot", llm.spec or llm.model)
    start = time.monotonic()
    try:
        gen = _ask(llm, prompts.static_generator_messages(question, schema), episode, ("cypher",))
        result = tools.run_cypher(gen.cypher)
        observation = result.render()
        episode.steps.append(Step(1, gen.thoughts, gen.cypher, result.to_dict(), observation))
        final = _ask(llm, prompts.synthesizer_messages(
            question, gen.cypher, _clip(observation, RESULT_CHARS)), episode, ("final_answer",))
        episode.answer = final.answer
    except (LLMError, ActionParseError) as exc:
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
            llm: ChatClient, tools: GraphTools, max_steps: int = MAX_STEPS) -> Episode:
    episode = Episode(question_id, question, "agentic", llm.spec or llm.model)
    start = time.monotonic()
    try:
        for index in range(1, max_steps + 1):
            messages = prompts.agent_messages(
                question, schema, _previous_attempts(episode.steps),
                max_steps=max_steps, max_rows=tools.max_rows, timeout=tools.timeout_s,
            )
            completion = llm.complete(messages)
            episode.account(completion)
            try:
                action = parse_action(completion.text)
            except ActionParseError as exc:
                # Costs a step, like any other wasted action; the next prompt says why.
                episode.steps.append(Step(index, parse_error=str(exc)))
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

"""IFC-Bench answer judge, ported from Hellin et al. (2026).

The prompt is ``EvaluateResponse`` from ``answer_verifier.baml`` and the verdict
rule is ``derive_binary_classification`` from ``answer_verifier.py`` in
https://github.com/sylvainHellin/cobbie ((c) 2025-2026 Sylvain Hellin, TU
Munich, CC BY 4.0), so accuracies are comparable to the published ones:

  * abstention                         -> "abstained"
  * all four criteria == Yes           -> "correct"
  * anything else (including any 'Na') -> "wrong"

Accuracy is correct / all questions, abstentions included in the denominator.

The judge sees the question, its category, the reference answer and the
system's final answer -- *not* the execution trace -- so every source and
method has to be cited in the answer text itself.

Hellin's primary judge is GLM-4.7 at temperature 0.1 (``clients.baml``). The
only change is the output instruction: BAML's ``ctx.output_format`` is replaced
by an equivalent JSON schema with the same field descriptions (``schemas.baml``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.actions import first_json_object
from src.agent.llm import ChatClient, Completion, LLMError, client_from_spec

DEFAULT_JUDGE = "openrouter:z-ai/glm-4.7"
CRITERIA = ("faithfulness", "completeness", "transparency", "relevance")
VALUES = ("Yes", "No", "Na")
PARSE_RETRIES = 2

JUDGE_SYSTEM = """You are evaluating a BIM information extraction system's response to a question using a multi-criteria assessment framework.

## QUESTION CATEGORIES (Background)

Category 1 (Direct Retrieval): Direct information from BIM element properties, no computation
Category 2 (Computational Aggregation): Simple unambiguous computations (counting, summing, averaging)
Category 3 (Geometric/Spatial): Complex geometric computations from spatial relationships
Category 4 (Incomplete Information): Required information not in model, may need assumptions

---

## EVALUATION WORKFLOW

**Step 1: Determine if system ABSTAINED or PROVIDED AN ANSWER**

ABSTAINED if response contains explicit refusal:
- "I cannot determine..."
- "Insufficient information to answer..."
- "This data is not available and I cannot provide an answer..."
- "I am unable to answer this question..."

NOT abstained:
- Expressions of uncertainty while providing answer
- Conditional statements ("If X, then Y...")
- Partial information with explanation of limitations

**If ABSTAINED:**
- Set abstention = true
- Set ALL other criteria to 'Na'
- Provide brief justification
- STOP evaluation

**If PROVIDED AN ANSWER:**
- Set abstention = false
- Proceed to evaluate 4 quality criteria independently

---

## CRITERION 1: FAITHFULNESS

Are all claims grounded in valid sources and numerically accurate for this question category?

**Category-Specific Numerical Accuracy Requirements:**

**Category 1 & 2 (Deterministic):**
- Numerical values must match reference answer within tolerance
- These categories involve deterministic retrieval/aggregation → should produce same results
- Tolerance rules:
  * Continuous measurements (lengths, areas, volumes): within 2% of reference
    Formula: |system_value - reference_answer | / reference_answer ≤ 0.02
  * Discrete counts (elements, rooms, floors): exact match required
- Fail if: numerical mismatch beyond tolerance

**Category 3 (Geometric):**
- Numerical values can differ from reference answer
- Multiple valid geometric calculation approaches exist
- Require: methodology/approach must be explicitly disclosed
- Fail if: values are implausible OR methodology not disclosed

**Category 4 (Incomplete Information):**
- Numerical values can differ from reference answer
- Answers may involve assumptions leading to different results
- Require: ALL assumptions MUST be disclosed verbatim in answer
  * "Assuming X..." or "Based on assumption that X..." required
  * Implied/hedged assumptions (e.g., "typically...", "generally...") → FAIL
- Fail if: assumptions not explicitly stated OR values implausible

**Source/Method Disclosure (All Categories):**
- All claims should cite sources (properties, functions, calculations)
- Note: This is primarily assessed in Transparency criterion
- For faithfulness, focus on numerical accuracy per category rules above

**Result:**
- Yes: Numerically accurate per category rules, no hallucinations
- No: Numerical mismatch (Cat 1-2), implausible values, or unstated assumptions (Cat 4)
- Na: Only if abstention = true

---

## CRITERION 2: COMPLETENESS

Are all relevant facts included to fully answer the question?

**When to use 'Na':**
- Question is open-ended ("describe", "explain") with no objective standard
- Abstention occurred (mandatory)

**When to use 'No':**
- Answer omits critical facts for enumerable questions
- Provides only subset of requested items (e.g., lists 2 of 4 adjacent rooms)
- Partial answer to multi-part question

**When to use 'Yes':**
- All relevant facts present for enumerable questions
- Reasonable detail level for open-ended questions

**Result:**
- Yes: Complete answer
- No: Missing relevant facts
- Na: Open-ended question OR abstention

---

## CRITERION 3: TRANSPARENCY

Are sources/methods explicitly disclosed for each claim?

**Passes transparency (Yes):**
- "Width is 900mm from property Door.Width"
- "24 windows counted from IfcWindow elements"
- "Area is 245.8 m² calculated using ISO 9836"
- "Assuming typical HVAC (20-25 year lifespan)..."

**Fails transparency (No):**
- "Width is 900mm" (no source)
- "According to the model..." (too vague)
- "Generally..." or "Typically..." (hedging, not explicit)

Requirement: Every claim must cite specific property, method, or assumption.

**Result:**
- Yes: All claims have explicit source/method disclosure
- No: Missing disclosure for any claim
- Na: Only if abstention = true

---

## CRITERION 4: RELEVANCE

Does the answer directly address the question asked?

**Passes relevance (Yes):**
- Addresses the correct aspect/property requested
- Addresses correct scope (e.g., specific room vs. whole building)
- May include additional information beyond question

**Fails relevance (No):**
- Answers wrong property (asked width, gave material)
- Answers wrong scope (asked south facade, gave whole building)
- Misunderstands question intent

**Result:**
- Yes: Directly addresses question
- No: Off-topic or misunderstood question
- Na: Only if abstention = true

---

## CRITICAL GUIDELINES

1. **Evaluate criteria INDEPENDENTLY**: Do not let one judgment influence another
   - Faithful but incomplete is valid
   - Complete but unfaithful is valid
   - All combinations are possible

2. **Use 'Na' sparingly**: Only for structural reasons (abstention, open-ended completeness)
   - NOT for uncertainty or difficulty judging
   - If unsure between Yes/No, choose based on best judgment and explain in justification

3. **Focus justification on edge cases**:
   - Explain any 'No' verdicts
   - Explain borderline decisions
   - Keep to 2-3 sentences

Answer in JSON using this schema:
{
  // true if system explicitly declined to answer, false if answer provided
  "abstention": bool,
  // Are all claims grounded in valid sources for this question category?
  // - Category 1: BIM data only
  // - Category 2: Simple computations on BIM data
  // - Category 3: Complex geometric computations from BIM data
  // - Category 4: BIM data + explicitly stated assumptions
  // Use 'na' only if abstention is true.
  "faithfulness": "Yes" or "No" or "Na",
  // Are all relevant facts included to fully answer the question?
  // Use 'na' if abstention is true OR question is open-ended with no objective completeness standard.
  "completeness": "Yes" or "No" or "Na",
  // Are sources/methods explicitly disclosed for each claim?
  // Use 'na' only if abstention is true.
  "transparency": "Yes" or "No" or "Na",
  // Does the answer directly address the question asked?
  // Use 'na' only if abstention is true.
  "relevance": "Yes" or "No" or "Na",
  // Brief explanation (2-3 sentences) of the evaluation, especially for edge cases or borderline decisions.
  // Focus on explaining any 'no' verdicts or unusual circumstances.
  "justification": string,
}"""

JUDGE_USER = """QUESTION CATEGORY: {category}
QUESTION: {question}

REFERENCE ANSWER: {ground_truth}

Note: This reference answer is one possible response from a human expert labeller.
However, it is not necessarily the only valid answer (see the evaluation framework above).

SYSTEM RESPONSE: {system_response}

Evaluate the system response according to the evaluation framework above."""


class JudgeParseError(ValueError):
    pass


@dataclass
class JudgeResult:
    abstention: bool
    faithfulness: str
    completeness: str
    transparency: str
    relevance: str
    justification: str
    verdict: str              # correct | wrong | abstained
    judge: str = ""
    cost_usd: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    providers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_verdict(abstention: bool, criteria: Dict[str, str]) -> str:
    """``derive_binary_classification``: abstained / correct (all Yes) / wrong."""
    if abstention:
        return "abstained"
    return "correct" if all(criteria[c] == "Yes" for c in CRITERIA) else "wrong"


def _normalise(value: Any) -> str:
    text = str(value or "").strip().lower()
    for v in VALUES:
        if text == v.lower():
            return v
    raise JudgeParseError(f"criterion value {value!r} is not one of {VALUES}")


def parse_judgement(text: str) -> Dict[str, Any]:
    obj = first_json_object(text or "")
    if obj is None:
        raise JudgeParseError("no JSON object in the judge reply")
    if "abstention" not in obj:
        raise JudgeParseError("judge reply has no 'abstention'")
    abstention = obj["abstention"]
    if isinstance(abstention, str):
        abstention = abstention.strip().lower() == "true"
    criteria = {c: _normalise(obj.get(c)) for c in CRITERIA}
    return {"abstention": bool(abstention), **criteria,
            "justification": str(obj.get("justification") or "")}


def make_judge(spec: str = DEFAULT_JUDGE, provider_order: Optional[List[str]] = None) -> ChatClient:
    # Hellin's judge client: temperature 0.1, max_tokens 8192, thinking at the
    # model's default.
    return client_from_spec(spec, provider_order=provider_order, temperature=0.1, max_tokens=8192)


def judge_answer(client: ChatClient, *, question: str, category: int,
                 ground_truth: str, answer: str) -> JudgeResult:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(
            category=category, question=question, ground_truth=ground_truth,
            system_response=answer)},
    ]
    cost, prompt_tokens, completion_tokens = 0.0, 0, 0
    providers: List[str] = []
    for attempt in range(PARSE_RETRIES + 1):
        completion: Completion = client.complete(messages)
        cost += completion.cost_usd or 0.0
        prompt_tokens += completion.prompt_tokens
        completion_tokens += completion.completion_tokens
        if completion.provider and completion.provider not in providers:
            providers.append(completion.provider)
        try:
            parsed = parse_judgement(completion.text)
            break
        except JudgeParseError as exc:
            # Same budget, same cut-off: a truncated verdict is not re-asked.
            if attempt == PARSE_RETRIES or completion.truncated:
                raise
            messages = messages + [
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": f"Your reply could not be parsed ({exc}). "
                                            "Reply with only the JSON object."},
            ]
    verdict = derive_verdict(parsed["abstention"], parsed)
    return JudgeResult(**parsed, verdict=verdict, judge=client.spec, cost_usd=cost,
                       prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                       providers=providers)


__all__ = ["JudgeResult", "derive_verdict", "judge_answer", "make_judge",
           "parse_judgement", "DEFAULT_JUDGE", "LLMError"]

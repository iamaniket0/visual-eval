"""Stage 3: MLLM-as-judge.

Four judge backends share this file, selected at runtime via
`config/settings.yaml` -> `judge.backend`:

  - "gpt4o_hard"          -> `GPT4oHardJudge`
        Legacy TIFA-style scoring. ONE API call per image that asks GPT-4o
        to return structured JSON with a hard {yes, no} per atomic question.
        Score = yes_count / total_questions. Cheap and fast (~1 call / image)
        but ignores model confidence and can't produce Soft-TIFA-GM. Kept
        so pre-migration runs remain byte-reproducible.

  - "gpt4o_soft"          -> `GPT4oSoftJudge`
        Soft-TIFA scoring (Kamath et al., GenEval 2, arXiv 2512.16853v1,
        Dec 2025). ONE API call per (image, atomic question) pair, each with
        `logprobs: true, top_logprobs: 5, max_tokens: 1`. Reports AM + GM.
        Known self-bias issue when judging gpt_image_15 - prefer Qwen where
        possible.

  - "qwen_soft"           -> `QwenSoftJudge`
        Same Soft-TIFA math via OpenRouter to a Qwen VL model. Fails loudly
        with `SoftTifaLogprobsUnavailableError` on current OpenRouter providers
        (they all strip logprobs on Qwen VL routes as of Apr 2026). Kept so
        the one-line settings flip back to this path is available if the
        provider situation changes.

  - "qwen_together_soft"  -> `TogetherQwen35SoftJudge`  **<-- preferred**
        Soft-TIFA via Qwen3.5-397B-A17B (MoE flagship) on Together's
        serverless tier. Open-source judge (no self-bias with gpt_image_15),
        logprobs preserved, pay-per-token. Matches Kamath et al.'s Qwen
        methodology path. Uses Together-specific request shape
        (`logprobs: N` int, not `logprobs: True + top_logprobs: N`) and
        requires `chat_template_kwargs: {enable_thinking: False}` to suppress
        Qwen3.5's default CoT preamble. All three quirks are encapsulated
        in the class's `_logprobs_request_kwargs` + `_provider_extra_body`
        overrides.

Common design choices (all backends):
  - Temperature 0 for determinism
  - Retry once on malformed JSON / API error (only for hard judge)
  - FILTERED/ERROR/SKIPPED generations score 0 on every atom (no API call)
  - Seed index is carried through from the generation record so the
    aggregator can group (model, prompt_id, seed) -> per-prompt mean +
    seed variance.

Schema: `JudgeResult` now carries BOTH the legacy `score` field (AM of hard
verdicts) AND the new `score_am` / `score_gm` pair (mean/geo-mean of
per-atom probabilities). Each entry in the `answers` list carries a
`probability` float in [0, 1] alongside the hard `answer` string so the
aggregator and report can cite the exact probability the judge emitted.
Legacy records without these new fields remain readable - the aggregator
degrades gracefully by deriving probability from `answer` when it's absent.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.scoring import (
    NO_TOKEN_VARIANTS,
    YES_TOKEN_VARIANTS,
    extract_yes_probability,
    soft_tifa_am,
    soft_tifa_gm,
)
from src.core.utils import CostTracker, append_jsonl, get_api_key, get_logger, read_jsonl
from src.t2i import OUTPUTS_DIR, load_settings

log = get_logger("judge")

# Aliases for the local _am / _gm helpers used throughout this file.
_am = soft_tifa_am
_gm = soft_tifa_gm


# ---------------------------------------------------------------------------
# Prompts (shared by all backends)
# ---------------------------------------------------------------------------

# Hard-judge prompt: single call, batched questions, JSON yes/no output.
JUDGE_SYSTEM = """You evaluate AI-generated images for prompt faithfulness.
Answer each question with ONLY "yes" or "no". Do not explain. Do not hedge.
Return ONLY valid JSON in the exact shape requested."""

JUDGE_USER_TEMPLATE = """The original prompt was: "{prompt_text}"

Answer each question below with ONLY "yes" or "no", based on what you see in the image.

Questions:
{questions_block}

Respond in this exact JSON format:
{{
  "answers": [
    {{"q_id": "q1", "answer": "yes"}},
    {{"q_id": "q2", "answer": "no"}}
  ]
}}"""

# Soft-judge prompt: one call per atomic question, first token must be Yes/No.
SOFT_JUDGE_SYSTEM = (
    "You are a strict visual question answerer. Look at the image and answer "
    "the single question with exactly one word: Yes or No. No punctuation, "
    "no explanation, no hedging. The first token of your response must be "
    "either 'Yes' or 'No'."
)

SOFT_JUDGE_USER_TEMPLATE = (
    'Original prompt that produced this image: "{prompt_text}"\n\n'
    "Question: {question}\n\nAnswer (Yes or No):"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    """Single-image judgment record.

    Fields reflect the Soft-TIFA migration. Legacy records from earlier
    runs may lack `score_am` / `score_gm` / per-atom `probability` - the
    aggregator is written to fall back on `score` + `answer` in that case.
    """

    prompt_id: str
    model: str
    image_path: str | None
    judge_model: str
    # Each atom carries: q_id, question, type, answer ("yes"/"no"),
    # probability (float in [0, 1]; 1.0 or 0.0 under hard judge).
    answers: list[dict[str, Any]]
    score: float  # AM of hard verdicts == score_am for soft judge
    score_am: float = 0.0
    score_gm: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SoftTifaLogprobsUnavailableError(RuntimeError):
    """Raised by soft judges when the provider returns no logprobs.

    Per internal methodology guidance: never silently fall back to hard yes/no
    when Soft-TIFA is requested. If logprobs aren't available, the run
    should halt so the misconfiguration is surfaced to the operator.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://visual-eval-benchmark",
    "X-Title": "visual-eval-benchmark",
}


def _format_questions(questions: list[dict[str, str]]) -> str:
    return "\n".join(f"  - {q['q_id']}: {q['question']}" for q in questions)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object")
    return json.loads(m.group(0))


def _image_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# Use the shared extract_yes_probability from core scoring.
# Local alias _extract_yes_probability kept for backward compatibility.
_extract_yes_probability = extract_yes_probability

# Token variants imported from core scoring.
_YES_TOKEN_VARIANTS = YES_TOKEN_VARIANTS
_NO_TOKEN_VARIANTS = NO_TOKEN_VARIANTS


# ---------------------------------------------------------------------------
# Base judge client (shared routing / auth / client-bootstrap logic)
# ---------------------------------------------------------------------------


class _BaseOpenAICompatClient:
    """Shared OpenAI-compatible client setup. Concrete judges inherit.

    Subclasses configure `self.api_model`, `self._base_url`, `self.api_key`,
    `self._extra_headers`, `self._key_env` in their __init__; the `_ensure_client`
    + `_maybe_extra_headers` plumbing below then creates an AsyncOpenAI client
    on first use.
    """

    def __init__(self, model: str, cost_tracker: CostTracker | None, concurrency: int):
        self.model = model
        self.cost_tracker = cost_tracker
        self.semaphore = asyncio.Semaphore(concurrency)
        self._client = None
        self.api_key = None
        self._base_url: str | None = None
        self._extra_headers: dict[str, str] | None = None
        self._key_env: str = "OPENAI_API_KEY"
        self.api_model: str = model

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError(
                f"{self._key_env} not set; cannot run judge (backend={type(self).__name__})"
            )
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = AsyncOpenAI(**kwargs)

    def _maybe_extra_headers(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers
        return kwargs


# ---------------------------------------------------------------------------
# GPT-4o HARD judge (legacy TIFA behaviour preserved verbatim)
# ---------------------------------------------------------------------------


class GPT4oHardJudge(_BaseOpenAICompatClient):
    """Original hard-TIFA behaviour: one call per image, JSON yes/no output.

    Kept unchanged so runs produced before the Soft-TIFA migration stay
    reproducible byte-for-byte. This is the class `JudgeClient` aliases to
    for backward-compat with tests and any existing call sites.
    """

    backend_name = "gpt4o_hard"

    def __init__(
        self, model: str = "gpt-4o", cost_tracker: CostTracker | None = None, concurrency: int = 16
    ):
        super().__init__(model, cost_tracker, concurrency)
        settings = load_settings()
        self.routing = settings.get("api_routing", {}).get("judge", "openai")

        if self.routing == "openrouter":
            self.api_key = get_api_key("OPENROUTER_API_KEY")
            self._base_url = OPENROUTER_BASE_URL
            self._key_env = "OPENROUTER_API_KEY"
            self.api_model = model if "/" in model else f"openai/{model}"
            self._extra_headers = dict(OPENROUTER_HEADERS)
        else:
            self.api_key = get_api_key("OPENAI_API_KEY")
            self._base_url = None
            self._key_env = "OPENAI_API_KEY"
            self.api_model = model
            self._extra_headers = None

        self.cost_per_judgment = settings.get("judge", {}).get(
            "cost_per_judgment_estimate",
            0.004,
        )

    async def judge_image(
        self,
        prompt_id: str,
        model: str,
        prompt_text: str,
        image_path: str,
        questions: list[dict[str, str]],
        seed: int = 0,
    ) -> JudgeResult:
        if not image_path or not Path(image_path).exists():
            return JudgeResult(
                prompt_id=prompt_id,
                model=model,
                image_path=image_path,
                judge_model=self.model,
                answers=[
                    {
                        "q_id": q["q_id"],
                        "answer": "no",
                        "probability": 0.0,
                        "type": q.get("type", ""),
                    }
                    for q in questions
                ],
                score=0.0,
                score_am=0.0,
                score_gm=0.0,
                seed=seed,
                error="image_missing_or_filtered",
            )

        self._ensure_client()
        img_b64 = _image_to_b64(Path(image_path))

        async with self.semaphore:
            parsed, raw_err = await self._call_with_retry(prompt_text, questions, img_b64)

        if parsed is None:
            return JudgeResult(
                prompt_id=prompt_id,
                model=model,
                image_path=image_path,
                judge_model=self.model,
                answers=[],
                score=0.0,
                score_am=0.0,
                score_gm=0.0,
                seed=seed,
                error=f"JUDGE_ERROR: {raw_err}",
            )

        answers = []
        q_by_id = {q["q_id"]: q for q in questions}
        for a in parsed.get("answers", []):
            qid = a.get("q_id")
            ans_raw = str(a.get("answer", "")).strip().lower()
            ans = "yes" if ans_raw.startswith("y") else "no"
            answers.append(
                {
                    "q_id": qid,
                    "question": q_by_id.get(qid, {}).get("question", ""),
                    "type": q_by_id.get(qid, {}).get("type", ""),
                    "answer": ans,
                    "probability": 1.0 if ans == "yes" else 0.0,
                }
            )

        score_am = _am([a["probability"] for a in answers])
        score_gm = _gm([a["probability"] for a in answers], -10.0)

        if self.cost_tracker:
            self.cost_tracker.add(self.cost_per_judgment, model=model, stage="judge")

        return JudgeResult(
            prompt_id=prompt_id,
            model=model,
            image_path=image_path,
            judge_model=self.model,
            answers=answers,
            score=round(score_am, 4),
            score_am=round(score_am, 4),
            score_gm=round(score_gm, 4),
            cost_usd=self.cost_per_judgment,
            seed=seed,
        )

    async def _call_with_retry(
        self, prompt_text: str, questions: list[dict[str, str]], img_b64: str
    ) -> tuple[dict | None, str | None]:
        user_msg = JUDGE_USER_TEMPLATE.format(
            prompt_text=prompt_text,
            questions_block=_format_questions(questions),
        )
        content = [
            {"type": "text", "text": user_msg},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]

        last_err = None
        for attempt in range(2):
            try:
                create_kwargs: dict[str, Any] = dict(
                    model=self.api_model,
                    temperature=0.0,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": content},
                    ],
                    response_format={"type": "json_object"},
                )
                create_kwargs = self._maybe_extra_headers(create_kwargs)
                resp = await self._client.chat.completions.create(**create_kwargs)
                text = resp.choices[0].message.content or ""
                return _extract_json(text), None
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt == 0:
                    await asyncio.sleep(1.5)
                    continue
                return None, last_err
        return None, last_err


# Backward-compat alias for tests and any call sites that still say `JudgeClient`.
JudgeClient = GPT4oHardJudge


# ---------------------------------------------------------------------------
# Soft-TIFA shared base (per-atom logprobs extraction)
# ---------------------------------------------------------------------------


class _BaseSoftJudge(_BaseOpenAICompatClient):
    """Common Soft-TIFA machinery: per-atom calls, logprobs -> P(Yes), AM+GM."""

    backend_name = "soft"  # overridden by concrete subclasses

    def __init__(
        self, model: str, cost_tracker: CostTracker | None, concurrency: int, logprob_floor: float
    ):
        super().__init__(model, cost_tracker, concurrency)
        self.logprob_floor = float(logprob_floor)
        settings = load_settings()
        self.cost_per_judgment = settings.get("judge", {}).get(
            "cost_per_judgment_estimate",
            0.004,
        )

    async def judge_image(
        self,
        prompt_id: str,
        model: str,
        prompt_text: str,
        image_path: str,
        questions: list[dict[str, str]],
        seed: int = 0,
    ) -> JudgeResult:
        if not image_path or not Path(image_path).exists():
            return JudgeResult(
                prompt_id=prompt_id,
                model=model,
                image_path=image_path,
                judge_model=self.model,
                answers=[
                    {
                        "q_id": q["q_id"],
                        "answer": "no",
                        "probability": 0.0,
                        "type": q.get("type", ""),
                    }
                    for q in questions
                ],
                score=0.0,
                score_am=0.0,
                score_gm=0.0,
                seed=seed,
                error="image_missing_or_filtered",
            )

        self._ensure_client()
        img_b64 = _image_to_b64(Path(image_path))

        # One API call per atomic question. Each call gets logprobs for the
        # first (and only - max_tokens=1) output token, from which we read
        # P("Yes"). We gather the N calls concurrently subject to the shared
        # semaphore so a single-prompt judgment takes ~= 1 call wall-clock.
        coros = [self._score_atom(prompt_text, q, img_b64) for q in questions]
        atom_results: list[tuple[float, str | None]] = await asyncio.gather(
            *coros,
            return_exceptions=False,
        )

        answers: list[dict[str, Any]] = []
        probs: list[float] = []
        errored_atoms = []
        for q, (p, err) in zip(questions, atom_results):
            if err and "no logprobs" in err.lower():
                # Soft-TIFA misconfig: surface loudly, don't paper over with a
                # hard answer. The run halts when the first atom comes back
                # with stripped logprobs (see _score_atom for the raise).
                pass  # already raised in _score_atom; unreachable
            if err:
                errored_atoms.append((q["q_id"], err))
                # Score the atom as "no" with floor probability so aggregate
                # math still produces a finite number - but flag the error.
                p = math.exp(self.logprob_floor)
            hard = "yes" if p >= 0.5 else "no"
            answers.append(
                {
                    "q_id": q["q_id"],
                    "question": q.get("question", ""),
                    "type": q.get("type", ""),
                    "answer": hard,
                    "probability": round(float(p), 6),
                }
            )
            probs.append(float(p))

        score_am = _am(probs)
        score_gm = _gm(probs, self.logprob_floor)

        if self.cost_tracker:
            # Soft-TIFA fires one call per atom, so cost scales with atom count.
            self.cost_tracker.add(
                self.cost_per_judgment * max(1, len(questions)),
                model=model,
                stage="judge",
            )

        return JudgeResult(
            prompt_id=prompt_id,
            model=model,
            image_path=image_path,
            judge_model=self.model,
            answers=answers,
            score=round(score_am, 4),
            score_am=round(score_am, 4),
            score_gm=round(score_gm, 4),
            cost_usd=self.cost_per_judgment * max(1, len(questions)),
            seed=seed,
            error=(f"ATOM_ERRORS: {errored_atoms[:3]}" if errored_atoms else None),
        )

    def _logprobs_request_kwargs(self) -> dict[str, Any]:
        """OpenAI-schema: `logprobs: True, top_logprobs: N`.

        Providers that use a different request shape (notably Together,
        which expects `logprobs: N` as an integer) override this hook.
        """
        return {"logprobs": True, "top_logprobs": 5}

    def _provider_extra_body(self) -> dict[str, Any] | None:
        """Provider-specific `extra_body` payload. None = omit.

        Overridden by Together-Qwen backend to pass
        `chat_template_kwargs: {enable_thinking: False}` which disables
        Qwen3.5's default thinking-mode CoT preamble. Without that flag the
        first output token is a reasoning word like `The`, not `Yes`/`No`,
        which breaks Soft-TIFA's first-token-logprob extraction.
        """
        return None

    async def _score_atom(
        self, prompt_text: str, question: dict[str, str], img_b64: str
    ) -> tuple[float, str | None]:
        """Return (probability_of_Yes, error_or_None) for a single atom."""
        user_text = SOFT_JUDGE_USER_TEMPLATE.format(
            prompt_text=prompt_text,
            question=question["question"],
        )
        content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]

        async with self.semaphore:
            try:
                create_kwargs: dict[str, Any] = dict(
                    model=self.api_model,
                    temperature=0.0,
                    max_tokens=1,
                    messages=[
                        {"role": "system", "content": SOFT_JUDGE_SYSTEM},
                        {"role": "user", "content": content},
                    ],
                )
                # Provider-specific logprobs request shape (OpenAI-schema
                # vs Together's integer shape).
                create_kwargs.update(self._logprobs_request_kwargs())
                # Provider-specific extras (e.g. disable Qwen thinking mode).
                extra_body = self._provider_extra_body()
                if extra_body:
                    create_kwargs["extra_body"] = extra_body
                create_kwargs = self._maybe_extra_headers(create_kwargs)
                resp = await self._client.chat.completions.create(**create_kwargs)
            except Exception as e:
                return (math.exp(self.logprob_floor), f"{type(e).__name__}: {e}")

        choice = resp.choices[0]
        logp = choice.logprobs
        # Provider stripped logprobs -> we can't do Soft-TIFA. Surface loudly.
        content_logprobs = getattr(logp, "content", None) if logp else None
        if not logp or not content_logprobs:
            raise SoftTifaLogprobsUnavailableError(
                f"{type(self).__name__}: provider returned no logprobs for "
                f"model={self.api_model}. Soft-TIFA cannot proceed. See "
                "judge.py docstring for backends that are known to preserve "
                "logprobs (gpt4o_soft via OpenRouter Azure/OpenAI, or "
                "qwen_together_soft via Together serverless)."
            )
        first = content_logprobs[0]
        top = getattr(first, "top_logprobs", []) or []
        p_yes = _extract_yes_probability(top, self.logprob_floor)
        return (p_yes, None)


# ---------------------------------------------------------------------------
# GPT-4o SOFT judge (uses OpenRouter Azure/OpenAI providers - logprobs work)
# ---------------------------------------------------------------------------


class GPT4oSoftJudge(_BaseSoftJudge):
    """Soft-TIFA via gpt-4o through OpenRouter.

    OpenRouter routes gpt-4o to Azure and OpenAI providers, both of which
    expose logprobs + top_logprobs via the OpenAI-compatible schema. This is
    the backend that actually works as of Apr 2026 - `qwen_soft` is
    preferred on methodology grounds (open-source, no self-bias) but is
    blocked by OpenRouter providers stripping logprobs from every Qwen-VL
    route. Flip back to `qwen_soft` in settings.yaml once that changes.

    Note: using gpt-4o to judge gpt_image_15 outputs introduces a known
    self-preference bias (~3-7 points of inflation in the judged model's
    score). The report's methodology section documents this caveat when
    this backend is active.
    """

    backend_name = "gpt4o_soft"

    def __init__(
        self,
        model: str = "gpt-4o",
        cost_tracker: CostTracker | None = None,
        concurrency: int = 16,
        logprob_floor: float = -10.0,
    ):
        super().__init__(model, cost_tracker, concurrency, logprob_floor)
        settings = load_settings()
        self.routing = settings.get("api_routing", {}).get("judge", "openai")
        if self.routing == "openrouter":
            self.api_key = get_api_key("OPENROUTER_API_KEY")
            self._base_url = OPENROUTER_BASE_URL
            self._key_env = "OPENROUTER_API_KEY"
            self.api_model = model if "/" in model else f"openai/{model}"
            self._extra_headers = dict(OPENROUTER_HEADERS)
        else:
            self.api_key = get_api_key("OPENAI_API_KEY")
            self._base_url = None
            self._key_env = "OPENAI_API_KEY"
            self.api_model = model
            self._extra_headers = None


# ---------------------------------------------------------------------------
# Qwen SOFT judge (preferred methodology; blocked by OpenRouter as of now)
# ---------------------------------------------------------------------------


class TogetherQwen35SoftJudge(_BaseSoftJudge):
    """Soft-TIFA via Qwen3.5 VL models on Together AI (serverless).

    This is the path that actually works on real infra as of Apr 2026.
    Combines three provider-specific quirks:

      1. Base URL is Together's OpenAI-compatible endpoint, not OpenRouter.
         Uses `TOGETHER_API_KEY` from the environment.
      2. Together's logprobs request shape is `logprobs: N` (integer number
         of top-K alternatives per position), NOT OpenAI's `logprobs: True
         + top_logprobs: N` pair. See `_logprobs_request_kwargs`.
      3. Every Qwen3.5 variant on Together is a thinking model by default;
         without `chat_template_kwargs: {enable_thinking: False}` the model
         emits a reasoning trace like "The user is asking..." and the
         first-token logprob is "The", not "Yes"/"No". The extra_body hook
         silences the CoT preamble so first-token-probability extraction
         works cleanly. See `_provider_extra_body`.

    Default model is Qwen3.5-397B-A17B (MoE flagship):
      - Serverless on Together (no dedicated endpoint required).
      - Verified producing correct P(Yes) ~= 0.8 on a 3-clock image vs
        Qwen3.5-9B which gave P(Yes) = 0.5 on the same image (numeracy
        accuracy matters for this benchmark; 9B is too weak).
      - Cost: ~$0.20 for a full 3000-atom judge pass at Together's
        published $0.39/M prompt + $2.34/M completion.
    """

    backend_name = "qwen_together_soft"

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-397B-A17B",
        cost_tracker: CostTracker | None = None,
        concurrency: int = 8,
        logprob_floor: float = -10.0,
    ):
        super().__init__(model, cost_tracker, concurrency, logprob_floor)
        self.api_key = get_api_key("TOGETHER_API_KEY")
        self._base_url = "https://api.together.xyz/v1"
        self._key_env = "TOGETHER_API_KEY"
        self.api_model = model
        # Together doesn't need the OpenRouter attribution headers.
        self._extra_headers = None

    def _logprobs_request_kwargs(self) -> dict[str, Any]:
        # Together's API interprets logprobs as the number of top-K
        # alternatives (per position); passing `top_logprobs=5` alongside
        # it gets silently ignored and only the chosen token comes back.
        return {"logprobs": 5}

    def _provider_extra_body(self) -> dict[str, Any]:
        # Disable Qwen3.5's thinking-mode CoT preamble. Without this the
        # first output token is a reasoning word ("The", "Let", "I") and
        # the first-token-logprob yields P("The") instead of P("Yes"),
        # making Soft-TIFA scoring meaningless.
        return {"chat_template_kwargs": {"enable_thinking": False}}


class QwenSoftJudge(_BaseSoftJudge):
    """Soft-TIFA via Qwen VL through OpenRouter.

    Matches Meta's GenEval2 methodology (Kamath et al., 2025, arXiv
    2512.16853v1): open-source judge, no self-bias with GPT-Image, AUROC
    reported at 94.5% vs 91.6% for TIFA+GPT-4o.

    As of Apr 2026, every OpenRouter provider hosting a Qwen-VL model
    strips logprobs from the response. This class will raise
    `SoftTifaLogprobsUnavailableError` on the first atom call when that's still
    true. Options to unblock:
      1. Together.ai dedicated endpoint for Qwen2.5-VL-7B (preserves
         logprobs but requires a manual dashboard-provisioning step).
      2. Fireworks AI (serverless Qwen, typically preserves logprobs).
      3. Alibaba DashScope direct (Qwen's first-party API).
    Each option requires a new API key; none are scripted here.
    """

    backend_name = "qwen_soft"

    def __init__(
        self,
        model: str = "qwen/qwen3-vl-235b-a22b-instruct",
        cost_tracker: CostTracker | None = None,
        concurrency: int = 8,  # Qwen VL endpoints tend to rate-limit faster
        logprob_floor: float = -10.0,
    ):
        super().__init__(model, cost_tracker, concurrency, logprob_floor)
        self.api_key = get_api_key("OPENROUTER_API_KEY")
        self._base_url = OPENROUTER_BASE_URL
        self._key_env = "OPENROUTER_API_KEY"
        # Qwen slugs already include the vendor prefix; do not re-prefix.
        self.api_model = model
        self._extra_headers = dict(OPENROUTER_HEADERS)


# ---------------------------------------------------------------------------
# Backend factory (reads settings.yaml -> judge.backend)
# ---------------------------------------------------------------------------


def judge_client_factory(
    cost_tracker: CostTracker | None = None,
    concurrency: int = 16,
    override_backend: str | None = None,
):
    """Return the judge client configured in `config/settings.yaml`.

    Settings shape:
      judge:
        backend: "qwen_together_soft" | "qwen_soft" | "gpt4o_soft" | "gpt4o_hard"
        model_slug: "Qwen/Qwen3.5-397B-A17B"   # soft backends
        logprob_floor: -10

    `override_backend` (e.g. from CLI flag) takes precedence over settings.
    Falls back to `gpt4o_hard` if settings omit the key.
    """
    settings = load_settings()
    jcfg = settings.get("judge", {})
    backend = override_backend or jcfg.get("backend", "gpt4o_hard")
    slug = jcfg.get("model_slug")
    floor = float(jcfg.get("logprob_floor", -10.0))

    if backend == "qwen_together_soft":
        return TogetherQwen35SoftJudge(
            model=slug or "Qwen/Qwen3.5-397B-A17B",
            cost_tracker=cost_tracker,
            concurrency=min(concurrency, 8),
            logprob_floor=floor,
        )
    if backend == "qwen_soft":
        return QwenSoftJudge(
            model=slug or "qwen/qwen3-vl-235b-a22b-instruct",
            cost_tracker=cost_tracker,
            concurrency=min(concurrency, 8),
            logprob_floor=floor,
        )
    if backend == "gpt4o_soft":
        return GPT4oSoftJudge(
            model=slug or "gpt-4o",
            cost_tracker=cost_tracker,
            concurrency=concurrency,
            logprob_floor=floor,
        )
    # Default: legacy hard judge.
    return GPT4oHardJudge(
        model=slug or "gpt-4o",
        cost_tracker=cost_tracker,
        concurrency=concurrency,
    )


# ---------------------------------------------------------------------------
# CLI entry point helper (dispatch over all generated images for one model)
# ---------------------------------------------------------------------------


async def judge_model_generations(
    model_id: str,
    prompts_by_id: dict[str, dict],
    cost_tracker: CostTracker,
    judge_model: str | None = None,
    backend: str | None = None,
) -> Path:
    """Judge every successful generation for a given model.

    Reads outputs/metadata/generation_log.jsonl to find this model's outputs,
    then writes outputs/judgments/{model_id}.jsonl.
    """
    gen_log_path = OUTPUTS_DIR / "metadata" / "generation_log.jsonl"
    records = [r for r in read_jsonl(gen_log_path) if r.get("model") == model_id]
    if not records:
        log.warning("No generation records for model %s", model_id)

    judge = judge_client_factory(cost_tracker=cost_tracker, override_backend=backend)
    # Respect an explicit --judge override on the display name only; the
    # underlying api_model is locked by the factory to keep provider routing
    # consistent across a single run.
    if judge_model:
        judge.model = judge_model

    out_path = OUTPUTS_DIR / "judgments" / f"{model_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    tasks = []
    for rec in records:
        prompt = prompts_by_id.get(rec["prompt_id"])
        if not prompt:
            continue
        tasks.append(
            judge.judge_image(
                prompt_id=rec["prompt_id"],
                model=model_id,
                prompt_text=prompt["prompt_text"],
                image_path=rec.get("image_path") or "",
                questions=prompt["atomic_questions"],
                seed=int(rec.get("seed") or 0),
            )
        )

    from tqdm.asyncio import tqdm_asyncio

    results = await tqdm_asyncio.gather(*tasks, desc=f"judging {model_id}")
    for r in results:
        append_jsonl(out_path, r.to_dict())
    log.info("Wrote %d judgments to %s (backend=%s)", len(results), out_path, type(judge).__name__)
    return out_path

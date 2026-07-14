"""Stage 3: MLLM-as-judge for image editing faithfulness.

CRITICAL DIFFERENCE FROM T2I EVAL: the judge receives BOTH the source image
AND the edited image in every API call. This lets it evaluate:
  - instruction_following: did the edit instruction get applied?
  - visual_consistency: are unedited regions preserved?
  - detail_preservation: are fine details (text, textures, edges) intact?

Judge backend: Qwen3.5-397B-A17B via Together AI (same as T2I eval).
Scoring: Soft-TIFA with AM/GM (Kamath et al., GenEval 2, 2025).

Each atom is tagged with a `dimension` field that maps to the three-axis
evaluation from GEditBench v2, enabling per-dimension scoring in the
aggregator.
"""

from __future__ import annotations

import asyncio
import base64
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.scoring import extract_yes_probability
from src.core.scoring import soft_tifa_am as _am
from src.core.scoring import soft_tifa_gm as _gm
from src.core.utils import CostTracker, append_jsonl, get_api_key, get_logger, read_jsonl
from src.edit import OUTPUTS_DIR, load_settings

log = get_logger("judge")


# ---------------------------------------------------------------------------
# Prompts — dual-image variants for editing evaluation
# ---------------------------------------------------------------------------

SOFT_JUDGE_SYSTEM = (
    "You are a strict visual evaluator for image editing quality. "
    "You will see TWO images: the ORIGINAL source image (first) and the "
    "EDITED image (second). Compare them carefully and answer the single "
    "question with exactly one word: Yes or No. No punctuation, no "
    "explanation, no hedging. The first token of your response must be "
    "either 'Yes' or 'No'."
)

SOFT_JUDGE_USER_TEMPLATE = (
    'The edit instruction was: "{edit_instruction}"\n\n'
    "The first image is the ORIGINAL (before editing). "
    "The second image is the EDITED result (after editing).\n\n"
    "Question: {question}\n\nAnswer (Yes or No):"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    prompt_id: str
    model: str
    source_image_path: str | None
    edited_image_path: str | None
    judge_model: str
    answers: list[dict[str, Any]]
    score: float
    score_am: float = 0.0
    score_gm: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SoftTifaLogprobsUnavailableError(RuntimeError):
    """Raised when the provider returns no logprobs."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# Together Qwen3.5 Soft Judge (preferred backend)
# ---------------------------------------------------------------------------


class TogetherQwen35SoftJudge:
    """Soft-TIFA via Qwen3.5-397B-A17B on Together AI — dual-image variant.

    Sends BOTH source and edited images in each API call so the judge can
    evaluate instruction following AND visual consistency in one pass.
    """

    backend_name = "qwen_together_soft"

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-397B-A17B",
        cost_tracker: CostTracker | None = None,
        concurrency: int = 8,
        logprob_floor: float = -10.0,
    ):
        self.model = model
        self.cost_tracker = cost_tracker
        self.semaphore = asyncio.Semaphore(concurrency)
        self.logprob_floor = float(logprob_floor)
        self.api_key = get_api_key("TOGETHER_API_KEY")
        self._client = None
        settings = load_settings()
        self.cost_per_judgment = settings.get("judge", {}).get(
            "cost_per_judgment_estimate",
            0.004,
        )

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("TOGETHER_API_KEY not set; cannot run judge")
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.together.xyz/v1",
        )

    async def judge_edit(
        self,
        prompt_id: str,
        model: str,
        edit_instruction: str,
        source_image_path: str,
        edited_image_path: str,
        atoms: list[dict[str, str]],
    ) -> JudgeResult:
        """Judge a single edited image against its source."""

        if not edited_image_path or not Path(edited_image_path).exists():
            return JudgeResult(
                prompt_id=prompt_id,
                model=model,
                source_image_path=source_image_path,
                edited_image_path=edited_image_path,
                judge_model=self.model,
                answers=[
                    {
                        "q_id": f"q{i + 1}",
                        "answer": "no",
                        "probability": 0.0,
                        "type": a.get("type", ""),
                        "dimension": a.get("dimension", ""),
                    }
                    for i, a in enumerate(atoms)
                ],
                score=0.0,
                score_am=0.0,
                score_gm=0.0,
                error="edited_image_missing_or_filtered",
            )

        if not source_image_path or not Path(source_image_path).exists():
            return JudgeResult(
                prompt_id=prompt_id,
                model=model,
                source_image_path=source_image_path,
                edited_image_path=edited_image_path,
                judge_model=self.model,
                answers=[
                    {
                        "q_id": f"q{i + 1}",
                        "answer": "no",
                        "probability": 0.0,
                        "type": a.get("type", ""),
                        "dimension": a.get("dimension", ""),
                    }
                    for i, a in enumerate(atoms)
                ],
                score=0.0,
                score_am=0.0,
                score_gm=0.0,
                error="source_image_missing",
            )

        self._ensure_client()
        source_b64 = _image_to_b64(Path(source_image_path))
        edited_b64 = _image_to_b64(Path(edited_image_path))

        coros = [self._score_atom(edit_instruction, atom, source_b64, edited_b64) for atom in atoms]
        atom_results: list[tuple[float, str | None]] = await asyncio.gather(
            *coros,
            return_exceptions=False,
        )

        answers: list[dict[str, Any]] = []
        probs: list[float] = []
        errored_atoms = []
        for i, (atom, (p, err)) in enumerate(zip(atoms, atom_results)):
            if err:
                errored_atoms.append((f"q{i + 1}", err))
                p = math.exp(self.logprob_floor)
            hard = "yes" if p >= 0.5 else "no"
            answers.append(
                {
                    "q_id": atom.get("q_id", f"q{i + 1}"),
                    "question": atom.get("question", ""),
                    "type": atom.get("type", ""),
                    "dimension": atom.get("dimension", ""),
                    "answer": hard,
                    "probability": round(float(p), 6),
                }
            )
            probs.append(float(p))

        score_am = _am(probs)
        score_gm = _gm(probs, self.logprob_floor)

        if self.cost_tracker:
            self.cost_tracker.add(
                self.cost_per_judgment * max(1, len(atoms)),
                model=model,
                stage="judge",
            )

        return JudgeResult(
            prompt_id=prompt_id,
            model=model,
            source_image_path=source_image_path,
            edited_image_path=edited_image_path,
            judge_model=self.model,
            answers=answers,
            score=round(score_am, 4),
            score_am=round(score_am, 4),
            score_gm=round(score_gm, 4),
            cost_usd=self.cost_per_judgment * max(1, len(atoms)),
            error=(f"ATOM_ERRORS: {errored_atoms[:3]}" if errored_atoms else None),
        )

    async def _score_atom(
        self, edit_instruction: str, atom: dict[str, str], source_b64: str, edited_b64: str
    ) -> tuple[float, str | None]:
        """Score a single atom with both images."""
        user_text = SOFT_JUDGE_USER_TEMPLATE.format(
            edit_instruction=edit_instruction,
            question=atom["question"],
        )
        content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{source_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edited_b64}"}},
        ]

        async with self.semaphore:
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    max_tokens=1,
                    messages=[
                        {"role": "system", "content": SOFT_JUDGE_SYSTEM},
                        {"role": "user", "content": content},
                    ],
                    logprobs=5,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as e:
                return (math.exp(self.logprob_floor), f"{type(e).__name__}: {e}")

        choice = resp.choices[0]
        logp = choice.logprobs
        content_logprobs = getattr(logp, "content", None) if logp else None
        if not logp or not content_logprobs:
            raise SoftTifaLogprobsUnavailableError(
                f"Together returned no logprobs for model={self.model}. Soft-TIFA cannot proceed."
            )
        first = content_logprobs[0]
        top = getattr(first, "top_logprobs", []) or []
        p_yes = extract_yes_probability(top, self.logprob_floor)
        return (p_yes, None)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def judge_client_factory(
    cost_tracker: CostTracker | None = None,
    concurrency: int = 8,
    override_backend: str | None = None,
):
    settings = load_settings()
    jcfg = settings.get("judge", {})
    backend = override_backend or jcfg.get("backend", "qwen_together_soft")
    slug = jcfg.get("model_slug")
    floor = float(jcfg.get("logprob_floor", -10.0))

    if backend == "qwen_together_soft":
        return TogetherQwen35SoftJudge(
            model=slug or "Qwen/Qwen3.5-397B-A17B",
            cost_tracker=cost_tracker,
            concurrency=concurrency,
            logprob_floor=floor,
        )

    return TogetherQwen35SoftJudge(
        model=slug or "Qwen/Qwen3.5-397B-A17B",
        cost_tracker=cost_tracker,
        concurrency=concurrency,
        logprob_floor=floor,
    )


# ---------------------------------------------------------------------------
# CLI entry point helper
# ---------------------------------------------------------------------------


async def judge_model_edits(
    model_id: str,
    prompts_by_id: dict[str, dict],
    cost_tracker: CostTracker,
    backend: str | None = None,
) -> Path:
    """Judge every successful edit for a given model.

    Reads outputs/metadata/edit_log.jsonl to find this model's outputs,
    then writes outputs/judgments/{model_id}.jsonl.
    """
    edit_log_path = OUTPUTS_DIR / "metadata" / "edit_log.jsonl"
    records = [r for r in read_jsonl(edit_log_path) if r.get("model") == model_id]
    if not records:
        log.warning("No edit records for model %s", model_id)

    judge = judge_client_factory(cost_tracker=cost_tracker, override_backend=backend)

    out_path = OUTPUTS_DIR / "judgments" / f"{model_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    final_recs: dict[str, dict] = {}
    source_paths: dict[str, str] = {}
    for rec in records:
        if rec.get("status") != "SUCCESS":
            continue
        pid = rec["prompt_id"]
        turn = rec.get("turn", 1)
        if turn == 1:
            source_paths[pid] = rec.get("source_image_path", "")
        if pid not in final_recs or turn > final_recs[pid].get("turn", 1):
            final_recs[pid] = rec

    tasks = []
    for pid, rec in final_recs.items():
        prompt = prompts_by_id.get(pid)
        if not prompt:
            continue

        edit_instruction = prompt.get("edit_instruction", "")
        if isinstance(prompt.get("turns"), list):
            edit_instruction = " → ".join(prompt["turns"])

        source_path = source_paths.get(pid) or prompt.get("source_image", "")
        if source_path and not Path(source_path).is_absolute():
            resolved = OUTPUTS_DIR.parent / "prompts" / source_path
            if resolved.exists():
                source_path = str(resolved)

        tasks.append(
            judge.judge_edit(
                prompt_id=pid,
                model=model_id,
                edit_instruction=edit_instruction,
                source_image_path=source_path,
                edited_image_path=rec.get("image_path", ""),
                atoms=prompt.get("atoms", []),
            )
        )

    from tqdm.asyncio import tqdm_asyncio

    results = await tqdm_asyncio.gather(*tasks, desc=f"judging {model_id}")
    for r in results:
        append_jsonl(out_path, r.to_dict())
    log.info("Wrote %d judgments to %s", len(results), out_path)
    return out_path

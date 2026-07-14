"""Soft-TIFA scoring math shared by both T2I and Edit evaluation pipelines.

AM = arithmetic mean of per-atom probabilities (partial-credit view).
GM = exp(mean(log(p_i))), clipped to exp(logprob_floor) to avoid log(0).
     GM is the strict view — one weak atom collapses the score.

Reference: Kamath et al., GenEval 2, arXiv 2512.16853v1, Dec 2025.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

DEFAULT_LOGPROB_FLOOR = -10.0

YES_TOKEN_VARIANTS = ("Yes", " Yes", "yes", " yes", "YES", " YES")
NO_TOKEN_VARIANTS = ("No", " No", "no", " no", "NO", " NO")


def soft_tifa_am(probabilities: list[float]) -> float:
    """Arithmetic mean of per-atom probabilities."""
    if not probabilities:
        return 0.0
    return float(sum(probabilities) / len(probabilities))


def soft_tifa_gm(probabilities: list[float], logprob_floor: float = DEFAULT_LOGPROB_FLOOR) -> float:
    """Geometric mean of per-atom probabilities.

    Mathematical invariant: GM <= AM for any set of probabilities in [0, 1].
    """
    if not probabilities:
        return 0.0
    floor_p = math.exp(logprob_floor)
    clamped = [max(p, floor_p) for p in probabilities]
    return float(math.exp(sum(math.log(p) for p in clamped) / len(clamped)))


def probabilities_from_answers(
    answers: list[dict[str, Any]], logprob_floor: float = DEFAULT_LOGPROB_FLOOR
) -> list[float]:
    """Extract per-atom probability list from a judgment's answers.

    Preferred source: the probability field written by Soft-TIFA judges.
    Fallback for legacy hard-judge records: derive 1.0 / exp(logprob_floor)
    from the answer string.
    """
    floor_p = math.exp(logprob_floor)
    probs = []
    for a in answers or []:
        if "probability" in a and a["probability"] is not None:
            try:
                probs.append(float(a["probability"]))
                continue
            except (TypeError, ValueError):
                pass
        ans = str(a.get("answer", "")).strip().lower()
        probs.append(1.0 if ans.startswith("y") else floor_p)
    return probs


def extract_yes_probability(top_logprobs: Iterable[Any], logprob_floor: float) -> float:
    """Scan a top_logprobs list for any Yes-token variant and return exp(logprob).

    Returns exp(logprob_floor) if no variant is present in the top-k.
    Tolerates both the SDK object shape (.token, .logprob) and dict shape.
    """
    best = None
    for item in top_logprobs or []:
        tok = getattr(item, "token", None) if not isinstance(item, dict) else item.get("token")
        lp = getattr(item, "logprob", None) if not isinstance(item, dict) else item.get("logprob")
        if tok is None or lp is None:
            continue
        if tok in YES_TOKEN_VARIANTS:
            if best is None or lp > best:
                best = lp
    if best is None:
        best = logprob_floor
    return math.exp(best)

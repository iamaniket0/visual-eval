"""Judge-backend aware methodology and disclosure text."""

from __future__ import annotations

from src.t2i import load_settings
from src.t2i.report.constants import DISCLOSURE_LAYERS, THEME_MIN_N


def _current_judge_backend() -> str:
    try:
        return load_settings().get("judge", {}).get("backend", "gpt4o_hard")  # type: ignore[no-any-return]
    except Exception:
        return "gpt4o_hard"


def _methodology_text() -> str:
    """Methodology paragraph that matches the judge backend actually used."""
    backend = _current_judge_backend()
    base = (
        "Prompts are drawn from T2I-CompBench++ (Layer 1, 150 prompts) and a "
        "proprietary internally-authored set (Layer 2, 60 prompts). Each prompt is "
        "decomposed into atomic binary questions following the CompQuest "
        "pattern. "
    )
    if backend == "qwen_together_soft":
        # Actual backend used for the April 2026 run. Qwen3.5-397B-A17B on
        # Together AI serverless: open-source, preserves logprobs, no self-
        # preference bias against GPT Image family models.
        try:
            slug = load_settings().get("judge", {}).get("model_slug", "Qwen/Qwen3.5-397B-A17B")
        except Exception:
            slug = "Qwen/Qwen3.5-397B-A17B"
        judge = (
            f"Judge: {slug} (Qwen3.5 MoE, open-source) served on Together AI "
            "(serverless text+image endpoint). Scoring follows Soft-TIFA "
            "(Kamath et al., GenEval 2, arXiv 2512.16853v1, Dec 2025): per-atom "
            'probabilities are extracted from the judge\'s "Yes" token logprob '
            "and aggregated two ways. <b>AM</b> = atom-level arithmetic mean of "
            "probabilities (partial-credit view, comparable to legacy TIFA). "
            "<b>GM</b> = prompt-level geometric mean (exp(mean(log p_i)), "
            "clipped at exp(-10)); GM is the primary metric because it collapses "
            "whenever any single atom is weak - the stricter view Kamath et al. "
            "show correlates best with human-labelled alignment (AUROC 94.5%). "
            "Thinking-mode is disabled on the judge request so Qwen3.5 emits a "
            'single-token "Yes"/"No" answer and logprob extraction is clean.'
        )
    elif backend == "qwen_soft":
        judge = (
            "Judge: Qwen3-VL (open-source) via OpenRouter. Scoring follows "
            "Soft-TIFA (Kamath et al., GenEval 2, arXiv 2512.16853v1, Dec "
            "2025): per-atom probabilities are extracted from the judge's "
            '"Yes" token logprob and aggregated two ways. '
            "<b>AM</b> = atom-level arithmetic mean of probabilities "
            "(the partial-credit view, comparable to legacy TIFA). "
            "<b>GM</b> = prompt-level geometric mean "
            "(exp(mean(log p_i)), clipped at exp(-10)); GM is the "
            "primary metric here because it collapses whenever any single "
            "atom is weak - the stricter view Kamath et al. show correlates "
            "best with human-labelled alignment (AUROC 94.5%)."
        )
    elif backend == "gpt4o_soft":
        judge = (
            "Judge: GPT-4o via OpenRouter (temperature 0). Scoring follows "
            "Soft-TIFA (Kamath et al., GenEval 2, arXiv 2512.16853v1, Dec "
            "2025): per-atom probabilities are extracted from the judge's "
            '"Yes" token logprob. <b>AM</b> is the atom-level arithmetic '
            "mean (partial credit, comparable to legacy TIFA). <b>GM</b> is "
            "the prompt-level geometric mean, clipped at exp(-10) to avoid "
            "log(0), and is the primary metric here. "
            "Caveat: when GPT Image 1.5 (gpt_image_15) is in the benchmark, "
            "a known ~3-7 point self-preference bias inflates its judged "
            "score under this backend. The preferred open-source Qwen3-VL "
            "judge is blocked on provider logprob support as of this run; "
            "flip `judge.backend` to `qwen_soft` once that path opens."
        )
    else:  # gpt4o_hard or unknown
        judge = (
            "GPT-4o serves as the MLLM judge (temperature 0). Score per "
            "image = yes_count / total_questions (hard TIFA)."
        )
    tail = (
        " Human validation on 10% of images targets Cohen's kappa > 0.6. "
        f"Theme-level cuts apply an n&ge;{THEME_MIN_N} per-cell filter to "
        "the chart and top/bottom lists so statistically noisy themes "
        "don't dominate the narrative."
    )
    return base + judge + tail


def _disclosure_text() -> str:
    """Disclosure paragraph. Adds the Soft-TIFA comparability note when
    the judge backend is soft so readers know old + new runs aren't
    directly comparable."""
    backend = _current_judge_backend()
    parts = [DISCLOSURE_LAYERS]
    if backend in ("qwen_soft", "gpt4o_soft", "qwen_together_soft"):
        parts.append(
            "Scoring methodology: Soft-TIFA (Kamath et al., arXiv "
            "2512.16853v1, Dec 2025). Meta's paper reports "
            "Soft-TIFA-GM with Qwen3-VL at 94.5% AUROC on "
            "human-judged alignment versus 91.6% for legacy "
            "TIFA+GPT-4o. Previous runs of this benchmark used hard "
            "TIFA with GPT-4o and are not directly comparable to "
            "current results."
        )
    return "  ".join(parts)


def _pitch_backend_caveat() -> str:
    """A sentence to append to model-card data pitch when relevant."""
    if _current_judge_backend() == "gpt4o_soft":
        return (
            " Note: scores under GPT-4o as judge carry a documented "
            "self-preference bias when evaluating GPT Image family models; "
            "migration to an open-source Qwen judge is planned."
        )
    return ""

# Visual Eval — Design Document

**Author:** Aniket  
**Last updated:** 2026-06-20  
**Status:** Accepted

---

## Motivation

Standard T2I benchmarks (DrawBench, PartiPrompts, T2I-CompBench) are saturating — frontier models all score >90% and the rankings barely move quarter to quarter. But anyone who's actually used these models knows they still fail hard on compositional prompts: count 5 objects, place them spatially, add constraints. The benchmarks don't test for this because their prompts are too easy.

I wanted to build something that actually separates models on the hard stuff, using a scoring method that doesn't give free partial credit for getting 4 out of 5 things right while completely hallucinating the 5th.

After reading through GenEval 2 (Kamath et al., arXiv 2512.16853), Soft-TIFA stood out as the right scoring approach — atomic binary decomposition + logprob extraction. The geometric mean is the key insight: one confident miss tanks the whole score, which matches how humans actually judge these images ("it got most of it right but the elephant is clearly not there").

## Goals

1. Benchmark T2I generation AND image editing models in a single unified pipeline
2. Use hard/adversarial prompts that actually differentiate frontier models
3. Implement Soft-TIFA scoring (AM + GM) with logprob extraction from an MLLM judge
4. Produce leaderboards, per-category breakdowns, and PDF scorecards
5. Keep costs under control ($300 hard cap for T2I, $200 for edit)
6. Make the pipeline resume-friendly — don't regenerate existing images or re-judge existing results

## Non-Goals

- Real-time evaluation / live benchmarking service
- Human preference ratings (HITL is for validation, not scoring)
- Supporting models without API access (Midjourney has an API now but it's unreliable)
- A web app — Streamlit dashboard is good enough for exploration

## Architecture

### Three-Layer Prompt Design

I went back and forth on this. Originally was going to do two layers (gold benchmark + custom), but realized we need a third layer specifically for adversarial/hard prompts that really stress-test compositional understanding.

**L1 — Gold Standard** (~150 prompts): Drawn from established benchmarks (T2I-CompBench++, GenEval). These are the "control group" — models should score well here, and if they don't, something is broken in our pipeline.

**L2 — Proprietary** (~60 prompts): Custom prompts with controlled difficulty. 3-8 atomic constraints per prompt. Mix of easy/medium/hard. These fill coverage gaps in L1.

**L3 — Adversarial** (~50 prompts): Deliberately hard. Multi-type counting ("exactly 3 red and 2 blue"), negation ("no visible shadows"), causal physics ("a wine glass mid-fall, about to shatter"), spatial conflicts. These are where models diverge.

For edits, the prompt structure is similar but with 12 sub-categories (color change, object add/remove, background swap, etc.) and difficulty is determined by the edit complexity rather than the number of atomic constraints.

### Scoring Pipeline

```
Prompt → Atomic Decomposition → Image Generation → MLLM Judge → Score Aggregation
                                                     │
                                            Extract P(Yes) from
                                            first-token logprobs
                                                     │
                                              AM = mean(pᵢ)
                                              GM = exp(mean(log(pᵢ)))
```

The decomposition step breaks each prompt into binary yes/no questions. For T2I this happens offline (during prompt creation). For edits, the questions are structured around three dimensions: instruction following, visual consistency, and detail preservation.

### Judge Backend Selection

Considered several options:

| Backend | Pros | Cons | Decision |
|---------|------|------|----------|
| GPT-4o | Best vision, reliable logprobs | Self-bias when judging GPT Image outputs | Fallback only |
| Qwen3.5-397B (Together) | Open-source, no self-bias, logprobs work | Together serverless sometimes slow | **Primary** |
| Qwen via OpenRouter | Single API key | Providers strip logprobs from Qwen VL routes | Blocked (April 2026) |
| Claude | Good vision | No logprob access | Not viable |

The self-bias issue with GPT-4o is real — in early testing it consistently scored GPT Image outputs 3-5% higher than Qwen did, while scoring other models similarly. Using an open-source judge eliminates this.

**UPDATE (July 2026):** Had to fall back to GPT-4o via OpenRouter for the actual eval run because the Together Qwen endpoint requires a paid dedicated instance for consistent logprob access. The self-bias is noted in the results. If I redo this evaluation, I'd get a dedicated Together endpoint.

### Cost Control

Every API call goes through a CostTracker that:
- Accumulates costs by model and by stage (generation vs judging)
- Alerts at 80% of the hard cap
- Hard-stops at the cap itself

Generation is the expensive part ($0.04-0.08 per image), judging is cheap (~$0.004 per judgment). At 50 prompts × 5 models × 1 seed, generation costs ~$15. Judging is ~$3. Total well under the $300 cap.

### Resume-Friendly Design

The pipeline checks for existing files before making API calls:
- Generation: if `outputs/t2i/generations/{model}/{prompt_id}.png` exists, skip
- Judging: if the prompt_id already has an entry in `judgments/{model}.jsonl`, skip
- Editing: if `outputs/edit/edits/{model}/{prompt_id}.png` exists, skip

This means you can kill a run and restart without wasting money. It also means incremental expansion is free — add more prompt IDs and only the new ones get generated.

## Edit Evaluation — 3-Axis Scoring

Adapted from GEditBench v2 (NTU, 2026). Each edited image is judged on three orthogonal dimensions:

1. **Instruction Following** — Did the requested edit actually happen?
2. **Visual Consistency** — Are unedited regions preserved?
3. **Detail Preservation** — Are fine details (text, textures, edges) intact?

This matters because there's a fundamental tension: a model that aggressively follows instructions tends to clobber unrelated parts of the image. A model that preserves everything tends to apply edits too timidly. The 3-axis scoring surfaces this tradeoff rather than hiding it behind a single number.

## Technology Choices

- **Python 3.10+** — async/await for concurrent API calls, type hints throughout
- **Registry pattern for models** — `@register("model_name")` decorator, no factory switch statements
- **ReportLab for PDFs** — better than matplotlib for structured report layouts
- **Streamlit for dashboard** — zero-config, good enough for data exploration
- **pytest + pytest-asyncio** — TDD from the start, async test support

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| API rate limits | Configurable concurrency per model, exponential backoff |
| Content moderation filtering | Captured as FILTERED status, scored 0, not retried |
| Judge model changes scoring | Pin model version, include backend in judgment metadata |
| Costs exceed cap | CostTracker with hard cutoff |
| Prompt contamination (model trained on benchmark prompts) | L2/L3 prompts are custom and not public |

## Open Questions (Resolved)

- ~~Should we retry filtered prompts with modified wording?~~ **No — this would compromise benchmark integrity. A filter is a genuine failure mode.**
- ~~Multi-seed variance: how many seeds per prompt?~~ **Started with 3, dropped to 1 for the portfolio run. 50 prompts × 1 seed is statistically sufficient. Can always add more seeds later.**
- ~~Should GM be computed per-prompt then averaged, or across all atoms?~~ **Per-prompt then averaged. This matches the GenEval 2 methodology.**

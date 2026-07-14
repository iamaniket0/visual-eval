# AGENTS.md — Visual Eval (T2I + Edit)

## Verification rules for all research/methodology claims

When responding with technical claims involving papers, benchmarks, models, or numerical baselines:

1. **Citation format required.** Every factual claim must cite source URL + publication date. Examples:
   - Paper: "X scores Y on benchmark Z (arXiv:2511.10136, Nov 2025)"
   - Repo: "Cloned from github.com/org/repo (last commit: 2026-04-15)"
   - No bare claims like "the literature shows" or "recent papers suggest"

2. **If you cannot verify a source via direct search, say so.**
   - Do not generate plausible-looking citations
   - Acceptable: "I'm not able to verify a specific paper for this claim"
   - Not acceptable: producing arXiv IDs that fit the format but don't correspond to real papers

3. **Methodology claims must match the actual code.**
   - Before describing what the pipeline does, read the relevant config and adapter
   - T2I judge: read `src/t2i/judge.py` and `config/t2i/settings.yaml`
   - Edit judge: read `src/edit/judge.py` and `config/edit/settings.yaml`
   - When describing scoring, run a single judgment and inspect the output schema

4. **For benchmark comparisons:**
   - Default to running prompts pulled directly from published SOTA benchmarks before authoring our own
   - T2I benchmarks: GenEval 2, T2I-CompBench++, SpatialGenEval, R2I-Bench
   - Edit benchmarks: GEditBench v2, Complex-Edit, CompAlign/CompQuest
   - Flag any score that diverges >15 points from published baselines — this is a sign of methodology mismatch

5. **Before sending output:**
   - Reread the response and verify every specific claim (paper IDs, model names, percentages)
   - If a citation is fabricated, the entire response is suspect — re-verify

---

## SOTA Benchmark Prompts First

**ALWAYS default to published, peer-reviewed benchmark prompts before generating our own.**

Priority order for prompt sources:
1. Published benchmark datasets with downloadable prompts:
   - T2I: GenEval 2 (800), SpatialGenEval (1,230), R2I-Bench (3,068), T2I-CompBench++
   - Edit: GEditBench v2 (NTU 2026), Complex-Edit (UCSC 2025), CompAlign/CompQuest
2. Published paper examples with verified failure rates
3. Claude-generated adversarial prompts (only after 1 and 2 are exhausted)
4. Hand-written prompts (last resort)

If our scores are 20+ points above published baselines on the same category, assume **our prompts are too easy**, not that models improved overnight.

---

## Category Relevance Filter

Before spending time on ANY eval category, ask three questions in order:

1. **Does the model actually fail hard here?** (score below 0.80 on SOTA benchmark prompts)
2. **Is this fixable with training data?** (not architectural — data must be THE fix)
3. **Is this category high-impact?** (clear value, not incremental improvement)

If any answer is NO, skip the category.

### T2I categories
- PASS: counting/numeracy, multi-constraint binding, comparison/occlusion, spatial depth, compositional reasoning
- FAIL: negation (architectural fix), text rendering (different modality), character consistency (no identity data), commonsense (measures pipeline intelligence)

### Edit categories
- PASS: instruction following (object add/remove/replace), visual consistency (background preservation), detail preservation (texture fidelity)
- FAIL: style transfer (architectural), identity preservation (no identity data)

---

## Slack Communication Rules

- **Concise by default.** If a message is more than ~15 lines, it should be a Notion doc instead.
- **Bold numbered section headers** (`*1. Title*`) for any message with 2+ topics.
- **Blank lines between every bullet and paragraph.**
- **Don't ask for permission when not needed.** Execute and report results.
- **Don't send multiple messages when one will do.** Consolidate.

---

## Report / Notion Rules

- **Raw results + examples first.** Do not over-interpret.
- **Every number must be traceable** to a specific CSV or JSONL file on disk.
- **Every image reference must be verified** — run `ls` on the path before citing it.
- **Keep it lean.** Do not put too much interpretation in docs. Focus on raw results and examples — interpretation comes after.

---

## Judge Provenance

- Primary judge: `Qwen/Qwen3.5-397B-A17B` on Together AI serverless (Soft-TIFA)
- R2I-Bench replication judge: `GPT-4o` via OpenRouter (R2I-Score, T2I only)
- Backend setting: `qwen_together_soft` in `config/{t2i,edit}/settings.yaml`
- Every judgment JSONL must have `judge_model: "Qwen/Qwen3.5-397B-A17B"`
- `run_judge` overwrites the JSONL — there is no append mode
- Before describing the judge in any doc, verify against actual JSONL:
  - T2I: `head -1 outputs/t2i/judgments/MODEL.jsonl | python -m json.tool | grep judge_model`
  - Edit: `head -1 outputs/edit/judgments/MODEL.jsonl | python -m json.tool | grep judge_model`

### Edit Judge Differences

- Edit judge receives **two images** (source + edited) for comparison
- Scores on 3 axes: instruction_following, visual_consistency, detail_preservation
- Each axis decomposed into atomic binary questions (same Soft-TIFA math)

---

## Cost Awareness

- Estimate cost before running: `n_images × cost_per_image` for generation/editing, `n_judgments × ~$0.015` for Qwen, `n_judgments × ~$0.02` for GPT-4o
- Report actual costs after each run
- T2I full 6-model judge pass: ~$60-80
- Edit full 7-model judge pass: ~$50-70

---

## Model Version Verification

Before benchmarking any model, verify:
1. Is this the latest flagship version? (web search with date)
2. Is the API endpoint correct? (smoke test with 1 prompt)
3. Is the API key valid and has sufficient credits? (check balance)

---

## File Locations

```
config/
├── t2i/
│   ├── models.yaml              # T2I model configs + profiles
│   └── settings.yaml            # Judge backend, seeds, cost caps
└── edit/
    ├── models.yaml              # Edit model configs + profiles
    ├── settings.yaml            # Judge backend, dimensions, cost caps
    └── prompt_taxonomy.yaml     # Difficulty calibration (Complex-Edit C1-C8)

prompts/
├── t2i/prompt_set.json          # T2I prompts (L1 + L2 + L3 + external)
└── edit/
    ├── layer1_prompts.json      # Public benchmark prompts
    └── layer2_prompts.json      # Proprietary prompts

src/
├── core/                        # Shared: scoring math, utils, cost tracker
├── t2i/                         # T2I generators, judge, aggregator, report
└── edit/                        # Edit editors, judge, aggregator

scripts/
├── t2i/                         # T2I pipeline scripts
└── edit/                        # Edit pipeline scripts

outputs/
├── t2i/                         # T2I generations, judgments, scores, reports
└── edit/                        # Edit outputs, judgments, scores

tests/
├── test_core/                   # Scoring math tests
├── test_t2i/                    # T2I pipeline tests
└── test_edit/                   # Edit pipeline tests

external/
├── GenEval2/                    # Published GenEval 2 benchmark data
├── SpatialGenEval/              # Published SpatialGenEval benchmark data
└── R2I-Bench/                   # Published R2I-Bench data + eval script

AGENTS.md                        # This file
```

---

## What NOT to Do

- Do NOT generate plausible-looking citations — verify or say "unverified"
- Do NOT describe the judge model from memory — read the config/JSONL first
- Do NOT run negation as a primary eval category (architectural issue, not fixable with data)
- Do NOT generate prompts by hand when published benchmark data exists
- Do NOT cite numbers without verifying against the actual CSV/JSONL on disk
- Do NOT send verbose Slack messages — use Notion for deep dives
- Do NOT re-judge models without warning that existing numbers will change
- Do NOT skip the category relevance filter before investing time in a new category
- Do NOT present commonsense scores without the pipeline-intelligence caveat
- Do NOT report R2I concept_mixing scores externally (n=6-7, not meaningful)
- Do NOT compare T2I and Edit scores directly — different judge schemas

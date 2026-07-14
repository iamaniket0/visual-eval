# Monday post-flight runbook — Soft-TIFA execution

This runbook executes the Soft-TIFA migration against Canva + xAI Aurora
using **Qwen3.5-397B-A17B on Together serverless** as the judge (internal
preferred path after Friday's escalation). Code migration was written
Friday; this runbook is the Monday execution sequence.

Total wall clock: **~40 min**. Total cost: **~$7**.

---

## Prerequisites (3 min)

### 1. Code sanity

```bash
cd /Users/aniket_ml/Downloads/t2i-eval
git status                  # expect clean tree
python -m pytest tests/ -q  # expect 41 passed
```

If anything fails, stop and reconcile before touching real APIs.

### 2. Confirm judge backend config

```bash
grep -A 2 "^  backend:" config/settings.yaml
```

Expected output:
```
  backend: qwen_together_soft
  model_slug: Qwen/Qwen3.5-397B-A17B
```

If it's still on `gpt4o_hard` from pre-migration, edit settings.yaml and
flip to `qwen_together_soft`.

### 3. Confirm keys

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
for k in ['OPENROUTER_API_KEY', 'TOGETHER_API_KEY', 'XAI_API_KEY', 'LEONARDO_API_KEY']:
    v = (os.getenv(k) or '').strip()
    print(f'{k}: {\"SET\" if v and len(v) > 10 else \"MISSING\"}')"
```

All four should print SET. TOGETHER_API_KEY is load-bearing for the judge;
the other three complete the generation side.

### 4. Sanity-check Together + OpenRouter credit

```bash
# Together serverless - a few dollars is plenty
curl -sS -H "Authorization: Bearer $(grep ^TOGETHER_API_KEY= .env | cut -d= -f2-)" \
  https://api.together.xyz/v1/models | head -c 200 > /dev/null && echo "Together key OK"

# OpenRouter (still used for generation of gpt_image_15 if added)
curl -sS -H "Authorization: Bearer $(grep ^OPENROUTER_API_KEY= .env | cut -d= -f2-)" \
  https://openrouter.ai/api/v1/credits | python -m json.tool
```

Need ~$5 headroom on each (judge + gen combined).

---

## Step 1 — Finish the interrupted xAI 3-seed generation (~12 min, ~$5)

Friday's 3-seed run was interrupted with xAI partway through seeds 1 and 2.
Resume logic will skip the ~300 PNGs already on disk and generate only the
~24 missing xAI images.

```bash
python -m scripts.run_generation \
    --models canva_lucid_origin,xai_aurora \
    --layer 2 --seeds 3
```

Watch for rate-limit retries. It's the last step that touches image-gen
APIs, so monitor.

**Post-check:**

```bash
for m in canva_lucid_origin xai_aurora; do
  l1=$(ls outputs/generations/$m/L1_*.png 2>/dev/null | wc -l)
  l2_s0=$(ls outputs/generations/$m/L2_*.png 2>/dev/null | grep -v '__s' | wc -l)
  l2_s1=$(ls outputs/generations/$m/L2_*__s1.png 2>/dev/null | wc -l)
  l2_s2=$(ls outputs/generations/$m/L2_*__s2.png 2>/dev/null | wc -l)
  echo "$m: L1=$l1  L2_s0=$l2_s0  L2_s1=$l2_s1  L2_s2=$l2_s2"
done
```

Expected: **150, 60, 60, 60** per model. If anything short, rerun step 1.

---

## Step 2 — Archive the old hard-TIFA judgments (~5 sec)

```bash
mkdir -p outputs/judgments/archive
for f in outputs/judgments/*.jsonl; do
  mv "$f" "outputs/judgments/archive/$(basename ${f%.jsonl}).gpt4o_old.jsonl"
done
ls outputs/judgments/archive/
```

Expected: `canva_lucid_origin.gpt4o_old.jsonl`, `xai_aurora.gpt4o_old.jsonl`
in the archive subdirectory. Aggregator only reads `.jsonl` files directly
under `outputs/judgments/`, so these archived files won't pollute Monday's
numbers but remain available for retrospective diffs.

---

## Step 3 — Soft-TIFA re-judge via Qwen3.5-397B on Together (~20 min, ~$1)

```bash
python -m scripts.run_judge \
    --models canva_lucid_origin,xai_aurora \
    --backend qwen_together_soft
```

What to expect:
- 660 prompt-level judgments (2 models × [150 L1 + 60×3 L2 seeds]).
- Each fans out to ~1 API call per atomic question (mean 4.7 atoms/prompt
  ≈ **~3100 atom calls total**).
- Together serverless throughput is usually strong (~10-20 calls/sec);
  the 397B MoE is fast despite the param count (only 17B active per call).
- Cost: ~$0.20-0.40 at Together's $0.39/M prompt + $2.34/M completion
  rates for this model — cheaper than you'd think because the response is
  1 token.

**Watch for `SoftTifaLogprobsUnavailable`.** Should never fire on this
backend (verified Friday that Together preserves logprobs for Qwen3.5 with
`logprobs=N` shape). If it fires, something regressed server-side; stop
and probe Together directly.

**Watch for first-token-not-Yes/No.** Also should never happen thanks to
`chat_template_kwargs: {enable_thinking: False}`. If you see a lot of atom
errors with first_token="The" or similar reasoning-word tokens, the
thinking-mode-disable flag isn't being honored — check the request body
with a one-off curl.

---

## Step 4 — Aggregate + regen reports (~30 sec)

```bash
python -m scripts.run_aggregate
MPLCONFIGDIR=/tmp/matplotlib-cache python -m scripts.run_report
```

Check the numbers landed correctly:

```bash
cat outputs/scores/leaderboard.csv
cat outputs/scores/layer_comparison.csv | head
```

Leaderboard should show distinct AM and GM columns — this run, unlike the
pre-migration legacy data, produces real probabilities and therefore
different AM/GM values per model.

**L1_NUM_013 sanity spot-check** (the "cars/clocks/bottles/microwaves" case):

```bash
python -c "
import json
for line in open('outputs/judgments/canva_lucid_origin.jsonl'):
    r = json.loads(line)
    if r['prompt_id'] == 'L1_NUM_013':
        print(f\"AM={r['score_am']:.3f}  GM={r['score_gm']:.3f}\")
        for a in r['answers']:
            p = a.get('probability', 0)
            mark = '+' if p >= 0.5 else '-'
            print(f\"  {mark} p={p:.3f}  {a['question']}\")
        break
"
```

Expected: **AM ≈ 0.57** (close to the pre-migration hard score), **GM
noticeably lower, 0.15–0.35 range**. If GM ≥ AM there's a bug — halt and
check `soft_tifa_gm` in aggregator.py.

**Medium-objects gap sanity**:

```bash
python -c "
import pandas as pd
tb = pd.read_csv('outputs/scores/theme_breakdown.csv')
for t in ['few-objects', 'medium-objects', 'dense']:
    for m in ['canva_lucid_origin', 'xai_aurora']:
        row = tb[(tb['theme']==t) & (tb['model']==m)].iloc[0]
        print(f'{t:<16s} {m:<22s} GM={row[\"mean_score_gm\"]:.2f} AM={row[\"mean_score_am\"]:.2f} n={int(row[\"n_prompts\"])}')"
```

Under hard TIFA the Canva-vs-xAI gap on `medium-objects` was 0.25 under AM.
Under Qwen Soft-TIFA GM the gap usually widens (the whole point of GM).
If the gap narrows or inverts under GM but not under AM, that's a signal
something is off with Qwen's numeracy calibration — investigate via the
raw judgments.

**Open the PDFs for visual check**:

```bash
open outputs/reports/aggregate_report.pdf \
     outputs/reports/canva_lucid_origin_card.pdf \
     outputs/reports/xai_aurora_card.pdf
```

Verify:
- Aggregate leaderboard has both AM and GM columns
- Methodology section cites Kamath et al. 2025 + Qwen3.5 judge
- Disclosure notes "Previous runs used hard TIFA + GPT-4o and are not
  directly comparable"
- Per-model cards show per-atom probabilities in failure examples

---

## Step 5 — Ship results to Slack

Once PDFs look good:

> Monday Soft-TIFA run complete: Qwen3.5-397B-A17B judge via Together
> serverless (Kamath et al. 2025 methodology, no GPT self-bias).
>
> Canva Lucid Origin: AM=**{am}** GM=**{gm}**
> xAI Aurora:         AM=**{am}** GM=**{gm}**
>
> Medium-objects gap (was 0.25 under hard TIFA): **{gap_gm}** GM,
> **{gap_am}** AM.
>
> PDFs in outputs/reports/. Pre-migration GPT-4o-hard run archived in
> outputs/judgments/archive/*.gpt4o_old.jsonl for retrospective diffs.

---

## Known gaps / next work

- **gpt_image_15 still unbenchmarked.** Adding it is a follow-up: run
  `python -m scripts.run_generation --models gpt_image_15 --seeds 3`
  (~45 min, ~$25 OpenRouter), then re-judge. Gets you the OpenAI numbers
  Tim needs for the Altman pitch.
- **Adobe Firefly + Midjourney still blocked** on credential procurement.
- **The 5 corrupted `.env` keys** (BFL, STABILITY, BRIA, FREEPIK, GOOGLE)
  still have internal-whitespace contamination. Not tonight's problem but
  blocks the 8-model scale-up.
- **`qwen_soft` (OpenRouter Qwen VL) still blocked** on providers stripping
  logprobs. Flip the settings.yaml backend to `qwen_soft` once OpenRouter
  gets a provider that preserves them - code is wired + tested.

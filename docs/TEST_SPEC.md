# Test Specification

**Tracks:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)  
**Framework:** pytest + pytest-asyncio  
**Target:** 100% of public API surface, property-based invariants for scoring math

---

## Core Scoring (`tests/test_core/test_scoring.py`)

These tests validate the mathematical foundation. No mocks, no fixtures — pure function testing with known inputs and expected outputs.

### Arithmetic Mean (AM)

| Test | Input | Expected | Why |
|------|-------|----------|-----|
| All perfect | [1.0, 1.0, 1.0] | 1.0 | Baseline sanity |
| All zero | [0.0, 0.0, 0.0] | 0.0 | Floor behavior |
| Mixed | [1.0, 0.0, 0.5] | 0.5 | Partial credit works |
| Empty | [] | 0.0 | Edge case — no atoms |

### Geometric Mean (GM)

| Test | Input | Expected | Why |
|------|-------|----------|-----|
| All perfect | [1.0, 1.0, 1.0] | 1.0 | Baseline |
| Single miss | [1.0, 1.0, 0.0] | exp(-10/3) | Floor kicks in at 0, collapses score |
| Empty | [] | 0.0 | Edge case |
| GM <= AM | random × 1000 | assert gm <= am | **Invariant** — always holds by AM-GM inequality |
| Confident miss collapse | [0.95, 0.95, 0.01] | ~0.21 | One bad atom tanks the whole score |

### Probability Extraction

| Test | Input | Expected | Why |
|------|-------|----------|-----|
| Yes token found | {"Yes": -0.1, "No": -3.0} | exp(-0.1) ≈ 0.905 | Normal case |
| SDK nested shape | logprobs with content.token.logprob | correct P(Yes) | OpenAI SDK format |
| Yes not in top tokens | {"No": -0.01, "Maybe": -5.0} | fallback 0.0 | Graceful degradation |
| Empty logprobs | {} | 0.0 | Edge case |

### Answer Normalization

| Test | Input | Expected | Why |
|------|-------|----------|-----|
| Soft answers | [0.9, 0.5] | [0.9, 0.5] | Pass-through |
| Hard yes/no | ["Yes", "No"] | [1.0, 0.0] | Legacy TIFA compat |
| Mixed | [0.9, "Yes", "No"] | [0.9, 1.0, 0.0] | Real-world data |

## CostTracker (`tests/test_core/test_utils.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| Hard cap enforcement | Add costs exceeding cap | Raises CostCapExceeded |
| Summary structure | Track by model + stage | Dict with by_model, by_stage, total_usd |

## T2I Generators (`tests/test_t2i/test_generators.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| Registry complete | All config models present | Set equality with config keys |
| Filter detection (pos) | Moderation response | Returns FILTERED status |
| Filter detection (neg) | Normal response | Returns success |
| Missing API key | Unset env var | Returns SKIPPED (not exception) |

Design note: we don't mock API calls for generation tests. Instead we test the registration pattern and error handling. Actual generation is tested via integration runs.

## T2I Judge (`tests/test_t2i/test_judge.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| Extract JSON (fenced) | ` ```json {...}``` ` | Parsed dict |
| Format questions | List of question dicts | Formatted string |
| Factory → qwen_together_soft | Default settings | TogetherQwen35SoftJudge instance |
| Factory → gpt4o_hard | Explicit backend arg | GPT4oHardJudge instance |
| Factory override | Backend param overrides settings | Correct type |
| Result scoring | Mock judgment results | Correct AM/GM |

## T2I Prompt Loader (`tests/test_t2i/test_prompt_loader.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| Stratified sample size | Request N from pool | Exactly N returned |
| Small pool handling | Request more than available | Returns all available |
| Deterministic | Same seed twice | Same result |
| Prompt ID format | Generated IDs | Match L{1,2,3}_XXX_NNN pattern |
| JSON extraction (plain) | Raw JSON string | Parsed object |
| JSON extraction (fenced) | Markdown-fenced JSON | Parsed object |
| JSON extraction (none) | No JSON in string | Raises ValueError |
| Placeholder decomposition | Stub prompt | Returns list of question dicts |

## T2I Aggregator (`tests/test_t2i/test_aggregator.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| Output files | Run on sample judgments | All expected CSVs exist |
| Layer divergence | L1 scores > L3 scores | Detected in layer_comparison.csv |

## Edit Generators (`tests/test_edit/test_generators.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| Registry complete | All config editors present | Set equality |
| Registry count | Count registered editors | Matches expected |

## Edit Judge (`tests/test_edit/test_judge.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| Extract yes probability (found) | Logprobs with Yes | Correct probability |
| Extract yes probability (not found) | Logprobs without Yes | 0.0 |
| AM/GM invariant | Random probabilities | GM <= AM always |
| AM single value | [0.7] | 0.7 |
| GM single value | [0.7] | 0.7 |

## Edit Aggregator (`tests/test_edit/test_aggregator.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| AM basic | Known inputs | Correct mean |
| AM empty | [] | 0.0 |
| AM all ones | [1, 1, 1] | 1.0 |
| GM basic | Known inputs | Correct geometric mean |
| GM with zero | Includes 0.0 | Near-zero (floor applied) |
| GM empty | [] | 0.0 |
| GM <= AM | Random inputs | Invariant holds |
| Soft answer normalization | Probability values | Pass-through |
| Hard answer normalization | "Yes"/"No" strings | 1.0/0.0 |

---

## Running Tests

```bash
# Full suite
pytest tests/ -v

# By module
pytest tests/test_core/ -v      # 14 tests — scoring, utils
pytest tests/test_t2i/ -v       # 21 tests — generators, judge, aggregator, prompts
pytest tests/test_edit/ -v      # 16 tests — editors, judge, aggregator

# Single test
pytest tests/test_core/test_scoring.py::test_gm_collapses_on_single_confident_miss -v
```

## Coverage Goals

- All scoring functions: 100% branch coverage
- All factory/registry functions: 100%
- Error paths: missing keys, empty inputs, malformed logprobs
- Invariants: GM <= AM across random inputs (property-based)
- No mock-heavy tests — test real behavior, not mock behavior

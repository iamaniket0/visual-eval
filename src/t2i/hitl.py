"""Stage 6: Human-in-the-loop validation.

Samples 10% of images (stratified: per-model x per-sub-category),
produces two interfaces for Dani:
    - CSV export at outputs/t2i/hitl/hitl_sample.csv (edit, re-import)
    - Flask web UI at src/t2i/hitl_webui.py

Re-import uses outputs/t2i/hitl/hitl_human.jsonl. Cohen's kappa is computed
judge-vs-human per answer.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.core.utils import get_logger, read_jsonl, append_jsonl
from src.t2i import OUTPUTS_DIR, load_settings
from src.t2i.prompt_loader import load_prompt_set

log = get_logger("hitl")

HITL_DIR = OUTPUTS_DIR / "hitl"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@dataclass
class HitlSampleRow:
    prompt_id: str
    model: str
    sub_category: str
    image_path: str
    prompt_text: str
    questions: list[dict[str, str]]   # [{q_id, question, type}]
    judge_answers: list[dict[str, str]]  # [{q_id, answer}]


def build_sample(seed: int = 42) -> list[HitlSampleRow]:
    """Stratified sample across all (model, sub_category) cells.

    Target: ~7 images per model per sub-category at 10% of 2100 = 210.
    """
    settings = load_settings()
    frac = settings["hitl"]["sample_fraction"]

    prompts_by_id = {p["prompt_id"]: p for p in load_prompt_set()}
    rng = random.Random(seed)

    # Collect all judged records grouped by (model, sub_category)
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for path in (OUTPUTS_DIR / "judgments").glob("*.jsonl"):
        model = path.stem
        for rec in read_jsonl(path):
            p = prompts_by_id.get(rec["prompt_id"])
            if not p or not rec.get("image_path"):
                continue
            if not Path(rec["image_path"]).exists():
                continue
            key = (model, p["sub_category"])
            by_cell.setdefault(key, []).append(rec)

    sampled: list[HitlSampleRow] = []
    for (model, sub), recs in by_cell.items():
        n = max(1, int(round(len(recs) * frac)))
        picks = rng.sample(recs, min(n, len(recs)))
        for rec in picks:
            p = prompts_by_id[rec["prompt_id"]]
            sampled.append(HitlSampleRow(
                prompt_id=rec["prompt_id"],
                model=model,
                sub_category=sub,
                image_path=rec["image_path"],
                prompt_text=p["prompt_text"],
                questions=p["atomic_questions"],
                judge_answers=[{"q_id": a["q_id"], "answer": a["answer"]}
                               for a in rec.get("answers", [])],
            ))
    log.info("HITL sample: %d images across %d cells", len(sampled), len(by_cell))
    return sampled


def save_sample(sample: list[HitlSampleRow]) -> Path:
    HITL_DIR.mkdir(parents=True, exist_ok=True)
    path = HITL_DIR / "hitl_sample.json"
    with open(path, "w") as f:
        json.dump([asdict(r) for r in sample], f, indent=2)
    return path


def load_sample() -> list[HitlSampleRow]:
    path = HITL_DIR / "hitl_sample.json"
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)
    return [HitlSampleRow(**r) for r in raw]


# ---------------------------------------------------------------------------
# CSV export / re-import
# ---------------------------------------------------------------------------

def export_csv(sample: list[HitlSampleRow]) -> Path:
    """One row per (image, question). Dani fills the `human_answer` column."""
    HITL_DIR.mkdir(parents=True, exist_ok=True)
    path = HITL_DIR / "hitl_sample.csv"
    rows = []
    for s in sample:
        judge_map = {a["q_id"]: a["answer"] for a in s.judge_answers}
        for q in s.questions:
            rows.append({
                "prompt_id": s.prompt_id,
                "model": s.model,
                "sub_category": s.sub_category,
                "image_path": s.image_path,
                "prompt_text": s.prompt_text,
                "q_id": q["q_id"],
                "question": q["question"],
                "judge_answer": judge_map.get(q["q_id"], ""),
                "human_answer": "",   # Dani fills yes/no
                "annotator": "",
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info("Wrote HITL CSV with %d rows to %s", len(rows), path)
    return path


def import_csv(csv_path: Path | None = None) -> Path:
    """Read back the completed CSV and write hitl_human.jsonl."""
    csv_path = Path(csv_path) if csv_path else (HITL_DIR / "hitl_sample.csv")
    df = pd.read_csv(csv_path)
    df["human_answer"] = df["human_answer"].fillna("").str.strip().str.lower()
    df = df[df["human_answer"].isin(["yes", "no"])]

    out_path = HITL_DIR / "hitl_human.jsonl"
    if out_path.exists():
        out_path.unlink()

    grouped = df.groupby(["prompt_id", "model"])
    for (pid, model), g in grouped:
        rec = {
            "prompt_id": pid,
            "model": model,
            "annotator": g["annotator"].iloc[0] if "annotator" in g else "dani",
            "human_answers": [
                {"q_id": row["q_id"], "answer": row["human_answer"]}
                for _, row in g.iterrows()
            ],
        }
        append_jsonl(out_path, rec)
    log.info("Imported %d (image, question) judgments into %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Agreement analysis
# ---------------------------------------------------------------------------

def compute_agreement() -> dict[str, Any]:
    """Compare judge vs human on the overlapping (prompt_id, q_id) cells."""
    human_recs = read_jsonl(HITL_DIR / "hitl_human.jsonl")
    if not human_recs:
        log.warning("No human annotations yet. Import the CSV first.")
        return {}

    human_map: dict[tuple[str, str, str], str] = {}
    for rec in human_recs:
        for a in rec["human_answers"]:
            human_map[(rec["prompt_id"], rec["model"], a["q_id"])] = a["answer"]

    pairs: list[tuple[str, str]] = []  # (judge, human)
    per_row = []
    for path in (OUTPUTS_DIR / "judgments").glob("*.jsonl"):
        model = path.stem
        for rec in read_jsonl(path):
            pid = rec["prompt_id"]
            judge_answers = rec.get("answers", [])
            matches = 0
            total = 0
            for a in judge_answers:
                key = (pid, model, a["q_id"])
                if key not in human_map:
                    continue
                judge_ans = a["answer"]
                human_ans = human_map[key]
                pairs.append((judge_ans, human_ans))
                matches += int(judge_ans == human_ans)
                total += 1
            if total:
                per_row.append({
                    "prompt_id": pid, "model": model,
                    "n_questions": total,
                    "agreement": round(matches / total, 4),
                })

    if not pairs:
        log.warning("No overlap between judge and human answers")
        return {}

    judge_labels = [p[0] for p in pairs]
    human_labels = [p[1] for p in pairs]
    kappa = cohen_kappa_score(judge_labels, human_labels)
    raw_agreement = sum(1 for j, h in pairs if j == h) / len(pairs)

    out = {
        "n_pairs": len(pairs),
        "raw_agreement": round(raw_agreement, 4),
        "cohen_kappa": round(kappa, 4),
        "target_kappa": load_settings()["hitl"]["target_cohen_kappa"],
        "per_image": per_row,
    }
    out_path = HITL_DIR / "agreement.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Cohen's kappa: %.3f (target > %.2f) across %d answer pairs",
             kappa, out["target_kappa"], len(pairs))
    return out

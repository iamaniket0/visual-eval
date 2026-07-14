"""Generate dark-themed PDF model reports using Jinja2 + WeasyPrint.

Usage:
    python -m scripts.generate_design_report
    python -m scripts.generate_design_report --model lucid_origin
"""

from __future__ import annotations

import argparse
import base64
import ctypes.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# WeasyPrint needs pango/gobject from Homebrew — set the fallback path
# before any cffi import triggers dlopen.
if sys.platform == "darwin":
    _brew_lib = "/opt/homebrew/lib"
    _existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if _brew_lib not in _existing:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{_brew_lib}:{_existing}"
    # cffi checks ctypes.util.find_library which doesn't honour DYLD_FALLBACK;
    # monkeypatch it so it finds the Homebrew dylibs.
    _orig_find = ctypes.util.find_library

    def _patched_find(name):
        result = _orig_find(name)
        if result:
            return result
        candidate = Path(_brew_lib) / f"lib{name}.dylib"
        if candidate.exists():
            return str(candidate)
        for p in Path(_brew_lib).glob(f"lib{name}*dylib"):
            return str(p)
        return None

    ctypes.util.find_library = _patched_find

import subprocess

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from PIL import Image

from src.core.utils import get_logger, read_jsonl
from src.t2i import OUTPUTS_DIR, load_settings

log = get_logger("design_report")

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
REPORTS_DIR = OUTPUTS_DIR / "reports" / "designed"
SCORES_DIR = OUTPUTS_DIR / "scores"

COMPANY_COLORS = {
    "lucid_origin": {"hex": "#00C4CC", "rgb": "0, 196, 204"},
    "xai_aurora": {"hex": "#FAFAFA", "rgb": "250, 250, 250"},
    "gpt_image_15": {"hex": "#10A37F", "rgb": "16, 163, 127"},
    "gpt_image_2": {"hex": "#10A37F", "rgb": "16, 163, 127"},
    "flux2_max": {"hex": "#8B5CF6", "rgb": "139, 92, 246"},
    "bria_fibo": {"hex": "#F59E0B", "rgb": "245, 158, 11"},
    "default": {"hex": "#1ED760", "rgb": "30, 215, 96"},
}

DISPLAY_NAMES = {
    "lucid_origin": "Lucid Origin",
    "xai_aurora": "xAI Aurora",
    "gpt_image_15": "GPT Image 1.5",
    "gpt_image_2": "GPT Image 2",
    "flux2_max": "FLUX.2 [max]",
    "bria_fibo": "Bria FIBO",
}

PITCH_TEXTS = {
    "lucid_origin": (
        "Lucid Origin is structurally weak on 2 of 3 compositional axes: "
        "numeracy (0.59 GM, 42% atom failure rate) and 3D spatial relationships (0.75 GM). "
        "Multi-object scenes with verified counts AND depth-labeled data — "
        "one training pipeline closes two independent weaknesses."
    ),
    "xai_aurora": (
        "xAI Aurora is the most reliable model in the benchmark (100% coverage) but its one "
        "weakness is sharp: 3D spatial relationships (0.79 GM, 33% atom failure on occlusion). "
        "Depth-and-occlusion-labeled data for front/back ordering and precise "
        "occlusion outlines on overlapping objects can address this."
    ),
    "gpt_image_15": (
        "GPT Image 1.5's intent router misclassifies 11% of standard compositional prompts "
        "as text-chat requests. On covered prompts, quality is strong (0.87 GM). "
        "Intent-router training data AND compositional-refinement data can address this."
    ),
    "gpt_image_2": (
        "GPT Image 2 is the quality ceiling (0.92 GM covered) but inherits the same 12% "
        "intent-routing failure as its predecessor. Intent-router training data "
        "for declarative-sentence image prompts AND compositional-refinement data for the "
        "remaining precision gaps can address this."
    ),
    "flux2_max": (
        "FLUX.2 [max] is the full-set leader (0.875 GM, 100% coverage). Its numeracy score "
        "(0.83) trails GPT Image 2 by 8 points. Verified multi-object counting data "
        "can close that gap and push FLUX.2 into clear first on all axes."
    ),
    "bria_fibo": (
        "Bria FIBO scores 0.857 GM — effectively tied with xAI Aurora. Numeracy at 0.81 "
        "is the gap: GPT Image 2 leads by 10 points. Numeracy training data can "
        "close that gap and push FIBO into a clear top-3 position."
    ),
}


def get_logo_b64() -> str:
    logo_path = TEMPLATE_DIR / "logo.svg"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_text().encode()).decode()
    return ""


def score_color(val: float) -> str:
    if val >= 0.80:
        return "#1ED760"
    if val >= 0.50:
        return "#F59E0B"
    return "#EF4444"


def pill_class(val: float) -> str:
    if val >= 0.80:
        return "pill-green"
    if val >= 0.50:
        return "pill-amber"
    return "pill-red"


def cell_class(val: float, col_values: list[float]) -> str:
    if val == max(col_values):
        return "cell-top"
    if val == min(col_values):
        return "cell-low"
    return "cell-mid"


def image_to_b64(path: str | Path, max_width: int = 400) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        img = Image.open(p)
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        img = img.convert("RGB")
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def build_leaderboard_bars(lb: pd.DataFrame, target: str, company_hex: str) -> list[dict]:
    col = "overall_gm_covered" if "overall_gm_covered" in lb.columns else "overall_gm"
    lb_sorted = lb.sort_values(col, ascending=False, na_position="last").reset_index(drop=True)
    max_score = lb_sorted[col].max()
    bars = []
    for _, row in lb_sorted.iterrows():
        model = row["model"]
        score = row[col]
        is_target = model == target
        bars.append(
            {
                "model": DISPLAY_NAMES.get(model, model),
                "width": min(100, max(2, (score / max(max_score, 0.01)) * 95)),
                "color": company_hex if is_target else "#A1A1AA",
                "label": f"{score:.3f}",
            }
        )
    return bars


def build_leaderboard_table(lb: pd.DataFrame, target: str) -> list[dict]:
    cov_col = "overall_gm_covered" if "overall_gm_covered" in lb.columns else "overall_gm"
    am_col = "overall_am_covered" if "overall_am_covered" in lb.columns else "overall_am"
    lb_sorted = lb.sort_values(cov_col, ascending=False, na_position="last").reset_index(drop=True)
    rows = []
    for _, row in lb_sorted.iterrows():
        model = row["model"]
        n_cov = int(row.get("n_covered", row.get("n_prompts", 210)))
        n_tot = int(row.get("n_total", row.get("n_prompts", 210)))
        rows.append(
            {
                "model": DISPLAY_NAMES.get(model, model),
                "gm_cov": f"{row[cov_col]:.3f}",
                "am_cov": f"{row[am_col]:.3f}" if pd.notna(row.get(am_col)) else "—",
                "gm_full": f"{row['overall_gm']:.3f}",
                "coverage": f"{n_cov}/{n_tot}",
                "is_target": model == target,
            }
        )
    return rows


def build_subcat_bars(psc: pd.DataFrame, target: str, company_hex: str) -> list[dict]:
    gm_cols = [c for c in psc.columns if c.endswith("__gm") and c != "overall_gm"]
    target_row = psc[psc["model"] == target]
    if target_row.empty:
        return []
    bars = []
    for col in sorted(gm_cols):
        name = col.replace("__gm", "")
        val = float(target_row[col].iloc[0])
        bars.append(
            {
                "model": name,
                "width": min(100, max(2, val * 95)),
                "color": company_hex,
                "label": f"{val:.2f}",
            }
        )
    return bars


def build_subcat_table(psc: pd.DataFrame, target: str) -> list[dict]:
    num_col = "numeracy__gm"
    sp_col = "spatial_3d__gm"
    cmp_col = "complex_compositions__gm"
    ov_col = "overall_gm"

    cols_needed = [num_col, sp_col, cmp_col, ov_col]
    for c in cols_needed:
        if c not in psc.columns:
            return []

    num_vals = psc[num_col].tolist()
    sp_vals = psc[sp_col].tolist()
    cmp_vals = psc[cmp_col].tolist()

    rows = []
    for _, row in psc.sort_values(ov_col, ascending=False).iterrows():
        model = row["model"]
        rows.append(
            {
                "model": DISPLAY_NAMES.get(model, model),
                "numeracy": f"{row[num_col]:.3f}",
                "spatial": f"{row[sp_col]:.3f}",
                "complex": f"{row[cmp_col]:.3f}",
                "overall": f"{row[ov_col]:.3f}",
                "num_class": cell_class(row[num_col], num_vals),
                "sp_class": cell_class(row[sp_col], sp_vals),
                "cmp_class": cell_class(row[cmp_col], cmp_vals),
                "is_target": model == target,
            }
        )
    return rows


def build_failures(model: str, n: int = 3) -> list[dict]:
    jpath = OUTPUTS_DIR / "judgments" / f"{model}.jsonl"
    if not jpath.exists():
        return []

    prompts = {}
    pset_path = Path("prompts/prompt_set.json")
    if pset_path.exists():
        prompts = {p["prompt_id"]: p for p in json.load(open(pset_path))}

    records = [r for r in read_jsonl(jpath) if not r.get("error") and r.get("answers")]
    records.sort(key=lambda r: r.get("score_gm", r.get("score", 0)))

    seen_pids = set()
    failures = []
    for r in records:
        pid = r["prompt_id"]
        if pid in seen_pids:
            continue
        seen_pids.add(pid)

        p = prompts.get(pid, {})
        img_path = r.get("image_path", "")
        atoms = []
        for a in r.get("answers", []):
            prob = a.get("probability")
            atoms.append(
                {
                    "question": a.get("question", ""),
                    "prob": f"{prob:.2f}" if prob is not None else "?",
                    "passed": prob is not None and prob >= 0.50,
                }
            )

        weakest = (
            min(atoms, key=lambda a: float(a["prob"]) if a["prob"] != "?" else 999)
            if atoms
            else None
        )
        diagnosis = ""
        diagnosis_type = ""
        if weakest and not weakest["passed"]:
            diagnosis_type = "Primary failure"
            diagnosis = f'Weakest atom: "{weakest["question"]}" at p={weakest["prob"]}.'

        failures.append(
            {
                "prompt_id": pid,
                "sub_category": p.get("sub_category", "unknown"),
                "prompt_text": p.get("prompt_text", ""),
                "gm": f"{r.get('score_gm', 0):.2f}",
                "am": f"{r.get('score_am', 0):.2f}",
                "image_b64": image_to_b64(img_path) if img_path else "",
                "atoms": atoms,
                "diagnosis": diagnosis,
                "diagnosis_type": diagnosis_type,
            }
        )

        if len(failures) >= n:
            break
    return failures


def generate_model_report(model: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lb = pd.read_csv(SCORES_DIR / "leaderboard.csv")
    psc = pd.read_csv(SCORES_DIR / "per_subcategory.csv")

    if model not in lb["model"].values:
        log.warning("Model %s not in leaderboard", model)
        return None

    colors = COMPANY_COLORS.get(model, COMPANY_COLORS["default"])
    cov_col = "overall_gm_covered" if "overall_gm_covered" in lb.columns else "overall_gm"
    am_cov_col = "overall_am_covered" if "overall_am_covered" in lb.columns else "overall_am"
    lb_sorted = lb.sort_values(cov_col, ascending=False, na_position="last").reset_index(drop=True)
    rank = int(lb_sorted[lb_sorted["model"] == model].index[0]) + 1

    model_row = lb[lb["model"] == model].iloc[0]
    gm_cov = float(model_row.get(cov_col, model_row["overall_gm"]))
    am_cov = float(model_row.get(am_cov_col, model_row["overall_am"]))
    n_covered = int(model_row.get("n_covered", model_row.get("n_prompts", 210)))
    n_total = int(model_row.get("n_total", model_row.get("n_prompts", 210)))
    coverage_pct = f"{100 * n_covered / max(n_total, 1):.0f}"
    uncovered_count = n_total - n_covered

    gm_cols = [c for c in psc.columns if c.endswith("__gm") and c != "overall_gm"]
    target_psc = psc[psc["model"] == model]
    weakest_subcat = ""
    weakest_subcat_gm = ""
    if not target_psc.empty and gm_cols:
        worst_col = min(gm_cols, key=lambda c: float(target_psc[c].iloc[0]))
        weakest_subcat = worst_col.replace("__gm", "")
        weakest_subcat_gm = f"{float(target_psc[worst_col].iloc[0]):.2f}"

    uncovered_explanation = ""
    if uncovered_count > 0 and "gpt_image" in model:
        uncovered_explanation = (
            "OpenAI's unified model misroutes declarative-sentence prompts as text-chat "
            "requests, returning grammar feedback instead of images. This is an intent-classification "
            "issue, not a safety filter."
        )
    elif uncovered_count > 0:
        uncovered_explanation = f"{uncovered_count} prompts returned no image after all retries."

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False)
    template = env.get_template("model_report.html")

    settings = load_settings()
    judge_slug = settings.get("judge", {}).get("model_slug", "Qwen/Qwen3.5-397B-A17B")

    methodology_text = (
        f"Prompts drawn from T2I-CompBench++ (Layer 1, 150 prompts) and proprietary set "
        f"(Layer 2, 60 prompts). Each prompt decomposed into atomic binary questions. "
        f"Judge: {judge_slug} on Together AI serverless. Scoring: Soft-TIFA "
        f"(Kamath et al., GenEval 2, arXiv 2512.16853v1). GM = geometric mean (primary, strict); "
        f"AM = arithmetic mean (diagnostic, partial credit). 3 seeds per prompt. "
        f"Thinking-mode disabled for clean logprob extraction."
    )

    disclosure_text = (
        "Layer 1 prompts are from T2I-CompBench++ (NeurIPS 2023), a public benchmark. "
        "Frontier models may have been calibrated against these. Layer 2 uses proprietary, "
        "unpublished prompts. Soft-TIFA GM with Qwen3-VL achieves 94.5% AUROC on human-judged "
        "alignment (Meta GenEval 2). Previous hard-TIFA runs are not directly comparable."
    )

    data = {
        "logo_b64": get_logo_b64(),
        "company_color": colors["hex"],
        "company_color_rgb": colors["rgb"],
        "run_date": datetime.now().strftime("%B %d, %Y"),
        "model_name": DISPLAY_NAMES.get(model, model),
        "gm_covered": f"{gm_cov:.3f}",
        "am_covered": f"{am_cov:.3f}",
        "gm_color": score_color(gm_cov),
        "rank": rank,
        "total_models": len(lb),
        "n_covered": n_covered,
        "n_total": n_total,
        "coverage_pct": coverage_pct,
        "weakest_subcat": weakest_subcat,
        "weakest_subcat_gm": weakest_subcat_gm,
        "leaderboard_bars": build_leaderboard_bars(lb, model, colors["hex"]),
        "leaderboard_table": build_leaderboard_table(lb, model),
        "subcat_bars": build_subcat_bars(psc, model, colors["hex"]),
        "subcat_table": build_subcat_table(psc, model),
        "uncovered_count": uncovered_count,
        "uncovered_explanation": uncovered_explanation,
        "uncovered_rate": f"{100 * uncovered_count / max(n_total, 1):.0f}",
        "failures": build_failures(model, n=3),
        "pitch_text": PITCH_TEXTS.get(
            model, "Targeted training data calibrated to this model's failure modes."
        ),
        "methodology_text": methodology_text,
        "disclosure_text": disclosure_text,
    }

    html_str = template.render(**data)

    html_path = REPORTS_DIR / f"{model}_designed.html"
    html_path.write_text(html_str, encoding="utf-8")

    pdf_path = REPORTS_DIR / f"{model}_designed.pdf"

    # Use Chrome headless to convert HTML → PDF (avoids WeasyPrint pango issues).
    # Try common Chrome/Chromium paths on macOS.
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Cursor.app/Contents/Frameworks/Chromium Embedded Framework.framework/Helpers/chrome",
    ]
    chrome = None
    for p in chrome_paths:
        if Path(p).exists():
            chrome = p
            break
    if not chrome:
        import shutil

        chrome = shutil.which("google-chrome") or shutil.which("chromium")

    if chrome:
        cmd = [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--print-to-pdf=" + str(pdf_path),
            "--print-to-pdf-no-header",
            "--no-pdf-header-footer",
            str(html_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            log.warning("Chrome PDF failed (rc=%d): %s", result.returncode, result.stderr[:200])
    else:
        log.warning("No Chrome/Chromium found. HTML saved but PDF not generated.")

    if pdf_path.exists():
        log.info("Wrote designed report: %s", pdf_path)
    else:
        log.info("HTML only (no PDF renderer): %s", html_path)
    return pdf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Generate for one model only")
    args = ap.parse_args()

    lb = pd.read_csv(SCORES_DIR / "leaderboard.csv")

    if args.model:
        models = [args.model]
    else:
        models = lb["model"].tolist()

    for model in models:
        try:
            path = generate_model_report(model)
            if path:
                print(f"Wrote: {path}")
        except Exception as e:
            log.error("Failed to generate report for %s: %s", model, e)
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()

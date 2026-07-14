"""Minimal Flask web UI for HITL annotation.

Run: python -m src.t2i.hitl_webui
Opens at http://localhost:5000

Each page shows one image with its atomic binary questions. Dani clicks
yes/no for each, then Submit. Progress is persisted to
outputs/t2i/hitl/hitl_human.jsonl on every submit.
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file, url_for

from src.t2i.hitl import HITL_DIR, build_sample, compute_agreement, load_sample, save_sample
from src.core.utils import append_jsonl, get_logger, read_jsonl

log = get_logger("hitl_webui")
app = Flask(__name__)


PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>HITL - T2I Eval</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px;
         margin: 30px auto; padding: 0 20px; color: #222; }
  h1 { font-size: 20px; margin: 0 0 4px 0; }
  .meta { color: #666; font-size: 13px; margin-bottom: 16px; }
  .progress { background: #eee; height: 6px; border-radius: 3px; margin-bottom: 20px; }
  .progress > div { background: #4a90e2; height: 100%; border-radius: 3px; }
  img { max-width: 100%; max-height: 520px; border: 1px solid #ccc;
        border-radius: 6px; display: block; margin: 10px 0; }
  .prompt { background: #f7f7f9; padding: 10px 14px; border-radius: 6px;
            font-style: italic; margin-bottom: 16px; }
  .q { display: flex; align-items: center; margin-bottom: 10px;
       padding: 10px; background: #fafafa; border-radius: 6px; }
  .q .text { flex: 1; }
  .q .btns { display: flex; gap: 6px; }
  .q button { padding: 6px 14px; border: 1px solid #ccc; background: white;
              border-radius: 4px; cursor: pointer; font-size: 14px; }
  .q button.sel-yes { background: #d4f0d4; border-color: #5cb85c; font-weight: bold; }
  .q button.sel-no  { background: #f5d4d4; border-color: #d9534f; font-weight: bold; }
  .actions { display: flex; justify-content: space-between; margin-top: 20px; }
  .actions button { padding: 10px 20px; font-size: 15px; border-radius: 6px;
                    border: none; cursor: pointer; }
  #submit { background: #4a90e2; color: white; }
  #submit:disabled { background: #aaa; cursor: not-allowed; }
  .done { text-align: center; padding: 60px 20px; }
  .kappa { font-size: 28px; color: #4a90e2; font-weight: bold; }
</style></head>
<body>
{% if done %}
  <div class="done">
    <h1>All done. Thanks Dani.</h1>
    <p>{{ n_annotated }} images annotated.</p>
    {% if kappa is not none %}
      <p>Cohen's kappa (judge vs human): <span class="kappa">{{ "%.3f"|format(kappa) }}</span></p>
      <p>Target: &gt; {{ target_kappa }}
         {% if kappa >= target_kappa %} Passed.{% else %} Below target.{% endif %}
      </p>
    {% endif %}
    <p><a href="/reset">Start over</a></p>
  </div>
{% else %}
  <h1>Image {{ idx + 1 }} of {{ total }}</h1>
  <div class="meta">
    <b>{{ row.prompt_id }}</b> &middot; model: {{ row.model }} &middot;
    sub-category: {{ row.sub_category }}
  </div>
  <div class="progress"><div style="width: {{ (idx/total*100)|round(1) }}%"></div></div>

  <div class="prompt">"{{ row.prompt_text }}"</div>
  <img src="{{ url_for('image', prompt_id=row.prompt_id, model=row.model) }}" />

  <form id="f" method="post" action="{{ url_for('submit', idx=idx) }}">
    {% for q in row.questions %}
      <div class="q">
        <div class="text"><b>{{ q.q_id }}</b>: {{ q.question }}</div>
        <div class="btns">
          <button type="button" data-q="{{ q.q_id }}" data-a="yes">Yes</button>
          <button type="button" data-q="{{ q.q_id }}" data-a="no">No</button>
        </div>
        <input type="hidden" name="{{ q.q_id }}" value="" required />
      </div>
    {% endfor %}
    <div class="actions">
      {% if idx > 0 %}
        <a href="{{ url_for('page', idx=idx-1) }}"><button type="button">Back</button></a>
      {% else %}<span></span>{% endif %}
      <button id="submit" type="submit" disabled>Submit &rarr;</button>
    </div>
  </form>

  <script>
    const btns = document.querySelectorAll('.q button');
    const form = document.getElementById('f');
    const submitBtn = document.getElementById('submit');
    btns.forEach(b => b.addEventListener('click', () => {
      const qid = b.dataset.q, a = b.dataset.a;
      form.querySelector(`input[name="${qid}"]`).value = a;
      b.parentElement.querySelectorAll('button').forEach(x => {
        x.classList.remove('sel-yes', 'sel-no');
      });
      b.classList.add(a === 'yes' ? 'sel-yes' : 'sel-no');
      const allFilled = [...form.querySelectorAll('input[type=hidden]')]
        .every(i => i.value);
      submitBtn.disabled = !allFilled;
    }));
  </script>
{% endif %}
</body></html>
"""


def _sample_or_build():
    sample = load_sample()
    if not sample:
        sample = build_sample()
        save_sample(sample)
    return sample


def _annotated_ids() -> set[tuple[str, str]]:
    recs = read_jsonl(HITL_DIR / "hitl_human.jsonl")
    return {(r["prompt_id"], r["model"]) for r in recs}


@app.route("/")
def index():
    sample = _sample_or_build()
    done_ids = _annotated_ids()
    for i, row in enumerate(sample):
        if (row.prompt_id, row.model) not in done_ids:
            return redirect(url_for("page", idx=i))
    return redirect(url_for("page", idx=len(sample)))


@app.route("/page/<int:idx>")
def page(idx: int):
    sample = _sample_or_build()
    total = len(sample)
    if idx >= total:
        agreement = compute_agreement()
        return render_template_string(
            PAGE, done=True, n_annotated=len(_annotated_ids()),
            kappa=agreement.get("cohen_kappa"),
            target_kappa=agreement.get("target_kappa", 0.6),
        )
    if idx < 0:
        abort(404)
    return render_template_string(PAGE, done=False, idx=idx, total=total,
                                   row=sample[idx])


@app.route("/submit/<int:idx>", methods=["POST"])
def submit(idx: int):
    sample = _sample_or_build()
    if idx >= len(sample):
        abort(404)
    row = sample[idx]
    answers = []
    for q in row.questions:
        val = request.form.get(q["q_id"], "").strip().lower()
        if val not in ("yes", "no"):
            abort(400, f"Missing answer for {q['q_id']}")
        answers.append({"q_id": q["q_id"], "answer": val})
    append_jsonl(HITL_DIR / "hitl_human.jsonl", {
        "prompt_id": row.prompt_id,
        "model": row.model,
        "annotator": "dani",
        "human_answers": answers,
    })
    return redirect(url_for("page", idx=idx + 1))


@app.route("/image/<prompt_id>/<model>")
def image(prompt_id: str, model: str):
    sample = _sample_or_build()
    for row in sample:
        if row.prompt_id == prompt_id and row.model == model:
            p = Path(row.image_path)
            if p.exists():
                return send_file(p)
    abort(404)


@app.route("/reset")
def reset():
    for name in ("hitl_human.jsonl", "agreement.json"):
        path = HITL_DIR / name
        if path.exists():
            path.unlink()
    return redirect(url_for("index"))


@app.route("/status")
def status():
    sample = _sample_or_build()
    done = _annotated_ids()
    return jsonify({
        "total": len(sample),
        "annotated": len(done),
        "remaining": len(sample) - len(done),
    })


def main():
    HITL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Starting HITL web UI at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()

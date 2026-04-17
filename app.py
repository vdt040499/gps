# app.py — Gradio web demo (GPS-VI)
import json
from pathlib import Path

import core.warnings_config  # noqa: F401

import gradio as gr

from core.ga_engine import GeneticPromptSearch
from core.scorer import PromptScorer
from mutation import PromptMutator

_REPO_ROOT = Path(__file__).resolve().parent
_DATA_DIR  = _REPO_ROOT / "data"
_DEV_PATH  = _DATA_DIR / "vi_sentiment_dev.json"
_SEEDS_PATH = _DATA_DIR / "seed_prompts.json"
_LABELS_VI  = ("tích cực", "tiêu cực", "trung lập")

scorer  = PromptScorer()
mutator = PromptMutator(scorer=scorer)
with open(_DEV_PATH,   encoding="utf-8") as f: dev_set = json.load(f)
with open(_SEEDS_PATH, encoding="utf-8") as f: seeds   = json.load(f)

gen_logs: list[dict] = []

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Force full width and dark background */
body, .gradio-container, #root {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #0d1117 !important;
    max-width: 100% !important;
    width: 100% !important;
}
.main.svelte-1kyws56, .wrap.svelte-1kyws56 {
    max-width: 100% !important;
    width: 100% !important;
}

/* === Hero === */
.gps-hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 55%, #24243e 100%);
    border-radius: 16px;
    padding: 32px 40px 28px;
    margin-bottom: 24px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
.gps-hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse at 25% 50%, rgba(99,102,241,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 50%, rgba(16,185,129,0.10) 0%, transparent 55%);
    pointer-events: none;
}
.gps-hero h1 {
    position: relative;
    margin: 0 0 8px;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    background: linear-gradient(120deg, #e0e7ff 0%, #a5b4fc 45%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.gps-hero .hero-sub {
    position: relative;
    color: rgba(255,255,255,0.48);
    font-size: 0.9rem;
    margin: 0 0 16px;
}
.gps-hero .badges {
    position: relative;
    display: flex; gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
}
.gps-hero .hb {
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 600;
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.6);
    border: 1px solid rgba(255,255,255,0.1);
}

/* === Section label === */
.sec-lbl {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #818cf8;
    margin: 20px 0 10px;
    padding-bottom: 7px;
    border-bottom: 1px solid rgba(129,140,248,0.15);
}

/* === Status banner === */
.st-banner {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 18px;
    border-radius: 10px;
    font-size: 0.87rem; font-weight: 500;
    margin-bottom: 14px;
}
.st-run  { background: rgba(59,130,246,0.10); border: 1px solid rgba(59,130,246,0.22); color: #93c5fd; }
.st-done { background: rgba(16,185,129,0.10); border: 1px solid rgba(16,185,129,0.22); color: #6ee7b7; }
@keyframes spin { to { transform: rotate(360deg); } }
.sp {
    width: 14px; height: 14px;
    border: 2px solid rgba(59,130,246,0.25);
    border-top-color: #60a5fa;
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
    flex-shrink: 0;
}

/* === 3-col stat cards === */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.s-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.s-card .sv { font-size: 1.55rem; font-weight: 800; line-height: 1.2; }
.s-card .sl { font-size: 0.69rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-top: 3px; }
.c-blue  { color: #60a5fa; }
.c-amber { color: #fbbf24; }
.c-green { color: #34d399; }

/* === Best prompt card === */
.bp-card {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.18);
    border-left: 4px solid #34d399;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.bp-lbl {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: #34d399; margin-bottom: 8px;
}
.bp-text {
    font-size: 0.93rem; color: #f1f5f9;
    line-height: 1.6; font-weight: 500;
    word-break: break-word;
}
.bp-chips { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
.bp-chip {
    padding: 3px 10px; border-radius: 8px;
    font-size: 0.72rem; font-weight: 600;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    color: #cbd5e1;
}

/* === Table === */
.ptbl {
    width: 100%;
    border-collapse: separate; border-spacing: 0;
    font-size: 0.84rem;
    border-radius: 10px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
}
.ptbl th {
    background: rgba(255,255,255,0.05);
    color: #94a3b8;
    font-weight: 700; text-transform: uppercase;
    font-size: 0.67rem; letter-spacing: 0.06em;
    padding: 11px 13px; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.ptbl td {
    padding: 11px 13px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    vertical-align: middle; color: #e2e8f0;
}
.ptbl tbody tr:hover td { background: rgba(99,102,241,0.05); }
.ptbl .t1 td { background: rgba(16,185,129,0.07) !important; }
.ptbl .rc {
    text-align: center; font-weight: 800;
    width: 44px; color: #64748b;
}
.ptbl .t1 .rc { color: #34d399; }
.ptbl .pc {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.ptbl .sc { width: 120px; }

/* Score bar */
.sb { display: flex; align-items: center; gap: 7px; }
.sb-bg { flex: 1; height: 5px; background: rgba(255,255,255,0.07); border-radius: 3px; overflow: hidden; }
.sb-fill { height: 100%; border-radius: 3px; transition: width .3s ease; }
.sb-v { font-weight: 700; font-size: 0.79rem; min-width: 40px; text-align: right; font-variant-numeric: tabular-nums; }

/* Naturalness pill */
.np { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 0.73rem; font-weight: 700; }
.np-h { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.25); }
.np-m { background: rgba(251,191,36,0.12); color: #fde68a; border: 1px solid rgba(251,191,36,0.2); }
.np-l { background: rgba(239,68,68,0.12);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.2); }

/* Empty state */
.es { text-align: center; padding: 48px 20px; color: #64748b; }
.es .ei { font-size: 2.2rem; margin-bottom: 10px; opacity: 0.4; }
.es .et { font-size: 0.88rem; font-weight: 500; }
.es .esu{ font-size: 0.76rem; color: #475569; margin-top: 4px; }

/* Test result */
.tr-box {
    margin-top: 10px; padding: 16px 20px;
    border-radius: 10px;
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.16);
}
.tr-pred { font-size: 1.25rem; font-weight: 800; }
.tr-pos { color: #34d399; }
.tr-neg { color: #f87171; }
.tr-neu { color: #fbbf24; }
.tr-rendered { margin-top: 10px; font-size: 0.8rem; color: #94a3b8; line-height: 1.55; word-break: break-word; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sc(v: float) -> str:
    if v >= 0.6: return "#34d399"
    if v >= 0.4: return "#fbbf24"
    return "#f87171"

def _np(v: float) -> str:
    cls = "np-h" if v >= 0.5 else ("np-m" if v >= 0.3 else "np-l")
    return f'<span class="np {cls}">{v:.2f}</span>'

def _stats_html(scores: dict) -> str:
    vals = [v for v in scores.values() if isinstance(v, dict)]
    if not vals: return ""
    ba = max(vals, key=lambda s: s["accuracy"])
    bn = max(vals, key=lambda s: s["naturalness"])
    bc = max(vals, key=lambda s: s["combined"])
    return (
        '<div class="stats-row">'
        f'<div class="s-card"><div class="sv c-blue">{ba["accuracy"]:.1%}</div><div class="sl">Best Accuracy</div></div>'
        f'<div class="s-card"><div class="sv c-amber">{bn["naturalness"]:.2f}</div><div class="sl">Best Naturalness</div></div>'
        f'<div class="s-card"><div class="sv c-green">{bc["combined"]:.1%}</div><div class="sl">Best Combined</div></div>'
        '</div>'
    )

def _best_html(scores: dict) -> str:
    ranked = sorted(scores.items(), key=lambda x: x[1]["combined"] if isinstance(x[1], dict) else 0, reverse=True)
    if not ranked: return ""
    p, d = ranked[0]
    if not isinstance(d, dict): return ""
    return (
        '<div class="bp-card">'
        '<div class="bp-lbl">🏆 PROMPT TỐT NHẤT</div>'
        f'<div class="bp-text">{p}</div>'
        '<div class="bp-chips">'
        f'<span class="bp-chip"><span style="color:#60a5fa">●</span> Acc {d["accuracy"]:.1%}</span>'
        f'<span class="bp-chip"><span style="color:#fbbf24">●</span> Nat {d["naturalness"]:.2f}</span>'
        f'<span class="bp-chip"><span style="color:#34d399">●</span> Combined {d["combined"]:.1%}</span>'
        '</div></div>'
    )

def build_pop_html(scores: dict) -> str:
    """Stats + best-prompt card + full table."""
    if not scores:
        return ('<div class="es"><div class="ei">🧬</div>'
                '<div class="et">Chưa có dữ liệu</div>'
                '<div class="esu">Chọn cấu hình và nhấn Chạy GPS để bắt đầu</div></div>')

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1]["combined"] if isinstance(x[1], dict) else x[1],
        reverse=True,
    )

    rows = ""
    for i, (prompt, sd) in enumerate(ranked):
        r = i + 1
        rc = ' class="t1"' if r == 1 else ""
        rd = "👑" if r == 1 else str(r)
        tr = (prompt[:90] + "…") if len(prompt) > 90 else prompt
        esc = prompt.replace('"', "&quot;")

        if isinstance(sd, dict):
            a, n, c = sd["accuracy"], sd["naturalness"], sd["combined"]
        else:
            a, n, c = float(sd), 0.0, float(sd)

        ac, cc = _sc(a), _sc(c)
        rows += (
            f'<tr{rc}>'
            f'<td class="rc">{rd}</td>'
            f'<td class="pc" title="{esc}">{tr}</td>'
            f'<td class="sc"><div class="sb"><div class="sb-bg"><div class="sb-fill" style="width:{max(a*100,2):.0f}%;background:{ac}"></div></div>'
            f'<span class="sb-v" style="color:{ac}">{a:.1%}</span></div></td>'
            f'<td style="text-align:center">{_np(n)}</td>'
            f'<td class="sc"><div class="sb"><div class="sb-bg"><div class="sb-fill" style="width:{max(c*100,2):.0f}%;background:{cc}"></div></div>'
            f'<span class="sb-v" style="color:{cc}">{c:.1%}</span></div></td>'
            f'</tr>'
        )

    tbl = (
        '<table class="ptbl"><thead><tr>'
        '<th style="width:44px">#</th><th>Prompt</th>'
        '<th style="width:120px">Accuracy</th><th style="width:90px">Natural</th><th style="width:120px">Combined</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return _stats_html(scores) + _best_html(scores) + tbl


def _status(gen: int, total: int, running: bool) -> str:
    if running:
        return (f'<div class="st-banner st-run"><div class="sp"></div>'
                f'Đang chạy Generation {gen+1}/{total} — scoring + mutation…</div>')
    return f'<div class="st-banner st-done">✅ Hoàn thành {total}/{total} generations</div>'

# ─────────────────────────────────────────────────────────────────────────────
# GPS runner — threading-based generator
# ─────────────────────────────────────────────────────────────────────────────

def run_gps(n_iter, top_k, strategy, alpha):
    """Generator: yields (status_html, pop_html, best_text, gen_slider_update)."""
    import queue, threading

    global gen_logs
    gen_logs = []
    n_iter, alpha = int(n_iter), float(alpha)

    gps = GeneticPromptSearch(scorer, mutator, top_k=int(top_k))
    q: queue.Queue = queue.Queue()

    def on_gen(gen, prompts, scores, all_scores):
        entry = {"gen": gen, "prompts": prompts, "scores": scores, "all_scores": all_scores}
        gen_logs.append(entry)
        q.put(("gen", entry))

    def _run():
        best, _, _ = gps.run(
            seed_prompts=seeds, dev_set=dev_set,
            n_iter=n_iter, strategy=strategy, alpha=alpha,
            callback=on_gen,
        )
        q.put(("done", best))

    # — Initial placeholder —
    empty = ('<div class="es"><div class="ei">⏳</div>'
             '<div class="et">Đang scoring generation đầu tiên…</div></div>')
    yield _status(0, n_iter, True), empty, "", gr.update(visible=False)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    best_prompts = []
    while True:
        kind, payload = q.get()

        if kind == "gen":
            entry   = payload
            g       = entry["gen"] + 1
            html    = build_pop_html(entry["all_scores"])
            ranked  = sorted(
                entry["all_scores"].items(),
                key=lambda x: x[1]["combined"] if isinstance(x[1], dict) else x[1],
                reverse=True,
            )
            best    = ranked[0][0] if ranked else "—"
            is_last = g >= n_iter

            yield (
                _status(g, n_iter, not is_last),
                html,
                best,
                gr.update(maximum=n_iter, value=g, visible=True),
            )

        else:  # "done"
            best_prompts = payload or []
            break

    thread.join()

    # — Final repaint with best result —
    if gen_logs:
        last   = gen_logs[-1]
        html   = build_pop_html(last["all_scores"])
        ranked = sorted(
            last["all_scores"].items(),
            key=lambda x: x[1]["combined"] if isinstance(x[1], dict) else x[1],
            reverse=True,
        )
        best = best_prompts[0] if best_prompts else (ranked[0][0] if ranked else "—")
        yield (
            _status(n_iter, n_iter, False),
            html,
            best,
            gr.update(maximum=n_iter, value=n_iter, visible=True),
        )


def update_pop(gen_idx: int):
    idx = int(gen_idx) - 1
    if idx < 0 or idx >= len(gen_logs):
        return ('<div class="es"><div class="ei">📭</div>'
                '<div class="et">Generation chưa có dữ liệu</div></div>')
    return build_pop_html(gen_logs[idx]["all_scores"])


def test_prompt(prompt_text: str, test_input: str) -> str:
    if not prompt_text or not test_input:
        return '<div class="tr-box" style="color:#94a3b8">⚠️ Nhập câu cần test và chọn prompt trước.</div>'
    pred = scorer.predict_label(prompt_text, test_input, list(_LABELS_VI))
    cmap = {"tích cực": "tr-pos", "tiêu cực": "tr-neg", "trung lập": "tr-neu"}
    cls  = cmap.get(pred, "")
    rendered = prompt_text.replace("{{văn_bản}}", f'<strong style="color:#e0e7ff">{test_input}</strong>')
    return (
        f'<div class="tr-box">'
        f'<div class="tr-pred {cls}">→ {pred}</div>'
        f'<div class="tr-rendered">{rendered}</div>'
        f'</div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="GPS-VI — Tìm kiếm Prompt Tự động",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:

    # ── Hero ──────────────────────────────────────────────────────────────
    gr.HTML(
        '<div class="gps-hero">'
        '<h1>🧬 GPS — Tìm kiếm Prompt Tự động</h1>'
        '<p class="hero-sub">Genetic Prompt Search · Tối ưu prompt tiếng Việt bằng thuật toán di truyền</p>'
        '<div class="badges">'
        '<span class="hb">🎯 Accuracy + Naturalness</span>'
        '<span class="hb">🔬 mt0-large</span>'
        '<span class="hb">🇻🇳 bartpho</span>'
        '<span class="hb">📊 UIT-VSFC</span>'
        '</div></div>'
    )

    # ── Settings ──────────────────────────────────────────────────────────
    gr.HTML('<div class="sec-lbl">⚙️ CÀI ĐẶT THUẬT TOÁN</div>')

    with gr.Row():
        with gr.Column(scale=1):
            n_iter = gr.Slider(1, 9, value=4, step=1, label="Số vòng lặp")
        with gr.Column(scale=1):
            top_k = gr.Slider(2, 10, value=5, step=1, label="Top-K selection")
        with gr.Column(scale=1):
            alpha = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="α — Accuracy vs Naturalness",
                              info="α=1 → chỉ accuracy · α=0 → chỉ naturalness")

    with gr.Row():
        with gr.Column(scale=2):
            strategy = gr.Radio(
                ["bt", "sc", "cloze", "all"],
                value="sc",
                label="Chiến lược đột biến",
                info="bt = Back Translation · sc = Sentence Continuation · cloze = Cloze",
            )
        with gr.Column(scale=1):
            run_btn = gr.Button("🚀  Chạy GPS", variant="primary", size="lg")
            gen_slider = gr.Slider(1, 4, value=1, step=1,
                                   label="📂  Xem Generation", visible=False)

    # ── Results ───────────────────────────────────────────────────────────
    gr.HTML('<div class="sec-lbl">📊 KẾT QUẢ</div>')
    status_html = gr.HTML(value="")
    pop_table   = gr.HTML(
        value=(
            '<div class="es"><div class="ei">🧬</div>'
            '<div class="et">Chưa có dữ liệu</div>'
            '<div class="esu">Chọn cấu hình và nhấn Chạy GPS để bắt đầu</div></div>'
        )
    )
    best_prompt_box = gr.Textbox(visible=False, label="best")   # hidden state

    # ── Test ──────────────────────────────────────────────────────────────
    gr.HTML('<div class="sec-lbl">🧪 TEST THỬ PROMPT</div>')
    with gr.Row():
        with gr.Column(scale=3):
            test_input = gr.Textbox(
                label="Nhập câu tiếng Việt",
                lines=2,
                placeholder="VD: Sản phẩm rất tốt, tôi rất hài lòng!",
            )
        with gr.Column(scale=1, min_width=160):
            test_btn = gr.Button("⚡  Dự đoán", variant="secondary", size="lg")
    test_output = gr.HTML()

    # ── Wiring ────────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_gps,
        inputs=[n_iter, top_k, strategy, alpha],
        outputs=[status_html, pop_table, best_prompt_box, gen_slider],
    )
    gen_slider.change(fn=update_pop, inputs=[gen_slider], outputs=[pop_table])
    test_btn.click(fn=test_prompt, inputs=[best_prompt_box, test_input], outputs=[test_output])

if __name__ == "__main__":
    demo.launch(share=True)

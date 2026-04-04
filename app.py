# app.py — Gradio web demo
import json
from pathlib import Path

import core.warnings_config  # noqa: F401 — before gradio/torch (see core/warnings_config.py)

import gradio as gr
import plotly.graph_objects as go

from core.ga_engine import GeneticPromptSearch
from core.scorer import PromptScorer
from mutation import PromptMutator

_REPO_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _REPO_ROOT / "data"
_DEV_PATH = _DATA_DIR / "vi_sentiment_dev.json"
_SEEDS_PATH = _DATA_DIR / "seed_prompts.json"

_LABELS_VI = ("tích cực", "tiêu cực", "trung lập")

# Loaded once at startup
scorer = PromptScorer()
mutator = PromptMutator()
with open(_DEV_PATH, encoding="utf-8") as f:
    dev_set = json.load(f)
with open(_SEEDS_PATH, encoding="utf-8") as f:
    seeds = json.load(f)

gen_logs: list[dict] = []
is_running = False

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* Header */
.gps-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 16px;
    text-align: center;
}
.gps-header h1 {
    margin: 0 0 6px 0;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.gps-header p {
    margin: 0;
    opacity: 0.75;
    font-size: 0.95rem;
}

/* Card sections */
.card-section {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 12px;
}

/* Population table */
.pop-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.88rem;
    border-radius: 8px;
    overflow: hidden;
}
.pop-table th {
    background: #f8fafc;
    color: #475569;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 2px solid #e2e8f0;
}
.pop-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: middle;
}
.pop-table tr:nth-child(even) td {
    background: #f8fafc;
}
.pop-table tr:hover td {
    background: #f0f9ff;
}
.pop-table .top-row td {
    background: linear-gradient(90deg, #fefce8 0%, #fef9c3 100%) !important;
    font-weight: 600;
}
.pop-table .rank-cell {
    text-align: center;
    font-weight: 700;
    width: 50px;
}
.pop-table .score-cell {
    width: 160px;
}

/* Accuracy bar */
.acc-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
}
.acc-bar-bg {
    flex: 1;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
}
.acc-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}
.acc-pct {
    font-weight: 600;
    font-size: 0.85rem;
    min-width: 48px;
    text-align: right;
}

/* Best prompt box */
.best-prompt-card {
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 16px 20px;
}

/* Status banner */
.status-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 18px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 10px;
}
.status-running {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
}
.status-done {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
}
@keyframes spin { to { transform: rotate(360deg); } }
.spinner {
    width: 16px; height: 16px;
    border: 2.5px solid #bfdbfe;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}

/* Override dark mode if needed */
.dark .pop-table th { background: #1e293b; color: #94a3b8; border-bottom-color: #334155; }
.dark .pop-table td { border-bottom-color: #1e293b; }
.dark .pop-table tr:nth-child(even) td { background: #0f172a; }
.dark .pop-table .top-row td { background: linear-gradient(90deg, #422006 0%, #713f12 100%) !important; }
.dark .status-running { background: #1e293b; border-color: #1e3a5f; color: #93c5fd; }
.dark .status-done { background: #052e16; border-color: #166534; color: #86efac; }
"""


def _bar_color(score: float) -> str:
    if score >= 0.6:
        return "#10b981"
    if score >= 0.4:
        return "#f59e0b"
    return "#ef4444"


def build_population_html(all_scores: dict[str, float]) -> str:
    """Build an HTML table for the full population, sorted descending by score."""
    if not all_scores:
        return "<p style='color:#94a3b8;text-align:center;padding:24px;'>Chưa có dữ liệu. Hãy chạy GPS trước.</p>"

    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    rows_html = ""
    for i, (prompt, score) in enumerate(ranked):
        rank = i + 1
        is_top = rank == 1
        row_class = 'class="top-row"' if is_top else ""
        rank_display = f"🥇 {rank}" if is_top else str(rank)
        truncated = (prompt[:100] + "…") if len(prompt) > 100 else prompt
        color = _bar_color(score)
        bar_width = max(score * 100, 2)

        rows_html += f"""
        <tr {row_class}>
            <td class="rank-cell">{rank_display}</td>
            <td title="{prompt.replace('"', '&quot;')}">{truncated}</td>
            <td class="score-cell">
                <div class="acc-bar-wrap">
                    <div class="acc-bar-bg">
                        <div class="acc-bar-fill" style="width:{bar_width:.1f}%;background:{color};"></div>
                    </div>
                    <span class="acc-pct" style="color:{color};">{score:.1%}</span>
                </div>
            </td>
        </tr>"""

    return f"""
    <table class="pop-table">
        <thead>
            <tr><th style="width:50px;">#</th><th>Prompt</th><th style="width:160px;">Accuracy</th></tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>"""


def _status_html(gen_done: int, total: int, running: bool) -> str:
    """Small banner showing current GPS progress."""
    if running:
        return (
            f'<div class="status-banner status-running">'
            f'<div class="spinner"></div>'
            f'Đang chạy Generation {gen_done + 1}/{total} — scoring + mutation…'
            f'</div>'
        )
    return (
        f'<div class="status-banner status-done">'
        f'Hoàn thành {total}/{total} generations'
        f'</div>'
    )


def _build_chart(logs):
    """Build convergence chart from gen_logs so far."""
    if not logs:
        return go.Figure()
    gens = [h["gen"] for h in logs]
    top1s = [max(h["scores"].values()) for h in logs]
    avg_s = [sum(h["scores"].values()) / len(h["scores"]) for h in logs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens, y=top1s, name="Top-1",
        mode="lines+markers",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=avg_s, name="Mean top-K",
        mode="lines+markers",
        line=dict(color="#6366f1", width=1.5, dash="dot"),
        marker=dict(size=5),
    ))
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Accuracy",
        yaxis_tickformat=".0%",
        template="simple_white",
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )
    return fig


def run_gps(n_iter, top_k, strategy, progress=gr.Progress(track_tqdm=True)):
    """Generator: yield (chart, pop_html, best_prompt, gen_slider, status) after each generation."""
    import queue
    import threading

    global gen_logs, is_running
    gen_logs, is_running = [], True
    n_iter = int(n_iter)

    gps = GeneticPromptSearch(scorer, mutator, top_k=int(top_k))
    gen_queue: queue.Queue = queue.Queue()

    def on_gen(gen, prompts, scores, all_scores):
        entry = {
            "gen": gen,
            "prompts": prompts,
            "scores": scores,
            "all_scores": all_scores,
        }
        gen_logs.append(entry)
        gen_queue.put(entry)

    result_holder: list = []

    def _run():
        best, history, _final_scores = gps.run(
            seed_prompts=seeds,
            dev_set=dev_set,
            n_iter=n_iter,
            strategy=strategy,
            callback=on_gen,
        )
        result_holder.append(best)
        gen_queue.put(None)  # sentinel: done

    # Initial status: scoring first generation
    yield (
        go.Figure(),
        "<p style='color:#94a3b8;text-align:center;padding:24px;'>Đang scoring generation đầu tiên…</p>",
        "",
        gr.update(visible=False),
        _status_html(0, n_iter, running=True),
    )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        entry = gen_queue.get()
        if entry is None:
            break

        gen_num = entry["gen"] + 1  # 1-based
        fig = _build_chart(gen_logs)
        pop_html = build_population_html(entry["all_scores"])

        ranked = sorted(entry["all_scores"].items(), key=lambda x: x[1], reverse=True)
        best_text = ranked[0][0] if ranked else "—"
        slider_update = gr.update(maximum=n_iter, value=gen_num, visible=True)

        is_last = gen_num >= n_iter
        if is_last:
            # Last generation scored, but final rescore still running
            status = _status_html(gen_num, n_iter, running=True)
            yield fig, pop_html, best_text, slider_update, status
        else:
            # Show completed gen, then immediately show "running next gen" spinner
            status = _status_html(gen_num, n_iter, running=True)
            yield fig, pop_html, best_text, slider_update, status

    thread.join()
    is_running = False

    # Final yield — done
    if gen_logs:
        fig = _build_chart(gen_logs)
        last = gen_logs[-1]
        pop_html = build_population_html(last["all_scores"])
        ranked = sorted(last["all_scores"].items(), key=lambda x: x[1], reverse=True)
        best_text = result_holder[0][0] if result_holder and result_holder[0] else ranked[0][0]
        slider_update = gr.update(maximum=n_iter, value=n_iter, visible=True)
        status = _status_html(n_iter, n_iter, running=False)
        yield fig, pop_html, best_text, slider_update, status


def update_population_table(gen_idx):
    """Rebuild population table for the selected generation."""
    idx = int(gen_idx) - 1  # slider is 1-based
    if idx < 0 or idx >= len(gen_logs):
        return "<p style='color:#94a3b8;text-align:center;'>Generation chưa có dữ liệu.</p>"
    return build_population_html(gen_logs[idx]["all_scores"])


def test_prompt(prompt_text: str, test_input: str) -> str:
    """Run one-shot prediction with the best prompt template."""
    if not prompt_text or not test_input:
        return "Nhập câu cần test và chọn prompt trước."
    rendered = prompt_text.replace("{{văn_bản}}", test_input)
    pred = scorer.predict_label(prompt_text, test_input, list(_LABELS_VI))
    return f"Dự đoán: **{pred}**\n\nPrompt được render:\n{rendered}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="GPS-VI — Tìm kiếm Prompt Tự động", css=CUSTOM_CSS) as demo:
    # Header
    gr.HTML("""
    <div class="gps-header">
        <h1>GPS: Tìm kiếm Prompt Tự động cho Tiếng Việt</h1>
        <p>Genetic Prompt Search — thuật toán di truyền tối ưu prompt cho NLP tiếng Việt</p>
    </div>
    """)

    # Controls + Chart
    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            gr.HTML('<div style="font-weight:600;margin-bottom:8px;">Cài đặt</div>')
            n_iter = gr.Slider(1, 9, value=4, step=1, label="Số vòng lặp (generations)")
            top_k = gr.Slider(2, 10, value=5, step=1, label="Top-K selection")
            strategy = gr.Radio(
                ["bt", "sc", "cloze", "all"],
                value="sc",
                label="Chiến lược đột biến",
                info="bt = Back Translation · sc = Sentence Continuation · cloze = T5 Cloze",
            )
            run_btn = gr.Button("Chạy GPS", variant="primary", size="lg")

            gr.HTML('<div style="font-weight:600;margin:16px 0 8px;">Chọn Generation</div>')
            gen_slider = gr.Slider(
                1, 4, value=4, step=1,
                label="Generation",
                visible=False,
            )

        with gr.Column(scale=3):
            status_html = gr.HTML(value="")
            conv_plot = gr.Plot(label="Biểu đồ hội tụ", show_label=False)

    # Population table
    gr.HTML('<div style="font-weight:600;font-size:1.05rem;margin:8px 0;">Quần thể prompts</div>')
    pop_table = gr.HTML(
        value="<p style='color:#94a3b8;text-align:center;padding:24px;'>Chưa có dữ liệu. Hãy chạy GPS trước.</p>"
    )

    # Best prompt
    gr.HTML('<div style="font-weight:600;font-size:1.05rem;margin:8px 0;">Prompt tốt nhất (#1)</div>')
    best_prompt_box = gr.Textbox(
        label="Prompt #1",
        lines=3,
        show_label=False,
        elem_classes=["best-prompt-card"],
    )

    # Test section
    gr.HTML('<div style="font-weight:600;font-size:1.05rem;margin:16px 0 8px;">Test thử prompt</div>')
    with gr.Row():
        with gr.Column(scale=2):
            test_input = gr.Textbox(
                label="Nhập câu tiếng Việt",
                lines=2,
                placeholder="Sản phẩm rất tốt, tôi rất hài lòng!",
            )
        with gr.Column(scale=1, min_width=120):
            test_btn = gr.Button("Dự đoán", variant="secondary")
    test_output = gr.Markdown(label="Kết quả")

    # Wiring
    run_btn.click(
        run_gps,
        [n_iter, top_k, strategy],
        [conv_plot, pop_table, best_prompt_box, gen_slider, status_html],
    )
    gen_slider.change(
        update_population_table,
        [gen_slider],
        [pop_table],
    )
    test_btn.click(test_prompt, [best_prompt_box, test_input], [test_output])

if __name__ == "__main__":
    demo.launch(share=True)

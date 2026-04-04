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


def run_gps(n_iter, top_k, strategy, progress=gr.Progress()):
    """Run GPS and return table rows, convergence figure, and best prompt text."""
    global gen_logs, is_running
    gen_logs, is_running = [], True

    gps = GeneticPromptSearch(scorer, mutator, top_k=int(top_k))

    def on_gen(gen, prompts, scores):
        gen_logs.append({"gen": gen, "prompts": prompts, "scores": scores})

    progress(0, desc="Starting GPS...")
    best, history, final_scores = gps.run(
        seed_prompts=seeds,
        dev_set=dev_set,
        n_iter=int(n_iter),
        strategy=strategy,
        callback=on_gen,
    )
    is_running = False

    rows = [(p[:80] + "…", f"{final_scores[p]:.1%}") for p in best]

    gens = [h["gen"] for h in history]
    top1s = [max(h["scores"].values()) for h in history]
    avg_s = [sum(h["scores"].values()) / len(h["scores"]) for h in history]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gens,
            y=top1s,
            name="Top-1",
            mode="lines+markers",
            line=dict(color="#1D9E75", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gens,
            y=avg_s,
            name="Mean",
            mode="lines+markers",
            line=dict(color="#378ADD", width=1.5, dash="dot"),
        )
    )
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Accuracy",
        yaxis_tickformat=".0%",
        template="simple_white",
        margin=dict(l=40, r=20, t=20, b=40),
    )

    return rows, fig, best[0] if best else "—"


def test_prompt(prompt_text: str, test_input: str) -> str:
    """Run one-shot prediction with the best prompt template."""
    if not prompt_text or not test_input:
        return "Nhập câu cần test và chọn prompt trước."
    rendered = prompt_text.replace("{{văn_bản}}", test_input)
    pred = scorer.predict_label(prompt_text, test_input, list(_LABELS_VI))
    return f"Dự đoán: **{pred}**\n\nPrompt được render:\n{rendered}"


with gr.Blocks(title="GPS Tiếng Việt — Đồ án Thuật toán Tiến hóa") as demo:
    gr.Markdown("## GPS: Tìm kiếm Prompt Tự động cho Tiếng Việt")
    gr.Markdown(
        "Genetic Prompt Search — áp dụng thuật toán di truyền để tối ưu prompt NLP"
    )

    with gr.Row():
        with gr.Column(scale=1):
            n_iter = gr.Slider(1, 9, value=4, step=1, label="Số vòng lặp (generations)")
            top_k = gr.Slider(2, 10, value=5, step=1, label="Top-K selection")
            strategy = gr.Radio(
                ["bt", "sc", "cloze", "all"],
                value="sc",
                label="Chiến lược đột biến",
                info="bt=Back Translation, sc=Sentence Continuation, cloze=T5 Cloze",
            )
            run_btn = gr.Button("Chạy GPS", variant="primary")

        with gr.Column(scale=2):
            result_table = gr.Dataframe(
                headers=["Prompt", "Accuracy"],
                label="Prompt tốt nhất tìm được",
            )
            conv_plot = gr.Plot(label="Biểu đồ hội tụ qua các generation")

    with gr.Row():
        best_prompt_box = gr.Textbox(label="Prompt #1 tốt nhất", lines=3)

    gr.Markdown("### Test thử prompt trực tiếp")
    with gr.Row():
        test_input = gr.Textbox(
            label="Nhập câu tiếng Việt để test",
            lines=2,
            placeholder="Sản phẩm rất tốt, tôi rất hài lòng!",
        )
        test_output = gr.Markdown(label="Kết quả")
    test_btn = gr.Button("Dự đoán")

    run_btn.click(
        run_gps,
        [n_iter, top_k, strategy],
        [result_table, conv_plot, best_prompt_box],
    )
    test_btn.click(test_prompt, [best_prompt_box, test_input], [test_output])

if __name__ == "__main__":
    demo.launch(share=True)

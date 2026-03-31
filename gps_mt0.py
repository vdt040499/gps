#!/usr/bin/env python3
"""
Demo GPS (Genetic Prompt Search) — Hướng B: Model cục bộ

Điểm khác biệt so với hướng A (demo_gps_vietnamese.py):
  - Scoring : dùng bigscience/mt0-base (local, ~580MB) — KHÔNG cần API
  - Cách chấm: log-probability của từng answer choice (giống GPS gốc dùng T0)
  - Mutation : predefined templates (offline) hoặc OpenAI nếu có key

Cách chạy:
    pip install transformers torch
    python demo_gps_local_b.py
"""

import os
import time
import copy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# ─── MÀUSẮC TERMINAL ───────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def h(t): return f"{BOLD}{t}{RESET}"
def g(t): return f"{GREEN}{t}{RESET}"
def y(t): return f"{YELLOW}{t}{RESET}"
def c(t): return f"{CYAN}{t}{RESET}"
def d(t): return f"{DIM}{t}{RESET}"

# ─── MODEL ─────────────────────────────────────────────────────
MODEL_NAME = "bigscience/mt0-base"  # ~580MB, T0 đa ngữ (46 ngôn ngữ gồm tiếng Việt)

# ─── DỮ LIỆU TIẾNG VIỆT ────────────────────────────────────────
DATASET = [
    # Tích cực (label=0)
    {"text": "Sản phẩm chất lượng cao, đúng như mô tả, rất hài lòng.",          "label": 0},
    {"text": "Giao hàng nhanh, đóng gói cẩn thận, sản phẩm đẹp hơn mong đợi.", "label": 0},
    {"text": "Dùng được 1 tuần, hoạt động tốt, sẽ mua lại lần sau.",            "label": 0},
    {"text": "Giá cả hợp lý, chất lượng ổn, nhân viên tư vấn nhiệt tình.",     "label": 0},
    {"text": "Hàng y như hình, màu sắc đẹp, size vừa vặn, rất thích.",          "label": 0},
    {"text": "Mua lần thứ 2 rồi, vẫn tốat như lần đầu, shop uy tín.",           "label": 0},
    {"text": "Giao hàng đúng hẹn, sản phẩm ok, sẽ giới thiệu cho bạn bè.",     "label": 0},
    {"text": "Chất liệu tốt, bền bỉ, xứng đáng với đồng tiền bỏ ra.",          "label": 0},
    {"text": "Thiết kế tinh tế, rất ưng ý, mua về là thích ngay.",              "label": 0},
    {"text": "Mua tặng ba, ba thích lắm, cảm ơn shop nhiều!",                   "label": 0},
    # Tiêu cực (label=1)
    {"text": "Hàng kém chất lượng, không như quảng cáo, rất thất vọng.",       "label": 1},
    {"text": "Giao hàng trễ hơn 1 tuần, shop không phản hồi tin nhắn.",        "label": 1},
    {"text": "Sản phẩm bị lỗi ngay từ đầu, liên hệ hoàn tiền không được.",     "label": 1},
    {"text": "Chất liệu rẻ tiền, màu sắc khác xa hình, không mua lần 2.",      "label": 1},
    {"text": "Đóng gói cẩu thả, hàng bị móp méo khi nhận, bực bội.",           "label": 1},
    {"text": "Mua về dùng được 3 ngày là hỏng ngay, chất lượng quá tệ.",       "label": 1},
    {"text": "Size sai hoàn toàn, đổi hàng mất gần 2 tuần, không chuyên nghiệp.", "label": 1},
    {"text": "Hàng không đúng mô tả, nghi là hàng giả, tránh xa shop này.",    "label": 1},
    {"text": "Dịch vụ khách hàng kém, thái độ nhân viên hỗn láo.",              "label": 1},
    {"text": "Sản phẩm không đáng giá tiền, mua về chỉ thấy hối hận.",         "label": 1},
]

# ─── PROMPT BAN ĐẦU (Step 0) ───────────────────────────────────
# answer_choices[0] = tích cực (label 0), answer_choices[1] = tiêu cực (label 1)
INITIAL_PROMPTS = [
    {
        "id": "p1",
        "name": "hoi_truc_tiep",
        "template": (
            "Đánh giá sản phẩm dưới đây là tích cực hay tiêu cực?\n\n"
            "{{text}}"
        ),
        "answer_choices": ["tích cực", "tiêu cực"],
    },
    {
        "id": "p2",
        "name": "phan_tich_camxuc",
        "template": (
            "Hãy phân tích cảm xúc của đoạn nhận xét sản phẩm sau:\n\n"
            "{{text}}\n\n"
            "Cảm xúc là"
        ),
        "answer_choices": ["tích cực", "tiêu cực"],
    },
    {
        "id": "p3",
        "name": "khach_hang_hai_long",
        "template": (
            "Dựa vào nhận xét này, khách hàng cảm thấy thế nào về sản phẩm?\n\n"
            "{{text}}"
        ),
        "answer_choices": ["tích cực", "tiêu cực"],
    },
]

# ─── MUTATION PREDEFINED (offline) ─────────────────────────────
# Mô phỏng những gì một mô hình paraphrase sẽ sinh ra.
# Mỗi prompt gốc có sẵn 3 biến thể khác nhau về cách hỏi.
PREDEFINED_MUTATIONS: dict[str, list[dict]] = {
    "p1": [
        {
            "id": "p1_m0", "name": "hoi_truc_tiep_mut1",
            "template": "Cảm xúc trong đánh giá dưới đây là tích cực hay tiêu cực?\n\n{{text}}",
            "answer_choices": ["tích cực", "tiêu cực"],
        },
        {
            "id": "p1_m1", "name": "hoi_truc_tiep_mut2",
            "template": "Nhận xét này thể hiện sự hài lòng hay không hài lòng của người mua?\n\n{{text}}",
            "answer_choices": ["tích cực", "tiêu cực"],
        },
        {
            "id": "p1_m2", "name": "hoi_truc_tiep_mut3",
            "template": "Đọc đánh giá sau và phân loại thái độ của khách hàng:\n\n{{text}}",
            "answer_choices": ["tích cực", "tiêu cực"],
        },
    ],
    "p2": [
        {
            "id": "p2_m0", "name": "phan_tich_camxuc_mut1",
            "template": "Xác định thái độ của người viết trong đánh giá sản phẩm sau:\n\n{{text}}\n\nThái độ là",
            "answer_choices": ["tích cực", "tiêu cực"],
        },
        {
            "id": "p2_m1", "name": "phan_tich_camxuc_mut2",
            "template": "Sau khi mua sản phẩm, người viết nhận xét này cảm thấy:\n\n{{text}}\n\nCảm giác",
            "answer_choices": ["tích cực", "tiêu cực"],
        },
        {
            "id": "p2_m2", "name": "phan_tich_camxuc_mut3",
            "template": "Mức độ hài lòng trong đánh giá dưới đây là gì?\n\n{{text}}\n\nMức độ hài lòng:",
            "answer_choices": ["tích cực", "tiêu cực"],
        },
    ],
    "p3": [
        {
            "id": "p3_m0", "name": "khach_hang_hai_long_mut1",
            "template": "Sau khi trải nghiệm sản phẩm, khách hàng này có hài lòng không?\n\n{{text}}",
            "answer_choices": ["tích cực", "tiêu cực"],
        },
        {
            "id": "p3_m1", "name": "khach_hang_hai_long_mut2",
            "template": "Đánh giá này phản ánh trải nghiệm mua sắm tốt hay xấu?\n\n{{text}}",
            "answer_choices": ["tích cực", "tiêu cực"],
        },
        {
            "id": "p3_m2", "name": "khach_hang_hai_long_mut3",
            "template": "Nhìn vào nhận xét, người mua có muốn mua lại sản phẩm không?\n\n{{text}}",
            "answer_choices": ["tích cực", "tiêu cực"],
        },
    ],
}


# ─── TẢI MODEL ─────────────────────────────────────────────────
def load_model(model_name: str):
    print(f"\n  Đang tải model {h(model_name)} ...")
    print(d("  (Lần đầu tải ~580MB về cache, các lần sau dùng ngay)"))
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    elapsed = time.time() - t0
    print(g(f"  ✓ Model sẵn sàng [{device}] — {elapsed:.1f}s"))
    return model, tokenizer, device


# ─── SCORING: LOG-PROBABILITY ───────────────────────────────────
def compute_log_prob(model, tokenizer, input_text: str, target_text: str, device) -> float:
    """
    Tính log P(target_text | input_text).

    Đây chính xác là cách GPS gốc chấm điểm với T0:
      Input  → prompt đã điền dữ liệu ("Đánh giá này là... [text]")
      Target → answer choice           ("tích cực" hoặc "tiêu cực")
      Score  → log P(target | input)  →  chọn answer có score cao nhất
    """
    enc = tokenizer(
        input_text, return_tensors="pt",
        max_length=512, truncation=True
    ).to(device)
    dec = tokenizer(
        target_text, return_tensors="pt",
        max_length=32, truncation=True
    ).input_ids.to(device)

    with torch.no_grad():
        loss = model(**enc, labels=dec).loss   # mean NLL per token
    # Nhân length để ra tổng log-prob (so sánh công bằng giữa các choices)
    return -loss.item() * dec.shape[1]


def score_prompt(prompt_obj: dict, dataset: list, model, tokenizer, device) -> float:
    """Chấm điểm một prompt trên toàn bộ dataset, trả về accuracy."""
    answer_choices = prompt_obj["answer_choices"]
    correct = 0
    for example in dataset:
        filled = prompt_obj["template"].replace("{{text}}", example["text"])
        log_probs = [
            compute_log_prob(model, tokenizer, filled, choice, device)
            for choice in answer_choices
        ]
        predicted = log_probs.index(max(log_probs))
        if predicted == example["label"]:
            correct += 1
    return correct / len(dataset)


# ─── MUTATION ──────────────────────────────────────────────────
def get_mutations(prompt_obj: dict, openai_client=None) -> list:
    """
    Lấy danh sách biến thể của prompt.
    Ưu tiên: OpenAI API (nếu có) → predefined templates (offline).
    """
    # Tìm predefined dựa trên id gốc (bỏ phần _m... của các prompt con)
    base_id = prompt_obj["id"].split("_m")[0]
    predefined = PREDEFINED_MUTATIONS.get(base_id, [])

    if openai_client is not None:
        try:
            from demo_gps_vietnamese import mutate_prompt as api_mutate
            variations = api_mutate(prompt_obj, openai_client, n=3)
            if variations:
                for v in variations:
                    v["answer_choices"] = prompt_obj["answer_choices"]
                return variations
        except Exception:
            pass  # fallback to predefined

    return predefined


# ─── GA LOOP CHÍNH ─────────────────────────────────────────────
def ga_demo(max_steps: int = 3, top_k: int = 2):
    model, tokenizer, device = load_model(MODEL_NAME)

    # Kiểm tra OpenAI (tùy chọn)
    openai_client = None
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            openai_client = OpenAI()
            print(g("  ✓ Tìm thấy OPENAI_API_KEY — mutation sẽ dùng GPT-4o-mini"))
        except ImportError:
            pass
    else:
        print(y("  ℹ Không có OPENAI_API_KEY — mutation dùng predefined templates"))

    print(f"\n{'═'*62}")
    print(h("  GPS Demo B — Scoring bằng model cục bộ (mt0-base)"))
    print(f"{'═'*62}")
    print(f"  Scoring : {c('log P(answer | prompt)')} với {h('mt0-base')}  [{device}]")
    print(f"  Mutation: {'OpenAI GPT-4o-mini' if openai_client else 'Predefined templates (offline)'}")
    print(f"  Dataset : {len(DATASET)} ví dụ")
    print(f"  GA      : {max_steps} bước | giữ top-{top_k}")
    print(f"{'═'*62}")
    print()
    print(d("  Tại sao dùng log-probability thay vì generate?"))
    print(d("  Demo A: GPT đọc prompt → sinh ra chữ → parse → đếm đúng/sai"))
    print(d("  Demo B: mt0 tính log P('tích cực'|prompt) so với log P('tiêu cực'|prompt)"))
    print(d("          → chọn answer nào có xác suất cao hơn (nhanh hơn, chính xác hơn)"))
    print(d("          → đây là cách GPS gốc làm với T0 (xem run_all_eval.py:515-525)"))

    all_scored: dict[str, dict] = {}
    current_prompts = copy.deepcopy(INITIAL_PROMPTS)
    step0_best = 0.0

    for step in range(max_steps):
        print(f"\n{y(h(f'  BƯỚC {step}'))}  ({'prompts ban đầu' if step == 0 else 'sau mutation'})")
        print(f"  {'─'*58}")
        print(f"  {len(current_prompts)} prompt × {len(DATASET)} ví dụ\n")

        step_scores: dict[str, float] = {}

        for p in current_prompts:
            # Dùng cache nếu đã chấm rồi
            if p["id"] in all_scored:
                score = all_scored[p["id"]]["score"]
                print(d(f"  ↩ [{p['name']}] (cache) {score:.1%}"))
                step_scores[p["id"]] = score
                continue

            print(f"  ► [{c(p['name'])}]")

            # Minh họa log-prob với ví dụ đầu tiên
            ex0 = DATASET[0]
            filled0 = p["template"].replace("{{text}}", ex0["text"])
            lp_pos = compute_log_prob(model, tokenizer, filled0, "tích cực", device)
            lp_neg = compute_log_prob(model, tokenizer, filled0, "tiêu cực", device)
            pred_ok = (lp_pos > lp_neg) == (ex0["label"] == 0)
            result_str = g("✓ đúng") if pred_ok else f"\033[91m✗ sai\033[0m"
            print(d(f"    Ví dụ 1: \"{ex0['text'][:48]}\""))
            print(d(f"    log P('tích cực') = {lp_pos:6.3f}  |  log P('tiêu cực') = {lp_neg:6.3f}  → {result_str}"))

            # Chấm toàn dataset
            t0 = time.time()
            score = score_prompt(p, DATASET, model, tokenizer, device)
            elapsed = time.time() - t0

            color = g if score >= 0.75 else (y if score >= 0.6 else "\033[91m")
            print(f"    Accuracy: {color}{h(f'{score:.1%}')}{RESET}  {d(f'({elapsed:.1f}s)')}\n")

            all_scored[p["id"]] = {"prompt": p, "score": score, "step": step}
            step_scores[p["id"]] = score

        if step == 0:
            step0_best = max(step_scores.values())

        # ── Selection ────────────────────────────────────────
        sorted_p = sorted(current_prompts, key=lambda p: step_scores[p["id"]], reverse=True)
        selected = sorted_p[:top_k]

        print(f"\n  {h('▶ Selection')} — top-{top_k}:")
        for rank, p in enumerate(selected, 1):
            sc = step_scores[p["id"]]
            bar = "█" * int(sc * 20) + "░" * (20 - int(sc * 20))
            print(f"  {rank}. [{c(p['name'])}]  {g(bar)}  {g(h(f'{sc:.1%}'))}")
            short = p["template"].split("{{text}}")[0].strip().replace("\n", " ")
            print(f"     {d(short[:68])}")

        # ── Mutation ─────────────────────────────────────────
        if step < max_steps - 1:
            print(f"\n  {h('▶ Mutation')}:")
            new_prompts = copy.deepcopy(selected)
            for p in selected:
                mutations = get_mutations(p, openai_client)
                print(f"\n  ✦ Biến thể của [{c(p['name'])}]:")
                for v in mutations:
                    short_v = v["template"].split("{{text}}")[0].strip().replace("\n", " ")
                    print(f"    [{y(v['name'])}]  {d(short_v[:62])}")
                    new_prompts.append(v)
            current_prompts = new_prompts
        else:
            current_prompts = selected

    # ── Kết quả ──────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(h("  KẾT QUẢ"))
    print(f"{'═'*62}")

    all_ranked = sorted(all_scored.values(), key=lambda x: x["score"], reverse=True)
    print(f"\n  {'Rank':<5} {'Tên prompt':<34} {'Bước':<6} {'Accuracy'}")
    print(f"  {'─'*57}")
    for i, obj in enumerate(all_ranked[:8], 1):
        sc = obj["score"]
        clr = g if sc >= 0.75 else (y if sc >= 0.6 else "\033[91m")
        mark = g("  ◄ BEST") if i == 1 else ""
        print(f"  {i:<5} {obj['prompt']['name']:<34} step {obj['step']:<2}  {clr}{sc:.1%}{RESET}{mark}")

    best = all_ranked[0]
    delta = best["score"] - step0_best
    sign = "+" if delta >= 0 else ""
    clr = g if delta > 0 else (y if delta == 0 else "\033[91m")

    print(f"\n  Baseline (step 0, tốt nhất): {y(f'{step0_best:.1%}')}")
    best_score_str = f"{best['score']:.1%}"
    print(f"  Sau GPS  (tốt nhất):          {g(h(best_score_str))}")
    print(f"  Cải thiện:                    {clr}{h(f'{sign}{delta:.1%}')}{RESET}")

    print(f"\n  {h('Prompt tốt nhất:')}")
    print(f"  {'─'*57}")
    for line in best["prompt"]["template"].split("\n"):
        display = line.replace("{{text}}", d("[nội dung đánh giá]"))
        print(f"  {c(display)}")
    print(f"\n{'═'*62}\n")


if __name__ == "__main__":
    ga_demo(max_steps=3, top_k=2)

#!/usr/bin/env python3
"""
Demo GPS (Genetic Prompt Search) bằng tiếng Việt
Task: Phân loại cảm xúc đánh giá sản phẩm (Tích cực / Tiêu cực)

Cách chạy:
    export OPENAI_API_KEY="sk-..."
    python demo_gps_vietnamese.py
"""

import os
import time
import copy
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── MÀUSẮC TERMINAL ───────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def h(text): return f"{BOLD}{text}{RESET}"
def g(text): return f"{GREEN}{text}{RESET}"
def y(text): return f"{YELLOW}{text}{RESET}"
def c(text): return f"{CYAN}{text}{RESET}"
def r(text): return f"{RED}{text}{RESET}"
def d(text): return f"{DIM}{text}{RESET}"

# ─── DỮ LIỆU TIẾNG VIỆT ────────────────────────────────────────
# 20 đánh giá sản phẩm: 10 tích cực, 10 tiêu cực
DATASET = [
    # Tích cực (label=0)
    {"text": "Sản phẩm chất lượng cao, đúng như mô tả, rất hài lòng.",          "label": 0},
    {"text": "Giao hàng nhanh, đóng gói cẩn thận, sản phẩm đẹp hơn mong đợi.", "label": 0},
    {"text": "Dùng được 1 tuần, hoạt động tốt, sẽ mua lại lần sau.",            "label": 0},
    {"text": "Giá cả hợp lý, chất lượng ổn, nhân viên tư vấn nhiệt tình.",     "label": 0},
    {"text": "Hàng y như hình, màu sắc đẹp, size vừa vặn, rất thích.",          "label": 0},
    {"text": "Mua lần thứ 2 rồi, vẫn tốt như lần đầu, shop uy tín.",           "label": 0},
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
# 3 prompt có chất lượng khác nhau để thấy rõ sự cải thiện
INITIAL_PROMPTS = [
    {
        "id": "p1",
        "name": "hoi_truc_tiep",
        "template": (
            "Đánh giá sản phẩm dưới đây là tích cực hay tiêu cực?\n\n"
            "{{text}}\n\n"
            "Chỉ trả lời đúng một từ: 'tích cực' hoặc 'tiêu cực'."
        ),
    },
    {
        "id": "p2",
        "name": "phan_tich_camxuc",
        "template": (
            "Hãy phân tích cảm xúc của đoạn nhận xét sản phẩm sau:\n\n"
            "{{text}}\n\n"
            "Cảm xúc là tích cực hay tiêu cực? Trả lời ngắn gọn."
        ),
    },
    {
        "id": "p3",
        "name": "khach_hang_hai_long",
        "template": (
            "Đọc nhận xét này và cho biết khách hàng có hài lòng không?\n\n"
            "{{text}}\n\n"
            "Trả lời: 'tích cực' nếu hài lòng, 'tiêu cực' nếu không hài lòng."
        ),
    },
]

# ─── SCORING ───────────────────────────────────────────────────
def score_prompt(prompt_obj: dict, dataset: list, client: OpenAI) -> float:
    """Chấm điểm một prompt trên toàn bộ dataset, trả về accuracy."""
    correct = 0
    for example in dataset:
        filled = prompt_obj["template"].replace("{{text}}", example["text"])
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=20,
                temperature=0,
                messages=[{"role": "user", "content": filled}],
            )
            answer = resp.choices[0].message.content.lower().strip()

            # Parse: tìm từ khoá trong câu trả lời
            has_pos = "tích cực" in answer or "hài lòng" in answer or "positive" in answer
            has_neg = "tiêu cực" in answer or "không hài lòng" in answer or "negative" in answer

            if has_pos and not has_neg:
                predicted = 0
            elif has_neg:
                predicted = 1
            else:
                predicted = -1  # không rõ

            if predicted == example["label"]:
                correct += 1

        except Exception as e:
            print(r(f"    ⚠ API error: {e}"))

        time.sleep(0.05)  # tránh rate limit

    return correct / len(dataset)


# ─── MUTATION ──────────────────────────────────────────────────
def mutate_prompt(prompt_obj: dict, client: OpenAI, n: int = 3) -> list:
    """Dùng GPT sinh ra n biến thể mới từ một prompt."""
    template = prompt_obj["template"]

    # Tách phần instruction (trước {{text}}) và phần suffix (sau {{text}})
    parts = template.split("{{text}}")
    instruction = parts[0].strip()
    suffix = parts[1].strip() if len(parts) > 1 else ""

    mutation_request = (
        f"Bạn đang giúp cải thiện câu lệnh (prompt) cho bài toán phân loại cảm xúc "
        f"đánh giá sản phẩm tiếng Việt (tích cực / tiêu cực).\n\n"
        f"Câu lệnh gốc:\n\"{instruction}\"\n\n"
        f"Hãy viết {n} cách diễn đạt KHÁC về ý nghĩa tương tự. "
        f"Mỗi biến thể phải:\n"
        f"- Yêu cầu trả lời 'tích cực' hoặc 'tiêu cực'\n"
        f"- Ngắn gọn, rõ ràng\n"
        f"- Khác hẳn nhau về từ ngữ\n\n"
        f"Liệt kê theo số thứ tự (1. 2. 3.):"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            temperature=0.9,
            messages=[{"role": "user", "content": mutation_request}],
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        print(r(f"    ⚠ Mutation API error: {e}"))
        return []

    # Parse danh sách có số thứ tự
    new_prompts = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and ". " in line:
            new_instruction = line.split(". ", 1)[1].strip()
            # Ghép lại thành template hoàn chỉnh
            if suffix:
                new_template = f"{new_instruction}\n\n{{{{text}}}}\n\n{suffix}"
            else:
                new_template = f"{new_instruction}\n\n{{{{text}}}}"
            new_prompts.append({
                "id": f"{prompt_obj['id']}_m{len(new_prompts)}",
                "name": f"{prompt_obj['name']}_mut{len(new_prompts)+1}",
                "template": new_template,
                "parent": prompt_obj["id"],
            })
    return new_prompts[:n]


# ─── GA LOOP CHÍNH ─────────────────────────────────────────────
def ga_demo(max_steps: int = 3, top_k: int = 2, mutations_per_prompt: int = 3):
    client = OpenAI()  # lấy OPENAI_API_KEY từ biến môi trường

    print(f"\n{'═'*62}")
    print(h(" GPS Demo — Genetic Prompt Search (Tiếng Việt)"))
    print(f"{'═'*62}")
    print(f" Task    : Phân loại cảm xúc đánh giá sản phẩm")
    print(f" Dataset : {len(DATASET)} ví dụ ({sum(1 for d in DATASET if d['label']==0)} tích cực, "
          f"{sum(1 for d in DATASET if d['label']==1)} tiêu cực)")
    print(f" GA      : {max_steps} bước | giữ top-{top_k} | {mutations_per_prompt} mutation/prompt")
    print(f"{'═'*62}")

    # Lưu toàn bộ lịch sử
    all_scored: dict[str, dict] = {}   # id -> {prompt_obj, score, step}
    current_prompts = copy.deepcopy(INITIAL_PROMPTS)
    step0_best = 0.0

    for step in range(max_steps):
        print(f"\n{y(h(f'  BƯỚC {step}'))}  ({'đánh giá prompts ban đầu' if step == 0 else 'đánh giá prompts sau mutation'})")
        print(f"  {'─'*58}")
        print(f"  Có {len(current_prompts)} prompt cần đánh giá trên {len(DATASET)} ví dụ...\n")

        step_scores: dict[str, float] = {}

        for p in current_prompts:
            if p["id"] in all_scored:
                # đã chấm rồi, dùng lại kết quả
                score = all_scored[p["id"]]["score"]
                print(d(f"  ↩ [{p['name']}] (đã có) accuracy = {score:.1%}"))
            else:
                print(f"  ► Chấm điểm [{c(p['name'])}]")
                # Hiển thị template để người đọc thấy
                preview = p["template"].replace("{{text}}", d("[nội dung đánh giá]"))
                for line in preview.split("\n"):
                    print(f"    {d(line)}")

                score = score_prompt(p, DATASET, client)

                color = g if score >= 0.75 else (y if score >= 0.6 else r)
                print(f"    → Accuracy: {color(h(f'{score:.1%}'))}\n")

                all_scored[p["id"]] = {"prompt": p, "score": score, "step": step}

            step_scores[p["id"]] = all_scored[p["id"]]["score"]

        if step == 0:
            step0_best = max(step_scores.values())

        # ── Selection: giữ top-K ────────────────────────────
        sorted_prompts = sorted(current_prompts, key=lambda p: step_scores[p["id"]], reverse=True)
        selected = sorted_prompts[:top_k]

        print(f"\n  {h('▶ Selection')} — chọn top-{top_k} prompt:")
        for rank, p in enumerate(selected, 1):
            sc = step_scores[p["id"]]
            bar = "█" * int(sc * 20) + "░" * (20 - int(sc * 20))
            print(f"  {rank}. [{c(p['name'])}]  {g(bar)}  {g(h(f'{sc:.1%}'))}")
            short = p["template"].split("{{text}}")[0].strip().replace("\n", " ")
            print(f"     {d(short[:70] + '...' if len(short) > 70 else short)}")

        # ── Mutation ─────────────────────────────────────────
        if step < max_steps - 1:
            print(f"\n  {h('▶ Mutation')} — sinh {mutations_per_prompt} biến thể từ mỗi prompt được chọn:")
            new_prompts = copy.deepcopy(selected)  # giữ lại prompt cha

            for p in selected:
                print(f"\n  ✦ Đang biến đổi [{c(p['name'])}]...")
                variations = mutate_prompt(p, client, mutations_per_prompt)

                for v in variations:
                    short_v = v["template"].split("{{text}}")[0].strip().replace("\n", " ")
                    print(f"    → [{y(v['name'])}]: {d(short_v[:65] + '...' if len(short_v) > 65 else short_v)}")
                    new_prompts.append(v)

            current_prompts = new_prompts
        else:
            current_prompts = selected  # bước cuối chỉ giữ top-K

    # ── Kết quả tổng kết ─────────────────────────────────────
    print(f"\n{'═'*62}")
    print(h("  KẾT QUẢ"))
    print(f"{'═'*62}")

    # Sắp xếp tất cả prompt theo điểm
    all_ranked = sorted(all_scored.values(), key=lambda x: x["score"], reverse=True)

    print(f"\n  {'Rank':<5} {'Tên prompt':<30} {'Bước':<6} {'Accuracy'}")
    print(f"  {'─'*55}")
    for i, obj in enumerate(all_ranked[:8], 1):
        sc = obj["score"]
        step_label = f"step {obj['step']}"
        color = g if sc >= 0.75 else (y if sc >= 0.6 else r)
        marker = " ◄ BEST" if i == 1 else ""
        print(f"  {i:<5} {obj['prompt']['name']:<30} {step_label:<6}  {color(f'{sc:.1%}')}{g(marker)}")

    best_score = all_ranked[0]["score"]
    best_prompt = all_ranked[0]["prompt"]

    print(f"\n  Baseline (bước 0, tốt nhất): {y(f'{step0_best:.1%}')}")
    print(f"  Sau GPS  (tốt nhất):          {g(h(f'{best_score:.1%}'))}")
    delta = best_score - step0_best
    sign = "+" if delta >= 0 else ""
    color = g if delta > 0 else (y if delta == 0 else r)
    print(f"  Cải thiện:                    {color(h(f'{sign}{delta:.1%}'))}")

    print(f"\n  {h('Prompt tốt nhất sau GPS:')}")
    print(f"  {'─'*55}")
    for line in best_prompt["template"].split("\n"):
        print(f"  {c(line)}")

    print(f"\n{'═'*62}\n")


# ─── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print(r("\n⚠  Chưa set OPENAI_API_KEY!"))
        print("   Chạy lệnh:  export OPENAI_API_KEY='sk-...'  rồi thử lại.\n")
        exit(1)

    ga_demo(
        max_steps=3,           # 3 bước GA
        top_k=2,               # giữ top 2 prompt mỗi bước
        mutations_per_prompt=3 # sinh 3 biến thể từ mỗi prompt được chọn
    )

import json
from typing import List, Dict
import os

from openai import OpenAI

# =========================
# CONFIG
# =========================

# Model for scoring
MODEL = "gpt-5-mini"   # reasoning model; see OpenAI docs

# Input JSON with [{"target": "...", "pred": "..."}]
INPUT_JSON = "/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/gpt_files/merge_result_3b_caption_5s.json"

# Output with per-example scores
OUTPUT_SCORED_JSON = "scored_output_3b_caption_gpt_5s_merge.json"

# Use environment variable OPENAI_API_KEY (recommended)
# export OPENAI_API_KEY="your_key_here"  (bash)
client = OpenAI()


# =========================
# LOAD DATA
# =========================

def load_examples_from_json(path: str) -> List[Dict[str, str]]:
    """
    Expects a JSON file structured as:
    [
      {"target": "text...", "pred": "text..."},
      ...
    ]
    """
    with open(path, "r") as f:
        data = json.load(f)

    for i, ex in enumerate(data):
        if "target" not in ex or "pred" not in ex:
            raise ValueError(f"Item {i} missing 'target' or 'pred' keys: {ex}")

    return data


# =========================
# GPT SCORING
# =========================

SCORING_INSTRUCTIONS = """
You are an expert rater for infant audio captions.

Given a TARGET caption and a PRED caption, you must output:

- accuracy: 0–10, how factually correct the PRED is relative to TARGET
- quality: 0–10, how clear, grammatical, and usable the PRED is
- completeness: 0–10, how many of the key facts in TARGET are covered by PRED

Rules:
- Ignore special tokens such as <s>, </s>, <unk>.
- Only compare semantics, not formatting.
- Ignore "when" the predicted vocalization is happening in calculating score.
- Return ONLY a JSON object on a single line, no explanations, no extra keys:
  {"accuracy": int, "quality": int, "completeness": int}
"""
def make_prompt(target: str, pred: str) -> str:
    return f"""{SCORING_INSTRUCTIONS}

TARGET: {target}
PRED: {pred}
"""
def score_batched(prompts: List[str], batch_size: int = 50) -> List[Dict[str, int]]:
    all_scores = []

    for start in range(0, 200, batch_size):
        end = start + batch_size
        batch = prompts[start:end]

        print(f"Sending batch {start//batch_size + 1} "
              f"({len(batch)} items)...")

        resp = client.responses.create(
            model=MODEL,
            input=batch
        )

        # Each response.outputs[i] corresponds to input[i]
        for out in resp.outputs:
            text = out.output_text.strip()
            scores = json.loads(text)
            all_scores.append({
                "accuracy": int(scores["accuracy"]),
                "quality": int(scores["quality"]),
                "completeness": int(scores["completeness"]),
            })

    return all_scores


# =========================
# AGGREGATION
# =========================

def compute_weighted_overall(scores_list, weights):
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Sum of weights is zero.")

    acc = sum(s["accuracy"] * w for s, w in zip(scores_list, weights)) / total_weight
    qual = sum(s["quality"] * w for s, w in zip(scores_list, weights)) / total_weight
    comp = sum(s["completeness"] * w for s, w in zip(scores_list, weights)) / total_weight

    return {"accuracy": acc, "quality": qual, "completeness": comp}


# =========================
# MAIN
# =========================

def main():
    # Load examples
    examples = load_examples_from_json(INPUT_JSON)
    print(f"Loaded {len(examples)} examples")

    # Build all prompts
    prompts = [make_prompt(ex["target"], ex["pred"]) for ex in examples]

    # Run batched inference
    all_scores = score_batched(prompts, batch_size=50)
    assert len(all_scores) == len(examples)

    # Uniform weights
    weights = [1.0] * len(all_scores)

    # Compute weighted overall score
    overall = compute_weighted_overall(all_scores, weights)
    print("\nWeighted overall scores:")
    print(json.dumps(overall, indent=2))

    # Save scored results
    scored_output = []
    for ex, sc in zip(examples, all_scores):
        scored_output.append({
            "target": ex["target"],
            "pred": ex["pred"],
            "scores": sc
        })

    with open(OUTPUT_SCORED_JSON, "w") as f:
        json.dump(scored_output, f, indent=2)

    print(f"\nScored results written to {OUTPUT_SCORED_JSON}")


if __name__ == "__main__":
    main()
# def score_pair_with_gpt(target: str, pred: str) -> Dict[str, int]:
#     """Call GPT-5-mini to get (accuracy, quality, completeness) for one pair."""
#     prompt = f"""{SCORING_INSTRUCTIONS}

# TARGET: {target}
# PRED: {pred}
# """

#     resp = client.responses.create(
#         model=MODEL,
#         input=prompt
#     )

#     # Text is in resp.output[0].content[0].text for Responses API
#     text = resp.output_text.strip()

#     scores = json.loads(text)
#     return {
#         "accuracy": int(scores["accuracy"]),
#         "quality": int(scores["quality"]),
#         "completeness": int(scores["completeness"]),
#     }


# # =========================
# # AGGREGATION
# # =========================

# def compute_weighted_overall(scores_list: List[Dict[str, int]],
#                              weights: List[float]) -> Dict[str, float]:
#     """
#     scores_list: list of {"accuracy": int, "quality": int, "completeness": int}
#     weights:     list of floats, same length as scores_list

#     Returns weighted average for each category.
#     """
#     assert len(scores_list) == len(weights), "scores_list and weights must match length."

#     total_weight = sum(weights)
#     if total_weight == 0:
#         raise ValueError("Sum of weights is zero.")

#     accum = {"accuracy": 0.0, "quality": 0.0, "completeness": 0.0}

#     for s, w in zip(scores_list, weights):
#         accum["accuracy"]     += w * s["accuracy"]
#         accum["quality"]      += w * s["quality"]
#         accum["completeness"] += w * s["completeness"]

#     return {
#         "accuracy":     accum["accuracy"] / total_weight,
#         "quality":      accum["quality"] / total_weight,
#         "completeness": accum["completeness"] / total_weight,
#     }


# # =========================
# # MAIN PIPELINE
# # =========================

# def main():
#     # 1) Load examples from JSON
#     examples = load_examples_from_json(INPUT_JSON)
#     print(f"Loaded {len(examples)} examples from {INPUT_JSON}")

#     # 2) Uniform weights (modify if you want duration-based weights, etc.)
#     weights = [1.0 for _ in examples]

#     all_scores: List[Dict[str, int]] = []

#     # 3) Score each pair with GPT-5-mini
#     for i, ex in enumerate(examples):
#         target = ex["target"]
#         pred = ex["pred"]

#         print(f"Scoring example {i+1}/{len(examples)}...")
#         scores = score_pair_with_gpt(target, pred)
#         all_scores.append(scores)
#         print(f"  -> {scores}")

#     # 4) Compute weighted overall scores
#     overall = compute_weighted_overall(all_scores, weights)
#     print("\nWeighted overall scores:")
#     print(json.dumps(overall, indent=2))

#     # 5) Save per-example scores
#     scored = []
#     for ex, s in zip(examples, all_scores):
#         item = {
#             "target": ex["target"],
#             "scores": s,
#         }
#         scored.append(item)

#     with open(OUTPUT_SCORED_JSON, "w") as f:
#         json.dump(scored, f, indent=2)

#     print(f"\nPer-example scores written to {OUTPUT_SCORED_JSON}")


# if __name__ == "__main__":
#     main()

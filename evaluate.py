import os
import json
from rapidfuzz.distance import Levenshtein

from utils import build_train_datasets, normalize_latex
from inference import load_base_model, load_lora_model, generate_latex


def exact_match(y_true, y_pred):
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)


def normalized_edit_distance(y_true, y_pred):
    scores = []
    for a, b in zip(y_true, y_pred):
        a = normalize_latex(a)
        b = normalize_latex(b)
        denom = max(len(a), len(b), 1)
        dist = Levenshtein.distance(a, b)
        scores.append(dist / denom)
    return sum(scores) / max(len(scores), 1)


def evaluate_setup(processor, model, test_dataset, one_shot_example=None):
    refs = []
    preds = []
    rows = []

    for idx, ex in enumerate(test_dataset):
        image = ex["image"]
        ref = normalize_latex(ex["label"])

        try:
            pred = generate_latex(
                processor,
                model,
                image,
                one_shot_example=one_shot_example,
            )
            pred = normalize_latex(pred)
        except Exception as e:
            pred = f"[ERROR] {type(e).__name__}: {e}"

        refs.append(ref)
        preds.append(pred)

        rows.append(
            {
                "index": idx,
                "reference": ref,
                "prediction": pred,
            }
        )

        print(f"Processed {idx + 1}/{len(test_dataset)}")

    safe_preds = [p if not p.startswith("[ERROR]") else "" for p in preds]

    return {
        "status": "completed",
        "exact_match": exact_match(refs, safe_preds),
        "normalized_edit_distance": normalized_edit_distance(refs, safe_preds),
        "predictions": preds,
        "references": refs,
        "rows": rows,
    }


def build_missing_result(reason):
    return {
        "status": "not_run",
        "reason": reason,
        "exact_match": None,
        "normalized_edit_distance": None,
        "predictions": [],
        "references": [],
        "rows": [],
    }


def print_results(name, result):
    if result["status"] != "completed":
        print(f"{name}: NOT RUN ({result.get('reason', 'no reason provided')})")
        return

    print(
        f"{name}: "
        f"exact_match={result['exact_match']:.4f}, "
        f"normalized_edit_distance={result['normalized_edit_distance']:.4f}"
    )


def save_results_json(path, results_dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)


def save_summary_md(path, results_dict):
    lines = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append("Evaluation dataset: `linxy/LaTeX_OCR:test[:70]`")
    lines.append("")
    lines.append("Main metric: **Normalized Edit Distance (lower is better)**")
    lines.append("Additional metric: **Exact Match (higher is better)**")
    lines.append("")
    lines.append("| Setup | Status | Exact Match | Normalized Edit Distance | Notes |")
    lines.append("|---|---|---:|---:|---|")

    ordered_keys = [
        ("zero_shot", "Zero-shot inference"),
        ("one_shot", "One-shot inference"),
        ("sft_latex_ocr", "SFT using linxy/LaTeX_OCR:train"),
        ("sft_combined", "SFT using linxy/LaTeX_OCR:train + deepcopy/MathWriting-human"),
    ]

    for key, title in ordered_keys:
        result = results_dict.get(key)
        if result is None:
            lines.append(f"| {title} | NOT RUN | - | - | Missing result entry |")
            continue

        if result["status"] != "completed":
            note = result.get("reason", "Not run")
            lines.append(f"| {title} | NOT RUN | - | - | {note} |")
        else:
            lines.append(
                f"| {title} | COMPLETED | "
                f"{result['exact_match']:.4f} | "
                f"{result['normalized_edit_distance']:.4f} | - |"
            )

    lines.append("")
    lines.append("## Note on setup availability")
    lines.append("")
    lines.append(
        "The setup **SFT using `linxy/LaTeX_OCR:train`** was not executed because no separate "
        "trained adapter for this setup was available in the project artifacts. "
        "The only available trained adapter corresponds to the combined training run "
        "(`linxy/LaTeX_OCR:train + deepcopy/MathWriting-human`)."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("Loading datasets...")
    train_latex_ocr, train_combined, test_dataset = build_train_datasets(
        max_mathwriting_samples=20000
    )

    print(f"Train LaTeX_OCR size: {len(train_latex_ocr)}")
    print(f"Train combined size: {len(train_combined)}")
    print(f"Test size: {len(test_dataset)}")  # должно быть 70

    results = {}

    # 1) Zero-shot
    print("\n=== Loading base model for zero-shot ===")
    processor_base, model_base = load_base_model()

    print("\n=== Evaluating zero-shot ===")
    zero_shot_results = evaluate_setup(
        processor_base,
        model_base,
        test_dataset,
        one_shot_example=None,
    )
    results["zero_shot"] = zero_shot_results
    print_results("Zero-shot", zero_shot_results)

    # 2) One-shot
    print("\n=== Evaluating one-shot ===")
    one_shot_ex = (
        train_latex_ocr[0]["image"],
        normalize_latex(train_latex_ocr[0]["label"]),
    )
    one_shot_results = evaluate_setup(
        processor_base,
        model_base,
        test_dataset,
        one_shot_example=one_shot_ex,
    )
    results["one_shot"] = one_shot_results
    print_results("One-shot", one_shot_results)

    # 3) SFT on LaTeX_OCR only
    # Честно отмечаем как отсутствующий режим
    results["sft_latex_ocr"] = build_missing_result(
        "No separate trained adapter is available for `SFT using linxy/LaTeX_OCR:train`. "
        "The folder `qwen_latexocr_sft` was empty, and the available checkpoints belong "
        "to the combined training run only."
    )
    print_results("SFT LaTeX_OCR", results["sft_latex_ocr"])

    # 4) SFT on combined data
    adapter_combined = os.environ.get(
        "SFT_COMBINED_ADAPTER",
        "./qwen_latexocr_mathwriting_sft"
    )

    if not os.path.isdir(adapter_combined):
        results["sft_combined"] = build_missing_result(
            f"Combined adapter folder not found: {adapter_combined}"
        )
    elif not os.listdir(adapter_combined):
        results["sft_combined"] = build_missing_result(
            f"Combined adapter folder is empty: {adapter_combined}"
        )
    else:
        print(f"\n=== Loading LoRA adapter for combined SFT: {adapter_combined} ===")
        processor_sft2, model_sft2 = load_lora_model(adapter_combined)

        print("\n=== Evaluating SFT on combined data ===")
        sft2_results = evaluate_setup(
            processor_sft2,
            model_sft2,
            test_dataset,
            one_shot_example=None,
        )
        results["sft_combined"] = sft2_results

    print_results("SFT combined", results["sft_combined"])

    save_results_json("evaluation_results.json", results)
    save_summary_md("evaluation_summary.md", results)

    print("\nSaved results to evaluation_results.json")
    print("Saved summary to evaluation_summary.md")


if __name__ == "__main__":
    main()
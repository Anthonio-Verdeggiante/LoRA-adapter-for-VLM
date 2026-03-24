import re
from datasets import load_dataset, concatenate_datasets


SYSTEM_PROMPT = "Convert the handwritten mathematical formula into valid LaTeX. Return only LaTeX."


def normalize_latex(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s


def unify_latex_ocr_split(split):
    """
    Приводит split датасета linxy/LaTeX_OCR к схеме:
    {
        "image": PIL.Image,
        "label": str
    }
    """
    def _map(ex):
        label = ex.get("label", ex.get("text", ""))
        return {
            "image": ex["image"],
            "label": label,
        }

    mapped = split.map(_map)
    keep_cols = [c for c in mapped.column_names if c in ["image", "label"]]
    remove_cols = [c for c in mapped.column_names if c not in keep_cols]
    if remove_cols:
        mapped = mapped.remove_columns(remove_cols)
    return mapped


def unify_mathwriting_split(split):
    """
    Приводит split датасета deepcopy/MathWriting-human к схеме:
    {
        "image": PIL.Image,
        "label": str
    }
    """
    def _map(ex):
        label = ex.get("label", ex.get("latex", ex.get("text", "")))
        return {
            "image": ex["image"],
            "label": label,
        }

    mapped = split.map(_map)
    keep_cols = [c for c in mapped.column_names if c in ["image", "label"]]
    remove_cols = [c for c in mapped.column_names if c not in keep_cols]
    if remove_cols:
        mapped = mapped.remove_columns(remove_cols)
    return mapped


def build_train_datasets(max_mathwriting_samples=20000):
    """
    Возвращает:
    - train_latex_ocr: train split из linxy/LaTeX_OCR
    - train_combined: train_latex_ocr + часть MathWriting-human
    - test_latex_ocr: первые 70 примеров official linxy/LaTeX_OCR:test
    """

    latex_ocr = load_dataset("linxy/LaTeX_OCR", name="human_handwrite")
    mathwriting = load_dataset("deepcopy/MathWriting-human")
    
    print("linxy/LaTeX_OCR splits:", latex_ocr)
    
    if "train" not in latex_ocr:
        raise ValueError("Dataset linxy/LaTeX_OCR does not contain 'train' split.")
    if "test" not in latex_ocr:
        raise ValueError(
            "Dataset linxy/LaTeX_OCR does not contain 'test' split. "
            "Проверь актуальную структуру датасета."
        )

    train_latex_ocr = unify_latex_ocr_split(latex_ocr["train"])
    test_latex_ocr = unify_latex_ocr_split(latex_ocr["test"])

    # Требование задания: использовать test subset из 70 примеров
    if len(test_latex_ocr) < 70:
        raise ValueError(
            f"Official linxy/LaTeX_OCR:test contains only {len(test_latex_ocr)} examples, "
            f"but task requires 70."
        )

    test_latex_ocr = test_latex_ocr.select(range(70))

    if "train" not in mathwriting:
        raise ValueError("Dataset deepcopy/MathWriting-human does not contain 'train' split.")

    train_mathwriting = unify_mathwriting_split(mathwriting["train"])

    if max_mathwriting_samples is not None:
        max_mathwriting_samples = min(max_mathwriting_samples, len(train_mathwriting))
        train_mathwriting = train_mathwriting.select(range(max_mathwriting_samples))

    train_combined = concatenate_datasets([train_latex_ocr, train_mathwriting])

    return train_latex_ocr, train_combined, test_latex_ocr
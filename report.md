# Task 1 Technical Report

## Model
I used `Qwen/Qwen2.5-VL-3B-Instruct` as the base Vision-Language Model for the handwritten formula to LaTeX task.

## Task formulation
The task was formulated as an image-to-text generation problem:
“Convert the handwritten mathematical formula into valid LaTeX. Return only LaTeX.”

## Datasets
Training datasets:
- `linxy/LaTeX_OCR:train`
- `deepcopy/MathWriting-human` (reduced subset for computational efficiency)

Evaluation dataset:
- `linxy/LaTeX_OCR`, config `human_handwrite`, split `test[:70]`

# Compared setups

The following setups were evaluated:

1. **Zero-shot inference**
2. **One-shot inference**
3. **SFT using `linxy/LaTeX_OCR:train`** — not available
4. **SFT using `linxy/LaTeX_OCR:train + deepcopy/MathWriting-human`**

## Fine-tuning method
I used parameter-efficient fine-tuning (LoRA).

## Hyperparameters
- Learning rate: 2e-4
- Epochs: 2
- Per-device batch size: 1
- Gradient accumulation steps: 8
- Quantization: 4-bit
- Max generation length: 128

## Metrics
Primary metric:
- Normalized Edit Distance (lower is better)

Additional metric:
- Exact Match Accuracy (higher is better)

## Evaluation

Evaluation dataset:
- `linxy/LaTeX_OCR:test[:70]`

This evaluation setup uses the official `linxy/LaTeX_OCR` test subset restricted to 70 examples, as required by the task.

### Metrics
Main metric:
- **Normalized Edit Distance (NED)**

Additional metric:
- **Exact Match (EM)**

Why NED:
- the model output is a LaTeX string;
- exact equality alone is too strict;
- edit distance captures how close the generated LaTeX is to the reference;
- normalization makes scores comparable across formulas of different lengths.

## Results

Evaluation dataset: `linxy/LaTeX_OCR:test[:70]`

Main metric: **Normalized Edit Distance (lower is better)**
Additional metric: **Exact Match (higher is better)**

| Setup | Status | Exact Match | Normalized Edit Distance | Notes |
|---|---|---:|---:|---|
| Zero-shot inference | COMPLETED | 0.0000 | 0.2257 | - |
| One-shot inference | COMPLETED | 0.0857 | 0.2084 | - |
| SFT using linxy/LaTeX_OCR:train | NOT RUN | - | - |
| SFT using linxy/LaTeX_OCR:train + deepcopy/MathWriting-human | COMPLETED | 0.1714 | 0.4510 | - |

## Note on setup availability

The setup **SFT using `linxy/LaTeX_OCR:train`** was not executed because no separate trained adapter for this setup was available in the project artifacts. The only available trained adapter corresponds to the combined training run (`linxy/LaTeX_OCR:train + deepcopy/MathWriting-human`).

### Note
A separate adapter for **SFT using `linxy/LaTeX_OCR:train`** was not available in the provided artifacts.  
The folder `qwen_latexocr_sft` remained empty, while all saved checkpoints belonged to the combined training run only.
Since `linxy/LaTeX_OCR` contains multiple dataset configurations, the evaluation used the `human_handwrite` configuration explicitly to access the official handwritten test split.

## Streamlit demo
The application accepts an image of a handwritten formula and returns:
1. predicted LaTeX
2. rendered formula

## Real-photo test
A real handwritten formula photo was tested in the Streamlit app.

## Repository contents
- training code
- evaluation code
- Streamlit app
- checkpoint links
- screenshots

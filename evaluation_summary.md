# Evaluation Summary

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
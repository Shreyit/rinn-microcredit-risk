<p align="center">
  <img src="docs/assets/banner.svg" alt="Rinn — Microcredit Default Risk" width="100%">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="LaTeX" src="https://img.shields.io/badge/report-LaTeX-008080?logo=latex&logoColor=white">
  <img alt="uv" src="https://img.shields.io/badge/env-uv-DE5FE9">
  <img alt="tests" src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

<p align="center"><b>Can a lender predict which microloans will default — and how should such a model be judged?</b></p>

---

**Rinn** (ऋण, *loan*) is an end-to-end, reproducible study of default prediction on a simulated rural
microloan portfolio. It compares five classical classifiers against a majority-class baseline, guards
rigorously against data leakage, and evaluates models the way a lender actually should: through
**recall, precision, ROC-AUC and the cost of a missed defaulter**, not through accuracy alone.

The project pairs a **machine-verified LaTeX report** (theory and hand derivations) with a
**leakage-safe Python pipeline**, so every number in the write-up can be reproduced from one command.

> [!NOTE]
> The data are **synthetic**: they are produced by a seeded generator and describe no real borrowers.
> Findings illustrate method, not real-world lending policy.

## <img src="docs/assets/icons/highlights.svg" width="20" height="20" align="top" alt=""> Highlights

- <img src="docs/assets/icons/lock.svg" width="16" height="16" align="top" alt=""> **Leakage-safe by construction.** Imputation medians and scaling parameters are learned on the fit partition only, and the test set is opened exactly once. Unit tests enforce both.
- <img src="docs/assets/icons/balance.svg" width="16" height="16" align="top" alt=""> **Imbalance-aware evaluation.** Every model is compared with a trivial baseline, which scores 80 % accuracy and 0 % recall here. Metrics come from a hand-written `metrics_from_cm`, checked against scikit-learn.
- <img src="docs/assets/icons/chart.svg" width="16" height="16" align="top" alt=""> **Model behaviour, explained.** K and depth sweeps show the bias–variance trade-off, an SVM C sweep shows how the margin responds, and the tree's root split is backed by an information-gain calculation done by hand.
- <img src="docs/assets/icons/cost.svg" width="16" height="16" align="top" alt=""> **Cost-sensitive decisions.** The decision threshold is treated as a business lever. It is chosen on validation data, never on the test set.
- <img src="docs/assets/icons/verify.svg" width="16" height="16" align="top" alt=""> **Verified maths.** Section II hand calculations are recomputed in Python, and the LaTeX build fails if any result disagrees.
- <img src="docs/assets/icons/report.svg" width="16" height="16" align="top" alt=""> **Academic report** set in LaTeX in a European style: A4 title page, declaration of authorship, author–year references.

## <img src="docs/assets/icons/pipeline.svg" width="20" height="20" align="top" alt=""> Pipeline

<p align="center">
  <img src="docs/assets/pipeline.svg" alt="Pipeline" width="100%">
</p>

```mermaid
flowchart TD
    M[(Master dataset<br/>3 000 loans)] -->|roll → SHA-256 seed| S[Personal sample<br/>900 loans · 20 % default]
    S -->|stratified 80/20| TR[Train 720]
    S -->|stratified 80/20| TE[Test 180 · locked]
    TR --> FIT[Fit 576] & VAL[Validation 144]
    FIT & VAL --> P1[prepare_features<br/>fit-only median + scaler]
    P1 --> TUNE[K · depth · C sweeps]
    TR & TE --> P2[prepare_features]
    P2 --> MODELS[LR · KNN · GNB · Tree · SVM · Baseline]
    TUNE -. informs .-> MODELS
    MODELS --> EVAL[Confusion matrices · ROC · results table]
    EVAL --> DEC[Threshold & cost reasoning]

    classDef lock fill:#fee2e2,stroke:#e11d48,color:#7f1d1d
    class TE lock
```

## <img src="docs/assets/icons/results.svg" width="20" height="20" align="top" alt=""> Results

> *Populated once Section III is complete.*

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | — | — | — | — | — | — |
| Logistic regression | — | — | — | — | — | — |
| KNN (K = 9) | — | — | — | — | — | — |
| Gaussian NB | — | — | — | — | — | — |
| Decision tree (depth 4) | — | — | — | — | — | — |
| Linear SVM | — | — | — | — | — | — |

## <img src="docs/assets/icons/folder.svg" width="20" height="20" align="top" alt=""> Repository structure

```text
rinn-microcredit-risk/
├── src/rinn/          reusable library: data · features · models · evaluation · viz · hand_calcs
├── notebooks/         01 graded analysis · 02 extended portfolio analysis
├── report/written/    LaTeX source for the written report (Sections I & II)
├── tests/             leakage, metric, and hand-calculation tests
├── scripts/           word-count, verification, submission builder
├── results/           generated figures and tables
└── docs/assets/       README graphics
```

The full design is in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## <img src="docs/assets/icons/terminal.svg" width="20" height="20" align="top" alt=""> Quick start

```bash
git clone <repo-url> rinn-microcredit-risk && cd rinn-microcredit-risk
make env        # create .venv with uv (Python 3.11, pinned libs)
make test       # leakage / metric / hand-calc tests
make notebooks  # execute notebooks top-to-bottom
make report     # compile LaTeX → report/written/build/main.pdf
make all        # everything above
```

## <img src="docs/assets/icons/methods.svg" width="20" height="20" align="top" alt=""> Methods at a glance

| Concept | Where it appears |
|---|---|
| Linear scores, perceptron updates | Report § I.1, § II.E1–E2 |
| Logistic link, thresholds | Report § I.2, § II.E3 · Notebook Task 7 |
| Confusion-matrix metrics, imbalance | Report § I.3, § II.E4 · Notebook Tasks 3 & 6 |
| KNN & feature scaling | Report § I.4, § II.E5 · Notebook Tasks 4.2 & 5.1 |
| Decision trees, information gain | Report § I.4 · Notebook Tasks 4.3 & 5.2 |
| Max-margin SVM, regularisation C | Report § I.5 · Notebook Task 5.3 |
| Naive Bayes | Report § II.E6 · Notebook Task 4 |

## <img src="docs/assets/icons/context.svg" width="20" height="20" align="top" alt=""> Context

Built for *Introduction to Machine Learning* (BS Analytics, School of Analytics, Tata Institute of Social Sciences)
and extended into a portfolio project. AI assistance was used and is disclosed in the report, as the course policy requires.

## <img src="docs/assets/icons/license.svg" width="20" height="20" align="top" alt=""> License

MIT. See [LICENSE](LICENSE). The assignment brief and the provided notebook cells belong to the course instructors and are not covered by this licence.

# CLAUDE.md — Rinn (rinn-microcredit-risk)

Graded assignment for TISS *Introduction to Machine Learning* (BS Analytics, Sem V), polished into a
résumé-grade data-science project. Student: **Shreyit Sinha**, roll **M2024BSASS027**.
Full blueprint: `ARCHITECTURE.md` (read it first). Original brief: `assignment/BS_Assignment_Graded.pdf`.

## Individualised config (verified locally)
| Key | Value |
|---|---|
| Roll number | `M2024BSASS027` |
| Version ID | `20AA1102C3` |
| Assigned K (KNN) | 9 |
| Assigned max depth | 4 |
| Sample | 900 loans, 180 defaults; train 720 / test 180 (36 positives); fit 576 / val 144 |
| Master checksum | `2AFAEA208FCBF28D` (matches reference with numpy 2.4.6, pandas 3.0.6, sklearn 1.9.1) |

⚠️ `notebooks/01_section_III_submission.ipynb` still shows Task 0 output from a **wrong roll (…002)**.
It must be re-run with `M2024BSASS027` (Task 0 uses `input()`, so run interactively or feed stdin).
Version ID must match in the notebook, `assignment_config.json` and page 1 of the written PDF.

## Hard rules (marks or integrity are at stake)
- **Never edit `PROVIDED` cells**, seeds, or assigned K/depth in the graded notebook. Only fill `YOUR CODE HERE` cells and answer markdown cells.
- **Test set is used only in Task 6/7.** All tuning on `X_fit`/`X_val`. `prepare_features` fits medians + scaler on the fit part only.
- **Respect word limits** in every written answer (equations/tables excluded).
- Section I: answer **exactly five** (plan: Q1–Q5; Q6 backup lives in `report/written/sections/11_section1_q6_backup.tex`, not included).
- Section II: show *formula → substitution → result*, 4 d.p.; cross-check numbers with code.
- AI use is allowed **with disclosure**; the student must be able to explain everything. Fill the AI declaration in `report/written/metadata.tex` and the notebook's final cell.
- Repo stays **private until grading is done**; `assignment/`, `submission/`, `assignment_config.json` are git-ignored.
- Extra portfolio analysis goes in `notebooks/02_extended_analysis.ipynb`, never in the graded notebook.

## Layout
- `notebooks/01_section_III_submission.ipynb` — graded notebook (working copy; pristine template in `assignment/`)
- `report/written/` — LaTeX for Sections I & II: `main.tex`, `metadata.tex` (name/roll/Version ID/AI declaration), `rinn-report.sty` (house style), `sections/*.tex`, `references.bib`
- `src/rinn/` — reusable library (only stubs so far), `tests/`, `scripts/` (planned: verify_hand_calcs, wordcount, build_submission)
- `docs/assets/` — animated SVGs for README; `results/` — generated figures/tables

## Commands
```bash
make env       # uv venv (.venv, Python 3.11) + deps — already created
make report    # pdflatex+bibtex → report/written/build/main.pdf
make test      # pytest
make notebooks # execute notebooks in place
```
- LaTeX is **BasicTeX**: `enumitem`, `siunitx`, `titlesec`, `cleveref`, `tcolorbox`, `lastpage`, `multirow`, `csquotes` are NOT installed — don't use them. No `₹` glyph under pdflatex; write `INR`.
- Report macros: `\question{1.1}{2 marks}{text}` restates a question, `\wordlimit{300}`, `\pending` marks unfinished parts (remove before submission), `\finalans{}`.

## Status (2026-10-21)
- ✅ P0 setup: architecture, README + SVGs, LaTeX skeleton compiles (all answers `\pending`), env built.
- ⏭️ Next: P1 Section III notebook (Tasks 0–7) → P2 Section II → P3 Section I → library/tests → extended notebook → zip.
- Missing info: course instructor name (`<Course Instructor>` in `metadata.tex`).

## Submission (`M2024BSASS027.zip`)
`M2024BSASS027_ML_Written.pdf`, `M2024BSASS027_ML_Notebook.ipynb` (outputs visible), `M2024BSASS027_ML_Notebook.pdf`, `assignment_config.json`.

# Manuscript readiness checklist (BAI301 Option B)

This is the **course-level, journal-styled working draft** based on the completed 342-run experiment. It is not an accepted paper or a claim that a *Neural Networks* journal submission would currently meet novelty/generalization standards.

## Done

- [x] Two-column `elsarticle` layout modeled on the appearance of the user's arXiv:1811.08687 reference (which is an *Engineering Applications of Artificial Intelligence* paper, not the journal's required template).
- [x] Abstract, introduction, focused literature review, methods, results, limitations and conclusion.
- [x] In-text numbered citations, `references.bib` and Overleaf-ready media.
- [x] Actual logged results, including positive performance results **and** catastrophic finite ESN electricity outliers.
- [x] Training choices, architecture widths and measured run counts reconciled to the underlying code.
- [x] Actual-data example plot with corrected Mackey–Glass sampling labels.
- [x] Visual distinction between ImageGen conceptual artwork and actual numerical data/plots.
- [x] Separate GitHub-ready reproducibility ZIP including notebook, source, CSVs, detailed logs, and README.

## Before submitting to the instructor

- [ ] Upload the contents of this ZIP to Overleaf, build the PDF and check all figures, citations and table placement. Local LaTeX compilation was not available during packaging.
- [ ] Coauthors confirm exact author order, spelling, affiliations and individual contributions.
- [ ] Both authors check equations, descriptions, tables, figure captions and references against the notebooks and saved results.
- [ ] Confirm the instructor's acceptable report format (standard IEEE course report versus longer journal-style extra-credit draft).
- [ ] Execute at least the notebook's built-in tests or a smoke run in Kaggle/PyTorch to validate the extracted standalone repository scripts.

## Extra work needed before claiming journal submission readiness

- [ ] Evaluate at least two **additional independent real-world datasets** (Mackey–Glass is a synthetic dynamics benchmark, and the current electricity dataset is reduced to one aggregate signal).
- [ ] Harmonize ESN-vs-gradient-model compute budgets or explicitly limit cross-family efficiency conclusions.
- [ ] Increase independent seeds/series, prespecify hypotheses, and use uncertainty estimates compatible with overlapping windows and seed dependence.
- [ ] Characterize ESN finite rollout instability via a systematic stability-focused sweep.
- [ ] Obtain a persistent GitHub/repository link with environment lockfile and independently reproduced training run.
- [ ] Reconfirm Elsevier/**target-journal** instructions, review/anonymization requirements and generative-image rules. The ImageGen artwork is in-article explanatory illustration, **not a generated graphical abstract**.

## Fast Overleaf import

1. Create a blank Overleaf project.
2. Upload every file here (including all PNG/PDF visuals and `.bib`) into the project root.
3. Set `main.tex` as the main file and choose **pdfLaTeX**.
4. Recompile, inspect warnings and fix author metadata before submitting a PDF.

The result tables and full code are in the **companion GitHub-ready ZIP**; the raw source electricity archive is not redistributed in either package pending terms verification.

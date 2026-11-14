# Revised recurrent forecasting paper (Elsevier two-column 5p working draft)

This Overleaf source package rewrites the **342-run BAI301 time-series study** using the two-column *Elsevier* appearance of the paper at https://arxiv.org/abs/1811.08687. That reference paper is associated with **Engineering Applications of Artificial Intelligence**. Its PDF is a style reference, **not** the official template for the journal *Neural Networks*.

## Overleaf steps

1. Start a blank Overleaf project and upload **all files** in this ZIP at the project root, or choose the official Elsevier `elsarticle` template and replace its example `main.tex` and assets with this package.
2. Set `main.tex` as the main document. Select **pdfLaTeX** and compile twice after BibTeX has generated the bibliography (Overleaf normally runs this sequence automatically).
3. The class file `elsarticle.cls` and bibliography style `elsarticle-num.bst` are included in standard Overleaf installations. The package intentionally does not duplicate third-party Elsevier-owned class files.
4. Figures appear at the top of columns/pages according to the two-column float algorithm; the first full-width hero figure may move to the following page. All images referenced in `main.tex` are included.
5. Check titles, affiliations, author order, image-generation disclosure and references with both authors before distributing.

## Research integrity

- **Diffusion is used only to generate explanatory artwork.** No U-Net, diffusion forecaster, learned feature-map encoder, or 128-hidden-unit recurrent model was evaluated. The new hero figure fixes those inaccuracies in the previously shared illustration.
- The hero and training-feedback images are AI-generated **conceptual visuals**. Their illustrative signal curves and network icons should not be read as real data or exact computation graphs.
- Under Elsevier's June 2026 policy, AI-supported **explanatory figures in the article** may be used with human verification and disclosure. However, general-purpose AI image generation is **not permitted for a separate graphical abstract**. Treat the hero as an in-article explanatory figure only; do not repurpose it as the journal's graphical abstract. Check the current journal policy again before submission.
- `data_examples_measured.png` shows **actual saved time-series values**. The three PDF result figures (`accuracy_time.pdf`, `forcing_interaction.pdf`, `electricity_esn_instability.pdf`) were generated from experimental CSV records. They are not synthetic AI plots.
- Recurrent widths: RNN **64**, LSTM **31**, GRU **36**. ESN reservoirs: **200, 500, 1,000**. Total logged runs: **324 gradient-trained + 18 ESN**.
- Different supervision objectives and hardware paths make ESN CPU fitting seconds **not a fair speed comparison** to neural GPU training seconds; memory metrics are also incompatible.
- All research data and numbers in the rewritten manuscript originate in the previously supplied `BAI301_full_results.zip`. The ZIP is not bundled here because it is a separate, large experimental-results archive.

## What this is not

This is a carefully qualified **journal-oriented working manuscript**, not an accepted or submission-ready journal paper. A companion GitHub-ready archive now contains reproducibility code and complete logged results, but it still needs a verified public release and actual independent re-execution. Scientific journal readiness would additionally require independent real datasets, stronger training-budget controls, predeclared stability analyses, final metadata checking, and author sign-off. For actual submission, confirm the target journal's required review layout, anonymization and figure policies; the two-column 5p layout is chosen here to replicate the user's reference PDF.

## Included files

- `main.tex` — complete rewritten manuscript (~4,100 words), 2-column numbered citation style.
- `references.bib` — 15 relevant bibliography entries.
- `hero_pipeline_ai.png` — corrected navy/orange scientific pipeline artwork.
- `data_examples_measured.png` — numerical examples from actual two datasets.
- `feedback_rollout_ai.png` — conceptual training vs evaluation feedback illustration.
- `accuracy_time.pdf` — logged accuracy vs fitting time.
- `forcing_interaction.pdf` — logged architecture × feedback interactions.
- `electricity_esn_instability.pdf` — all electricity ESN runs and finite outliers.

**Build status:** Local TeX preview is unavailable in this runtime because its pdfLaTeX format file is missing (`pdflatex.fmt`). The source has been packaged for Overleaf but a successful compile has **not** been claimed.

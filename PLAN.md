# Plan: Improving Lab Report Extraction

## 1. Problem Breakdown & Approach
Our goal is to diagnose and measurably improve the baseline lab report extractor (`extract.py`) while preserving the CLI interface (`python extract.py <file>`), optimizing accuracy, cost, and latency. We break this down into four sequential stages:

1. **Establish Evaluation Metric (`eval.py`)**: Define what constitutes a "correct" test row match between predictions and gold labels (`test_name`, `value`, `unit`, `reference_range`). We will measure field-level precision, recall, and overall row F1 score. For the 4 unlabelled reports (`04`, `07`, `10`, `12`), we will visually inspect the documents, curate verified ground-truth labels, and measure consistency.
2. **Benchmark the Baseline**: Run the baseline against the 8 labelled reports to record baseline accuracy, average latency per report, and token cost.
3. **Root-Cause Diagnosis**: Inspect extraction discrepancies across document modalities (digital PDF, scans, multi-page, photos) to pinpoint the top 3 structural failure causes.
4. **Targeted Pipeline Iteration**: Address root causes via image preprocessing, multi-page support, prompt calibration, and schema-constrained output, followed by comparative re-evaluation and documenting a negative experiment.

## 2. What We Will Look at First
- **PDF Preprocessing in `extract.py`**: The baseline hardcodes `pdftoppm -f 1 -l 1` at `DPI = 60`. We will immediately inspect multi-page reports (e.g. `report_11.pdf`) and dense tables to evaluate how much information is lost before the model ever sees the document.
- **The Baseline Prompt & Parser Constraints**: The baseline forces `to_number()` on `value`, hallucinates missing reference ranges, and prompts for "standard medical names". We will compare this with the actual entries in `gold/` and `schema.json`.
- **Gold Labels Quality**: The brief notes labels were "made in a hurry". We will check for formatting conventions, missing rows, and normalization discrepancies.

## 3. What We Expect to Find
- **Severe Recall Drop on Multi-Page Documents**: Any report longer than 1 page will have low recall because pages 2+ are completely discarded by the baseline converter.
- **OCR/Resolution Failures at 60 DPI**: 60 DPI is insufficient for small tabular fonts, decimal points, and minus signs, resulting in corrupted numbers or missed lines.
- **Data Loss on Qualitative / Non-Numeric Tests**: The baseline assumes `value` must be a float. Urine analysis, serology, and blood group tests with values like "Positive", "Clear", or "Reactive" will be wiped or parsed as `None`.
- **Hallucinated Reference Ranges**: The prompt explicitly commands the model to invent reference ranges if omitted. We expect artificial ranges that do not match the report or gold labels.
- **Name Mismatch Penalties**: Asking the model to use "standard medical name" likely produces canonical clinical terms that differ from the literal verbatim lab labels recorded in the gold set.

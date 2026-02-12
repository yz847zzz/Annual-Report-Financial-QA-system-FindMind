# Project Gap Audit (What is Missing to Run End-to-End)

This audit is based on the current repository snapshot and code-path inspection.

## 1) High-impact missing assets

The following required runtime assets are **not present in the repository**:

- `data/` directory (entire runtime workspace is absent)
- Annual report corpus (PDF files, ~70GB per README)
- `data/pdf_info.json`
- `data/test/C-list-question.json`
- `data/test/C-list-pdf-name.txt`
- `data/check/test.pdf` (used by `check_paths()`)
- Extracted intermediate tables:
  - `data/basic_info.json`
  - `data/employee_info.json`
  - `data/dev_info.json`
  - `data/cbs_info.json`
  - `data/cis_info.json`
  - `data/cscf_info.json`
- Built tabular outputs:
  - `data/CompanyTable.csv`
  - `data/key_count.json`
  - `data/key_map.json`
- Pretrained/fine-tuned model artifacts:
  - `data/pretrained_models/chatglm2-6b`
  - `data/pretrained_models/text2vec-base-chinese`
  - `ptuning/CLASSIFY_PTUNING/.../checkpoint-*`
  - `ptuning/KEYWORDS_PTUNING/.../checkpoint-*`
  - `ptuning/NL2SQL_PTUNING/.../checkpoint-*`
- External tool folder:
  - `xpdf/bin64/pdftotext`

## 2) Environment/config gaps

- `config/cfg.py` currently uses a **local Windows path**:
  - `BASE_DIR = r'D:\lecture\lecture1\lecture\llm\FINDMIND'`
  - This will fail on other machines/containers unless edited.
- `qwen_ptuning.py` assumes an online service endpoint:
  - `URL = "http://localhost:8089/v1/completions"`
  - A serving process for the Qwen model must be started separately.
- `qwen_ptuning.py` references `logger.error(...)` without importing/defining `logger` (runtime bug path if request fails).

## 3) Python dependency gaps

`requirements.txt` is incomplete vs. actual imports used in code.

### Present in `requirements.txt`
- `camelot-py`, `ghostscript`, `pdfplumber`, `loguru`, `langchain`, `transformers`, `sentencepiece`, `opencv-python`, `parse`, `fastbm25`

### Imported in code but missing from `requirements.txt`
- `torch`
- `pandas`
- `numpy`
- `requests`
- `text2vec`
- `matplotlib`

### Potentially missing local/custom module
- `chinese_text_splitter` is imported in `recall_report_text.py` but not found in repo root.

## 4) File-by-file check (root Python files)

## Core pipeline entry files

- `main.py`
  - Needs full `data/`, xpdf toolchain, and ChatGLM checkpoints.
- `lora_main.py`
  - Needs full `data/`, xpdf toolchain, and a running Qwen inference service.

## Model wrappers

- `chatglm_ptuning.py`
  - Needs local `chatglm2-6b` model and 3 p-tuning checkpoint sets.
- `qwen_ptuning.py`
  - Needs HTTP model server at `localhost:8089`; currently has logger import bug.

## Data extraction / preprocessing

- `preprocess.py`
  - Needs raw annual-report PDFs + xpdf + camelot/ghostscript runtime.
- `pdf2txt.py`
  - Needs readable PDF inputs.
- `financial_state.py`
  - Needs extracted page text and report table files.
- `pdf_util.py`
  - Utility module, depends on parsed PDF structure being available.

## Data load / transform / storage

- `file.py`
  - Central loader, expects `data/pdf_info.json`, extracted json tables, and test files.
- `company_table.py`
  - Requires extracted table jsons to build `CompanyTable.csv`.

## QA logic

- `generate_answer_with_classify.py`
  - Requires classification outputs, keyword outputs, SQL outputs, and table data.
- `question_util.py`
  - Requires `pdf_info` metadata.
- `type1.py`
  - Prompt construction, depends on upstream retrieved background.
- `type2.py`
  - Formula/calc QA, depends on retrieved structured data.
- `prompt_util.py`
  - Prompt templates only; no large assets by itself.
- `recall_report_text.py`
  - Needs vector index build dependencies and `chinese_text_splitter`.
- `recall_report_names.py`
  - Needs test question and report name files.

## Validation / scoring / misc

- `check.py`
  - Needs `data/check/test.pdf` and processed outputs to validate.
- `test_score.py`
  - Needs reference answer file and prediction file + text2vec model.
- `sql_correct_util.py`
  - Utility, no large file by itself.
- `re_util.py`
  - Utility, no large file by itself.
- `tuning_data_util.py`
  - Contains hardcoded local path `F:\mantoutech\dataset\prompt.jsonl`; not portable.
- `version.py`
  - Utility.
- `__init__.py`
  - Package marker.

## 5) What to upload (minimum useful package)

If you cannot upload full 70GB+ assets/models, upload at least:

1. A **small demo dataset bundle** (2–5 PDFs + matching `pdf_info.json`).
2. A **`data/test/` sample** with a tiny question set and expected schema.
3. A **model setup guide** with either:
   - download scripts/checkpoint links, or
   - explicit "not open-source" placeholders + interface contract.
4. A `.env.example` / config template replacing machine-specific absolute paths.
5. A completed `requirements.txt` (or `environment.yml`) with pinned versions.

This makes the repository reproducible for reviewers even without full private assets.

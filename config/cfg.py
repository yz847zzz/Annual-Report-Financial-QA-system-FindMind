import os

# ========== Project Directory Setup ==========
# Prefer an explicit environment variable, and fall back to the repository root.
_DEFAULT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.environ.get('FINDMIND_BASE_DIR', _DEFAULT_BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, "data")

# ========== PDF Related ==========
PDF_TEXT_DIR = 'pdf_docs'
ERROR_PDF_DIR = 'error_pdfs'

# ========== Model Settings ==========
CLASSIFY_PTUNING_PRE_SEQ_LEN = 512
KEYWORDS_PTUNING_PRE_SEQ_LEN = 256
NL2SQL_PTUNING_PRE_SEQ_LEN = 128
NL2SQL_PTUNING_MAX_LENGTH = 2200

# ========== Model Checkpoint Paths ==========
CLASSIFY_CHECKPOINT_PATH = os.path.join(BASE_DIR, "ptuning", "CLASSIFY_PTUNING", "output", "Fin-Train-chatglm2-6b-pt-512-2e-2", "checkpoint-400")
NL2SQL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "ptuning", "NL2SQL_PTUNING", "output", "Fin-Train-chatglm2-6b-pt-128-2e-2", "checkpoint-600")
KEYWORDS_CHECKPOINT_PATH = os.path.join(BASE_DIR, "ptuning", "KEYWORDS_PTUNING", "output", "Fin-Train-chatglm2-6b-pt-256-2e-2", "checkpoint-250")

# ========== Pretrained LLM Model Path ==========
LLM_MODEL_DIR = os.path.join(DATA_PATH, "pretrained_models", "chatglm2-6b")

# ========== Other ==========
XPDF_PATH = os.path.join(BASE_DIR, "xpdf", "bin64")
NUM_PROCESSES = 64

# updated config.py
import os
import torch
import logging
import random
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────
# Visualization constants (consumed by visualization.py)
# ─────────────────────────────────────────────
OPTION_COLS  = ["A", "B", "C", "D", "E"]
PLOT_STYLE   = "seaborn-v0_8-whitegrid"
COLORS       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
                "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
FIGURE_DPI   = 150
SAVE_PLOTS   = True
OUTPUT_DIR   = "/kaggle/working/outputs"
PLOT_DIR     = "/kaggle/working/outputs/plots"
PREBUILT_TFIDF_MODEL_PATH = Path("/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026outputs/models/tfidf_vectorizer.pkl")
PREBUILT_W2V_MODEL_PATH   = Path("/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026outputs/models/word2vec.model")


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def get_logger(name: str = "MCQSolver") -> logging.Logger:
    """Return a consistently formatted logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


logger = get_logger("Config")


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# Master Config dataclass
# ─────────────────────────────────────────────
@dataclass
class Config:
    """
    Centralised configuration object.
    Instantiate once in main.py; pass the instance everywhere.
    """

    # ── Paths ──────────────────────────────────────────────────────
    data_dir      : Path = Path("/kaggle/input/competitions/smart-mcq-solver-challenge")
    output_dir    : Path = Path("/kaggle/working/outputs")
    model_dir     : Path = Path("/kaggle/working/outputs/models")         
    submission_dir: Path = Path("/kaggle/working/outputs/submissions")
    processed_dir : Path = Path("/kaggle/working/outputs/processed_files")  
    plot_dir      : Path = Path("/kaggle/working/outputs/plots")

    # ── Raw file names ─────────────────────────────────────────────
    train_file: str = "train.csv"
    test_file : str = "test.csv"

    # ── Column schema ──────────────────────────────────────────────
    options   : List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E"])
    answer_col: str       = "answer"
    id_col    : str       = "id"
    prompt_col: str       = "prompt"
    top_k     : int       = 3   # MAP@3

    # ── TF-IDF ─────────────────────────────────────────────────────
    tfidf_max_features: int            = 50_000
    tfidf_ngram_range : Tuple[int,int] = (1, 3)
    tfidf_min_df      : int            = 1

    # ── Word2Vec ───────────────────────────────────────────────────
    w2v_vector_size: int = 300
    w2v_window     : int = 5
    w2v_min_count  : int = 1
    w2v_epochs     : int = 10

    # ── Sentence-BERT ──────────────────────────────────────────────
    sbert_model     : str = "all-MiniLM-L6-v2"
    sbert_batch_size: int = 64

    # ── Fine-tuning (Phase 4 – placeholder) ───────────────────────
    finetune_model              : str   = "microsoft/deberta-v3-small"
    max_length                  : int   = 512
    learning_rate               : float = 2e-5
    weight_decay                : float = 0.01
    num_epochs                  : int   = 3
    train_batch_size            : int   = 4
    eval_batch_size             : int   = 8
    gradient_accumulation_steps : int   = 8
    warmup_ratio                : float = 0.1
    fp16                        : bool  = True
    gradient_checkpointing      : bool  = True

    # ── LoRA ───────────────────────────────────────────────────────
    lora_r             : int       = 16
    lora_alpha         : int       = 32
    lora_dropout       : float     = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query_proj", "value_proj"]
    )

    # ── RAG ────────────────────────────────────────────────────────
    rag_top_k          : int = 3
    rag_retrieval_model: str = "all-mpnet-base-v2"

    # ── Ensemble ───────────────────────────────────────────────────
    ensemble_temperature: float = 1.0

    # ── W&B ────────────────────────────────────────────────────────
    wandb_project: str            = "22f3001418-t22026"
    wandb_entity : Optional[str]  = None
    use_wandb    : bool           = True

    # ── Hardware ───────────────────────────────────────────────────
    seed       : int  = 42
    num_workers: int  = 4
    pin_memory : bool = True
    device     : str  = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    n_gpus     : int  = field(default_factory=torch.cuda.device_count)

    # ── Persistence flags ──────────────────────────────────────────
    # Set to False to force re-training even when cached files exist
    use_cached_models   : bool = True
    use_cached_processed: bool = True

    def __post_init__(self) -> None:
        """Create all output directories; log hardware info."""
        set_seed(self.seed)
        dirs = [
            self.output_dir, self.model_dir,
            self.submission_dir, self.processed_dir, self.plot_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Keep module-level constants in sync with the dataclass
        global OUTPUT_DIR, PLOT_DIR, SAVE_PLOTS
        OUTPUT_DIR = str(self.output_dir)
        PLOT_DIR = str(self.plot_dir)

        logger.info(f"Device : {self.device}")
        logger.info(f"GPUs   : {self.n_gpus}")
        if torch.cuda.is_available():
            for i in range(self.n_gpus):
                name = torch.cuda.get_device_name(i)
                mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")

    # ── Derived path helpers ───────────────────────────────────────
    def _resolve_model_path(self, prebuilt_path: Path, fallback_name: str) -> Path:
        """Prefer the prebuilt Kaggle input artifact when it exists."""
        if prebuilt_path.exists():
            return prebuilt_path
        return Path(self.model_dir) / fallback_name

    @property
    def train_path(self) -> Path:
        return Path(self.data_dir) / self.train_file

    @property
    def test_path(self) -> Path:
        return Path(self.data_dir) / self.test_file

    @property
    def processed_train_path(self) -> Path:
        return Path(self.processed_dir) / "train_processed.csv"

    @property
    def processed_test_path(self) -> Path:
        return Path(self.processed_dir) / "test_processed.csv"

    @property
    def tfidf_model_path(self) -> Path:
        return self._resolve_model_path(
            PREBUILT_TFIDF_MODEL_PATH,
            "tfidf_vectorizer.pkl",
        )

    @property
    def w2v_model_path(self) -> Path:
        return self._resolve_model_path(
            PREBUILT_W2V_MODEL_PATH,
            "word2vec.model",
        )
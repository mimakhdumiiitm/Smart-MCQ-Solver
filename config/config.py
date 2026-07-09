# config/config.py
# updated for mielstone 2


import os
import torch

# ==================================================================
# PATHS — KAGGLE ENVIRONMENT
# ==================================================================

DATA_DIR        = "data"
OUTPUT_DIR      = "/kaggle/working/outputs"
MODEL_DIR       = "/kaggle/working/models"

# Kaggle Competition Input
KAGGLE_COMP_DIR = "/kaggle/input/competitions/smart-mcq-solver-challenge"
TRAIN_PATH      = os.path.join(KAGGLE_COMP_DIR, "train.csv")
TEST_PATH       = os.path.join(KAGGLE_COMP_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(KAGGLE_COMP_DIR, "sample_submission.csv")

# Processed Data Paths (output from EDA pipeline)
PROCESSED_DIR        = "/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026/outputs/processed_files/"
TRAIN_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "train_processed.csv")
TEST_PROCESSED_PATH  = os.path.join(PROCESSED_DIR, "test_processed.csv")

# Processed files output directory
PROCESSED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "processed_files")
os.makedirs(PROCESSED_OUTPUT_DIR, exist_ok=True)
TRAIN_OUTPUT_PATH = os.path.join(PROCESSED_OUTPUT_DIR, "train_processed.csv")
TEST_OUTPUT_PATH = os.path.join(PROCESSED_OUTPUT_DIR, "test_processed.csv")

# Pre-trained Model Inputs (from previous notebook)
W2V_MODEL_PATH    = (
    "/kaggle/input/notebooks/mimakhdumiiitm/"
    "dl-22f3001418-notebook-t22026/models/w2v.model"
)
TFIDF_INPUT_PATH  = (
    "/kaggle/input/notebooks/mimakhdumiiitm/"
    "dl-22f3001418-notebook-t22026/models/tfidf.pkl"
)
TFIDF_OUTPUT_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")

# Submission Output
SUBMISSION_OUT_PATH = "/kaggle/working/submission.csv"
RESULTS_PLOT_PATH   = "/kaggle/working/results_plot.png"

# ==================================================================
# COLUMN NAMES
# ==================================================================

ID_COL      = "id"
PROMPT_COL  = "prompt"
ANSWER_COL  = "answer"
OPTION_COLS = ["A", "B", "C", "D", "E"]
TEXT_COLS   = [PROMPT_COL] + OPTION_COLS

# ==================================================================
# TEXT PREPROCESSING  (EDA pipeline)
# ==================================================================

REMOVE_STOPWORDS = True
LEMMATIZE        = True
STEM             = False
MIN_TOKEN_LENGTH = 2

# ==================================================================
# TF-IDF SETTINGS  (EDA pipeline)
# ==================================================================

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 1
TFIDF_MAX_DF       = 0.95
TFIDF_SUBLINEAR_TF = True

# ==================================================================
# WORD2VEC SETTINGS  (EDA pipeline)
# ==================================================================

W2V_VECTOR_SIZE = 100
W2V_WINDOW      = 5
W2V_MIN_COUNT   = 1
W2V_SG          = 1        # 1 = Skip-gram, 0 = CBOW
W2V_EPOCHS      = 10
W2V_WORKERS     = 4
W2V_SEED        = 42

# ==================================================================
# GENERAL
# ==================================================================

RANDOM_SEED = 42
TOP_K       = 3            # MAP@K evaluation

# ==================================================================
# VISUALIZATION
# ==================================================================

PLOT_STYLE  = "seaborn-v0_8-whitegrid"
COLORS      = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
FIGURE_DPI  = 150
SAVE_PLOTS  = False

# ==================================================================
# GPU / DEVICE SETTINGS  (Transformer pipeline)
# ==================================================================

# CUDA environment flags — set before any model is loaded
CUDA_LAUNCH_BLOCKING = "1"     # Better CUDA error reporting
TORCH_USE_CUDA_DSA   = "1"     # Device-side assertions

# Minimum compute capability to use CUDA (6.0 = P100 compatible)
MIN_COMPUTE_CAPABILITY = 60    # cc = major*10 + minor  (6.0 → 60)

# Precision settings
# P100 has limited float16 support → use float32
USE_FP16 = False
TORCH_DTYPE = torch.float32

# ==================================================================
# TRANSFORMER MODEL SETTINGS  (Transformer pipeline)
# ==================================================================

# Sentence-BERT embedding model (P100-compatible)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# NLI zero-shot model (P100-compatible, float32 stable)
ZEROSHOT_MODEL  = "typeform/distilbert-base-uncased-mnli"

# Inference settings
TRANSFORMER_BATCH_SIZE = 16    
MAX_SEQ_LENGTH         = 128   

# ==================================================================
# ENSEMBLE SETTINGS  (Transformer pipeline)
# ==================================================================

STRATEGY         = "ensemble"
ENSEMBLE_WEIGHTS = {"embedding": 0.45, "zeroshot": 0.55}

# ==================================================================
# WEIGHTS & BIASES SETTINGS
# ==================================================================

WANDB_PROJECT = "22f3001418-t22026"
WANDB_RUN     = "transformer-embeddings-zeroshot"
WANDB_TAGS    = ["transformer", "sentence-bert", "zero-shot", "map@3"]

# W&B secret key name in Kaggle secrets
WANDB_SECRET_KEY_NAME = "WANDB_API_KEY"

print("Config loaded successfully for Kaggle Environment.")
# config.py
# Central configuration file for Smart MCQ Solver
# Edit these values to customize behavior across all modules

import os

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
DATA_DIR        = "data"
OUTPUT_DIR      = "/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026/outputs/"
MODEL_DIR       = "/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026/models/"

TRAIN_PATH      = os.path.join(DATA_DIR, "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv")
TEST_PATH       = os.path.join(DATA_DIR, "/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "/kaggle/input/competitions/smart-mcq-solver-challenge/sample_submission.csv")

# ------------------------------------------------------------------
# COLUMN NAMES
# ------------------------------------------------------------------
ID_COL      = "id"
PROMPT_COL  = "prompt"
ANSWER_COL  = "answer"
OPTION_COLS = ["A", "B", "C", "D", "E"]
TEXT_COLS   = [PROMPT_COL] + OPTION_COLS

# ------------------------------------------------------------------
# TEXT PREPROCESSING
# ------------------------------------------------------------------
REMOVE_STOPWORDS = True
LEMMATIZE        = True
STEM             = False
MIN_TOKEN_LENGTH = 2

# ------------------------------------------------------------------
# TF-IDF SETTINGS
# ------------------------------------------------------------------
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 1
TFIDF_MAX_DF       = 0.95
TFIDF_SUBLINEAR_TF = True

# ------------------------------------------------------------------
# WORD2VEC SETTINGS
# ------------------------------------------------------------------
W2V_VECTOR_SIZE = 100
W2V_WINDOW      = 5
W2V_MIN_COUNT   = 1
W2V_SG          = 1        # 1 = Skip-gram, 0 = CBOW
W2V_EPOCHS      = 10
W2V_WORKERS     = 4
W2V_SEED        = 42

# ------------------------------------------------------------------
# GENERAL
# ------------------------------------------------------------------
RANDOM_SEED = 42
TOP_K       = 3            # MAP@K evaluation

# ------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------
PLOT_STYLE  = "seaborn-v0_8-whitegrid"
COLORS      = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
FIGURE_DPI  = 150
SAVE_PLOTS  = True
print("done")
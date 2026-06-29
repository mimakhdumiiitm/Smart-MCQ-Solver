# updated config.py
# Central configuration file for Smart MCQ Solver (Kaggle Dedicated)
import os

DATA_DIR        = "data"
OUTPUT_DIR      = "/kaggle/working/outputs"
MODEL_DIR       = "/kaggle/working/models"

# Directory for saving processed dataframe CSVs (changeable from one place)

# Corrected Kaggle Input Data Paths (removed /competitions/ if standard mount)
KAGGLE_COMP_DIR = "/kaggle/input/competitions/smart-mcq-solver-challenge"
TRAIN_PATH      = os.path.join(KAGGLE_COMP_DIR, "train.csv")
TEST_PATH       = os.path.join(KAGGLE_COMP_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(KAGGLE_COMP_DIR, "sample_submission.csv")

# Corrected Pre-trained Word2Vec Path (Removed the URL path junk)
# Corrected Pre-trained Word2Vec Path (Removed the URL path junk)
W2V_MODEL_PATH  = "/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026/models/w2v.model"

# Corrected Pre-trained TF-IDF Paths
TFIDF_INPUT_PATH  = "/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026/models/tfidf.pkl"
TFIDF_OUTPUT_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")

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
SAVE_PLOTS  = False

print("Config loaded successfully for Kaggle Environment.")
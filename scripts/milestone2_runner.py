# milestone2_runner.py

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.transformer_ranker import (
    TransformerEmbeddingRanker,
    ZeroShotMCQRanker,
    _compute_metrics,
    _load_or_none,
    _scores_exist,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Priority artifact directory – checked FIRST before any other path
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# Ordered list of artifact roots to probe (highest priority first).
# The first directory that actually exists wins.
_ARTIFACT_ROOTS: List[Path] = [
    Path("/kaggle/input/notebooks/mimakhdumiiitm"
         "/dl-22f3001418-notebook-t22026/outputs"),        
    Path("/kaggle/input/project-artifacts/outputs"),        
]


def _resolve_artifact_root() -> Optional[Path]:
    """
    Return the first existing artifact root from ``_ARTIFACT_ROOTS``,
    or ``None`` when none of them exist on this machine.
    """
    for root in _ARTIFACT_ROOTS:
        if root.exists():
            logger.info(f"[artifact] Using artifact root: {root}")
            return root
    logger.info("[artifact] No pre-existing artifact root found – will run from scratch.")
    return None


# Resolve once at import time so every helper can reuse it cheaply.
_ARTIFACT_ROOT: Optional[Path] = _resolve_artifact_root()


def _artifact_path(filename: str) -> Optional[Path]:
    """
    Return ``<artifact_root>/<filename>`` when it exists, else ``None``.
    Searches ALL roots in priority order so a file missing from root-1 can
    still be found in root-2.
    """
    for root in _ARTIFACT_ROOTS:
        candidate = root / filename
        if candidate.exists():
            logger.debug(f"[artifact] Found {filename} at {candidate}")
            return candidate
    return None


def _artifact_subpath(subdir: str, filename: str) -> Optional[Path]:
    """
    Like ``_artifact_path`` but looks inside ``<root>/<subdir>/<filename>``.
    Useful for processed_files sub-folder.
    """
    for root in _ARTIFACT_ROOTS:
        candidate = root / subdir / filename
        if candidate.exists():
            logger.debug(f"[artifact] Found {subdir}/{filename} at {candidate}")
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# W&B helpers  (delegate to utils/wandb_init.py where possible)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from utils.wandb_init import (
        init_wandb,
        log_model_metrics,
        finish_run,
        _WANDB_AVAILABLE,
        REQUIRED_METRICS,
        COMPARABLE_MODELS,
    )
    _WANDB_UTILS_AVAILABLE = True
    logger.info("[W&B] utils/wandb_init imported successfully.")
except ImportError:
    _WANDB_UTILS_AVAILABLE = False
    logger.warning(
        "[W&B] utils/wandb_init not found – falling back to inline helpers."
    )


def _wandb_init(
    project    : str,
    run_name   : str,
    config_dict: dict,
    cfg        = None,
    model_tag  : str = "custom",
) -> object:
    """
    Start a W&B run.

    Strategy
    ────────
    • If utils/wandb_init is available AND cfg is supplied AND model_tag is a
      recognised phase-1 baseline tag → delegate to ``init_wandb``.
    • Otherwise fall back to the inline wandb.init call (preserves behaviour
      for zero-shot / transformer runs that are not covered by init_wandb).

    Returns the run object (or None if W&B is absent / disabled).
    """
    # ── delegate path ────────────────────────────────────────────────────────
    if _WANDB_UTILS_AVAILABLE and cfg is not None and model_tag in COMPARABLE_MODELS:
        try:
            run = init_wandb(cfg, run_name=run_name, model_tag=model_tag)
            if run is not None:
                logger.info(f"[W&B] Run started via init_wandb → {run.url}")
            return run
        except Exception as exc:
            logger.warning(
                f"[W&B] init_wandb failed ({exc}). Trying inline fallback."
            )

    # ── inline fallback ──────────────────────────────────────────────────────
    try:
        import wandb
        run = wandb.init(
            project = project,
            name    = run_name,
            config  = config_dict,
            reinit  = True,
        )
        logger.info(f"[W&B] Run started (inline) → {run.url}")
        return run
    except Exception as exc:
        logger.warning(
            f"[W&B] wandb.init failed ({exc}). Metrics will not be uploaded."
        )
        return None


def _wandb_finish(run) -> None:
    """
    Safely finish a W&B run.

    Delegates to ``finish_run`` from utils/wandb_init when available,
    otherwise falls back to inline call.
    """
    if run is None:
        return

    if _WANDB_UTILS_AVAILABLE:
        try:
            finish_run(run)
            return
        except Exception as exc:
            logger.warning(f"[W&B] finish_run() failed: {exc}")

    # inline fallback
    try:
        run.finish()
        logger.info("[W&B] Run finished.")
    except Exception as exc:
        logger.warning(f"[W&B] run.finish() failed: {exc}")


def _wandb_log(run, metrics: dict) -> None:
    """
    Log metrics to a W&B run.

    Delegates to ``log_model_metrics`` from utils/wandb_init when available
    (which also validates that REQUIRED_METRICS are present).
    Falls back to inline run.log otherwise.
    """
    if run is None:
        return

    if _WANDB_UTILS_AVAILABLE:
        try:
            log_model_metrics(run, metrics)
            return
        except Exception as exc:
            logger.warning(f"[W&B] log_model_metrics failed: {exc}")

    # inline fallback
    try:
        run.log(metrics)
    except Exception as exc:
        logger.warning(f"[W&B] run.log failed: {exc}")


def _build_required_metrics_payload(raw_metrics: dict) -> dict:
    """
    Map internal metric keys produced by ``_compute_metrics`` to the
    canonical REQUIRED_METRICS names expected by utils/wandb_init:

        map_at_3   → map_at_k
        accuracy   → accuracy
        f1_macro   → f1_score
        precision  → precision   (if present)
        recall     → recall      (if present)

    Any extra keys are passed through unchanged.
    """
    mapping = {
        "map_at_3": "map_at_k",
        "f1_macro": "f1_score",
    }
    payload = {}
    for k, v in raw_metrics.items():
        payload[mapping.get(k, k)] = v

    # Ensure all REQUIRED_METRICS keys exist (set to None if missing)
    if _WANDB_UTILS_AVAILABLE:
        for req in REQUIRED_METRICS:
            payload.setdefault(req, None)

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Inline minimal Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class _Evaluator:
    """Self-contained evaluator (no external dependency needed)."""

    def scores_to_top_k_predictions(
        self,
        scores      : np.ndarray,
        option_cols : List[str],
        k           : int = 3,
    ) -> List[List[str]]:
        preds = []
        for row in scores:
            top_idx = np.argsort(row)[::-1][:k]
            preds.append([option_cols[i] for i in top_idx])
        return preds

    def evaluate(
        self,
        df    : pd.DataFrame,
        preds : List[List[str]],
        cfg,
        split : str = "val",
    ) -> Dict[str, float]:
        if "answer" not in df.columns:
            logger.warning("No 'answer' column – MAP@3 set to 0.")
            return {f"{split}/map@3": 0.0}

        aps = []
        for pred_list, true in zip(preds, df["answer"].tolist()):
            score, hits = 0.0, 0
            for rank, p in enumerate(pred_list, 1):
                if p == str(true):
                    hits  += 1
                    score += hits / rank
            aps.append(score)

        return {f"{split}/map@3": float(np.mean(aps))}


# ─────────────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_clean_cols(df: pd.DataFrame, option_cols: List[str]) -> pd.DataFrame:
    if "prompt_clean" not in df.columns:
        prompt_col = next(
            (c for c in ["prompt", "question", "Question"] if c in df.columns),
            df.columns[0],
        )
        df["prompt_clean"] = df[prompt_col].astype(str).str.strip()

    for opt in option_cols:
        if f"{opt}_clean" not in df.columns and opt in df.columns:
            df[f"{opt}_clean"] = df[opt].astype(str).str.strip()

    return df


def _auto_load_data(
    cfg,
    val_df  : Optional[pd.DataFrame],
    test_df : Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load validation and test DataFrames.

    Search order for each file
    ──────────────────────────
    1. Caller-supplied DataFrame  (skip search entirely)
    2. Priority artifact root     (/kaggle/input/notebooks/…/outputs)
    3. Secondary artifact root    (/kaggle/input/project-artifacts/outputs)
    4. cfg.output_dir             (local processed file)
    5. cfg.data_dir / cfg.*_file  (raw CSV)
    """
    option_cols = cfg.options

    # ── Validation set ────────────────────────────────────────────────────────
    if val_df is None:
        # 1. artifact roots (priority order via _artifact_subpath / _artifact_path)
        artifact_val = (
            _artifact_subpath("processed_files", "train_processed.csv")
            or _artifact_path("train_processed.csv")
        )

        # 2. local processed file
        processed_local = cfg.output_dir / "train_processed.csv"

        # 3. raw CSV
        raw = cfg.data_dir / cfg.train_file

        if artifact_val is not None:
            logger.info(f"[data] Loading val_df from artifact: {artifact_val}")
            full   = pd.read_csv(artifact_val)
            val_df = full.iloc[int(len(full) * 0.8):].reset_index(drop=True)
        elif processed_local.exists():
            logger.info(f"[data] Loading val_df from local: {processed_local}")
            val_df = pd.read_csv(processed_local)
        elif raw.exists():
            logger.info(f"[data] Loading val_df from raw: {raw}")
            full   = pd.read_csv(raw)
            val_df = full.iloc[int(len(full) * 0.8):].reset_index(drop=True)
        else:
            raise FileNotFoundError(
                "Cannot find validation data. "
                f"Tried artifact={artifact_val}, "
                f"local={processed_local}, raw={raw}"
            )

    # ── Test set ──────────────────────────────────────────────────────────────
    if test_df is None:
        artifact_test = (
            _artifact_subpath("processed_files", "test_processed.csv")
            or _artifact_path("test_processed.csv")
        )

        processed_local = cfg.output_dir / "test_processed.csv"
        raw             = cfg.data_dir / cfg.test_file

        if artifact_test is not None:
            logger.info(f"[data] Loading test_df from artifact: {artifact_test}")
            test_df = pd.read_csv(artifact_test)
        elif processed_local.exists():
            logger.info(f"[data] Loading test_df from local: {processed_local}")
            test_df = pd.read_csv(processed_local)
        elif raw.exists():
            logger.info(f"[data] Loading test_df from raw: {raw}")
            test_df = pd.read_csv(raw)
        else:
            logger.warning("[data] No test file found – using val_df as placeholder.")
            test_df = val_df.copy()

    val_df  = _ensure_clean_cols(val_df,  option_cols)
    test_df = _ensure_clean_cols(test_df, option_cols)

    return val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Artifact-aware baseline score loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_baseline_scores(out_dir, provided) -> Dict[str, np.ndarray]:
    """
    Load Phase-1 baseline .npy score arrays.

    Search order per file
    ─────────────────────
    1. Caller-supplied dict (``provided``)
    2. Priority artifact root   (/kaggle/input/notebooks/…/outputs)
    3. Secondary artifact root  (/kaggle/input/project-artifacts/outputs)
    4. cfg.output_dir           (local run outputs)
    """
    if provided:
        return provided

    scores  : Dict[str, np.ndarray] = {}
    mapping = {
        "tfidf_val": "tfidf_val_scores.npy",
        "w2v_val"  : "w2v_val_scores.npy",
        "sbert_val": "sbert_val_scores.npy",
    }

    for key, filename in mapping.items():
        # Try every artifact root in priority order, then local output dir.
        artifact_hit = _artifact_path(filename)
        local        = out_dir / filename

        if artifact_hit is not None:
            scores[key] = np.load(artifact_hit)
            logger.info(f"[baseline] Loaded {key} from artifact: {artifact_hit}")
        elif local.exists():
            scores[key] = np.load(local)
            logger.info(f"[baseline] Loaded {key} from local: {local}")
        else:
            logger.warning(
                f"[baseline] {filename} not found in any artifact root or "
                f"locally ({local}). Will skip {key}."
            )

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Score cache helpers (artifact-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _load_scores_with_artifact_fallback(
    local_val_path : Path,
    local_test_path: Path,
    artifact_val_filename : str,
    artifact_test_filename: str,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Try to load a pair of score .npy files.

    Search order
    ────────────
    1. Local output dir  (e.g. outputs/zs_val_scores.npy)
    2. Artifact roots    (priority → secondary) for each filename

    Returns ``(val_scores, test_scores)`` on success, ``None`` when neither
    source has both files.
    """
    # ── 1. local cache ────────────────────────────────────────────────────────
    if _scores_exist(local_val_path, local_test_path):
        logger.info(
            f"[cache] Reusing local scores: {local_val_path.name}, "
            f"{local_test_path.name}"
        )
        return np.load(local_val_path), np.load(local_test_path)

    # ── 2. artifact roots ─────────────────────────────────────────────────────
    artifact_val  = _artifact_path(artifact_val_filename)
    artifact_test = _artifact_path(artifact_test_filename)

    if artifact_val is not None and artifact_test is not None:
        logger.info(
            f"[cache] Reusing artifact scores: {artifact_val}, {artifact_test}"
        )
        return np.load(artifact_val), np.load(artifact_test)

    if artifact_val is not None and artifact_test is None:
        logger.info(
            f"[cache] Found artifact val scores ({artifact_val}) but no test "
            f"scores – will recompute both to stay consistent."
        )

    return None  # nothing cached → caller must recompute


# ─────────────────────────────────────────────────────────────────────────────
# Run helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_zero_shot(cfg, val_df, test_df, wandb_run):
    try:
        ranker = ZeroShotMCQRanker(
            cfg,
            model_key  = cfg.zs_model_key,
            batch_size = cfg.batch_size,
            wandb_run  = wandb_run,
        )
        val_scores  = ranker.predict_scores(val_df)
        test_scores = ranker.predict_scores(test_df)
        ranker.free()
        return val_scores, test_scores
    except Exception as exc:
        logger.error(f"ZeroShotMCQRanker failed: {exc}")
        dummy = np.zeros((len(val_df), len(cfg.options)))
        return dummy, np.zeros((len(test_df), len(cfg.options)))


def _run_transformer_embeddings(cfg, val_df, test_df, wandb_run):
    try:
        ranker = TransformerEmbeddingRanker(
            cfg,
            model_key  = cfg.tr_model_key,
            batch_size = cfg.batch_size,
            max_length = cfg.max_length,
            wandb_run  = wandb_run,
        )
        val_scores  = ranker.predict_scores(val_df)
        test_scores = ranker.predict_scores(test_df)
        ranker.free()
        return val_scores, test_scores
    except Exception as exc:
        logger.error(f"TransformerEmbeddingRanker failed: {exc}")
        dummy = np.zeros((len(val_df), len(cfg.options)))
        return dummy, np.zeros((len(test_df), len(cfg.options)))


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison_table(all_metrics: Dict[str, dict]) -> None:
    rows = []
    for method, m in all_metrics.items():
        rows.append(
            {
                "Method"    : method,
                "MAP@3"     : m.get("map_at_3",  float("nan")),
                "Accuracy"  : m.get("accuracy",  float("nan")),
                "F1 (macro)": m.get("f1_macro",  float("nan")),
            }
        )
    df = (
        pd.DataFrame(rows)
        .sort_values("MAP@3", ascending=False)
        .reset_index(drop=True)
    )
    for col in ["MAP@3", "Accuracy", "F1 (macro)"]:
        df[col] = df[col].map(
            lambda x: f"{x:.4f}"
            if not (isinstance(x, float) and np.isnan(x))
            else "N/A"
        )

    sep = "─" * 68
    print(f"\n{sep}")
    print("  MAP@3 / Accuracy / F1  –  All methods compared")
    print(sep)
    print(df.to_string(index=False))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_milestone2(
    cfg             = None,
    val_df          : Optional[pd.DataFrame]           = None,
    test_df         : Optional[pd.DataFrame]           = None,
    evaluator                                          = None,
    option_cols     : Optional[List[str]]              = None,
    baseline_scores : Optional[Dict[str, np.ndarray]] = None,
    wandb_project   : str                              = "smart-mcq-solver",
    wandb_api_key   : Optional[str]                   = None,
) -> Dict[str, Any]:
    """
    Run the full Milestone 2 pipeline.

    Parameters
    ──────────
    cfg             : Config object (built automatically if None)
    val_df          : Validation DataFrame (auto-loaded if None)
    test_df         : Test DataFrame (auto-loaded if None)
    evaluator       : Evaluator instance (built-in _Evaluator used if None)
    option_cols     : e.g. ['A','B','C','D','E']
    baseline_scores : dict of Phase-1 .npy arrays (auto-loaded if None)
    wandb_project   : W&B project name
    wandb_api_key   : W&B API key (set WANDB_API_KEY env var as alternative)

    Artifact search order
    ─────────────────────
    For every file this runner needs (CSVs, .npy scores) it probes:
      1. /kaggle/input/notebooks/mimakhdumiiitm/
             dl-22f3001418-notebook-t22026/outputs   ← NEW priority-1
      2. /kaggle/input/project-artifacts/outputs     ← original fallback
      3. cfg.output_dir                              ← local run outputs
      4. cfg.data_dir                                ← raw source files
    If nothing is found the relevant model is run from scratch and its
    outputs are saved to cfg.output_dir for future reuse.
    """

    # ── 0. W&B login ─────────────────────────────────────────────────────────
    if wandb_api_key:
        try:
            import wandb as _wandb
            _wandb.login(key=wandb_api_key, relogin=True)
            logger.info("[W&B] Logged in with provided API key.")
        except Exception as exc:
            logger.warning(f"[W&B] Login failed: {exc}")

    # ── 1. Config / evaluator defaults ───────────────────────────────────────
    if cfg is None:
        from config.config import Config
        cfg = Config()
        logger.info(f"[M2] device={cfg.device} | out={cfg.output_dir}")

    if evaluator is None:
        evaluator = _Evaluator()

    if option_cols is None:
        option_cols = cfg.options

    # Log which artifact root is active so users can confirm the right one
    # was picked at import time.
    if _ARTIFACT_ROOT is not None:
        logger.info(f"[M2] Active artifact root: {_ARTIFACT_ROOT}")
    else:
        logger.info("[M2] No artifact root found – all models run from scratch.")

    # ── 2. Load data ──────────────────────────────────────────────────────────
    val_df, test_df = _auto_load_data(cfg, val_df, test_df)
    logger.info(f"[M2] val={len(val_df)} rows | test={len(test_df)} rows")

    out                = cfg.output_dir
    results            : Dict[str, Any]  = {"metrics": {}}
    all_method_metrics : Dict[str, dict] = {}

    # ── 3. Phase-1 baseline scores ────────────────────────────────────────────
    baseline_scores = _load_baseline_scores(out, baseline_scores)

    # ─────────────────────────────────────────────────────────────────────────
    # RUN 1 : Zero-Shot NLI
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  W&B RUN 1 – Zero-Shot NLI Ranker")
    logger.info("═" * 60)

    zs_run = _wandb_init(
        project     = wandb_project,
        run_name    = "zero_shot_nli",
        config_dict = {
            "model_type"  : "zero_shot_nli",
            "model_key"   : cfg.zs_model_key,
            "batch_size"  : cfg.batch_size,
            "n_val"       : len(val_df),
            "n_test"      : len(test_df),
            "option_cols" : option_cols,
        },
        cfg       = cfg,
        model_tag = "zero_shot_nli",
    )

    try:
        zs_val_path  = out / "zs_val_scores.npy"
        zs_test_path = out / "zs_test_scores.npy"

        # ── artifact-aware cache check ────────────────────────────────────────
        cached = _load_scores_with_artifact_fallback(
            local_val_path        = zs_val_path,
            local_test_path       = zs_test_path,
            artifact_val_filename = "zs_val_scores.npy",
            artifact_test_filename= "zs_test_scores.npy",
        )

        if cached is not None:
            zs_val_scores, zs_test_scores = cached
            logger.info("[M2] Zero-shot scores loaded from cache/artifact.")
            _wandb_log(zs_run, {"cache_hit": True})
        else:
            logger.info("[M2] Running ZeroShotMCQRanker from scratch …")
            zs_val_scores, zs_test_scores = _run_zero_shot(
                cfg, val_df, test_df, zs_run
            )
            # Save locally for future runs.
            np.save(zs_val_path,  zs_val_scores)
            np.save(zs_test_path, zs_test_scores)
            logger.info(f"[M2] Zero-shot scores saved → {out}")
            _wandb_log(zs_run, {"cache_hit": False})

        zs_metrics         = _compute_metrics(zs_val_scores, val_df, option_cols)
        zs_required        = _build_required_metrics_payload(zs_metrics)
        zs_tagged          = {f"zeroshot_val/{k}": v for k, v in zs_metrics.items()}
        zs_required_tagged = {f"zeroshot_val/{k}": v for k, v in zs_required.items()}

        _wandb_log(zs_run, zs_required_tagged)

        results["metrics"].update(zs_tagged)
        results["zs_val_scores"]  = zs_val_scores
        results["zs_test_scores"] = zs_test_scores
        all_method_metrics["Zero-Shot NLI"] = zs_metrics

        print(
            f"[M2] Zero-Shot  MAP@3={zs_metrics['map_at_3']:.4f} "
            f"| Acc={zs_metrics['accuracy']:.4f} "
            f"| F1={zs_metrics['f1_macro']:.4f}"
        )
    finally:
        _wandb_finish(zs_run)

    # ─────────────────────────────────────────────────────────────────────────
    # RUN 2 : Transformer Embedding Ranker
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  W&B RUN 2 – Transformer Embedding Ranker")
    logger.info("═" * 60)

    tr_run = _wandb_init(
        project     = wandb_project,
        run_name    = "transformer_embed",
        config_dict = {
            "model_type"  : "transformer_embedding",
            "model_key"   : cfg.tr_model_key,
            "batch_size"  : cfg.batch_size,
            "max_length"  : cfg.max_length,
            "n_val"       : len(val_df),
            "n_test"      : len(test_df),
            "option_cols" : option_cols,
        },
        cfg       = cfg,
        model_tag = "transformer_embedding",
    )

    try:
        tr_val_path  = out / "transformer_val_scores.npy"
        tr_test_path = out / "transformer_test_scores.npy"

        # ── artifact-aware cache check ────────────────────────────────────────
        cached = _load_scores_with_artifact_fallback(
            local_val_path        = tr_val_path,
            local_test_path       = tr_test_path,
            artifact_val_filename = "transformer_val_scores.npy",
            artifact_test_filename= "transformer_test_scores.npy",
        )

        if cached is not None:
            transformer_val_scores, transformer_test_scores = cached
            logger.info("[M2] Transformer scores loaded from cache/artifact.")
            _wandb_log(tr_run, {"cache_hit": True})
        else:
            logger.info("[M2] Running TransformerEmbeddingRanker from scratch …")
            transformer_val_scores, transformer_test_scores = (
                _run_transformer_embeddings(cfg, val_df, test_df, tr_run)
            )
            np.save(tr_val_path,  transformer_val_scores)
            np.save(tr_test_path, transformer_test_scores)
            logger.info(f"[M2] Transformer scores saved → {out}")
            _wandb_log(tr_run, {"cache_hit": False})

        tr_metrics         = _compute_metrics(transformer_val_scores, val_df, option_cols)
        tr_required        = _build_required_metrics_payload(tr_metrics)
        tr_tagged          = {f"transformer_val/{k}": v for k, v in tr_metrics.items()}
        tr_required_tagged = {f"transformer_val/{k}": v for k, v in tr_required.items()}

        _wandb_log(tr_run, tr_required_tagged)

        results["metrics"].update(tr_tagged)
        results["transformer_val_scores"]  = transformer_val_scores
        results["transformer_test_scores"] = transformer_test_scores
        all_method_metrics["Transformer Embed"] = tr_metrics

        print(
            f"[M2] Transformer  MAP@3={tr_metrics['map_at_3']:.4f} "
            f"| Acc={tr_metrics['accuracy']:.4f} "
            f"| F1={tr_metrics['f1_macro']:.4f}"
        )
    finally:
        _wandb_finish(tr_run)

    # ─────────────────────────────────────────────────────────────────────────
    # RUN 3 : Phase-1 Baselines comparison
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  W&B RUN 3 – Phase-1 Baseline Comparison")
    logger.info("═" * 60)

    baseline_tag_map = {
        "tfidf_val": ("TF-IDF",   "tfidf"),
        "w2v_val"  : ("Word2Vec", "word2vec"),
        "sbert_val": ("SBERT",    "sbert"),
    }

    wandb_table_rows: List[list] = []

    for key, (display_tag, model_tag) in baseline_tag_map.items():
        if key not in baseline_scores:
            logger.warning(f"[baseline] {key} scores not available – skipping.")
            continue

        bl_run = _wandb_init(
            project     = wandb_project,
            run_name    = f"baseline_{model_tag}",
            config_dict = {
                "model_type"      : "phase1_baseline",
                "model_tag"       : model_tag,
                "baselines_found" : list(baseline_scores.keys()),
                "n_val"           : len(val_df),
                "option_cols"     : option_cols,
            },
            cfg       = cfg,
            model_tag = model_tag,
        )

        try:
            bm          = _compute_metrics(baseline_scores[key], val_df, option_cols)
            bm_required = _build_required_metrics_payload(bm)

            _wandb_log(bl_run, bm_required)

            tagged = {f"{display_tag}_val/{k}": v for k, v in bm.items()}
            results["metrics"].update(tagged)
            all_method_metrics[display_tag] = bm

            wandb_table_rows.append(
                [display_tag, bm["map_at_3"], bm["accuracy"], bm["f1_macro"]]
            )

            print(
                f"[M2] {display_tag:<10} MAP@3={bm['map_at_3']:.4f} "
                f"| Acc={bm['accuracy']:.4f} "
                f"| F1={bm['f1_macro']:.4f}"
            )
        finally:
            _wandb_finish(bl_run)

    # ── summary W&B Table ─────────────────────────────────────────────────────
    bl_summary_run = _wandb_init(
        project     = wandb_project,
        run_name    = "baseline_compare_summary",
        config_dict = {
            "model_type"      : "phase1_baselines_summary",
            "baselines_found" : list(baseline_scores.keys()),
            "n_val"           : len(val_df),
            "option_cols"     : option_cols,
        },
        cfg       = cfg,
        model_tag = "summary",
    )

    try:
        try:
            import wandb as _wandb
            for method, m in all_method_metrics.items():
                if method not in {t for t, _ in baseline_tag_map.values()}:
                    wandb_table_rows.append(
                        [method, m["map_at_3"], m["accuracy"], m["f1_macro"]]
                    )

            tbl = _wandb.Table(
                columns = ["Method", "MAP@3", "Accuracy", "F1_macro"],
                data    = wandb_table_rows,
            )
            if bl_summary_run is not None:
                bl_summary_run.log({"comparison_table": tbl})
                logger.info("[W&B] Comparison table logged.")
        except Exception as exc:
            logger.warning(f"[W&B] Could not log comparison table: {exc}")
    finally:
        _wandb_finish(bl_summary_run)

    # ── 6. Print final comparison table ──────────────────────────────────────
    _print_comparison_table(all_method_metrics)

    results["all_method_metrics"] = all_method_metrics
    return results
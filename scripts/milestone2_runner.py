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
# W&B helpers  (delegate to utils/wandb_init.py where possible)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from utils.wandb_init import (
        authenticate,           # ← the safe, non-interactive auth helper
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


# ─────────────────────────────────────────────────────────────────────────────
# One-time authentication
# ─────────────────────────────────────────────────────────────────────────────

def _do_wandb_auth(api_key: Optional[str] = None) -> None:
    """
    Authenticate with W&B exactly once before any run is created.

    Priority
    ────────
    1. api_key argument (caller-supplied)
    2. authenticate()  from utils/wandb_init  (Kaggle Secrets → env var → netrc)
    3. Inline env-var / netrc fallback (no interactive prompt)
    """
    if not _WANDB_UTILS_AVAILABLE:
        # Best-effort non-interactive login when the util module is absent
        if api_key:
            try:
                import wandb
                wandb.login(key=api_key, relogin=True)
                logger.info("[W&B] Logged in with provided API key (inline).")
            except Exception as exc:
                logger.warning(f"[W&B] Login failed: {exc}")
        else:
            try:
                import wandb
                wandb.login(anonymous="never", relogin=False)
            except Exception as exc:
                logger.warning(f"[W&B] Login failed: {exc}")
        return

    # ── utils/wandb_init is available ────────────────────────────────────────
    if api_key:
        try:
            import wandb
            wandb.login(key=api_key, relogin=True)
            logger.info("[W&B] Logged in with caller-supplied API key.")
            return
        except Exception as exc:
            logger.warning(
                f"[W&B] Login with supplied key failed ({exc}). "
                "Trying authenticate()."
            )

    # Kaggle Secrets → env var → netrc  (never interactive)
    authenticate()


# ─────────────────────────────────────────────────────────────────────────────
# Per-run W&B helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wandb_init(
    project    : str,
    run_name   : str,
    config_dict: dict,
    cfg        = None,
    model_tag  : str = "custom",
) -> object:
    """
    Start a W&B run.

    • model_tag in COMPARABLE_MODELS → delegates to init_wandb()
      (Kaggle-secret auth, group, job_type, tags — designed for phase-1 baselines)
    • everything else               → inline wandb.init() fallback
    Auth has already been performed by _do_wandb_auth(); no login() call here.
    """
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

    # ── inline fallback (auth already done) ──────────────────────────────────
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
    if run is None:
        return
    if _WANDB_UTILS_AVAILABLE:
        try:
            finish_run(run)
            return
        except Exception as exc:
            logger.warning(f"[W&B] finish_run() failed: {exc}")
    try:
        run.finish()
        logger.info("[W&B] Run finished.")
    except Exception as exc:
        logger.warning(f"[W&B] run.finish() failed: {exc}")


def _wandb_log(run, metrics: dict) -> None:
    if run is None:
        return
    if _WANDB_UTILS_AVAILABLE:
        try:
            log_model_metrics(run, metrics)
            return
        except Exception as exc:
            logger.warning(f"[W&B] log_model_metrics failed: {exc}")
    try:
        run.log(metrics)
    except Exception as exc:
        logger.warning(f"[W&B] run.log failed: {exc}")


def _build_required_metrics_payload(raw_metrics: dict) -> dict:
    """
    Translate _compute_metrics keys → REQUIRED_METRICS names.

        map_at_3  → map_at_k
        f1_macro  → f1_score
        (others passed through unchanged)

    Also seeds any still-missing REQUIRED_METRICS keys with None so that
    log_model_metrics never raises a missing-key warning for values we
    genuinely do not have.
    """
    mapping = {
        "map_at_3": "map_at_k",
        "f1_macro": "f1_score",
    }
    payload = {mapping.get(k, k): v for k, v in raw_metrics.items()}

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
    option_cols = cfg.options

    ARTIFACT_DIR = None
    try:
        from pathlib import Path
        candidate = Path("/kaggle/input/project-artifacts/outputs/processed_files")
        if candidate.exists():
            ARTIFACT_DIR = candidate
            logger.info(f"[artifact] Found processed files at {ARTIFACT_DIR}")
    except Exception:
        pass

    if val_df is None:
        processed_local    = cfg.output_dir / "train_processed.csv"
        processed_artifact = (
            ARTIFACT_DIR / "train_processed.csv" if ARTIFACT_DIR else None
        )
        raw = cfg.data_dir / cfg.train_file

        if processed_artifact and processed_artifact.exists():
            logger.info(f"[artifact] Loading val_df from {processed_artifact}")
            full   = pd.read_csv(processed_artifact)
            val_df = full.iloc[int(len(full) * 0.8):].reset_index(drop=True)
        elif processed_local.exists():
            logger.info(f"Loading val_df from {processed_local}")
            val_df = pd.read_csv(processed_local)
        elif raw.exists():
            logger.info(f"Loading val_df from raw {raw}")
            full   = pd.read_csv(raw)
            val_df = full.iloc[int(len(full) * 0.8):].reset_index(drop=True)
        else:
            raise FileNotFoundError(
                f"Cannot find validation data. "
                f"Tried artifact={processed_artifact}, "
                f"local={processed_local}, raw={raw}"
            )

    if test_df is None:
        processed_local    = cfg.output_dir / "test_processed.csv"
        processed_artifact = (
            ARTIFACT_DIR / "test_processed.csv" if ARTIFACT_DIR else None
        )
        raw = cfg.data_dir / cfg.test_file

        if processed_artifact and processed_artifact.exists():
            logger.info(f"[artifact] Loading test_df from {processed_artifact}")
            test_df = pd.read_csv(processed_artifact)
        elif processed_local.exists():
            logger.info(f"Loading test_df from {processed_local}")
            test_df = pd.read_csv(processed_local)
        elif raw.exists():
            logger.info(f"Loading test_df from raw {raw}")
            test_df = pd.read_csv(raw)
        else:
            logger.warning("No test file found – using val_df as test placeholder.")
            test_df = val_df.copy()

    val_df  = _ensure_clean_cols(val_df,  option_cols)
    test_df = _ensure_clean_cols(test_df, option_cols)

    return val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Artifact-aware baseline score loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_baseline_scores(out_dir, provided) -> Dict[str, np.ndarray]:
    if provided:
        return provided

    from pathlib import Path
    ARTIFACT_SCORE_DIR = Path("/kaggle/input/project-artifacts/outputs")

    scores  = {}
    mapping = {
        "tfidf_val": "tfidf_val_scores.npy",
        "w2v_val"  : "w2v_val_scores.npy",
        "sbert_val": "sbert_val_scores.npy",
    }

    for key, filename in mapping.items():
        local    = out_dir / filename
        artifact = ARTIFACT_SCORE_DIR / filename

        if local.exists():
            scores[key] = np.load(local)
            logger.info(f"[baseline] Loaded {key} from local {local}")
        elif artifact.exists():
            scores[key] = np.load(artifact)
            logger.info(f"[baseline] Loaded {key} from artifact {artifact}")
        else:
            logger.warning(
                f"[baseline] {filename} not found locally or in artifact."
            )

    return scores


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
    """

    # ── 0. Authenticate ONCE before any wandb.init call ───────────────────────
    # This is the only place login() is ever called; individual run helpers
    # must NOT call wandb.login() themselves to avoid interactive prompts.
    _do_wandb_auth(wandb_api_key)

    # ── 1. Config / evaluator defaults ───────────────────────────────────────
    if cfg is None:
        from config.config import Config
        cfg = Config()
        logger.info(f"[M2] device={cfg.device} | out={cfg.output_dir}")

    if evaluator is None:
        evaluator = _Evaluator()

    if option_cols is None:
        option_cols = cfg.options

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
            "model_type" : "zero_shot_nli",
            "model_key"  : cfg.zs_model_key,
            "batch_size" : cfg.batch_size,
            "n_val"      : len(val_df),
            "n_test"     : len(test_df),
            "option_cols": option_cols,
        },
        cfg       = cfg,
        model_tag = "zero_shot_nli",   # not in COMPARABLE_MODELS → inline path
    )

    try:
        zs_val_path  = out / "zs_val_scores.npy"
        zs_test_path = out / "zs_test_scores.npy"

        if _scores_exist(zs_val_path, zs_test_path):
            logger.info("[M2] Reusing cached zero-shot scores.")
            zs_val_scores  = np.load(zs_val_path)
            zs_test_scores = np.load(zs_test_path)
            _wandb_log(zs_run, {"cache_hit": True})
        else:
            logger.info("[M2] Running ZeroShotMCQRanker …")
            zs_val_scores, zs_test_scores = _run_zero_shot(
                cfg, val_df, test_df, zs_run
            )
            np.save(zs_val_path,  zs_val_scores)
            np.save(zs_test_path, zs_test_scores)
            logger.info(f"[M2] Zero-shot scores saved → {out}")
            _wandb_log(zs_run, {"cache_hit": False})

        zs_metrics         = _compute_metrics(zs_val_scores, val_df, option_cols)
        zs_required_tagged = {
            f"zeroshot_val/{k}": v
            for k, v in _build_required_metrics_payload(zs_metrics).items()
        }
        _wandb_log(zs_run, zs_required_tagged)

        results["metrics"].update(
            {f"zeroshot_val/{k}": v for k, v in zs_metrics.items()}
        )
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
            "model_type" : "transformer_embedding",
            "model_key"  : cfg.tr_model_key,
            "batch_size" : cfg.batch_size,
            "max_length" : cfg.max_length,
            "n_val"      : len(val_df),
            "n_test"     : len(test_df),
            "option_cols": option_cols,
        },
        cfg       = cfg,
        model_tag = "transformer_embedding",  # not in COMPARABLE_MODELS → inline
    )

    try:
        tr_val_path  = out / "transformer_val_scores.npy"
        tr_test_path = out / "transformer_test_scores.npy"

        if _scores_exist(tr_val_path, tr_test_path):
            logger.info("[M2] Reusing cached transformer scores.")
            transformer_val_scores  = np.load(tr_val_path)
            transformer_test_scores = np.load(tr_test_path)
            _wandb_log(tr_run, {"cache_hit": True})
        else:
            logger.info("[M2] Running TransformerEmbeddingRanker …")
            transformer_val_scores, transformer_test_scores = (
                _run_transformer_embeddings(cfg, val_df, test_df, tr_run)
            )
            np.save(tr_val_path,  transformer_val_scores)
            np.save(tr_test_path, transformer_test_scores)
            logger.info(f"[M2] Transformer scores saved → {out}")
            _wandb_log(tr_run, {"cache_hit": False})

        tr_metrics         = _compute_metrics(transformer_val_scores, val_df, option_cols)
        tr_required_tagged = {
            f"transformer_val/{k}": v
            for k, v in _build_required_metrics_payload(tr_metrics).items()
        }
        _wandb_log(tr_run, tr_required_tagged)

        results["metrics"].update(
            {f"transformer_val/{k}": v for k, v in tr_metrics.items()}
        )
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
    # RUN 3 : Phase-1 Baselines  (one W&B run per baseline model)
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
            model_tag = model_tag,   # in COMPARABLE_MODELS → uses init_wandb
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
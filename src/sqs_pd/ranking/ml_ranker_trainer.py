"""
Improved SQS Ranker using LambdaRank
=====================================

Directly optimizes ranking metrics (NDCG) for supercell recommendation.
Uses LightGBM Ranker with lambdarank objective.

Features:
- Group-based learning (per CIF)
- Relevance-based labels
- Direct NDCG optimization
- Comprehensive feature engineering

Usage:
  python scripts/train_sqs_ranker_improved.py

Outputs:
    - artifacts/models/ml_ranker/ranker_model.txt
    - artifacts/models/ml_ranker/ranker_report.json
    - artifacts/models/ml_ranker/feature_importance.csv
"""

from pathlib import Path
import shutil
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings("ignore")

from ..batching.pipeline_layout import resolve_project_path, DEFAULT_ML_DATASET_REL
from ..runtime.io_utils import ensure_dir, write_json_file

DEFAULT_TRAINED_MODEL_DIR = Path("artifacts/models/ml_ranker")
DEFAULT_PRODUCTION_MODEL_DIR = Path("artifacts/models/default_ml_ranker")


def promote_ranker_as_default(
    source_dir: str | Path = DEFAULT_TRAINED_MODEL_DIR,
    target_dir: str | Path = DEFAULT_PRODUCTION_MODEL_DIR,
) -> Path:
    """将训练完成的模型固化为默认生产模型。"""
    src = resolve_project_path(source_dir)
    dst = resolve_project_path(target_dir)

    required_files = [
        src / "ranker_model.txt",
        src / "ranker_inference_config.json",
        src / "ranker_report.json",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot promote model, required files missing: " + ", ".join(missing)
        )

    ensure_dir(dst)
    for file_name in [
        "ranker_model.txt",
        "ranker_inference_config.json",
        "ranker_report.json",
        "feature_importance.csv",
        "ranker_diagnostics.png",
        "per_cif_test_metrics.csv",
    ]:
        src_file = src / file_name
        if src_file.exists():
            shutil.copy2(src_file, dst / file_name)

    manifest = {
        "source_dir": str(src),
        "production_dir": str(dst),
        "model_file": str(dst / "ranker_model.txt"),
        "inference_config": str(dst / "ranker_inference_config.json"),
    }
    write_json_file(dst / "default_model_manifest.json", manifest)

    return dst


# Dynamic threshold for feature selection (based on CV average importance)
FEATURE_SELECTION_THRESHOLD = 10  # 'mean', 'median', or float value


class FeatureEngineer:
    """Feature engineering for supercell ranking - uses base features only"""

    def __init__(self):
        # Using selected default features from ml_dataset columns
        self.base_features = [
            "volume",  # 2161
            "sphericity",  # 1690
            # "face_sphericity",  # 848
            "lcmm",  # 1087
            "num_disordered_sites",  # 1650
            "valid_supercell_count",  # 1719
            "total_site_entropy",  # 2845
        ]

        self.feature_names = None

    def fit_transform(self, df):
        """Extract base features from dataframe"""
        X = df.copy()

        # Use only the explicitly defined base_features
        # (not all numeric columns - this ensures feature filtering is consistent)
        available_features = [f for f in self.base_features if f in X.columns]
        self.feature_names = available_features

        return X[available_features]


def select_features_cv(cv_rankers, feature_names):
    """Select features based on CV average importance"""
    # Collect importances from CV
    cv_importances = np.array([ranker.feature_importances_ for ranker in cv_rankers])
    avg_importance = np.mean(cv_importances, axis=0)

    # Determine threshold
    if FEATURE_SELECTION_THRESHOLD == "mean":
        threshold = np.mean(avg_importance)
    elif FEATURE_SELECTION_THRESHOLD == "median":
        threshold = np.median(avg_importance)
    else:
        threshold = FEATURE_SELECTION_THRESHOLD

    # Select features above threshold
    selected_mask = avg_importance >= threshold
    selected_features = [f for f, sel in zip(feature_names, selected_mask) if sel]
    selected_indices = [i for i, sel in enumerate(selected_mask) if sel]

    print(
        f"Selected {len(selected_features)} features above threshold ({threshold:.4f}) from CV average importance"
    )

    return selected_features, selected_indices


def _round_one_decimal_half_up(value: float) -> float:
    return float(
        Decimal(str(float(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    )


def _competition_ranks_ascending(values: np.ndarray) -> np.ndarray:
    """Competition ranks for ascending values, e.g. [1,1,3,4,4,6]."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int)

    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=int)

    i = 0
    while i < n:
        j = i + 1
        base_idx = int(order[i])
        base_val = float(arr[base_idx])
        while j < n and float(arr[int(order[j])]) == base_val:
            j += 1

        rank_value = i + 1
        for k in range(i, j):
            ranks[int(order[k])] = rank_value

        i = j

    return ranks


def create_ranking_labels(df):
    """Create relevance labels for ranking learning"""
    # Sort by CIF and objective (ascending, lower objective = better)
    df = df.sort_values(["cif_file", "objective"])

    # Tie rule: objective values that are equal after ROUND_HALF_UP to 1 decimal
    # are treated as the same rank level.
    df["objective_r1"] = df["objective"].astype(float).map(_round_one_decimal_half_up)

    # Competition ranking within each CIF (1,1,3,...) on rounded objective.
    comp_rank = (
        df.groupby("cif_file")["objective_r1"]
        .transform(
            lambda s: pd.Series(
                _competition_ranks_ascending(s.to_numpy()), index=s.index
            )
        )
        .astype(int)
    )
    df["competition_rank"] = comp_rank

    # Lower competition rank is better -> larger label/relevance.
    # Ties keep same rank and same label.
    max_rank_per_cif = df.groupby("cif_file")["competition_rank"].transform("max")
    df["label"] = (max_rank_per_cif - df["competition_rank"] + 1).astype(int)
    df["relevance"] = df["label"].astype(float)

    return df


def load_and_prepare_data(csv_path):
    """Load data and prepare for ranking"""
    df = pd.read_csv(csv_path)

    # Filter valid data
    df = df[df["success"] == True].copy()
    df = df[df["objective"].notna()].copy()

    # Create ranking labels
    df = create_ranking_labels(df)

    # Get group sizes (number of candidates per CIF)
    group_sizes = df.groupby("cif_file").size().values

    return df, group_sizes


def evaluate_ranking(true_objs, pred_scores, ks=(1, 3, 5)):
    """Evaluate ranking quality using standard IR/LTR metrics.

    Metrics families (4):
      - NDCG@k
      - MRR
      - HitRate@k
      - MAP@k
    """

    true_objs = np.asarray(true_objs, dtype=float)
    pred_scores = np.asarray(pred_scores, dtype=float)

    metrics: dict[str, float] = {}
    n = len(true_objs)

    if n == 0:
        metrics["mrr"] = np.nan
        for k in ks:
            metrics[f"ndcg@{k}"] = np.nan
            metrics[f"hit@{k}"] = np.nan
            metrics[f"map@{k}"] = np.nan
        return metrics

    rounded_objs = np.array(
        [_round_one_decimal_half_up(x) for x in true_objs], dtype=float
    )
    true_rank_comp = _competition_ranks_ascending(rounded_objs)

    # Ground-truth graded relevance from competition rank:
    # lower rank => higher relevance; rank gaps are preserved (1,1,3,5,...).
    relevance = (len(true_rank_comp) - true_rank_comp + 1).astype(float)

    top_relevance = float(np.max(relevance))
    top_relevant_set = {int(i) for i in np.where(relevance == top_relevance)[0]}

    # Predicted order (score descending)
    pred_order = np.argsort(-pred_scores)

    # ---- MRR (for tie-aware top set) ----
    first_hit_rank = None
    for rank_pos, idx in enumerate(pred_order, start=1):
        if int(idx) in top_relevant_set:
            first_hit_rank = rank_pos
            break
    metrics["mrr"] = 1.0 / float(first_hit_rank) if first_hit_rank else 0.0

    # ---- NDCG@k ----
    def dcg_at_k(rel: np.ndarray, order: np.ndarray, k: int) -> float:
        k = min(k, len(order))
        if k <= 0:
            return 0.0
        gains = 2 ** rel[order[:k]] - 1
        discounts = np.log2(np.arange(2, k + 2))
        return float(np.sum(gains / discounts))

    ideal_order = np.argsort(-relevance)

    # ---- HitRate@k and MAP@k ----
    for k in ks:
        k_eff = min(int(k), n)
        if k_eff <= 0:
            metrics[f"ndcg@{k}"] = 0.0
            metrics[f"hit@{k}"] = 0.0
            metrics[f"map@{k}"] = 0.0
            continue

        # NDCG@k
        dcg = dcg_at_k(relevance, pred_order, k_eff)
        idcg = dcg_at_k(relevance, ideal_order, k_eff)
        metrics[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

        # HitRate@k for tie-aware top set
        topk = pred_order[:k_eff]
        metrics[f"hit@{k}"] = (
            1.0 if set(int(x) for x in topk) & top_relevant_set else 0.0
        )

        # MAP@k: relevant items are candidates tied at best rounded objective.
        relevant = top_relevant_set
        hits = 0
        ap_sum = 0.0
        for rank_pos, idx in enumerate(topk, start=1):
            if int(idx) in relevant:
                hits += 1
                ap_sum += hits / rank_pos
        metrics[f"map@{k}"] = ap_sum / len(relevant) if relevant else 0.0

    return metrics


def _build_test_diagnostics(
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_cif_rows = []
    pair_rows = []
    regret_rows = []
    topk_rows = []
    topk_max_k = 10

    for cif, group in df_test.groupby("cif_file"):
        g = group.reset_index(drop=True)
        rounded_objs = np.array(
            [_round_one_decimal_half_up(x) for x in g["objective"].values],
            dtype=float,
        )

        pred_order = np.argsort(-g["pred"].values)

        true_rank = _competition_ranks_ascending(rounded_objs)
        pred_rank = np.empty(len(g), dtype=int)

        pred_rank[pred_order] = np.arange(1, len(g) + 1)

        metrics = evaluate_ranking(g["objective"].values, g["pred"].values)
        if len(g) > 0:
            best_value = float(np.min(rounded_objs))
            best_indices = [
                int(i)
                for i, value in enumerate(rounded_objs)
                if float(value) == best_value
            ]
            best_pred_rank = int(np.min(pred_rank[best_indices]))
        else:
            best_pred_rank = np.nan

        tie_top1_size = int(np.sum(true_rank == 1)) if len(g) > 0 else 0

        if len(g) > 0:
            pred_scores = g["pred"].to_numpy(dtype=float)
            pred_top_idx = int(pred_order[0])
            pred_top_obj = float(rounded_objs[pred_top_idx])
            objective_loss_r1 = float(pred_top_obj - best_value)
            if len(g) >= 2:
                confidence_margin = float(
                    pred_scores[int(pred_order[0])] - pred_scores[int(pred_order[1])]
                )
            else:
                confidence_margin = np.nan
            correct_top1 = int(pred_top_idx in set(best_indices))
            regret_rows.append(
                {
                    "cif_file": cif,
                    "num_candidates": int(len(g)),
                    "tie_top1_size": tie_top1_size,
                    "objective_loss_r1": objective_loss_r1,
                    "confidence_margin": confidence_margin,
                    "correct_top1": correct_top1,
                    "best_pred_rank": best_pred_rank,
                }
            )

        # Top-k per-CIF hit matrix (single-figure-friendly diagnostics)
        if len(g) > 0:
            top_relevant_set = {
                int(i)
                for i, value in enumerate(rounded_objs)
                if float(value) == best_value
            }
            topk_row = {
                "cif_file": cif,
                "best_pred_rank": best_pred_rank,
                "num_candidates": int(len(g)),
            }
            for k in range(1, topk_max_k + 1):
                k_eff = min(k, len(g))
                if k_eff <= 0:
                    topk_row[f"hit@{k}"] = np.nan
                else:
                    topk_pred = set(int(x) for x in pred_order[:k_eff])
                    topk_row[f"hit@{k}"] = (
                        1.0 if (topk_pred & top_relevant_set) else 0.0
                    )
            topk_rows.append(topk_row)

        per_cif_rows.append(
            {
                "cif_file": cif,
                "num_candidates": len(g),
                "tie_top1_size": tie_top1_size,
                "ndcg@1": metrics.get("ndcg@1", np.nan),
                "ndcg@3": metrics.get("ndcg@3", np.nan),
                "mrr": metrics.get("mrr", np.nan),
                "hit@1": metrics.get("hit@1", np.nan),
                "hit@3": metrics.get("hit@3", np.nan),
                "best_pred_rank": best_pred_rank,
            }
        )

        pair_rows.extend(
            {
                "cif_file": cif,
                "true_rank": int(tr),
                "pred_rank": int(pr),
            }
            for tr, pr in zip(true_rank, pred_rank)
        )

    topk_df = pd.DataFrame(topk_rows)
    if not topk_df.empty:
        topk_df = topk_df.sort_values(
            ["best_pred_rank", "num_candidates", "cif_file"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    return (
        pd.DataFrame(per_cif_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(regret_rows),
        topk_df,
    )


def _save_diagnostics_plot(
    out_path: Path,
    fi_df: pd.DataFrame,
    per_cif_df: pd.DataFrame,
    rank_pairs_df: pd.DataFrame,
    regret_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    cv_summary: dict | None = None,
    test_summary: dict | None = None,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: feature importance table (instead of bar chart)
    axs[0, 0].axis("off")
    if fi_df is not None and not fi_df.empty:
        fi_show = fi_df.head(12).reset_index(drop=True)
        fi_rows = [
            [str(i + 1), str(row["feature"]), f"{float(row['importance']):.1f}"]
            for i, row in fi_show.iterrows()
        ]
        table_fi = axs[0, 0].table(
            cellText=fi_rows,
            colLabels=["Rank", "Feature", "Importance"],
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table_fi.auto_set_font_size(False)
        table_fi.set_fontsize(8)
        table_fi.scale(1.0, 1.15)
        axs[0, 0].set_title(
            f"Top Feature Importances (Table)  |  Selected={len(fi_df)}",
            pad=8,
        )
    else:
        axs[0, 0].set_title("Top Feature Importances unavailable")

    # Top-right: CV/Test summary table (from console metrics)
    axs[0, 1].axis("off")
    metric_order = [
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "mrr",
        "hit@1",
        "hit@3",
        "hit@5",
        "map@1",
        "map@3",
        "map@5",
    ]

    def _fmt(value: float | None) -> str:
        if value is None:
            return "-"
        try:
            if np.isnan(float(value)):
                return "-"
            return f"{float(value):.4f}"
        except Exception:
            return "-"

    if cv_summary is not None and test_summary is not None:
        metric_rows = []
        for name in metric_order:
            cv_item = cv_summary.get(name, {}) if isinstance(cv_summary, dict) else {}
            cv_mean = cv_item.get("mean") if isinstance(cv_item, dict) else None
            cv_std = cv_item.get("std") if isinstance(cv_item, dict) else None
            cv_text = (
                f"{_fmt(cv_mean)} ± {_fmt(cv_std)}"
                if cv_mean is not None and cv_std is not None
                else "-"
            )
            metric_rows.append([name.upper(), cv_text, _fmt(test_summary.get(name))])

        table_metrics = axs[0, 1].table(
            cellText=metric_rows,
            colLabels=["Metric", "CV (mean ± std)", "Test"],
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table_metrics.auto_set_font_size(False)
        table_metrics.set_fontsize(8)
        table_metrics.scale(1.0, 1.12)
        axs[0, 1].set_title("CV/Test Summary (Table)", pad=8)
    else:
        axs[0, 1].set_title("CV/Test summary unavailable")

    if rank_pairs_df is not None and not rank_pairs_df.empty:
        x = rank_pairs_df["true_rank"].to_numpy(dtype=float)
        y = rank_pairs_df["pred_rank"].to_numpy(dtype=float)
        hb = axs[1, 0].hexbin(
            x,
            y,
            gridsize=45,
            mincnt=1,
            bins="log",
        )
        max_rank = int(max(float(np.nanmax(x)), float(np.nanmax(y))))
        axs[1, 0].plot([1, max_rank], [1, max_rank], linestyle="--", linewidth=1)
        axs[1, 0].set_title("Predicted Rank vs True Rank (Hexbin, log count)")
        axs[1, 0].set_xlabel("True Rank (competition rank; 1=best)")
        axs[1, 0].set_ylabel("Predicted Rank (1=best)")
        cbar = fig.colorbar(hb, ax=axs[1, 0], fraction=0.046, pad=0.04)
        cbar.set_label("log10(count)")

    # Per-CIF regret risk diagnostics (single-chart)
    if regret_df is not None and not regret_df.empty:
        loss_raw = regret_df["objective_loss_r1"].to_numpy(dtype=float)
        conf_margin = regret_df["confidence_margin"].to_numpy(dtype=float)
        correct = regret_df["correct_top1"].to_numpy(dtype=int)

        # Transform x-axis for better readability around zero-heavy regime.
        loss_nonneg = np.clip(loss_raw, 0.0, None)
        x = np.log10(1.0 + loss_nonneg)
        y = conf_margin

        finite_mask = np.isfinite(x) & np.isfinite(y)
        if np.any(finite_mask):
            hb = axs[1, 1].hexbin(
                x[finite_mask],
                y[finite_mask],
                gridsize=35,
                mincnt=1,
                bins="log",
                cmap="Greys",
                alpha=0.25,
            )
            cbar = fig.colorbar(hb, ax=axs[1, 1], fraction=0.046, pad=0.04)
            cbar.set_label("log10(count)")

        m_hit = (correct == 1) & finite_mask
        m_miss = (correct == 0) & finite_mask
        loss_threshold = 0.1
        x_threshold = float(np.log10(1.0 + loss_threshold))
        conf_finite = y[finite_mask]
        conf_threshold = (
            float(np.nanquantile(conf_finite, 0.75)) if len(conf_finite) > 0 else 0.5
        )

        m_miss_high_loss = m_miss & (loss_nonneg >= loss_threshold)
        m_miss_low_loss = m_miss & (loss_nonneg < loss_threshold)

        axs[1, 1].axvspan(-0.005, 0.02, color="#dbeafe", alpha=0.35, linewidth=0)

        axs[1, 1].scatter(
            x[m_hit],
            y[m_hit],
            s=14,
            alpha=0.35,
            label="Top1 hit",
            edgecolors="none",
        )
        axs[1, 1].scatter(
            x[m_miss_low_loss],
            y[m_miss_low_loss],
            s=18,
            alpha=0.6,
            label="Miss (low loss)",
            edgecolors="none",
        )
        axs[1, 1].scatter(
            x[m_miss_high_loss],
            y[m_miss_high_loss],
            s=28,
            alpha=0.85,
            label="Miss (high loss)",
            edgecolors="none",
        )

        axs[1, 1].axvline(x_threshold, linestyle="--", linewidth=1)
        axs[1, 1].axhline(conf_threshold, linestyle="--", linewidth=1)

        n_total = int(np.sum(finite_mask))
        n_miss = int(np.sum(m_miss))
        n_miss_high = int(np.sum(m_miss_high_loss))
        p_miss = (n_miss / n_total) if n_total > 0 else np.nan
        p_miss_high = (n_miss_high / n_total) if n_total > 0 else np.nan
        med_loss_miss = (
            float(np.nanmedian(loss_nonneg[m_miss])) if np.any(m_miss) else np.nan
        )
        med_conf_miss_high = (
            float(np.nanmedian(y[m_miss_high_loss]))
            if np.any(m_miss_high_loss)
            else np.nan
        )

        summary_text = (
            f"P(miss)={p_miss:.2f}\n"
            f"P(high-loss miss)={p_miss_high:.2f}\n"
            f"median(loss|miss)={med_loss_miss:.3f}\n"
            f"median(conf|high-loss miss)={med_conf_miss_high:.3f}"
        )
        axs[1, 1].text(
            0.98,
            0.98,
            summary_text,
            transform=axs[1, 1].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
        )

        axs[1, 1].set_title("Per-CIF Regret Risk: log10(1+loss) vs confidence")
        axs[1, 1].set_xlabel("log10(1 + objective loss)   (loss in rounded-0.1 bucket)")
        axs[1, 1].set_ylabel("Confidence margin (top1 - top2 score)")
        axs[1, 1].legend(frameon=False, loc="lower right")
    else:
        axs[1, 1].set_title("Regret diagnostics unavailable")
        axs[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _resolve_dataset_path(csv_path: str | Path | None) -> Path | None:
    """解析训练集路径（仅支持 artifacts 默认路径）。"""
    if csv_path is not None:
        p = resolve_project_path(csv_path)
        return p if p.exists() else None

    default_path = resolve_project_path(DEFAULT_ML_DATASET_REL)
    return default_path if default_path.exists() else None


def train_ranker(
    csv_path: str | Path | None = None,
    out_dir: str | Path = "artifacts/models/ml_ranker",
    set_as_default: bool = False,
    default_model_dir: str | Path = DEFAULT_PRODUCTION_MODEL_DIR,
):
    """Main training function (core reusable API)"""
    resolved_csv_path = _resolve_dataset_path(csv_path)
    if resolved_csv_path is None:
        print(
            "ML dataset not found. Run batch dataset generation first (e.g. run_sqs_all_candidates)."
        )
        return

    # Load and prepare data
    df, group_sizes = load_and_prepare_data(resolved_csv_path)
    print(f"Loaded {len(df)} candidates from {len(np.unique(df['cif_file']))} CIFs")

    # Analyze supercell specification distribution
    print("\nSupercell Specifications Distribution:")
    supercell_counts = df.groupby("cif_file").size()
    print(f"  Min supercells per CIF: {supercell_counts.min()}")
    print(f"  Max supercells per CIF: {supercell_counts.max()}")
    print(f"  Mean supercells per CIF: {supercell_counts.mean():.1f}")
    print(f"  Median supercells per CIF: {supercell_counts.median():.1f}")

    # Distribution by exact counts (not grouped by ranges)
    print("  Per-count distribution:")
    count_distribution = supercell_counts.value_counts().sort_index()
    for num_supercells, num_cifs in count_distribution.items():
        pct = (num_cifs / len(supercell_counts)) * 100
        print(f"    {num_supercells:3d} supercells: {num_cifs:3d} CIFs ({pct:5.1f}%)")

    # Feature engineering
    engineer = FeatureEngineer()
    X_full = engineer.fit_transform(df)
    feature_names = engineer.feature_names
    print(f"Created {len(feature_names)} features")

    # Split data (hold out 25% CIFs for testing)
    unique_cifs = np.unique(df["cif_file"])
    rng = np.random.default_rng(42)
    n_test = max(1, int(0.25 * len(unique_cifs)))
    test_cifs = rng.choice(unique_cifs, size=n_test, replace=False)

    is_test = df["cif_file"].isin(test_cifs)
    df_trainval = df[~is_test].reset_index(drop=True)
    df_test = df[is_test].reset_index(drop=True)

    # Prepare features and labels
    X_trainval = X_full[~is_test]
    X_test = X_full[is_test]
    y_trainval = df_trainval["label"].values
    y_test = df_test["label"].values

    # Label gain for ranking (required for integer labels beyond default range)
    max_label = int(max(y_trainval.max(), y_test.max()))
    label_gain = list(range(max_label + 1))

    # Group sizes for trainval and test
    group_trainval = df_trainval.groupby("cif_file").size().values
    group_test = df_test.groupby("cif_file").size().values

    print(
        f"Train/Val CIFs: {len(np.unique(df_trainval['cif_file']))}, Test CIFs: {len(test_cifs)}"
    )

    # Cross-validation
    gkf = GroupKFold(n_splits=5)
    cv_results = []
    cv_rankers = []  # Store rankers for feature selection

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X_trainval, y_trainval, df_trainval["cif_file"])
    ):
        print(f"\nFold {fold + 1}/5")

        X_train = X_trainval.iloc[train_idx].to_numpy(dtype=float)
        X_val = X_trainval.iloc[val_idx].to_numpy(dtype=float)
        y_train = y_trainval[train_idx]
        y_val = y_trainval[val_idx]

        # Group sizes for this fold
        train_groups = df_trainval.iloc[train_idx].groupby("cif_file").size().values
        val_groups = df_trainval.iloc[val_idx].groupby("cif_file").size().values

        # Train ranker
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[1, 3, 5],
            n_estimators=300,
            learning_rate=0.05,
            label_gain=label_gain,
            random_state=42,
            verbosity=-1,
        )

        ranker.fit(
            X_train,
            y_train,
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
        )

        # Evaluate on validation
        y_pred = ranker.predict(X_val)
        val_df = df_trainval.iloc[val_idx].reset_index(drop=True)
        val_df["pred"] = y_pred

        fold_metrics = {"fold": fold + 1}
        for cif, group in val_df.groupby("cif_file"):
            metrics = evaluate_ranking(group["objective"].values, group["pred"].values)
            for k, v in metrics.items():
                if k not in fold_metrics:
                    fold_metrics[k] = []
                fold_metrics[k].append(v)

        # Average metrics
        for k in [
            "ndcg@1",
            "ndcg@3",
            "ndcg@5",
            "mrr",
            "hit@1",
            "hit@3",
            "hit@5",
            "map@1",
            "map@3",
            "map@5",
        ]:
            fold_metrics[k] = float(np.nanmean(fold_metrics[k]))

        cv_results.append(fold_metrics)
        cv_rankers.append(ranker)  # Save for feature selection
        print(
            f"  NDCG@1: {fold_metrics['ndcg@1']:.4f}, NDCG@3: {fold_metrics['ndcg@3']:.4f}"
        )

    # CV summary
    cv_summary = {}
    for k in [
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "mrr",
        "hit@1",
        "hit@3",
        "hit@5",
        "map@1",
        "map@3",
        "map@5",
    ]:
        vals = [r[k] for r in cv_results]
        cv_summary[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    print("\nCV Summary:")
    for k, v in cv_summary.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Feature selection based on CV average importance
    selected_features, selected_indices = select_features_cv(cv_rankers, feature_names)

    # Prepare selected feature matrices
    X_trainval_selected = X_trainval.iloc[:, selected_indices].to_numpy(dtype=float)
    X_test_selected = X_test.iloc[:, selected_indices].to_numpy(dtype=float)

    # Train final model on selected features
    final_ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3, 5],
        n_estimators=400,
        learning_rate=0.05,
        label_gain=label_gain,
        random_state=42,
        verbosity=-1,
    )

    final_ranker.fit(
        X_trainval_selected,
        y_trainval,
        group=group_trainval,
        eval_set=[(X_test_selected, y_test)],
        eval_group=[group_test],
    )

    # Evaluate on test set
    y_test_pred = final_ranker.predict(X_test_selected)
    df_test["pred"] = y_test_pred

    test_metrics = {
        "ndcg@1": [],
        "ndcg@3": [],
        "ndcg@5": [],
        "mrr": [],
        "hit@1": [],
        "hit@3": [],
        "hit@5": [],
        "map@1": [],
        "map@3": [],
        "map@5": [],
    }
    for cif, group in df_test.groupby("cif_file"):
        metrics = evaluate_ranking(group["objective"].values, group["pred"].values)
        for k, v in metrics.items():
            test_metrics[k].append(v)

    test_summary = {k: float(np.nanmean(v)) for k, v in test_metrics.items()}

    per_cif_df, rank_pairs_df, regret_df, topk_df = _build_test_diagnostics(df_test)

    print("\nTest Results:")
    print(f"  NDCG@1: {test_summary['ndcg@1']:.4f}")
    print(f"  NDCG@3: {test_summary['ndcg@3']:.4f}")
    print(f"  NDCG@5: {test_summary['ndcg@5']:.4f}")
    print(f"  MRR: {test_summary['mrr']:.4f}")
    print(f"  Hit@1: {test_summary['hit@1']:.4f}")
    print(f"  Hit@3: {test_summary['hit@3']:.4f}")
    print(f"  Hit@5: {test_summary['hit@5']:.4f}")
    print(f"  MAP@1: {test_summary['map@1']:.4f}")
    print(f"  MAP@3: {test_summary['map@3']:.4f}")
    print(f"  MAP@5: {test_summary['map@5']:.4f}")

    # Save model and results
    out_dir = resolve_project_path(out_dir)
    ensure_dir(out_dir)

    final_ranker.booster_.save_model(str(out_dir / "ranker_model.txt"))

    fi_df = pd.DataFrame(
        {"feature": selected_features, "importance": final_ranker.feature_importances_}
    ).sort_values("importance", ascending=False)
    fi_df.to_csv(out_dir / "feature_importance.csv", index=False)

    inference_cfg = {
        "selected_features": selected_features,
    }
    write_json_file(out_dir / "ranker_inference_config.json", inference_cfg)

    per_cif_df.to_csv(out_dir / "per_cif_test_metrics.csv", index=False)

    diag_plot_file = out_dir / "ranker_diagnostics.png"
    _save_diagnostics_plot(
        diag_plot_file,
        fi_df,
        per_cif_df,
        rank_pairs_df,
        regret_df,
        topk_df,
        cv_summary,
        test_summary,
    )

    # Prepare results container
    results = {
        "cv_results": cv_results,
        "cv_summary": cv_summary,
        "test_results": test_summary,
        "feature_importance": fi_df.to_dict("records"),
        "selected_features": selected_features,
        "num_selected_features": len(selected_features),
        "diagnostic_plot": str(diag_plot_file),
        "per_cif_metrics_file": str(out_dir / "per_cif_test_metrics.csv"),
    }

    write_json_file(out_dir / "ranker_report.json", results)

    print(f"\nModel and results saved to {out_dir}")

    if set_as_default:
        promoted_dir = promote_ranker_as_default(
            source_dir=out_dir,
            target_dir=default_model_dir,
        )
        print(f"Default production model updated: {promoted_dir}")

    return out_dir


def main(argv=None):
    """CLI entry for core ranker trainer."""
    import argparse

    parser = argparse.ArgumentParser(description="Train SQS LambdaRank model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to ml_dataset.csv (optional; auto-discover if omitted)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/models/ml_ranker",
        help="Directory to save model and report",
    )
    parser.add_argument(
        "--set-default",
        action="store_true",
        help="Promote this trained model as default production ranker",
    )
    parser.add_argument(
        "--default-model-dir",
        type=str,
        default=str(DEFAULT_PRODUCTION_MODEL_DIR),
        help="Target directory for default production model",
    )
    args = parser.parse_args(argv)
    train_ranker(
        csv_path=args.dataset,
        out_dir=args.out_dir,
        set_as_default=args.set_default,
        default_model_dir=args.default_model_dir,
    )


if __name__ == "__main__":
    main()

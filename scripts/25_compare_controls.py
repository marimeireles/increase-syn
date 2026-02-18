#!/usr/bin/env python
"""
Three-way comparison: Base vs Metacog-FT vs Random-Control.

Disentangles metacognition-specific synergy changes from fine-tuning artifacts.

Generates:
1. Three-way overlaid PhiID profiles
2. Delta heatmap: random-control - base (fine-tuning artifact)
3. Delta heatmap: metacog-FT - random-control (metacognition-specific)
4. Three-way overlaid ablation curves
5. Statistical tests for all three pairwise comparisons
"""

import json
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging, get_result_path

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
FIG_DIR = os.path.join("results", "rm-confounding-factors", "figures")

MODEL_IDS = {
    "base": "gemma3_4b_it",
    "metacog": "gemma3_4b_it_ft",
    "random_ctrl": "gemma3_4b_it_random_ctrl",
}

LABELS = {
    "base": "Gemma 3 4B-IT (base)",
    "metacog": "Gemma 3 4B-IT (metacog FT)",
    "random_ctrl": "Gemma 3 4B-IT (random-confidence FT)",
}

COLORS = {
    "base": "#1f77b4",       # blue
    "metacog": "#d62728",    # red
    "random_ctrl": "#2ca02c",  # green
}

MARKERS = {
    "base": "o",
    "metacog": "s",
    "random_ctrl": "^",
}


def load_rankings(model_id):
    """Load head rankings CSV for a model."""
    path = get_result_path(RESULTS_DIR, "phiid_scores", model_id, "head_rankings.csv")
    if not os.path.exists(path):
        logger.error(f"Rankings not found: {path}")
        return None
    return pd.read_csv(path)


def load_ablation(model_id):
    """Load ablation results CSV for a model."""
    path = get_result_path(RESULTS_DIR, "ablation", model_id, "ablation.csv")
    if not os.path.exists(path):
        logger.warning(f"Ablation results not found: {path}")
        return None
    return pd.read_csv(path)


def compute_pairwise_stats(df_a, df_b, label_a, label_b):
    """Compute statistical comparison between two models."""
    results = {}

    # Per-layer means
    layer_a = df_a.groupby('layer')['syn_red_score'].mean().values
    layer_b = df_b.groupby('layer')['syn_red_score'].mean().values

    # Paired t-test on layer means
    t_stat, p_value = stats.ttest_rel(layer_a, layer_b)
    results['paired_ttest'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
    }

    # Cohen's d
    diff = layer_b - layer_a
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
    results['cohens_d'] = float(cohens_d)

    # Spearman correlation between head rankings
    scores_a = df_a.sort_values('head_idx')['syn_red_score'].values
    scores_b = df_b.sort_values('head_idx')['syn_red_score'].values
    rho, rho_p = stats.spearmanr(scores_a, scores_b)
    results['spearman'] = {'rho': float(rho), 'p_value': float(rho_p)}

    # Mean absolute difference per head
    mean_abs_diff = float(np.mean(np.abs(scores_b - scores_a)))
    results['mean_abs_head_diff'] = mean_abs_diff

    # Per-layer differences
    layer_diffs = layer_b - layer_a
    results['per_layer_diffs'] = {
        'mean': float(np.mean(layer_diffs)),
        'std': float(np.std(layer_diffs)),
        'max': float(np.max(np.abs(layer_diffs))),
    }

    results['comparison'] = f"{label_a} vs {label_b}"

    return results


def plot_three_way_profiles(dfs, save_path):
    """Overlaid PhiID profiles for all three models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for key in ["base", "metacog", "random_ctrl"]:
        df = dfs[key]
        layer_means = df.groupby('layer')['syn_red_score'].mean()
        num_layers = len(layer_means)
        x = np.arange(num_layers) / (num_layers - 1)
        ax.plot(
            x, layer_means.values,
            f'{MARKERS[key]}-', color=COLORS[key],
            linewidth=2, markersize=6, label=LABELS[key], alpha=0.9,
        )

    ax.set_xlabel('Normalized Layer Depth', fontsize=12)
    ax.set_ylabel('Average Syn-Red Score', fontsize=12)
    ax.set_title(
        'Synergy Profile: Base vs Metacog-FT vs Random-Confidence-FT',
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved three-way profiles to {save_path}")


def plot_delta_heatmap(base_df, other_df, num_layers, num_heads_per_layer,
                       title, save_path):
    """Heatmap of per-head syn_red_score change."""
    base_grid = np.zeros((num_layers, num_heads_per_layer))
    other_grid = np.zeros((num_layers, num_heads_per_layer))

    for _, row in base_df.iterrows():
        base_grid[int(row['layer']), int(row['head_in_layer'])] = row['syn_red_score']
    for _, row in other_df.iterrows():
        other_grid[int(row['layer']), int(row['head_in_layer'])] = row['syn_red_score']

    delta = other_grid - base_grid

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(abs(delta.min()), abs(delta.max()))
    if vmax < 1e-10:
        vmax = 0.1  # avoid zero range
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    im = ax.imshow(delta, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(num_heads_per_layer))
    ax.set_yticks(range(num_layers))
    fig.colorbar(im, ax=ax, label='Delta Syn-Red Score')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved delta heatmap to {save_path}")


def plot_three_way_ablation(ablations, save_path):
    """Overlaid ablation curves for all three models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for key in ["base", "metacog", "random_ctrl"]:
        abl_df = ablations[key]
        if abl_df is None:
            continue

        total_heads = abl_df['num_heads_removed'].max()

        # Synergistic order
        syn_data = abl_df[abl_df['order_type'] == 'syn_red']
        if not syn_data.empty:
            x = syn_data['num_heads_removed'].values / total_heads
            y = syn_data['mean_kl_div'].values
            ax.plot(
                x, y, '-', color=COLORS[key], linewidth=2,
                label=f'{LABELS[key]} (syn)', alpha=0.9,
            )

        # Random order
        random_data = abl_df[abl_df['order_type'].str.startswith('random')]
        if not random_data.empty:
            grouped = random_data.groupby('num_heads_removed')['mean_kl_div']
            x = np.array(
                sorted(random_data['num_heads_removed'].unique())
            ) / total_heads
            y_mean = grouped.mean().values
            ax.plot(
                x, y_mean, '--', color=COLORS[key], linewidth=1.5,
                label=f'{LABELS[key]} (random)', alpha=0.5,
            )

    ax.set_xlabel('Fraction of Heads Deactivated', fontsize=12)
    ax.set_ylabel('Behaviour Divergence (KL)', fontsize=12)
    ax.set_title('Ablation: Base vs Metacog-FT vs Random-Confidence-FT', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved three-way ablation to {save_path}")


def main():
    setup_logging()
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load rankings for all three models
    dfs = {}
    for key, model_id in MODEL_IDS.items():
        df = load_rankings(model_id)
        if df is None:
            logger.error(
                f"Cannot load rankings for {key} ({model_id}). "
                "Run the synergy pipeline first."
            )
            sys.exit(1)
        dfs[key] = df

    num_layers = dfs["base"]['layer'].nunique()
    num_heads_per_layer = dfs["base"]['head_in_layer'].nunique()

    # 1. Three-way overlaid profiles
    plot_three_way_profiles(
        dfs,
        save_path=os.path.join(FIG_DIR, "three_way_overlaid_profiles.png"),
    )

    # 2. Delta heatmap: random-ctrl - base (fine-tuning artifact)
    plot_delta_heatmap(
        dfs["base"], dfs["random_ctrl"], num_layers, num_heads_per_layer,
        title="Fine-tuning Effect (Random-Ctrl - Base)",
        save_path=os.path.join(FIG_DIR, "ctrl_delta_heatmap.png"),
    )

    # 3. Delta heatmap: metacog - random-ctrl (metacognition-specific)
    plot_delta_heatmap(
        dfs["random_ctrl"], dfs["metacog"], num_layers, num_heads_per_layer,
        title="Metacognition-Specific Effect (Metacog - Random-Ctrl)",
        save_path=os.path.join(FIG_DIR, "metacog_vs_ctrl_delta_heatmap.png"),
    )

    # 4. Ablation comparison
    ablations = {}
    for key, model_id in MODEL_IDS.items():
        ablations[key] = load_ablation(model_id)

    if all(a is not None for a in ablations.values()):
        plot_three_way_ablation(
            ablations,
            save_path=os.path.join(FIG_DIR, "three_way_ablation.png"),
        )
    else:
        logger.warning("Skipping ablation comparison — not all ablation results available")

    # 5. Statistical tests for all three pairwise comparisons
    logger.info("=" * 60)
    logger.info("STATISTICAL COMPARISONS")
    logger.info("=" * 60)

    comparisons = [
        ("base", "random_ctrl", "Base vs Random-Ctrl (fine-tuning effect)"),
        ("base", "metacog", "Base vs Metacog-FT (total effect)"),
        ("random_ctrl", "metacog", "Random-Ctrl vs Metacog-FT (metacognition effect)"),
    ]

    all_stats = {}
    for key_a, key_b, desc in comparisons:
        logger.info(f"\n--- {desc} ---")
        pair_stats = compute_pairwise_stats(
            dfs[key_a], dfs[key_b], LABELS[key_a], LABELS[key_b],
        )
        all_stats[f"{key_a}_vs_{key_b}"] = pair_stats

        t = pair_stats['paired_ttest']['t_statistic']
        p = pair_stats['paired_ttest']['p_value']
        d = pair_stats['cohens_d']
        rho = pair_stats['spearman']['rho']
        rho_p = pair_stats['spearman']['p_value']
        mad = pair_stats['mean_abs_head_diff']

        logger.info(f"  Paired t-test: t={t:.4f}, p={p:.6f}")
        logger.info(f"  Cohen's d: {d:.4f}")
        logger.info(f"  Spearman rho: {rho:.4f} (p={rho_p:.6f})")
        logger.info(f"  Mean abs head diff: {mad:.4f}")
        logger.info(
            f"  Per-layer diff: mean={pair_stats['per_layer_diffs']['mean']:.4f}, "
            f"std={pair_stats['per_layer_diffs']['std']:.4f}, "
            f"max={pair_stats['per_layer_diffs']['max']:.4f}"
        )

        # Interpret
        if p < 0.05 and abs(d) > 0.2:
            logger.info(f"  RESULT: Significant difference (p={p:.4f}, d={d:.4f})")
        elif p < 0.05:
            logger.info(
                f"  RESULT: Statistically significant but small effect "
                f"(p={p:.4f}, d={d:.4f})"
            )
        else:
            logger.info(f"  RESULT: No significant difference (p={p:.4f}, d={d:.4f})")

    # Key interpretation
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)

    ft_p = all_stats["base_vs_random_ctrl"]["paired_ttest"]["p_value"]
    mc_p = all_stats["random_ctrl_vs_metacog"]["paired_ttest"]["p_value"]
    ft_d = abs(all_stats["base_vs_random_ctrl"]["cohens_d"])
    mc_d = abs(all_stats["random_ctrl_vs_metacog"]["cohens_d"])

    if ft_p < 0.05 and ft_d > 0.2:
        logger.info(
            "Fine-tuning ITSELF causes significant synergy changes "
            f"(p={ft_p:.4f}, d={ft_d:.4f}). "
            "Some observed effects are fine-tuning artifacts."
        )
    else:
        logger.info(
            "Fine-tuning alone does NOT cause significant synergy changes "
            f"(p={ft_p:.4f}, d={ft_d:.4f}). "
            "Observed effects are likely metacognition-specific."
        )

    if mc_p < 0.05 and mc_d > 0.2:
        logger.info(
            "Metacognition training causes ADDITIONAL synergy changes beyond "
            f"fine-tuning (p={mc_p:.4f}, d={mc_d:.4f}). "
            "Calibrated confidence specifically reshapes the synergistic core."
        )
    else:
        logger.info(
            "No significant metacognition-SPECIFIC synergy changes detected "
            f"(p={mc_p:.4f}, d={mc_d:.4f}). "
            "Synergy changes may be primarily a fine-tuning artifact."
        )

    # Save all statistics
    stats_path = os.path.join(FIG_DIR, "control_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"\nSaved statistics to {stats_path}")

    logger.info("Three-way comparison complete.")


if __name__ == "__main__":
    main()

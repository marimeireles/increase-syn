#!/usr/bin/env python
"""
Phase F6: Compare synergy profiles between base IT and fine-tuned IT models.

Generates:
1. Overlaid PhiID profiles (layer depth vs syn-red score)
2. Delta heatmap (per-head change in syn_red_score)
3. Overlaid ablation curves
4. Statistical tests (paired t-test, Spearman correlation, Cohen's d)
"""

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
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


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


def plot_overlaid_profiles(base_df, ft_df, save_path):
    """Overlaid PhiID profiles: base IT vs fine-tuned IT."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for df, label, color, marker in [
        (base_df, 'Gemma 3 4B-IT (base)', '#1f77b4', 'o'),
        (ft_df, 'Gemma 3 4B-IT (fine-tuned)', '#d62728', 's'),
    ]:
        layer_means = df.groupby('layer')['syn_red_score'].mean()
        num_layers = len(layer_means)
        x = np.arange(num_layers) / (num_layers - 1)
        ax.plot(x, layer_means.values, f'{marker}-', color=color,
                linewidth=2, markersize=6, label=label, alpha=0.9)

    ax.set_xlabel('Normalized Layer Depth', fontsize=12)
    ax.set_ylabel('Average Syn-Red Score', fontsize=12)
    ax.set_title('Synergy Profile: Base vs Fine-tuned Gemma 3 4B-IT', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved overlaid profiles to {save_path}")


def plot_delta_heatmap(base_df, ft_df, num_layers, num_heads_per_layer, save_path):
    """Heatmap of per-head syn_red_score change (fine-tuned - base)."""
    base_grid = np.zeros((num_layers, num_heads_per_layer))
    ft_grid = np.zeros((num_layers, num_heads_per_layer))

    for _, row in base_df.iterrows():
        base_grid[int(row['layer']), int(row['head_in_layer'])] = row['syn_red_score']
    for _, row in ft_df.iterrows():
        ft_grid[int(row['layer']), int(row['head_in_layer'])] = row['syn_red_score']

    delta = ft_grid - base_grid

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(abs(delta.min()), abs(delta.max()))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    im = ax.imshow(delta, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Change in Syn-Red Score (Fine-tuned - Base)', fontsize=14)
    ax.set_xticks(range(num_heads_per_layer))
    ax.set_yticks(range(num_layers))
    fig.colorbar(im, ax=ax, label='Delta Syn-Red Score')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved delta heatmap to {save_path}")


def plot_overlaid_ablation(base_abl, ft_abl, save_path):
    """Overlaid ablation curves for both models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for abl_df, label, color in [
        (base_abl, 'Base IT', '#1f77b4'),
        (ft_abl, 'Fine-tuned IT', '#d62728'),
    ]:
        total_heads = abl_df['num_heads_removed'].max()

        # Synergistic order
        syn_data = abl_df[abl_df['order_type'] == 'syn_red']
        if not syn_data.empty:
            x = syn_data['num_heads_removed'].values / total_heads
            y = syn_data['mean_kl_div'].values
            ax.plot(x, y, '-', color=color, linewidth=2,
                    label=f'{label} (syn)', alpha=0.9)

        # Random order
        random_data = abl_df[abl_df['order_type'].str.startswith('random')]
        if not random_data.empty:
            grouped = random_data.groupby('num_heads_removed')['mean_kl_div']
            x = np.array(sorted(random_data['num_heads_removed'].unique())) / total_heads
            y_mean = grouped.mean().values
            ax.plot(x, y_mean, '--', color=color, linewidth=1.5,
                    label=f'{label} (random)', alpha=0.6)

    ax.set_xlabel('Fraction of Heads Deactivated', fontsize=12)
    ax.set_ylabel('Behaviour Divergence (KL)', fontsize=12)
    ax.set_title('Ablation: Base vs Fine-tuned Gemma 3 4B-IT', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved overlaid ablation to {save_path}")


def compute_statistics(base_df, ft_df):
    """Compute statistical comparisons between base and fine-tuned profiles."""
    results = {}

    # Per-layer means
    base_layer = base_df.groupby('layer')['syn_red_score'].mean().values
    ft_layer = ft_df.groupby('layer')['syn_red_score'].mean().values

    # Paired t-test on layer means
    t_stat, p_value = stats.ttest_rel(base_layer, ft_layer)
    results['paired_ttest'] = {'t_statistic': float(t_stat), 'p_value': float(p_value)}

    # Cohen's d (effect size)
    diff = ft_layer - base_layer
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
    results['cohens_d'] = float(cohens_d)

    # Spearman correlation between head rankings
    base_scores = base_df.sort_values('head_idx')['syn_red_score'].values
    ft_scores = ft_df.sort_values('head_idx')['syn_red_score'].values
    rho, rho_p = stats.spearmanr(base_scores, ft_scores)
    results['spearman'] = {'rho': float(rho), 'p_value': float(rho_p)}

    # Mean absolute difference per head
    mean_abs_diff = np.mean(np.abs(ft_scores - base_scores))
    results['mean_abs_head_diff'] = float(mean_abs_diff)

    # Per-layer mean differences
    layer_diffs = ft_layer - base_layer
    results['per_layer_diffs'] = {
        'mean': float(np.mean(layer_diffs)),
        'std': float(np.std(layer_diffs)),
        'max': float(np.max(np.abs(layer_diffs))),
    }

    return results


def main():
    setup_logging()
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load data
    base_df = load_rankings("gemma3_4b_it")
    ft_df = load_rankings("gemma3_4b_it_ft")

    if base_df is None or ft_df is None:
        logger.error("Cannot compare: missing rankings data. Run synergy pipelines first.")
        sys.exit(1)

    num_layers = base_df['layer'].nunique()
    num_heads_per_layer = base_df['head_in_layer'].nunique()

    # 1. Overlaid profiles
    plot_overlaid_profiles(
        base_df, ft_df,
        save_path=os.path.join(FIG_DIR, "metacog_overlaid_profiles.png"),
    )

    # 2. Delta heatmap
    plot_delta_heatmap(
        base_df, ft_df, num_layers, num_heads_per_layer,
        save_path=os.path.join(FIG_DIR, "metacog_delta_heatmap.png"),
    )

    # 3. Ablation comparison
    base_abl = load_ablation("gemma3_4b_it")
    ft_abl = load_ablation("gemma3_4b_it_ft")
    if base_abl is not None and ft_abl is not None:
        plot_overlaid_ablation(
            base_abl, ft_abl,
            save_path=os.path.join(FIG_DIR, "metacog_overlaid_ablation.png"),
        )

    # 4. Statistics
    logger.info("=" * 60)
    logger.info("STATISTICAL COMPARISON")
    logger.info("=" * 60)

    stat_results = compute_statistics(base_df, ft_df)

    logger.info(f"Paired t-test (layer means): t={stat_results['paired_ttest']['t_statistic']:.4f}, "
                f"p={stat_results['paired_ttest']['p_value']:.6f}")
    logger.info(f"Cohen's d (effect size): {stat_results['cohens_d']:.4f}")
    logger.info(f"Spearman rank correlation: rho={stat_results['spearman']['rho']:.4f}, "
                f"p={stat_results['spearman']['p_value']:.6f}")
    logger.info(f"Mean absolute head-level difference: {stat_results['mean_abs_head_diff']:.4f}")
    logger.info(f"Per-layer diff: mean={stat_results['per_layer_diffs']['mean']:.4f}, "
                f"std={stat_results['per_layer_diffs']['std']:.4f}, "
                f"max={stat_results['per_layer_diffs']['max']:.4f}")

    # Interpret
    p = stat_results['paired_ttest']['p_value']
    d = abs(stat_results['cohens_d'])
    if p < 0.05 and d > 0.2:
        logger.info("RESULT: Significant change in synergy profile after fine-tuning "
                     f"(p={p:.4f}, d={d:.4f})")
    elif p < 0.05:
        logger.info("RESULT: Statistically significant but small effect "
                     f"(p={p:.4f}, d={d:.4f})")
    else:
        logger.info("RESULT: No significant change in synergy profile "
                     f"(p={p:.4f}, d={d:.4f})")

    # Save statistics
    import json
    stats_path = os.path.join(FIG_DIR, "metacog_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stat_results, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")

    logger.info("Phase F6 complete.")


if __name__ == "__main__":
    main()

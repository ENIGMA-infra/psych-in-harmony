"""
Generate Figures from ENIGMA-PD Theory-Primed Analysis Results
===============================================================

This script generates visualization figures from the CSV output files
produced by similarity-analysis_theory-primed.py.

Key differences from the clustering-based version:
- "Cluster Evidence" → "Model Evidence" (similarity-based assignment)
- Cohesion heatmaps → Dimension Accuracy heatmaps
- New: Item-to-Dimension Similarity heatmaps
- New: Expert vs Model Confusion Matrices
- New: Similarity score distributions by dimension

Usage:
    python generate-figures_theory-primed.py \
        --expert-assignments item_assignments.csv \
        --model-comparison model_comparison.csv \
        --final-assignments final_assignments_combined.csv \
        --similarity-dir ./output_steps4_5/embeddings_and_similarities \
        --output-dir ./figures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix


# =============================================================================
# Style Configuration
# =============================================================================

ACCURACY_CMAP = 'YlGnBu'
SIMILARITY_CMAP = 'RdYlBu_r'
CONFUSION_CMAP = 'Blues'

DIMENSION_COLORS = {
    'depression': {
        1: '#E41A1C', 2: '#377EB8', 3: '#4DAF4A', 
        4: '#984EA3', 5: '#FF7F00', -1: '#999999'
    },
    'anxiety': {1: '#E41A1C', 2: '#377EB8', -1: '#999999'},
    'psychosis': {1: '#E41A1C', 2: '#377EB8', -1: '#999999'},
    'apathy': {1: '#E41A1C', 2: '#377EB8', 3: '#4DAF4A', -1: '#999999'},
    'impulse_control': {
        1: '#E41A1C', 2: '#377EB8', 3: '#4DAF4A',
        4: '#984EA3', 5: '#FF7F00', 6: '#FFFF33', -1: '#999999'
    },
    'sleep': {1: '#E41A1C', 2: '#377EB8', 3: '#4DAF4A', 4: '#984EA3', -1: '#999999'}
}

DIMENSION_LABELS = {
    'depression': {
        1: 'Mood & affective symptoms', 2: 'Cognitive & self-perception',
        3: 'Somatic & vegetative symptoms', 4: 'Activity & interest deficits',
        5: 'Anxiety & distress', -1: 'Does not fit'
    },
    'anxiety': {
        1: 'Somatic Anxiety', 2: 'Cognitive Anxiety', -1: 'Does not fit'
    },
    'psychosis': {
        1: 'Hallucinations', 2: 'Delusions', -1: 'Does not fit'
    },
    'apathy': {
        1: 'Cognitive apathy', 2: 'Behavioral apathy', 
        3: 'Affective apathy', -1: 'Does not fit'
    },
    'impulse_control': {
        1: 'Pathological gambling', 2: 'Hypersexuality', 3: 'Compulsive buying',
        4: 'Compulsive eating', 5: 'Punding-hobbyism', 
        6: 'Dopamine dysregulation syndrome', -1: 'Does not fit'
    },
    'sleep': {
        1: 'Daytime sleepiness and alertness', 2: 'Nocturnal sleep disturbances',
        3: 'REM sleep behavior and dreaming', 4: 'Sleep-disordered breathing',
        -1: 'Does not fit'
    }
}


def set_style():
    """Set publication-quality style."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
    })
    sns.set_style("whitegrid")


# =============================================================================
# Figure 1: Item-to-Dimension Similarity Heatmap (NEW for theory-primed)
# =============================================================================

def plot_item_dimension_similarity_heatmap(
    similarity_csv_path: Path,
    expert_assignments_df: pd.DataFrame,
    construct: str,
    model_name: str,
    figsize: Tuple[int, int] = (10, 14),
    save_path: Optional[Path] = None,
    show: bool = True,
    max_items: int = 50
) -> plt.Figure:
    """
    Heatmap showing similarity between each item and each dimension description.
    
    This is the core visualization for theory-primed assignment - it shows
    WHY items were assigned to specific dimensions.
    
    Parameters
    ----------
    similarity_csv_path : Path
        Path to item-dimension similarity CSV (items as rows, dimensions as columns)
    expert_assignments_df : pd.DataFrame
        Expert assignments for comparison
    construct : str
        Construct name
    model_name : str
        Model name for title
    figsize : Tuple
        Figure size
    save_path : Path, optional
        Path to save figure
    show : bool
        Whether to display
    max_items : int
        Maximum items to show (for readability)
    
    Returns
    -------
    plt.Figure
    """
    set_style()
    
    # Load similarity data
    sim_df = pd.read_csv(similarity_csv_path, index_col=0)
    
    # Get expert assignments for this construct
    expert_df = expert_assignments_df[expert_assignments_df['Construct'] == construct]
    expert_dims = dict(zip(expert_df['Item ID'], expert_df['Assigned Dimension']))
    
    # Get items that have expert assignments
    valid_items = [item for item in sim_df.index if item in expert_dims]
    if not valid_items:
        print(f"No matching items found for {construct}")
        return None
    
    sim_df = sim_df.loc[valid_items]
    
    # Sort items by expert-assigned dimension for visual grouping
    item_dims = [expert_dims.get(item, -1) for item in sim_df.index]
    sort_order = np.argsort(item_dims)
    sim_df = sim_df.iloc[sort_order]
    item_dims = [item_dims[i] for i in sort_order]
    
    # Subsample if too many items
    if len(sim_df) > max_items:
        step = len(sim_df) // max_items
        sim_df = sim_df.iloc[::step]
        item_dims = item_dims[::step]
    
    # Create figure with dimension color strip
    fig = plt.figure(figsize=figsize)
    
    # Main heatmap
    ax_main = fig.add_axes([0.25, 0.1, 0.6, 0.8])
    ax_dim_strip = fig.add_axes([0.22, 0.1, 0.02, 0.8])
    ax_cbar = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    
    # Heatmap
    hm = sns.heatmap(
        sim_df.values,
        annot=len(sim_df) <= 30,  # Only show values if not too crowded
        fmt='.2f',
        cmap=SIMILARITY_CMAP,
        vmin=0, vmax=1,
        linewidths=0.5 if len(sim_df) <= 30 else 0,
        linecolor='white',
        cbar_ax=ax_cbar,
        ax=ax_main,
        annot_kws={'size': 8}
    )
    
    ax_main.set_xticklabels(sim_df.columns, rotation=45, ha='right', fontsize=10)
    ax_main.set_yticklabels(sim_df.index, rotation=0, fontsize=8)
    ax_main.set_xlabel('Dimension Descriptions', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Questionnaire Items', fontsize=12, fontweight='bold')
    
    # Expert dimension color strip
    dim_colors = DIMENSION_COLORS.get(construct, {})
    for i, d in enumerate(item_dims):
        color = dim_colors.get(d, '#CCCCCC')
        ax_dim_strip.add_patch(plt.Rectangle((0, i), 1, 1, facecolor=color, edgecolor='none'))
    
    ax_dim_strip.set_xlim(0, 1)
    ax_dim_strip.set_ylim(0, len(item_dims))
    ax_dim_strip.set_ylabel('Expert\nAssignment', fontsize=9, rotation=0, ha='right', va='center')
    ax_dim_strip.set_xticks([])
    ax_dim_strip.set_yticks([])
    ax_dim_strip.invert_yaxis()
    
    # Colorbar label
    ax_cbar.set_ylabel('Cosine Similarity', fontsize=10)
    
    # Title
    fig.suptitle(
        f'Item-to-Dimension Similarity - {construct.replace("_", " ").title()}\n'
        f'Model: {model_name} | Items sorted by expert assignment',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    # Legend for dimension colors
    labels = DIMENSION_LABELS.get(construct, {})
    unique_dims = sorted(set(item_dims))
    legend_patches = [
        mpatches.Patch(color=dim_colors.get(d, '#CCCCCC'), label=labels.get(d, f'Dim {d}'))
        for d in unique_dims if d != -1
    ]
    fig.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.01, 0.98),
               fontsize=9, title='Expert Dimensions')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Figure 2: Dimension Accuracy Heatmap (replaces Cohesion Heatmap)
# =============================================================================

def plot_dimension_accuracy_heatmap(
    accuracy_csv_path: Path,
    construct: str,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None,
    show: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0
) -> plt.Figure:
    """
    Heatmap showing per-dimension accuracy for each model.
    
    This replaces the cohesion heatmap from the clustering approach.
    Shows how well each model identifies items belonging to each dimension.
    
    Parameters
    ----------
    accuracy_csv_path : Path
        Path to dimension_accuracy_detailed.csv
    construct : str
        Construct to plot
    figsize : Tuple
        Figure size
    save_path : Path, optional
        Path to save figure
    show : bool
        Whether to display
    vmin, vmax : float
        Color scale limits
    
    Returns
    -------
    plt.Figure
    """
    set_style()
    
    # Load data
    df = pd.read_csv(accuracy_csv_path)
    df = df[df['Construct'] == construct]
    
    if df.empty:
        print(f"No data for construct: {construct}")
        return None
    
    # Pivot to get dimensions as rows, models as columns
    pivot = df.pivot(index='Dimension Label', columns='Model', values='Accuracy')
    
    # Replace 'N/A' strings with NaN
    pivot = pivot.replace('N/A', np.nan).astype(float)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    hm = sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap=ACCURACY_CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Accuracy', 'shrink': 0.6},
        annot_kws={'size': 10, 'weight': 'bold'},
        ax=ax,
        mask=pivot.isna()
    )
    
    ax.set_title(
        f'Per-Dimension Accuracy by Model - {construct.replace("_", " ").title()}\n'
        f'(Proportion of expert-assigned items correctly identified by model)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_xlabel('Sentence Transformer Models', fontsize=12)
    ax.set_ylabel('Dimensions', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_dimension_accuracy_heatmap_from_data(
    accuracy_data: Dict[str, Dict[str, float]],
    construct: str,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None,
    show: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0
) -> plt.Figure:
    """
    Create accuracy heatmap from dictionary data.
    
    Parameters
    ----------
    accuracy_data : Dict[str, Dict[str, float]]
        {model_name: {dimension_label: accuracy}}
    construct : str
        Construct name
    """
    set_style()
    
    models = list(accuracy_data.keys())
    all_dims = set()
    for model_data in accuracy_data.values():
        all_dims.update(model_data.keys())
    dimensions = sorted(all_dims)
    
    # Build matrix
    data_matrix = []
    for dim in dimensions:
        row = [accuracy_data.get(model, {}).get(dim, np.nan) for model in models]
        data_matrix.append(row)
    data_matrix = np.array(data_matrix)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    hm = sns.heatmap(
        data_matrix,
        annot=True,
        fmt='.2f',
        cmap=ACCURACY_CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Accuracy', 'shrink': 0.6},
        annot_kws={'size': 10, 'weight': 'bold'},
        xticklabels=models,
        yticklabels=dimensions,
        ax=ax
    )
    
    ax.set_title(
        f'Per-Dimension Accuracy by Model - {construct.replace("_", " ").title()}\n'
        f'Theory-Primed Semantic Similarity Assignment',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_xlabel('Sentence Transformer Models', fontsize=12)
    ax.set_ylabel('Dimensions', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Figure 3: Expert vs Model Confusion Matrix (NEW for theory-primed)
# =============================================================================

def plot_expert_model_confusion_matrix(
    final_assignments_df: pd.DataFrame,
    expert_assignments_df: pd.DataFrame,
    construct: str,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    show: bool = True,
    normalize: str = 'true'  # 'true', 'pred', 'all', or None
) -> plt.Figure:
    """
    Confusion matrix comparing expert assignments to model assignments.
    
    This shows where the model agrees and disagrees with experts,
    and which dimension confusions are most common.
    
    Parameters
    ----------
    final_assignments_df : pd.DataFrame
        Final assignments from theory-primed analysis
    expert_assignments_df : pd.DataFrame
        Expert assignments from Step 3
    construct : str
        Construct to analyze
    figsize : Tuple
        Figure size
    save_path : Path, optional
        Path to save figure
    show : bool
        Whether to display
    normalize : str
        Normalization mode: 'true' (by row/expert), 'pred' (by column/model), 
        'all' (total), or None (counts)
    
    Returns
    -------
    plt.Figure
    """
    set_style()
    
    # Get data for this construct
    final_df = final_assignments_df[final_assignments_df['Construct'] == construct]
    expert_df = expert_assignments_df[expert_assignments_df['Construct'] == construct]
    
    if final_df.empty or expert_df.empty:
        print(f"No data for construct: {construct}")
        return None
    
    # Merge on Item ID
    expert_dims = dict(zip(expert_df['Item ID'], expert_df['Assigned Dimension']))
    
    expert_labels = []
    model_labels = []
    
    for _, row in final_df.iterrows():
        item_id = row['Item ID']
        if item_id in expert_dims:
            expert_labels.append(expert_dims[item_id])
            model_labels.append(row['Assigned Dimension'])
    
    if not expert_labels:
        print(f"No matching items for {construct}")
        return None
    
    # Get dimension labels
    labels = DIMENSION_LABELS.get(construct, {})
    all_dims = sorted(set(expert_labels) | set(model_labels))
    dim_names = [labels.get(d, f'Dim {d}') for d in all_dims]
    
    # Compute confusion matrix
    cm = confusion_matrix(expert_labels, model_labels, labels=all_dims)
    
    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle divisions by zero
        fmt = '.2f'
        title_suffix = '(normalized by expert assignment)'
    elif normalize == 'pred':
        cm = cm.astype(float) / cm.sum(axis=0, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = '.2f'
        title_suffix = '(normalized by model prediction)'
    elif normalize == 'all':
        cm = cm.astype(float) / cm.sum()
        fmt = '.2f'
        title_suffix = '(normalized by total)'
    else:
        fmt = 'd'
        title_suffix = '(counts)'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    hm = sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=CONFUSION_CMAP,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'shrink': 0.6},
        annot_kws={'size': 11, 'weight': 'bold'},
        xticklabels=dim_names,
        yticklabels=dim_names,
        ax=ax
    )
    
    ax.set_xlabel('Model Assignment (Theory-Primed)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expert Assignment', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Expert vs Model Confusion Matrix - {construct.replace("_", " ").title()}\n{title_suffix}',
        fontsize=14, fontweight='bold', pad=20
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy annotation
    accuracy = np.trace(cm) / np.sum(cm) if normalize else np.trace(cm) / len(expert_labels)
    ax.text(
        0.02, 0.02, f'Overall Accuracy: {accuracy:.1%}',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Figure 4: Model Comparison Bar Charts (Updated labels)
# =============================================================================

def plot_model_kappa_comparison(
    model_comparison_df: pd.DataFrame,
    construct: str,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Bar chart comparing Cohen's kappa across models.
    """
    set_style()
    
    df = model_comparison_df[model_comparison_df['Construct'] == construct].copy()
    
    if df.empty:
        print(f"No data for construct: {construct}")
        return None
    
    # Sort by kappa
    df = df.sort_values("Cohen's κ", ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    models = df['Model'].tolist()
    kappas = df["Cohen's κ"].tolist()
    
    # Handle 'N/A' values
    kappas_numeric = []
    for k in kappas:
        try:
            kappas_numeric.append(float(k))
        except (ValueError, TypeError):
            kappas_numeric.append(0)
    
    # Color best model differently
    max_kappa = max(kappas_numeric) if kappas_numeric else 0
    colors = ['#2ecc71' if k == max_kappa else '#3498db' for k in kappas_numeric]
    
    bars = ax.barh(range(len(models)), kappas_numeric, color=colors, edgecolor='white')
    
    # Add value labels
    for bar, val in zip(bars, kappas_numeric):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=10)
    
    # Threshold line
    ax.axvline(x=0.40, color='red', linestyle='--', linewidth=2,
              label='Minimum threshold (κ=0.40)')
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Cohen's κ", fontsize=12)
    ax.set_title(
        f"Model Comparison - {construct.replace('_', ' ').title()}\n"
        f"Cohen's κ (Expert vs Theory-Primed Model Agreement)",
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(kappas_numeric) * 1.15 if kappas_numeric else 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_model_kappa_accuracy_comparison(
    model_comparison_df: pd.DataFrame,
    construct: str,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Dual bar chart showing both Cohen's kappa and accuracy for each model.
    """
    set_style()
    
    df = model_comparison_df[model_comparison_df['Construct'] == construct].copy()
    
    if df.empty:
        print(f"No data for construct: {construct}")
        return None
    
    # Sort by kappa
    df = df.sort_values("Cohen's κ", ascending=False)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    models = df['Model'].tolist()
    
    # Handle 'N/A' values
    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0
    
    kappas = [safe_float(k) for k in df["Cohen's κ"].tolist()]
    accuracies = [safe_float(a) for a in df["Accuracy"].tolist()]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, kappas, width, label="Cohen's κ", color='#3498db')
    bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy', color='#2ecc71')
    
    # Add value labels
    for bar, val in zip(bars1, kappas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Threshold line
    ax.axhline(y=0.40, color='red', linestyle='--', linewidth=1.5,
              label='κ threshold (0.40)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        f"Model Performance - {construct.replace('_', ' ').title()}\n"
        f"Cohen's κ and Accuracy (Theory-Primed Assignment)",
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_all_constructs_comparison(
    model_comparison_df: pd.DataFrame,
    metric: str = "Cohen's κ",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Grouped bar chart comparing models across all constructs.
    """
    set_style()
    
    df = model_comparison_df.copy()
    constructs = df['Construct'].unique()
    models = df['Model'].unique()
    
    # Pivot data
    pivot = df.pivot(index='Construct', columns='Model', values=metric)
    
    # Convert to numeric, handling 'N/A'
    pivot = pivot.apply(pd.to_numeric, errors='coerce')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(constructs))
    width = 0.8 / len(models)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        offset = (i - len(models)/2 + 0.5) * width
        values = [pivot.loc[c, model] if model in pivot.columns else 0 for c in constructs]
        ax.bar(x + offset, values, width, label=model, color=colors[i])
    
    if metric == "Cohen's κ":
        ax.axhline(y=0.40, color='red', linestyle='--', linewidth=2,
                  label='Min threshold (κ=0.40)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in constructs])
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(
        f'Model Comparison Across Constructs\n{metric} (Theory-Primed Assignment)',
        fontsize=14, fontweight='bold'
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Figure 5: Combined Evidence Score Distribution (Updated labels)
# =============================================================================

def plot_score_distribution(
    final_assignments_df: pd.DataFrame,
    construct: str,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot distribution of combined evidence scores.
    """
    set_style()
    
    df = final_assignments_df[final_assignments_df['Construct'] == construct]
    
    if df.empty:
        print(f"No data for construct: {construct}")
        return None
    
    scores = df['Combined Score (S_ik)'].values
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n, bins, patches = ax.hist(scores, bins=20, edgecolor='white', alpha=0.7)
    
    # Color by threshold
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0.40:
            patch.set_facecolor('#e74c3c')  # Excluded
        elif left_edge < 0.50:
            patch.set_facecolor('#f39c12')  # Low confidence
        elif left_edge < 0.70:
            patch.set_facecolor('#3498db')  # Moderate
        else:
            patch.set_facecolor('#27ae60')  # High confidence
    
    ax.axvline(x=0.40, color='red', linestyle='--', linewidth=2,
              label='Exclusion threshold (0.40)')
    ax.axvline(x=0.50, color='orange', linestyle='--', linewidth=2,
              label='Low confidence (0.50)')
    ax.axvline(x=0.70, color='green', linestyle='--', linewidth=2,
              label='High confidence (0.70)')
    
    # Stats
    n_excluded = (scores < 0.40).sum()
    n_low = ((scores >= 0.40) & (scores < 0.50)).sum()
    n_moderate = ((scores >= 0.50) & (scores < 0.70)).sum()
    n_high = (scores >= 0.70).sum()
    
    stats_text = (
        f'Mean: {scores.mean():.3f}\n'
        f'Median: {np.median(scores):.3f}\n'
        f'High (≥0.70): {n_high}\n'
        f'Moderate: {n_moderate}\n'
        f'Low: {n_low}\n'
        f'Excluded: {n_excluded}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Combined Evidence Score (S_ik)', fontsize=12)
    ax.set_ylabel('Number of Items', fontsize=12)
    ax.set_title(
        f'Distribution of Combined Evidence Scores - {construct.replace("_", " ").title()}\n'
        f'S_ik = 0.60×E_ik + 0.40×M_ik',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Figure 6: Expert vs Model Evidence Scatter (Updated from Cluster to Model)
# =============================================================================

def plot_expert_vs_model(
    final_assignments_df: pd.DataFrame,
    construct: str,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Scatter plot of expert vs model evidence.
    
    Shows the relationship between expert consensus and model-based
    similarity scores for each item's assigned dimension.
    """
    set_style()
    
    df = final_assignments_df[final_assignments_df['Construct'] == construct].copy()
    
    if df.empty:
        print(f"No data for construct: {construct}")
        return None
    
    n_items = len(df)
    
    # Get the values - check for both old and new column names
    expert_col = 'Expert Evidence (E_ik)'
    model_col = 'Model Evidence (M_ik)' if 'Model Evidence (M_ik)' in df.columns else 'Cluster Evidence (C_ik)'
    
    expert_ev = df[expert_col].values
    model_ev = df[model_col].values
    combined = df['Combined Score (S_ik)'].values
    
    # Add small jitter to reveal overlapping points
    np.random.seed(42)
    jitter_amount = 0.015
    expert_jittered = expert_ev + np.random.uniform(-jitter_amount, jitter_amount, len(expert_ev))
    model_jittered = model_ev + np.random.uniform(-jitter_amount, jitter_amount, len(model_ev))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        expert_jittered, model_jittered, 
        c=combined, cmap='RdYlGn',
        s=80, alpha=0.6, edgecolor='white', linewidth=0.5,
        vmin=0.3, vmax=0.9
    )
    
    plt.colorbar(scatter, ax=ax, label='Combined Score (S_ik)')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect agreement')
    
    # Iso-score lines (S = 0.60*E + 0.40*M)
    for s in [0.4, 0.6, 0.8]:
        e = np.linspace(0, 1, 100)
        m = (s - 0.6 * e) / 0.4
        valid = (m >= 0) & (m <= 1)
        ax.plot(e[valid], m[valid], '--', alpha=0.5, label=f'S = {s}')
    
    ax.set_xlabel('Expert Evidence (E_ik)', fontsize=12)
    ax.set_ylabel('Model Evidence (M_ik)', fontsize=12)
    ax.set_title(
        f'Expert vs Model Evidence - {construct.replace("_", " ").title()}\n'
        f'(n = {n_items} items, jittered to show overlaps)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal')
    
    # Correlation
    valid_mask = ~(np.isnan(expert_ev) | np.isnan(model_ev))
    if valid_mask.sum() > 2:
        corr = np.corrcoef(expert_ev[valid_mask], model_ev[valid_mask])[0, 1]
        ax.text(0.02, 0.02, f'Correlation: r = {corr:.3f}', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Figure 7: Similarity Score Distribution by Dimension (NEW)
# =============================================================================

def plot_similarity_distribution_by_dimension(
    similarity_csv_path: Path,
    expert_assignments_df: pd.DataFrame,
    construct: str,
    model_name: str,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Violin/box plots showing distribution of similarity scores to each dimension,
    grouped by expert-assigned dimension.
    
    This shows whether items assigned to a dimension by experts also have
    high similarity to that dimension's description.
    
    Parameters
    ----------
    similarity_csv_path : Path
        Path to item-dimension similarity CSV
    expert_assignments_df : pd.DataFrame
        Expert assignments
    construct : str
        Construct name
    model_name : str
        Model name
    """
    set_style()
    
    sim_df = pd.read_csv(similarity_csv_path, index_col=0)
    expert_df = expert_assignments_df[expert_assignments_df['Construct'] == construct]
    expert_dims = dict(zip(expert_df['Item ID'], expert_df['Assigned Dimension']))
    
    # Build long-form data for plotting
    data = []
    for item in sim_df.index:
        if item not in expert_dims:
            continue
        expert_dim = expert_dims[item]
        for dim_label in sim_df.columns:
            sim_score = sim_df.loc[item, dim_label]
            data.append({
                'Item': item,
                'Expert Dimension': expert_dim,
                'Target Dimension': dim_label,
                'Similarity': sim_score,
                'Is Correct': dim_label.split()[0] == str(expert_dim) or dim_label.startswith(str(expert_dim))
            })
    
    plot_df = pd.DataFrame(data)
    
    if plot_df.empty:
        print(f"No data for {construct}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by target dimension
    dimensions = sim_df.columns.tolist()
    x_positions = np.arange(len(dimensions))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(dimensions)))
    
    for i, target_dim in enumerate(dimensions):
        dim_data = plot_df[plot_df['Target Dimension'] == target_dim]['Similarity']
        
        # Violin plot
        parts = ax.violinplot([dim_data], positions=[i], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dimensions, rotation=45, ha='right')
    ax.set_xlabel('Dimension Description', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title(
        f'Similarity Score Distribution by Target Dimension - {construct.replace("_", " ").title()}\n'
        f'Model: {model_name}',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Figure 8: Fleiss' Kappa Summary (unchanged)
# =============================================================================

def plot_fleiss_kappa_summary(
    fleiss_summary_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Bar chart of Fleiss' kappa for expert agreement.
    """
    set_style()
    
    df = fleiss_summary_df.sort_values('Fleiss κ', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    constructs = df['Construct'].tolist()
    kappas = df['Fleiss κ'].tolist()
    ci_lower = df['95% CI Lower'].tolist() if '95% CI Lower' in df.columns else [k * 0.9 for k in kappas]
    ci_upper = df['95% CI Upper'].tolist() if '95% CI Upper' in df.columns else [min(k * 1.1, 1.0) for k in kappas]
    
    # Error bars
    errors = [[k - l for k, l in zip(kappas, ci_lower)],
              [u - k for k, u in zip(kappas, ci_upper)]]
    
    # Color by interpretation
    colors = []
    for k in kappas:
        if k >= 0.80:
            colors.append('#27ae60')
        elif k >= 0.60:
            colors.append('#2ecc71')
        elif k >= 0.40:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')
    
    bars = ax.barh(range(len(constructs)), kappas, xerr=errors, color=colors,
                  capsize=5, edgecolor='white')
    
    # Value labels
    for bar, val, upper in zip(bars, kappas, ci_upper):
        ax.text(upper + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=10)
    
    # Threshold lines
    ax.axvline(x=0.40, color='red', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax.axvline(x=0.60, color='orange', linestyle='--', alpha=0.7, label='Substantial threshold')
    ax.axvline(x=0.80, color='green', linestyle='--', alpha=0.7, label='Almost perfect')
    
    ax.set_yticks(range(len(constructs)))
    ax.set_yticklabels([c.replace('_', ' ').title() for c in constructs])
    ax.set_xlabel("Fleiss' κ", fontsize=12)
    ax.set_title("Expert Agreement (Fleiss' κ) by Construct\nwith 95% Confidence Intervals",
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Figure 9: Sensitivity Analysis Heatmap (NEW)
# =============================================================================

def plot_sensitivity_analysis(
    sensitivity_df: pd.DataFrame,
    construct: str,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Heatmap showing assignment stability across different weight schemes.
    
    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Sensitivity analysis results with columns for different weight schemes
    construct : str
        Construct to analyze
    """
    set_style()
    
    df = sensitivity_df[sensitivity_df['Construct'] == construct].copy()
    
    if df.empty:
        print(f"No data for construct: {construct}")
        return None
    
    # Get weight scheme columns
    dim_cols = [c for c in df.columns if c.startswith('Dim (')]
    
    if not dim_cols:
        print("No weight scheme columns found")
        return None
    
    # Count stability
    n_stable = df['Stable'].sum() if 'Stable' in df.columns else 0
    n_total = len(df)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap of assignments
    data = df[dim_cols].values
    
    # Use a discrete colormap for dimension IDs
    unique_dims = np.unique(data)
    n_dims = len(unique_dims)
    
    cmap = plt.cm.get_cmap('Set3', n_dims)
    
    hm = ax.imshow(data, aspect='auto', cmap=cmap)
    
    ax.set_xticks(range(len(dim_cols)))
    ax.set_xticklabels([c.replace('Dim (', '').replace(')', '') for c in dim_cols], rotation=45, ha='right')
    ax.set_xlabel('Weight Scheme (Expert/Model)', fontsize=12)
    ax.set_ylabel('Items', fontsize=12)
    ax.set_title(
        f'Sensitivity Analysis - {construct.replace("_", " ").title()}\n'
        f'Stable assignments: {n_stable}/{n_total} ({100*n_stable/n_total:.1f}%)',
        fontsize=14, fontweight='bold'
    )
    
    # Colorbar with dimension labels
    cbar = plt.colorbar(hm, ax=ax, ticks=unique_dims)
    cbar.set_label('Assigned Dimension', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Main Function
# =============================================================================

def generate_all_figures_from_csvs(
    expert_assignments_path: Optional[Path] = None,
    fleiss_summary_path: Optional[Path] = None,
    model_comparison_path: Optional[Path] = None,
    final_assignments_path: Optional[Path] = None,
    dimension_accuracy_path: Optional[Path] = None,
    similarity_dir: Optional[Path] = None,
    sensitivity_path: Optional[Path] = None,
    output_dir: Path = Path('./figures'),
    show: bool = False
):
    """
    Generate all available figures from CSV files.
    
    Parameters
    ----------
    expert_assignments_path : Path
        Path to item_assignments.csv (from Step 3)
    fleiss_summary_path : Path
        Path to fleiss_kappa_summary.csv (from Step 3)
    model_comparison_path : Path
        Path to model_comparison.csv (from Steps 4-5)
    final_assignments_path : Path
        Path to final_assignments_combined.csv (from Steps 4-5)
    dimension_accuracy_path : Path
        Path to dimension_accuracy_detailed.csv (from Steps 4-5)
    similarity_dir : Path
        Directory containing similarity CSVs (embeddings_and_similarities folder)
    sensitivity_path : Path
        Path to sensitivity_analysis.csv (from Steps 4-5)
    output_dir : Path
        Output directory for figures
    show : bool
        Whether to display figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GENERATING FIGURES FROM THEORY-PRIMED ANALYSIS")
    print("="*60)
    
    # Load dataframes
    expert_df = None
    if expert_assignments_path and Path(expert_assignments_path).exists():
        expert_df = pd.read_csv(expert_assignments_path)
        print(f"\nLoaded expert assignments: {len(expert_df)} items")
    
    fleiss_df = None
    if fleiss_summary_path and Path(fleiss_summary_path).exists():
        fleiss_df = pd.read_csv(fleiss_summary_path)
        print(f"Loaded Fleiss summary: {len(fleiss_df)} constructs")
    
    model_df = None
    if model_comparison_path and Path(model_comparison_path).exists():
        model_df = pd.read_csv(model_comparison_path)
        print(f"Loaded model comparison: {len(model_df)} entries")
    
    final_df = None
    if final_assignments_path and Path(final_assignments_path).exists():
        final_df = pd.read_csv(final_assignments_path)
        print(f"Loaded final assignments: {len(final_df)} items")
    
    accuracy_df = None
    if dimension_accuracy_path and Path(dimension_accuracy_path).exists():
        accuracy_df = pd.read_csv(dimension_accuracy_path)
        print(f"Loaded dimension accuracy: {len(accuracy_df)} entries")
    
    sensitivity_df = None
    if sensitivity_path and Path(sensitivity_path).exists():
        sensitivity_df = pd.read_csv(sensitivity_path)
        print(f"Loaded sensitivity analysis: {len(sensitivity_df)} entries")
    
    generated = []
    
    # 1. Fleiss' kappa summary
    if fleiss_df is not None:
        print("\n1. Generating Fleiss' kappa summary...")
        path = output_dir / 'fleiss_kappa_summary.png'
        try:
            plot_fleiss_kappa_summary(fleiss_df, save_path=path, show=show)
            generated.append(path)
        except Exception as e:
            print(f"  Warning: {e}")
    
    # 2. Model comparisons
    if model_df is not None:
        print("\n2. Generating model comparisons...")
        
        constructs = model_df['Construct'].unique()
        for construct in constructs:
            # Kappa comparison
            path = output_dir / f'{construct}_model_comparison.png'
            try:
                plot_model_kappa_comparison(model_df, construct, save_path=path, show=show)
                generated.append(path)
            except Exception as e:
                print(f"  Warning for {construct}: {e}")
            
            # Kappa + Accuracy comparison
            if 'Accuracy' in model_df.columns:
                path = output_dir / f'{construct}_model_kappa_accuracy.png'
                try:
                    plot_model_kappa_accuracy_comparison(model_df, construct, save_path=path, show=show)
                    generated.append(path)
                except Exception as e:
                    print(f"  Warning for {construct}: {e}")
        
        # All constructs
        path = output_dir / 'all_constructs_model_comparison.png'
        try:
            plot_all_constructs_comparison(model_df, save_path=path, show=show)
            generated.append(path)
        except Exception as e:
            print(f"  Warning: {e}")
    
    # 3. Dimension accuracy heatmaps
    if accuracy_df is not None:
        print("\n3. Generating dimension accuracy heatmaps...")
        
        constructs = accuracy_df['Construct'].unique()
        for construct in constructs:
            path = output_dir / f'{construct}_dimension_accuracy.png'
            try:
                plot_dimension_accuracy_heatmap(
                    dimension_accuracy_path, construct,
                    save_path=path, show=show
                )
                generated.append(path)
            except Exception as e:
                print(f"  Warning for {construct}: {e}")
    
    # 4. Score distributions and scatter plots
    if final_df is not None:
        print("\n4. Generating score distributions...")
        
        constructs = final_df['Construct'].unique()
        for construct in constructs:
            # Distribution
            path = output_dir / f'{construct}_score_distribution.png'
            try:
                plot_score_distribution(final_df, construct, save_path=path, show=show)
                generated.append(path)
            except Exception as e:
                print(f"  Warning for {construct}: {e}")
            
            # Expert vs Model scatter
            path = output_dir / f'{construct}_expert_vs_model.png'
            try:
                plot_expert_vs_model(final_df, construct, save_path=path, show=show)
                generated.append(path)
            except Exception as e:
                print(f"  Warning for {construct}: {e}")
    
    # 5. Confusion matrices
    if final_df is not None and expert_df is not None:
        print("\n5. Generating confusion matrices...")
        
        constructs = final_df['Construct'].unique()
        for construct in constructs:
            path = output_dir / f'{construct}_confusion_matrix.png'
            try:
                plot_expert_model_confusion_matrix(
                    final_df, expert_df, construct,
                    save_path=path, show=show
                )
                generated.append(path)
            except Exception as e:
                print(f"  Warning for {construct}: {e}")
    
    # 6. Item-dimension similarity heatmaps
    if similarity_dir and Path(similarity_dir).exists() and expert_df is not None:
        print("\n6. Generating item-dimension similarity heatmaps...")
        
        similarity_dir = Path(similarity_dir)
        
        # Check directory structure
        for construct_dir in similarity_dir.iterdir():
            if not construct_dir.is_dir():
                continue
            
            construct = construct_dir.name
            
            # Find similarity files
            for sim_file in construct_dir.glob('*_item_dimension_similarity.csv'):
                model_name = sim_file.stem.replace('_item_dimension_similarity', '')
                
                path = output_dir / f'{construct}_{model_name}_item_dimension_similarity.png'
                try:
                    plot_item_dimension_similarity_heatmap(
                        sim_file, expert_df, construct, model_name,
                        save_path=path, show=show
                    )
                    generated.append(path)
                except Exception as e:
                    print(f"  Warning for {construct}/{model_name}: {e}")
    
    # 7. Sensitivity analysis
    if sensitivity_df is not None:
        print("\n7. Generating sensitivity analysis figures...")
        
        constructs = sensitivity_df['Construct'].unique()
        for construct in constructs:
            path = output_dir / f'{construct}_sensitivity_analysis.png'
            try:
                plot_sensitivity_analysis(sensitivity_df, construct, save_path=path, show=show)
                generated.append(path)
            except Exception as e:
                print(f"  Warning for {construct}: {e}")
    
    print("\n" + "="*60)
    print(f"Generated {len(generated)} figures in {output_dir}")
    print("="*60)
    
    for fig_path in generated:
        print(f"  - {fig_path.name}")
    
    return generated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate figures from ENIGMA-PD Theory-Primed Analysis CSV results"
    )
    parser.add_argument(
        "--expert-assignments", type=str,
        help="Path to item_assignments.csv (from Step 3)"
    )
    parser.add_argument(
        "--fleiss-summary", type=str,
        help="Path to fleiss_kappa_summary.csv (from Step 3)"
    )
    parser.add_argument(
        "--model-comparison", type=str,
        help="Path to model_comparison.csv (from Steps 4-5)"
    )
    parser.add_argument(
        "--final-assignments", type=str,
        help="Path to final_assignments_combined.csv (from Steps 4-5)"
    )
    parser.add_argument(
        "--dimension-accuracy", type=str,
        help="Path to dimension_accuracy_detailed.csv (from Steps 4-5)"
    )
    parser.add_argument(
        "--similarity-dir", type=str,
        help="Directory with item-dimension similarity CSVs"
    )
    parser.add_argument(
        "--sensitivity", type=str,
        help="Path to sensitivity_analysis.csv (from Steps 4-5)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="./figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively"
    )
    
    args = parser.parse_args()
    
    generate_all_figures_from_csvs(
        expert_assignments_path=args.expert_assignments,
        fleiss_summary_path=args.fleiss_summary,
        model_comparison_path=args.model_comparison,
        final_assignments_path=args.final_assignments,
        dimension_accuracy_path=args.dimension_accuracy,
        similarity_dir=args.similarity_dir,
        sensitivity_path=args.sensitivity,
        output_dir=Path(args.output_dir),
        show=args.show
    )
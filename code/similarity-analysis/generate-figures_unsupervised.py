"""
Generate Figures from ENIGMA-PD Analysis Results
=================================================

This script generates visualization figures from the CSV output files
produced by expert_mappings_analysis.py and semantic_similarity_integration.py.

It can work with:
1. Expert mappings results only (from Step 3)
2. Full integration results (from Steps 4 & 5)
3. Pre-computed similarity matrices

Usage:
    python generate_figures.py --expert-assignments item_assignments.csv \
                               --model-comparison model_comparison.csv \
                               --final-assignments final_assignments_combined.csv \
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
from scipy.cluster.hierarchy import dendrogram, linkage


# =============================================================================
# Style Configuration
# =============================================================================

COHESION_CMAP = 'YlGnBu'
SIMILARITY_CMAP = 'RdYlBu_r'

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
        5: 'Anxiety & distress', -1: 'None of the categories fit'
    },
    'anxiety': {
        1: 'Somatic Anxiety', 2: 'Cognitive Anxiety', -1: 'None of the categories fit'
    },
    'psychosis': {
        1: 'Hallucinations', 2: 'Delusions', -1: 'None of the categories fit'
    },
    'apathy': {
        1: 'Behavioral apathy', 2: 'Cognitive apathy', 
        3: 'Affective apathy', -1: 'None of the categories fit'
    },
    'impulse_control': {
        1: 'Pathological gambling', 2: 'Hypersexuality', 3: 'Compulsive buying',
        4: 'Compulsive eating', 5: 'Punding-hobbyism', 
        6: 'Dopamine dysregulation syndrome', -1: 'None of the categories fit'
    },
    'sleep': {
        1: 'Daytime sleepiness and alertness', 2: 'Nocturnal sleep disturbances',
        3: 'REM sleep behavior and dreaming', 4: 'Sleep-disordered breathing',
        -1: 'None of the categories fit'
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
# Figure 6: Question Pair Similarity Comparison (detailed model comparison)
# =============================================================================

def plot_question_pair_similarity_comparison(
    question_pairs: List[Tuple[str, str, str, str]],
    similarity_matrices: Dict[str, pd.DataFrame],
    construct: str,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None,
    show: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    max_text_length: int = 45
) -> plt.Figure:
    """
    Create a heatmap comparing how different models rate similarity between question pairs.
    
    This figure helps understand model behavior on specific, interpretable examples.
    
    Parameters
    ----------
    question_pairs : List[Tuple[str, str, str, str]]
        List of question pairs to compare. Each tuple contains:
        (item1_id, item1_text, item2_id, item2_text)
        Example: [('BAI_04', 'Unable to relax', 'HADS_11', 'I feel restless...'), ...]
    similarity_matrices : Dict[str, pd.DataFrame]
        Dictionary mapping model names to their similarity matrices.
        Each DataFrame should have item IDs as both index and columns.
    construct : str
        Construct name for the title
    figsize : Tuple
        Figure size
    save_path : Path, optional
        Path to save figure
    show : bool
        Whether to display
    vmin, vmax : float
        Color scale limits
    max_text_length : int
        Maximum characters for question text display
    
    Returns
    -------
    plt.Figure
    """
    set_style()
    
    models = list(similarity_matrices.keys())
    n_pairs = len(question_pairs)
    n_models = len(models)
    
    # Build data matrix (pairs x models)
    data_matrix = np.full((n_pairs, n_models), np.nan)
    pair_labels = []
    
    for i, (id1, text1, id2, text2) in enumerate(question_pairs):
        # Truncate text if needed
        text1_short = text1[:max_text_length] + '...' if len(text1) > max_text_length else text1
        text2_short = text2[:max_text_length] + '...' if len(text2) > max_text_length else text2
        
        # Extract questionnaire name from item ID (e.g., 'BAI' from 'BAI_04')
        q1 = id1.rsplit('_', 1)[0] if '_' in id1 else id1
        q2 = id2.rsplit('_', 1)[0] if '_' in id2 else id2
        
        # Create label
        label = f"{q1}: {text1_short}\nvs.\n{q2}: {text2_short}"
        pair_labels.append(label)
        
        # Get similarity from each model
        for j, model in enumerate(models):
            sim_df = similarity_matrices[model]
            
            # Try to find the similarity value
            try:
                if id1 in sim_df.index and id2 in sim_df.columns:
                    val = sim_df.loc[id1, id2]
                elif id2 in sim_df.index and id1 in sim_df.columns:
                    val = sim_df.loc[id2, id1]
                else:
                    val = np.nan
                
                # Clip to valid range (some models might give slightly negative values)
                if not np.isnan(val):
                    val = np.clip(val, 0.0, 1.0)
                data_matrix[i, j] = val
            except:
                data_matrix[i, j] = np.nan
    
    # Remove models with all NaN values (no data for any pair)
    valid_model_mask = ~np.all(np.isnan(data_matrix), axis=0)
    if not np.any(valid_model_mask):
        print("No valid similarity data found for any model")
        return None
    
    data_matrix = data_matrix[:, valid_model_mask]
    models = [m for m, valid in zip(models, valid_model_mask) if valid]
    n_models = len(models)
    
    # Create figure - adjust size based on number of models
    fig_width = max(10, n_models * 0.7)
    fig_height = max(6, n_pairs * 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create heatmap - use mask for any remaining NaN values
    mask = np.isnan(data_matrix)
    
    hm = sns.heatmap(
        data_matrix,
        annot=True,
        fmt='.2f',
        cmap=COHESION_CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': '', 'shrink': 0.6},
        annot_kws={'size': 11, 'weight': 'bold'},
        xticklabels=models,
        yticklabels=pair_labels,
        mask=mask,
        ax=ax
    )
    
    ax.set_title(f'Semantic Similarity Matrix: {construct.replace("_", " ").title()} Questionnaires',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sentence Transformer Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Question Pairs Between Questionnaires', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def select_interesting_question_pairs(
    expert_assignments_df: pd.DataFrame,
    similarity_matrices: Dict[str, pd.DataFrame],
    construct: str,
    n_pairs: int = 5,
    strategy: str = 'diverse'
) -> List[Tuple[str, str, str, str]]:
    """
    Automatically select interesting question pairs for the comparison plot.
    
    Parameters
    ----------
    expert_assignments_df : pd.DataFrame
        Expert assignments with Item ID, Question Text, Assigned Dimension, Questionnaire
    similarity_matrices : Dict[str, pd.DataFrame]
        Similarity matrices from different models
    construct : str
        Construct to analyze
    n_pairs : int
        Number of pairs to select
    strategy : str
        Selection strategy:
        - 'diverse': Mix of same-dimension and different-dimension pairs
        - 'same_dimension': Pairs expected to be similar (same expert dimension)
        - 'different_dimension': Pairs expected to be different
        - 'cross_questionnaire': Pairs from different questionnaires
        - 'high_variance': Pairs where models disagree most
    
    Returns
    -------
    List[Tuple[str, str, str, str]]
        Selected question pairs (id1, text1, id2, text2)
    """
    # Filter to construct
    items = expert_assignments_df[expert_assignments_df['Construct'] == construct].copy()
    
    # Get one similarity matrix for reference
    ref_model = list(similarity_matrices.keys())[0]
    ref_sim = similarity_matrices[ref_model]
    
    # Generate candidate pairs
    candidates = []
    item_list = items.to_dict('records')
    
    for i, item1 in enumerate(item_list):
        for j, item2 in enumerate(item_list):
            if i >= j:  # Avoid duplicates
                continue
            
            id1, id2 = item1['Item ID'], item2['Item ID']
            text1 = item1.get('Question Text', '')
            text2 = item2.get('Question Text', '')
            
            # Skip if texts are too similar (avoid trivial pairs)
            # Normalize texts for comparison
            text1_norm = text1.lower().strip()
            text2_norm = text2.lower().strip()
            if text1_norm == text2_norm:
                continue
            # Also skip if one is a substring of the other (very similar)
            if len(text1_norm) > 10 and len(text2_norm) > 10:
                if text1_norm in text2_norm or text2_norm in text1_norm:
                    continue
            
            # Skip if not in similarity matrix
            if id1 not in ref_sim.index or id2 not in ref_sim.columns:
                continue
            
            # Get questionnaire names
            q1 = item1.get('Questionnaire', id1.rsplit('_', 1)[0] if '_' in id1 else id1)
            q2 = item2.get('Questionnaire', id2.rsplit('_', 1)[0] if '_' in id2 else id2)
            
            # Skip pairs from the same questionnaire for more interesting comparisons
            if q1 == q2:
                continue
            
            # Get dimensions
            dim1 = item1.get('Assigned Dimension', -1)
            dim2 = item2.get('Assigned Dimension', -1)
            
            # Calculate variance across models (only for models that have this pair)
            sims = []
            for model, sim_df in similarity_matrices.items():
                if id1 in sim_df.index and id2 in sim_df.columns:
                    val = sim_df.loc[id1, id2]
                    if not np.isnan(val):
                        sims.append(val)
            
            # Skip if not enough models have this pair
            if len(sims) < len(similarity_matrices) * 0.5:  # At least 50% of models
                continue
                
            variance = np.var(sims)
            mean_sim = np.mean(sims)
            
            candidates.append({
                'id1': id1, 'text1': text1,
                'id2': id2, 'text2': text2,
                'q1': q1, 'q2': q2,
                'dim1': dim1, 'dim2': dim2,
                'same_dim': dim1 == dim2,
                'cross_q': q1 != q2,
                'variance': variance,
                'mean_sim': mean_sim,
                'n_models': len(sims)
            })
    
    if not candidates:
        print(f"  Warning: No valid question pairs found for {construct}")
        return []
    
    # Select based on strategy
    selected = []
    
    if strategy == 'diverse':
        # Mix of different types
        # 2 same dimension with high similarity (should be similar)
        same_dim = [c for c in candidates if c['same_dim']]
        same_dim.sort(key=lambda x: x['mean_sim'], reverse=True)
        selected.extend(same_dim[:2])
        
        # 2 different dimension with high variance (models disagree)
        diff_dim = [c for c in candidates if not c['same_dim']]
        diff_dim.sort(key=lambda x: x['variance'], reverse=True)
        for c in diff_dim:
            if c not in selected:
                selected.append(c)
                if len([s for s in selected if not s['same_dim']]) >= 2:
                    break
        
        # 1 high variance overall (interesting disagreement)
        candidates.sort(key=lambda x: x['variance'], reverse=True)
        for c in candidates:
            if c not in selected:
                selected.append(c)
                break
                
    elif strategy == 'high_variance':
        candidates.sort(key=lambda x: x['variance'], reverse=True)
        selected = candidates[:n_pairs]
        
    elif strategy == 'cross_questionnaire':
        cross_q = [c for c in candidates if c['cross_q']]
        cross_q.sort(key=lambda x: x['variance'], reverse=True)
        selected = cross_q[:n_pairs]
        
    elif strategy == 'same_dimension':
        same_dim = [c for c in candidates if c['same_dim']]
        same_dim.sort(key=lambda x: x['mean_sim'], reverse=True)
        selected = same_dim[:n_pairs]
        
    elif strategy == 'different_dimension':
        diff_dim = [c for c in candidates if not c['same_dim']]
        diff_dim.sort(key=lambda x: x['variance'], reverse=True)
        selected = diff_dim[:n_pairs]
    
    # Ensure we have n_pairs
    while len(selected) < n_pairs and len(candidates) > len(selected):
        for c in candidates:
            if c not in selected:
                selected.append(c)
                if len(selected) >= n_pairs:
                    break
    
    # Convert to tuple format
    return [(s['id1'], s['text1'], s['id2'], s['text2']) for s in selected[:n_pairs]]


def plot_question_pair_comparison_from_csvs(
    expert_assignments_csv: Path,
    similarity_dir: Path,
    construct: str,
    n_pairs: int = 5,
    question_pairs: Optional[List[Tuple[str, str, str, str]]] = None,
    models: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate question pair comparison heatmap from CSV files.
    
    Parameters
    ----------
    expert_assignments_csv : Path
        Path to item_assignments.csv (contains Item ID, Question Text, Construct, etc.)
    similarity_dir : Path
        Directory containing similarity matrix CSVs
        (files named like: {construct}_{model}_similarity.csv or {model}_similarity.csv)
    construct : str
        Construct to analyze
    n_pairs : int
        Number of question pairs to show
    question_pairs : List, optional
        Manually specified question pairs. If None, auto-selects interesting pairs.
    models : List[str], optional
        Specific models to include. If None, uses all available.
    figsize : Tuple
        Figure size
    save_path : Path
        Where to save figure
    show : bool
        Whether to display
    
    Returns
    -------
    plt.Figure
    """
    # Load expert assignments (has Item ID and Question Text)
    expert_df = pd.read_csv(expert_assignments_csv)
    
    # Load similarity matrices
    similarity_dir = Path(similarity_dir)
    similarity_matrices = {}
    
    # Try different naming patterns
    for sim_file in similarity_dir.glob('*similarity*.csv'):
        # Extract model name from filename
        name = sim_file.stem
        
        # Handle different naming conventions
        if construct in name:
            # Format: {construct}_{model}_similarity
            model_name = name.replace(f'{construct}_', '').replace('_similarity', '')
        else:
            # Format: {model}_similarity
            model_name = name.replace('_similarity', '')
        
        if models and model_name not in models:
            continue
            
        try:
            sim_df = pd.read_csv(sim_file, index_col=0)
            similarity_matrices[model_name] = sim_df
        except Exception as e:
            print(f"Warning: Could not load {sim_file}: {e}")
    
    if not similarity_matrices:
        print(f"No similarity matrices found in {similarity_dir}")
        return None
    
    print(f"Loaded {len(similarity_matrices)} similarity matrices: {list(similarity_matrices.keys())}")
    
    # Select question pairs if not provided
    if question_pairs is None:
        print(f"Auto-selecting {n_pairs} interesting question pairs...")
        question_pairs = select_interesting_question_pairs(
            expert_df, similarity_matrices, construct, n_pairs, strategy='diverse'
        )
    
    if not question_pairs:
        print("Could not find suitable question pairs")
        return None
    
    # Generate plot
    return plot_question_pair_similarity_comparison(
        question_pairs, similarity_matrices, construct,
        figsize=figsize, save_path=save_path, show=show
    )


def create_question_pair_figure_manual(
    similarity_data: Dict[str, Dict[Tuple[str, str], float]],
    question_pairs: List[Tuple[str, str, str, str]],
    construct: str,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create question pair comparison with manually specified similarity values.
    
    This is useful when you already have the similarity values extracted.
    
    Parameters
    ----------
    similarity_data : Dict[str, Dict[Tuple[str, str], float]]
        Nested dict: {model_name: {(item1_id, item2_id): similarity_value}}
        Example:
        {
            'ClinicalBert': {('BAI_04', 'HADS_11'): 0.60, ('HADS_05', 'HAMA_01'): 0.70},
            'MiniLM': {('BAI_04', 'HADS_11'): 0.34, ('HADS_05', 'HAMA_01'): 0.42},
        }
    question_pairs : List[Tuple[str, str, str, str]]
        List of (item1_id, item1_text, item2_id, item2_text)
    construct : str
        Construct name
    figsize, save_path, show : as above
    
    Returns
    -------
    plt.Figure
    
    Example
    -------
    >>> similarity_data = {
    ...     'ClinicalBert': {
    ...         ('BAI_04', 'HADS_11'): 0.60,
    ...         ('HADS_07', 'BAI_08'): 0.54,
    ...     },
    ...     'Harmony': {
    ...         ('BAI_04', 'HADS_11'): 0.32,
    ...         ('HADS_07', 'BAI_08'): 0.02,
    ...     },
    ... }
    >>> question_pairs = [
    ...     ('BAI_04', 'Unable to relax', 'HADS_11', 'I feel restless as I have to be on the move.'),
    ...     ('HADS_07', 'I can sit at ease and feel relaxed.', 'BAI_08', 'Fear of dying'),
    ... ]
    >>> fig = create_question_pair_figure_manual(similarity_data, question_pairs, 'anxiety')
    """
    set_style()
    
    models = list(similarity_data.keys())
    n_pairs = len(question_pairs)
    n_models = len(models)
    
    # Build data matrix
    data_matrix = np.zeros((n_pairs, n_models))
    pair_labels = []
    
    for i, (id1, text1, id2, text2) in enumerate(question_pairs):
        # Truncate text
        max_len = 45
        text1_short = text1[:max_len] + '...' if len(text1) > max_len else text1
        text2_short = text2[:max_len] + '...' if len(text2) > max_len else text2
        
        q1 = id1.rsplit('_', 1)[0] if '_' in id1 else id1
        q2 = id2.rsplit('_', 1)[0] if '_' in id2 else id2
        
        label = f"{q1}: {text1_short}\nvs.\n{q2}: {text2_short}"
        pair_labels.append(label)
        
        for j, model in enumerate(models):
            model_sims = similarity_data[model]
            # Try both orderings
            if (id1, id2) in model_sims:
                data_matrix[i, j] = model_sims[(id1, id2)]
            elif (id2, id1) in model_sims:
                data_matrix[i, j] = model_sims[(id2, id1)]
            else:
                data_matrix[i, j] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    hm = sns.heatmap(
        data_matrix,
        annot=True,
        fmt='.2f',
        cmap=COHESION_CMAP,
        vmin=0.0,
        vmax=1.0,
        linewidths=1,
        linecolor='white',
        cbar_kws={'shrink': 0.6},
        annot_kws={'size': 12, 'weight': 'bold'},
        xticklabels=models,
        yticklabels=pair_labels,
        ax=ax
    )
    
    ax.set_title(f'Semantic Similarity Matrix: {construct.replace("_", " ").title()} Questionnaires',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sentence Transformer Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Question Pairs Between Questionnaires', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=10)
    
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
# Figure 1: Cohesion Heatmap 
# =============================================================================

def plot_cohesion_heatmap_from_csv(
    cohesion_csv_path: Path,
    construct: str,
    method: str = 'Direct Embeddings',
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[Path] = None,
    show: bool = True,
    vmin: float = 0.5,
    vmax: float = 1.0
) -> plt.Figure:
    """
    Create cohesion heatmap from a CSV file.
    
    The CSV should have dimensions as rows (index) and models as columns.
    This is the format produced by save_cohesion_data_for_heatmap().
    
    Parameters
    ----------
    cohesion_csv_path : Path
        Path to the cohesion CSV file (e.g., anxiety_cohesion_by_model.csv)
    construct : str
        Construct name for the title
    method : str
        Method description for title
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
    
    # Load the CSV (dimensions as index, models as columns)
    df = pd.read_csv(cohesion_csv_path, index_col=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    hm = sns.heatmap(
        df,
        annot=True,
        fmt='.2f',
        cmap=COHESION_CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': '', 'shrink': 0.6},
        annot_kws={'size': 10, 'weight': 'bold'},
        ax=ax
    )
    
    ax.set_title(f'Category Cohesion by Model - {construct}\nMethod: {method}',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Different Sentence Transformer Models', fontsize=12)
    ax.set_ylabel('Construct Dimensions', fontsize=12)
    
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


def plot_cohesion_heatmap_from_df(
    model_comparison_df: pd.DataFrame,
    construct: str,
    method: str = 'Direct Embeddings',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create cohesion heatmap from model_comparison.csv data.
    
    Note: This requires dimension-level cohesion data. If you only have
    overall cohesion, use plot_cohesion_heatmap_from_detailed() instead.
    """
    set_style()
    
    df = model_comparison_df[model_comparison_df['Construct'] == construct].copy()
    
    if df.empty:
        print(f"No data found for construct: {construct}")
        return None
    
    # If we have dimension cohesion columns
    dim_cols = [c for c in df.columns if 'Cohesion' in c and c != 'Overall Cohesion' and c != 'Min Cohesion']
    
    if not dim_cols:
        print(f"No dimension-level cohesion data found. Using overall cohesion.")
        return plot_model_kappa_comparison(model_comparison_df, construct, save_path=save_path, show=show)
    
    models = df['Model'].tolist()
    
    # Build data matrix
    data_matrix = df[dim_cols].values.T
    dimension_names = [c.replace(' Cohesion', '') for c in dim_cols]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    hm = sns.heatmap(
        data_matrix,
        annot=True,
        fmt='.2f',
        cmap=COHESION_CMAP,
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Cohesion', 'shrink': 0.8},
        annot_kws={'size': 9, 'weight': 'bold'},
        xticklabels=models,
        yticklabels=dimension_names,
        ax=ax
    )
    
    ax.set_title(f'Category Cohesion by Model - {construct}\nMethod: {method}',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Different Sentence Transformer Models', fontsize=12)
    ax.set_ylabel('Construct Dimensions', fontsize=12)
    
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


def create_cohesion_heatmap_from_integration_results(
    integration_results_path: Path,
    construct: str,
    method: str = 'Direct Embeddings',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create cohesion heatmap by parsing the detailed integration results.
    
    This function can read dimension cohesion data if you saved it separately.
    """
    set_style()
    
    # This would need a specific file format with dimension cohesions
    # For now, redirect to the manual data entry function
    print("To create a cohesion heatmap like your example, use:")
    print("  plot_cohesion_heatmap_manual(cohesion_data, construct, dimension_labels)")
    print("\nWhere cohesion_data is a dict: {model_name: {dimension_label: cohesion_value}}")
    
    return None


def plot_cohesion_heatmap_manual(
    cohesion_data: Dict[str, Dict[str, float]],
    construct: str,
    method: str = 'Direct Embeddings',
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[Path] = None,
    show: bool = True,
    vmin: float = 0.5,
    vmax: float = 1.0
) -> plt.Figure:
    """
    Create cohesion heatmap with manually provided data.
    
    Parameters
    ----------
    cohesion_data : Dict[str, Dict[str, float]]
        {model_name: {dimension_label: cohesion_value}}
        Example:
        {
            'BGE-large': {'Somatic Anxiety': 0.81, 'Cognitive Anxiety': 0.95},
            'BioBERT': {'Somatic Anxiety': 0.58, 'Cognitive Anxiety': 0.54},
            ...
        }
    construct : str
        Construct name (e.g., 'anxiety')
    method : str
        Method description for title
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
    
    # Get models and dimensions
    models = list(cohesion_data.keys())
    
    # Get all unique dimensions
    all_dims = set()
    for model_data in cohesion_data.values():
        all_dims.update(model_data.keys())
    dimensions = sorted(all_dims)
    
    # Build data matrix (dimensions x models)
    data_matrix = []
    for dim in dimensions:
        row = []
        for model in models:
            val = cohesion_data.get(model, {}).get(dim, np.nan)
            row.append(val)
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    hm = sns.heatmap(
        data_matrix,
        annot=True,
        fmt='.2f',
        cmap=COHESION_CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': '', 'shrink': 0.6},
        annot_kws={'size': 10, 'weight': 'bold'},
        xticklabels=models,
        yticklabels=dimensions,
        ax=ax
    )
    
    # Customize
    ax.set_title(f'Category Cohesion by Model - {construct}\nMethod: {method}',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Different Sentence Transformer Models', fontsize=12)
    ax.set_ylabel('Construct Dimensions', fontsize=12)
    
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
# Figure 2: Model Comparison Bar Charts
# =============================================================================

def plot_model_kappa_comparison(
    model_comparison_df: pd.DataFrame,
    construct: str,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create bar chart comparing Cohen's kappa across models.
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
    
    # Color best model differently
    colors = ['#2ecc71' if k == max(kappas) else '#3498db' for k in kappas]
    
    bars = ax.barh(range(len(models)), kappas, color=colors, edgecolor='white')
    
    # Add value labels
    for bar, val in zip(bars, kappas):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=10)
    
    # Threshold line
    ax.axvline(x=0.40, color='red', linestyle='--', linewidth=2,
              label='Minimum threshold (κ=0.40)')
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Cohen's κ", fontsize=12)
    ax.set_title(f"Model Comparison - {construct.replace('_', ' ').title()}\nCohen's κ (Expert-Cluster Agreement)",
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(kappas) * 1.15)
    
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
    ax.set_title(f'Model Comparison Across Constructs\n{metric}',
                fontsize=14, fontweight='bold')
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
# Figure 3: Similarity Matrix Heatmap
# =============================================================================

def plot_similarity_matrix_from_csv(
    similarity_csv_path: Path,
    item_assignments_df: pd.DataFrame,
    construct: str,
    model_name: str,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot similarity matrix heatmap from CSV file.
    """
    set_style()
    
    sim_df = pd.read_csv(similarity_csv_path, index_col=0)
    item_ids = sim_df.index.tolist()
    similarity_matrix = sim_df.values
    
    # Get expert assignments
    df_expert = item_assignments_df[item_assignments_df['Construct'] == construct]
    item_dims = dict(zip(df_expert['Item ID'], df_expert['Assigned Dimension']))
    
    dims = [item_dims.get(item, -1) for item in item_ids]
    labels = DIMENSION_LABELS.get(construct, {})
    dim_colors = DIMENSION_COLORS.get(construct, {})
    
    # Reorder by dimension
    order = np.argsort(dims)
    similarity_matrix = similarity_matrix[order][:, order]
    item_ids = [item_ids[i] for i in order]
    dims = [dims[i] for i in order]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    ax_main = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    ax_cbar = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    ax_dim_left = fig.add_axes([0.12, 0.15, 0.02, 0.7])
    ax_dim_top = fig.add_axes([0.15, 0.87, 0.7, 0.02])
    
    im = ax_main.imshow(similarity_matrix, cmap=SIMILARITY_CMAP, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, cax=ax_cbar, label='Cosine Similarity')
    
    # Dimension strips
    for i, d in enumerate(dims):
        color = dim_colors.get(d, '#CCCCCC')
        ax_dim_left.add_patch(plt.Rectangle((0, i), 1, 1, facecolor=color, edgecolor='none'))
        ax_dim_top.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='none'))
    
    ax_dim_left.set_xlim(0, 1)
    ax_dim_left.set_ylim(0, len(dims))
    ax_dim_left.axis('off')
    ax_dim_top.set_xlim(0, len(dims))
    ax_dim_top.set_ylim(0, 1)
    ax_dim_top.axis('off')
    
    # Tick labels
    n_items = len(item_ids)
    if n_items > 30:
        tick_step = max(1, n_items // 15)
        ax_main.set_xticks(range(0, n_items, tick_step))
        ax_main.set_xticklabels([item_ids[i] for i in range(0, n_items, tick_step)], rotation=90, fontsize=8)
        ax_main.set_yticks(range(0, n_items, tick_step))
        ax_main.set_yticklabels([item_ids[i] for i in range(0, n_items, tick_step)], fontsize=8)
    
    ax_main.set_xlabel('Items', fontsize=12)
    ax_main.set_ylabel('Items', fontsize=12)
    
    fig.suptitle(f'Similarity Matrix - {construct.title()}\nModel: {model_name}',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Legend
    unique_dims = sorted(set(dims))
    legend_patches = [mpatches.Patch(color=dim_colors.get(d, '#CCCCCC'),
                                     label=labels.get(d, f'Dim {d}'))
                     for d in unique_dims]
    fig.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.01, 0.99),
              fontsize=9, title='Dimensions')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Figure 4: Combined Evidence Score Distribution
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
            patch.set_facecolor('#e74c3c')
        elif left_edge < 0.50:
            patch.set_facecolor('#f39c12')
        else:
            patch.set_facecolor('#3498db')
    
    ax.axvline(x=0.40, color='red', linestyle='--', linewidth=2,
              label='Exclusion threshold (0.40)')
    ax.axvline(x=0.50, color='orange', linestyle='--', linewidth=2,
              label='Low confidence (0.50)')
    
    # Stats
    n_excluded = (scores < 0.40).sum()
    stats_text = f'Mean: {scores.mean():.3f}\nMedian: {np.median(scores):.3f}\nExcluded: {n_excluded}/{len(scores)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Combined Evidence Score (S_ik)', fontsize=12)
    ax.set_ylabel('Number of Items', fontsize=12)
    ax.set_title(f'Distribution of Combined Evidence Scores - {construct.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
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


def plot_expert_vs_cluster(
    final_assignments_df: pd.DataFrame,
    construct: str,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Scatter plot of expert vs cluster evidence.
    
    Uses jitter and transparency to show overlapping points.
    """
    set_style()
    
    df = final_assignments_df[final_assignments_df['Construct'] == construct].copy()
    
    if df.empty:
        print(f"No data for construct: {construct}")
        return None
    
    n_items = len(df)
    print(f"  Plotting {n_items} items for {construct}")
    
    # Get the values
    expert_ev = df['Expert Evidence (E_ik)'].values
    cluster_ev = df['Cluster Evidence (C_ik)'].values
    combined = df['Combined Score (S_ik)'].values
    
    # Add small jitter to reveal overlapping points
    np.random.seed(42)  # For reproducibility
    jitter_amount = 0.015
    expert_jittered = expert_ev + np.random.uniform(-jitter_amount, jitter_amount, len(expert_ev))
    cluster_jittered = cluster_ev + np.random.uniform(-jitter_amount, jitter_amount, len(cluster_ev))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use smaller, more transparent points to show density
    scatter = ax.scatter(
        expert_jittered, cluster_jittered, 
        c=combined, cmap='RdYlGn',
        s=80, alpha=0.6, edgecolor='white', linewidth=0.5
    )
    
    plt.colorbar(scatter, ax=ax, label='Combined Score')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect agreement')
    
    # Iso-score lines
    for s in [0.4, 0.6, 0.8]:
        e = np.linspace(0, 1, 100)
        c = (s - 0.6 * e) / 0.4
        valid = (c >= 0) & (c <= 1)
        ax.plot(e[valid], c[valid], '--', alpha=0.5, label=f'S = {s}')
    
    ax.set_xlabel('Expert Evidence (E_ik)', fontsize=12)
    ax.set_ylabel('Cluster Evidence (C_ik)', fontsize=12)
    ax.set_title(f'Expert vs Cluster Evidence - {construct.replace("_", " ").title()}\n(n = {n_items} items, jittered to show overlaps)',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal')
    
    # Add text showing how many unique coordinate pairs
    unique_pairs = len(set(zip(np.round(expert_ev, 2), np.round(cluster_ev, 2))))
    ax.text(0.02, 0.02, f'Unique positions (rounded): {unique_pairs}', 
           transform=ax.transAxes, fontsize=9,
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
# Figure 5: Expert Mappings Summary (Fleiss' Kappa)
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
    ci_lower = df['95% CI Lower'].tolist()
    ci_upper = df['95% CI Upper'].tolist()
    
    # Error bars
    errors = [[k - l for k, l in zip(kappas, ci_lower)],
              [u - k for k, u in zip(kappas, ci_upper)]]
    
    # Color by interpretation
    colors = []
    for k in kappas:
        if k >= 0.80:
            colors.append('#27ae60')  # Almost perfect
        elif k >= 0.60:
            colors.append('#2ecc71')  # Substantial
        elif k >= 0.40:
            colors.append('#f39c12')  # Moderate
        else:
            colors.append('#e74c3c')  # Fair/Poor
    
    bars = ax.barh(range(len(constructs)), kappas, xerr=errors, color=colors,
                  capsize=5, edgecolor='white')
    
    # Value labels
    for bar, val, upper in zip(bars, kappas, ci_upper):
        ax.text(upper + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=10)
    
    # Threshold lines
    ax.axvline(x=0.40, color='red', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax.axvline(x=0.60, color='orange', linestyle='--', alpha=0.7, label='Substantial threshold')
    ax.axvline(x=0.80, color='green', linestyle='--', alpha=0.7, label='Almost perfect threshold')
    
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
# Main Function
# =============================================================================

def generate_all_figures_from_csvs(
    expert_assignments_path: Optional[Path] = None,
    fleiss_summary_path: Optional[Path] = None,
    model_comparison_path: Optional[Path] = None,
    final_assignments_path: Optional[Path] = None,
    similarity_dir: Optional[Path] = None,
    cohesion_heatmap_dir: Optional[Path] = None,
    n_question_pairs: int = 5,
    output_dir: Path = Path('./figures'),
    show: bool = False
):
    """
    Generate all available figures from CSV files.
    
    Parameters
    ----------
    expert_assignments_path : Path
        Path to item_assignments.csv (from Step 3) - contains Item ID, Question Text, Construct
    fleiss_summary_path : Path
        Path to fleiss_kappa_summary.csv (from Step 3)
    model_comparison_path : Path
        Path to model_comparison.csv (from Steps 4-5)
    final_assignments_path : Path
        Path to final_assignments_combined.csv (from Steps 4-5)
    similarity_dir : Path
        Directory containing similarity matrices (for similarity matrix plots and question pair comparisons)
    cohesion_heatmap_dir : Path
        Directory containing cohesion CSV files (from save_cohesion_data_for_heatmap)
    n_question_pairs : int
        Number of question pairs to show in the comparison figure (default: 5)
    output_dir : Path
        Output directory for figures
    show : bool
        Whether to display figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GENERATING FIGURES FROM CSV FILES")
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
    
    generated = []
    
    # 0. Cohesion heatmaps (like your example)
    if cohesion_heatmap_dir and Path(cohesion_heatmap_dir).exists():
        print("\n0. Generating cohesion heatmaps...")
        
        for cohesion_file in Path(cohesion_heatmap_dir).glob('*_cohesion_by_model.csv'):
            construct = cohesion_file.stem.replace('_cohesion_by_model', '')
            path = output_dir / f'{construct}_cohesion_heatmap_embeddings.png'
            try:
                plot_cohesion_heatmap_from_csv(
                    cohesion_file, construct,
                    method='Direct Embeddings',
                    save_path=path, show=show
                )
                generated.append(path)
            except Exception as e:
                print(f"  Warning: Could not generate cohesion heatmap for {construct}: {e}")
    
    # 1. Fleiss' kappa summary
    if fleiss_df is not None:
        print("\n1. Generating Fleiss' kappa summary...")
        path = output_dir / 'fleiss_kappa_summary.png'
        plot_fleiss_kappa_summary(fleiss_df, save_path=path, show=show)
        generated.append(path)
    
    # 2. Model comparisons
    if model_df is not None:
        print("\n2. Generating model comparisons...")
        
        constructs = model_df['Construct'].unique()
        for construct in constructs:
            path = output_dir / f'{construct}_model_comparison.png'
            plot_model_kappa_comparison(model_df, construct, save_path=path, show=show)
            generated.append(path)
        
        # All constructs
        path = output_dir / 'all_constructs_model_comparison.png'
        plot_all_constructs_comparison(model_df, save_path=path, show=show)
        generated.append(path)
    
    # 3. Score distributions and scatter plots
    if final_df is not None:
        print("\n3. Generating score distributions...")
        
        constructs = final_df['Construct'].unique()
        for construct in constructs:
            # Distribution
            path = output_dir / f'{construct}_score_distribution.png'
            plot_score_distribution(final_df, construct, save_path=path, show=show)
            generated.append(path)
            
            # Scatter
            path = output_dir / f'{construct}_expert_vs_cluster.png'
            plot_expert_vs_cluster(final_df, construct, save_path=path, show=show)
            generated.append(path)
    
    # 4. Similarity matrices
    if similarity_dir and Path(similarity_dir).exists() and expert_df is not None:
        print("\n4. Generating similarity matrices...")
        
        for sim_file in Path(similarity_dir).glob('*_similarity.csv'):
            parts = sim_file.stem.split('_similarity')[0].rsplit('_', 1)
            if len(parts) == 2:
                construct, model = parts
            else:
                continue
            
            path = output_dir / f'{construct}_similarity_matrix_{model}.png'
            try:
                plot_similarity_matrix_from_csv(
                    sim_file, expert_df, construct, model,
                    save_path=path, show=show
                )
                generated.append(path)
            except Exception as e:
                print(f"  Warning: Could not generate similarity matrix for {construct}/{model}: {e}")
    
    # 5. Question pair comparison heatmaps
    if similarity_dir and Path(similarity_dir).exists() and expert_df is not None:
        print("\n5. Generating question pair comparison heatmaps...")
        
        # Get constructs from similarity directory structure
        similarity_dir = Path(similarity_dir)
        
        # Check if similarity_dir contains subdirectories per construct
        construct_dirs = [d for d in similarity_dir.iterdir() if d.is_dir()]
        
        if construct_dirs:
            # Structure: similarity_dir/{construct}/*.csv
            for construct_dir in construct_dirs:
                construct = construct_dir.name
                path = output_dir / f'{construct}_question_pair_comparison.png'
                try:
                    plot_question_pair_comparison_from_csvs(
                        expert_assignments_csv=expert_assignments_path,
                        similarity_dir=construct_dir,
                        construct=construct,
                        n_pairs=n_question_pairs,
                        save_path=path,
                        show=show
                    )
                    generated.append(path)
                except Exception as e:
                    print(f"  Warning: Could not generate question pair comparison for {construct}: {e}")
        else:
            # Structure: similarity_dir/{construct}_{model}_similarity.csv
            # Group files by construct
            constructs = set()
            for sim_file in similarity_dir.glob('*_similarity.csv'):
                parts = sim_file.stem.replace('_similarity', '').rsplit('_', 1)
                if len(parts) >= 1:
                    # First part before last underscore is construct
                    construct = parts[0]
                    constructs.add(construct)
            
            for construct in constructs:
                path = output_dir / f'{construct}_question_pair_comparison.png'
                try:
                    plot_question_pair_comparison_from_csvs(
                        expert_assignments_csv=expert_assignments_path,
                        similarity_dir=similarity_dir,
                        construct=construct,
                        n_pairs=n_question_pairs,
                        save_path=path,
                        show=show
                    )
                    generated.append(path)
                except Exception as e:
                    print(f"  Warning: Could not generate question pair comparison for {construct}: {e}")
    
    print("\n" + "="*60)
    print(f"Generated {len(generated)} figures in {output_dir}")
    print("="*60)
    
    return generated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate figures from ENIGMA-PD CSV results")
    parser.add_argument("--expert-assignments", type=str, help="Path to item_assignments.csv (contains Item ID, Question Text, Construct)")
    parser.add_argument("--fleiss-summary", type=str, help="Path to fleiss_kappa_summary.csv")
    parser.add_argument("--model-comparison", type=str, help="Path to model_comparison.csv")
    parser.add_argument("--final-assignments", type=str, help="Path to final_assignments_combined.csv")
    parser.add_argument("--similarity-dir", type=str, help="Directory with similarity CSVs")
    parser.add_argument("--cohesion-heatmap-dir", type=str, help="Directory with cohesion CSVs for heatmaps")
    parser.add_argument("--n-question-pairs", type=int, default=5, help="Number of question pairs for comparison figure (default: 5)")
    parser.add_argument("--output-dir", "-o", type=str, default="./figures", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display figures")
    
    args = parser.parse_args()
    
    generate_all_figures_from_csvs(
        expert_assignments_path=args.expert_assignments,
        fleiss_summary_path=args.fleiss_summary,
        model_comparison_path=args.model_comparison,
        final_assignments_path=args.final_assignments,
        similarity_dir=args.similarity_dir,
        cohesion_heatmap_dir=args.cohesion_heatmap_dir,
        n_question_pairs=args.n_question_pairs,
        output_dir=Path(args.output_dir),
        show=args.show
    )
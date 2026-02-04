"""
Semantic Similarity Analysis and Integration of Mappings
=========================================================

ENIGMA-PD Questionnaire Harmonization - Steps 4 & 5

Step 4: Semantic Similarity Analysis and Unsupervised Clustering
- Compute embeddings using multiple sentence transformer models
- Calculate cosine similarity matrices
- Run unsupervised hierarchical clustering on embeddings and similarity matrices
- Generate data-driven item clusters

Step 5: Integration of Mappings
- Calculate agreement metrics (Cohen's kappa, dimension cohesion) for model selection
- Select best-performing embedding model
- Compute combined evidence scores integrating expert and cluster evidence
- Generate final item-to-dimension assignments

References:
- Cohen, J. (1960). A coefficient of agreement for nominal scales.
- Landis, J. R. & Koch, G. G. (1977). The measurement of observer agreement.
- Various embedding model papers (see registered report)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import normalize
import json

# Optional imports - will check availability
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# =============================================================================
# Configuration: Embedding Models to Test
# =============================================================================

# Models from the registered report 
EMBEDDING_MODELS = {
    # High-performance general models
    'bge-large': 'BAAI/bge-large-en-v1.5',
    'e5-large': 'intfloat/e5-large-v2',
    'instructor': 'hkunlp/instructor-large',
    'gtr-t5': 'sentence-transformers/gtr-t5-large',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'minilm': 'sentence-transformers/all-MiniLM-L12-v2',
    'simcse': 'princeton-nlp/sup-simcse-roberta-large',
    'sbert-similarity': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
    
    # Clinical/biomedical models
    'clinicalbert': 'medicalai/ClinicalBERT',
    'bio-clinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
    'biobert': 'dmis-lab/biobert-v1.1',
    'clinicalSentenceTransformer': 'Shobhank-iiitdwd/Clinical_sentence_transformers_mpnet_base_v2',
    
    # Mental health specific models
    'mentalroberta': 'mental/mental-roberta-base',
    'harmony': 'harmonydata/mental_health_harmonisation_1',
}

# Lighter subset for faster testing
EMBEDDING_MODELS_LITE = {
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'minilm': 'sentence-transformers/all-MiniLM-L12-v2',
    'sbert-similarity': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EmbeddingResult:
    """Results from embedding a set of items."""
    model_name: str
    model_path: str
    embeddings: np.ndarray  # Shape: (n_items, embedding_dim)
    item_ids: List[str]
    construct: str
    embedding_dim: int


@dataclass
class SimilarityResult:
    """Similarity matrix and related data."""
    model_name: str
    similarity_matrix: np.ndarray  # Shape: (n_items, n_items)
    item_ids: List[str]
    construct: str


@dataclass
class ClusteringResult:
    """Results from hierarchical clustering."""
    model_name: str
    construct: str
    cluster_labels: np.ndarray  # Cluster assignment for each item
    n_clusters: int
    item_ids: List[str]
    linkage_matrix: np.ndarray
    clustering_method: str  # 'embeddings' or 'similarity'
    linkage_method: str  # 'ward', 'average', etc.


@dataclass
class ModelAgreementMetrics:
    """Agreement metrics between expert and model-based assignments."""
    model_name: str
    construct: str
    cohens_kappa: float
    overall_cohesion: float
    min_cohesion: float
    dimension_cohesions: Dict[int, float]
    n_items: int
    expert_assignments: List[int]
    cluster_assignments: List[int]


@dataclass
class CombinedEvidenceScore:
    """Combined evidence score for an item-dimension pair."""
    item_id: str
    construct: str
    dimension: int
    dimension_label: str
    expert_evidence: float  # E_ik
    cluster_evidence: float  # C_ik
    combined_score: float  # S_ik
    is_assigned: bool  # True if this is the assigned dimension
    confidence: str  # 'high', 'moderate', 'low', 'excluded'


@dataclass 
class FinalAssignment:
    """Final item-to-dimension assignment using combined evidence."""
    item_id: str
    construct: str
    questionnaire: str
    question_text: str
    assigned_dimension: int
    dimension_label: str
    combined_score: float
    expert_evidence: float
    cluster_evidence: float
    is_excluded: bool  # True if S_ik < 0.40
    is_ambiguous: bool  # True if top two dimensions within 0.05
    competing_dimensions: List[int]
    all_scores: Dict[int, float]  # S_ik for all dimensions


@dataclass
class IntegrationResults:
    """Complete results from the integration step."""
    construct: str
    best_model: str
    model_metrics: Dict[str, ModelAgreementMetrics]
    final_assignments: List[FinalAssignment]
    n_excluded: int
    n_ambiguous: int
    sensitivity_results: Optional[Dict[str, List[FinalAssignment]]] = None


# =============================================================================
# Step 4: Embedding and Similarity Functions
# =============================================================================

def load_items_for_embedding(filepath: Union[str, Path], 
                             construct: Optional[str] = None) -> pd.DataFrame:
    """
    Load questionnaire items for embedding.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the items CSV file
    construct : str, optional
        Filter to specific construct
    
    Returns
    -------
    pd.DataFrame
        DataFrame with item information
    """
    df = pd.read_csv(filepath)
    
    if construct:
        df = df[df['construct'] == construct].copy()
    
    return df


def compute_embeddings(texts: List[str], 
                      model_name: str,
                      model_path: str,
                      instruction: Optional[str] = None) -> np.ndarray:
    """
    Compute embeddings for a list of texts using a sentence transformer model.
    
    Parameters
    ----------
    texts : List[str]
        List of text strings to embed
    model_name : str
        Short name of the model
    model_path : str
        HuggingFace model path
    instruction : str, optional
        Instruction prefix for instructor-style models
    
    Returns
    -------
    np.ndarray
        Embeddings matrix of shape (n_texts, embedding_dim)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    print(f"  Loading model: {model_name} ({model_path})")
    
    try:
        model = SentenceTransformer(model_path)
    except Exception as e:
        warnings.warn(f"Failed to load model {model_name}: {e}")
        return None
    
    # Handle instructor-style models
    if 'instructor' in model_name.lower() and instruction:
        texts = [[instruction, text] for text in texts]
    
    print(f"  Computing embeddings for {len(texts)} items...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix from embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings matrix of shape (n_items, embedding_dim)
    
    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_items, n_items)
    """
    # Normalize embeddings
    normalized = normalize(embeddings, axis=1)
    
    # Compute cosine similarity
    similarity = np.dot(normalized, normalized.T)
    
    return similarity


def run_hierarchical_clustering(data: np.ndarray,
                                n_clusters: int,
                                method: str = 'ward',
                                metric: str = 'euclidean',
                                is_similarity: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run hierarchical clustering on embeddings or similarity matrix.
    
    Parameters
    ----------
    data : np.ndarray
        Either embeddings (n_items, embedding_dim) or similarity matrix (n_items, n_items)
    n_clusters : int
        Number of clusters to form
    method : str
        Linkage method ('ward', 'average', 'complete', 'single')
    metric : str
        Distance metric (ignored if is_similarity=True)
    is_similarity : bool
        If True, data is a similarity matrix; convert to distance
    
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each item
    linkage_matrix : np.ndarray
        Linkage matrix for dendrogram plotting
    """
    if is_similarity:
        # Convert similarity to distance
        # Ensure values are in [0, 1] range
        data = np.clip(data, 0, 1)
        distance_matrix = 1 - data
        np.fill_diagonal(distance_matrix, 0)
        
        # Convert to condensed form
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Can't use 'ward' with precomputed distances
        if method == 'ward':
            method = 'average'
        
        linkage_matrix = linkage(condensed_dist, method=method)
    else:
        # Cluster on embeddings directly
        linkage_matrix = linkage(data, method=method, metric=metric)
    
    # Cut tree to get cluster labels
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    return labels, linkage_matrix


def embed_and_cluster_construct(df: pd.DataFrame,
                                construct: str,
                                models: Dict[str, str],
                                n_clusters: int,
                                linkage_method: str = 'ward') -> Dict[str, Dict]:
    """
    Compute embeddings and clustering for a construct using multiple models.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with items (filtered to construct)
    construct : str
        Name of the construct
    models : Dict[str, str]
        Dictionary mapping model names to HuggingFace paths
    n_clusters : int
        Number of clusters (typically number of dimensions)
    linkage_method : str
        Linkage method for hierarchical clustering
    
    Returns
    -------
    Dict[str, Dict]
        Results for each model containing embeddings, similarities, and clusters
    """
    df_construct = df[df['construct'] == construct].copy()
    texts = df_construct['question_text'].tolist()
    item_ids = df_construct['question_number'].tolist()
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\nProcessing model: {model_name}")
        
        try:
            # Compute embeddings
            embeddings = compute_embeddings(texts, model_name, model_path)
            
            if embeddings is None:
                continue
            
            # Compute similarity matrix
            similarity = compute_similarity_matrix(embeddings)
            
            # Cluster on embeddings
            labels_emb, linkage_emb = run_hierarchical_clustering(
                embeddings, n_clusters, method=linkage_method, is_similarity=False
            )
            
            # Cluster on similarity matrix
            labels_sim, linkage_sim = run_hierarchical_clustering(
                similarity, n_clusters, method='average', is_similarity=True
            )
            
            results[model_name] = {
                'embeddings': EmbeddingResult(
                    model_name=model_name,
                    model_path=model_path,
                    embeddings=embeddings,
                    item_ids=item_ids,
                    construct=construct,
                    embedding_dim=embeddings.shape[1]
                ),
                'similarity': SimilarityResult(
                    model_name=model_name,
                    similarity_matrix=similarity,
                    item_ids=item_ids,
                    construct=construct
                ),
                'clustering_embeddings': ClusteringResult(
                    model_name=model_name,
                    construct=construct,
                    cluster_labels=labels_emb,
                    n_clusters=n_clusters,
                    item_ids=item_ids,
                    linkage_matrix=linkage_emb,
                    clustering_method='embeddings',
                    linkage_method=linkage_method
                ),
                'clustering_similarity': ClusteringResult(
                    model_name=model_name,
                    construct=construct,
                    cluster_labels=labels_sim,
                    n_clusters=n_clusters,
                    item_ids=item_ids,
                    linkage_matrix=linkage_sim,
                    clustering_method='similarity',
                    linkage_method='average'
                )
            }
            
        except Exception as e:
            warnings.warn(f"Error processing model {model_name}: {e}")
            continue
    
    return results


# =============================================================================
# Step 5: Agreement Metrics and Model Selection
# =============================================================================

def get_expert_assignments(item_assignments_df: pd.DataFrame, 
                          construct: str) -> Dict[str, int]:
    """
    Get expert-assigned dimensions from the expert mappings output.
    
    Parameters
    ----------
    item_assignments_df : pd.DataFrame
        DataFrame from item_assignments.csv (output of expert_mappings_analysis.py)
    construct : str
        Name of the construct
    
    Returns
    -------
    Dict[str, int]
        Mapping from item_id to assigned dimension
    """
    df_construct = item_assignments_df[item_assignments_df['Construct'] == construct]
    return dict(zip(df_construct['Item ID'], df_construct['Assigned Dimension']))


def get_cluster_to_dimension_mapping(cluster_labels: np.ndarray,
                                     item_ids: List[str],
                                     expert_assignments: Dict[str, int]) -> Dict[int, int]:
    """
    Map each cluster to the dimension most represented in it.
    
    For each cluster, find which expert-assigned dimension has the most items.
    
    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster assignment for each item
    item_ids : List[str]
        Item identifiers
    expert_assignments : Dict[str, int]
        Expert-assigned dimension for each item
    
    Returns
    -------
    Dict[int, int]
        Mapping from cluster label to dimension
    """
    cluster_to_dim = {}
    
    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        # Get items in this cluster
        cluster_mask = cluster_labels == cluster
        cluster_items = [item_ids[i] for i in range(len(item_ids)) if cluster_mask[i]]
        
        # Count dimensions in this cluster
        dim_counts = {}
        for item in cluster_items:
            if item in expert_assignments:
                dim = expert_assignments[item]
                dim_counts[dim] = dim_counts.get(dim, 0) + 1
        
        # Assign cluster to most common dimension
        if dim_counts:
            cluster_to_dim[cluster] = max(dim_counts, key=dim_counts.get)
        else:
            cluster_to_dim[cluster] = -1  # No expert assignments found
    
    return cluster_to_dim


def calculate_cohens_kappa(expert_assignments: Dict[str, int],
                          cluster_labels: np.ndarray,
                          item_ids: List[str],
                          cluster_to_dim: Dict[int, int]) -> float:
    """
    Calculate Cohen's kappa between expert and cluster-based assignments.
    
    Parameters
    ----------
    expert_assignments : Dict[str, int]
        Expert-assigned dimension for each item
    cluster_labels : np.ndarray
        Cluster assignment for each item
    item_ids : List[str]
        Item identifiers
    cluster_to_dim : Dict[int, int]
        Mapping from cluster to dimension
    
    Returns
    -------
    float
        Cohen's kappa value
    """
    expert_labels = []
    cluster_dims = []
    
    for i, item_id in enumerate(item_ids):
        if item_id in expert_assignments:
            expert_labels.append(expert_assignments[item_id])
            cluster_dims.append(cluster_to_dim[cluster_labels[i]])
    
    if len(expert_labels) < 2:
        return np.nan
    
    return cohen_kappa_score(expert_labels, cluster_dims)


def calculate_dimension_cohesion(cluster_labels: np.ndarray,
                                 item_ids: List[str],
                                 expert_assignments: Dict[str, int]) -> Tuple[float, float, Dict[int, float]]:
    """
    Calculate dimension cohesion metrics.
    
    Dimension cohesion = (items in dominant cluster) / (total expert-assigned items)
    
    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster assignment for each item
    item_ids : List[str]
        Item identifiers  
    expert_assignments : Dict[str, int]
        Expert-assigned dimension for each item
    
    Returns
    -------
    overall_cohesion : float
        Average cohesion across dimensions
    min_cohesion : float
        Minimum cohesion across dimensions
    dimension_cohesions : Dict[int, float]
        Cohesion for each dimension
    """
    # Group items by expert-assigned dimension
    dim_to_items = {}
    for item_id, dim in expert_assignments.items():
        if dim not in dim_to_items:
            dim_to_items[dim] = []
        dim_to_items[dim].append(item_id)
    
    # Create item_id to index mapping
    item_to_idx = {item: i for i, item in enumerate(item_ids)}
    
    dimension_cohesions = {}
    
    for dim, items in dim_to_items.items():
        # Get cluster labels for items in this dimension
        clusters = []
        for item in items:
            if item in item_to_idx:
                clusters.append(cluster_labels[item_to_idx[item]])
        
        if not clusters:
            dimension_cohesions[dim] = 0.0
            continue
        
        # Find dominant cluster
        cluster_counts = {}
        for c in clusters:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
        
        max_count = max(cluster_counts.values())
        cohesion = max_count / len(clusters)
        dimension_cohesions[dim] = cohesion
    
    if dimension_cohesions:
        overall_cohesion = np.mean(list(dimension_cohesions.values()))
        min_cohesion = min(dimension_cohesions.values())
    else:
        overall_cohesion = 0.0
        min_cohesion = 0.0
    
    return overall_cohesion, min_cohesion, dimension_cohesions


def evaluate_model_agreement(clustering_result: ClusteringResult,
                            expert_assignments: Dict[str, int]) -> ModelAgreementMetrics:
    """
    Evaluate agreement between model clustering and expert assignments.
    
    Parameters
    ----------
    clustering_result : ClusteringResult
        Clustering results for a model
    expert_assignments : Dict[str, int]
        Expert-assigned dimensions
    
    Returns
    -------
    ModelAgreementMetrics
        Agreement metrics for this model
    """
    cluster_labels = clustering_result.cluster_labels
    item_ids = clustering_result.item_ids
    
    # Map clusters to dimensions
    cluster_to_dim = get_cluster_to_dimension_mapping(
        cluster_labels, item_ids, expert_assignments
    )
    
    # Calculate Cohen's kappa
    kappa = calculate_cohens_kappa(
        expert_assignments, cluster_labels, item_ids, cluster_to_dim
    )
    
    # Calculate dimension cohesion
    overall_coh, min_coh, dim_cohesions = calculate_dimension_cohesion(
        cluster_labels, item_ids, expert_assignments
    )
    
    # Get assignments for output
    expert_list = []
    cluster_list = []
    for i, item_id in enumerate(item_ids):
        if item_id in expert_assignments:
            expert_list.append(expert_assignments[item_id])
            cluster_list.append(cluster_to_dim[cluster_labels[i]])
    
    return ModelAgreementMetrics(
        model_name=clustering_result.model_name,
        construct=clustering_result.construct,
        cohens_kappa=kappa,
        overall_cohesion=overall_coh,
        min_cohesion=min_coh,
        dimension_cohesions=dim_cohesions,
        n_items=len(item_ids),
        expert_assignments=expert_list,
        cluster_assignments=cluster_list
    )


def select_best_model(model_metrics: Dict[str, ModelAgreementMetrics],
                     min_kappa_threshold: float = 0.40,
                     kappa_tolerance: float = 0.05) -> Tuple[str, ModelAgreementMetrics]:
    """
    Select the best-performing embedding model based on agreement metrics.
    
    Selection criteria (from registered report):
    1. Primary: Highest Cohen's kappa (minimum threshold κ ≥ 0.40)
    2. Secondary: If multiple models within 0.05 of highest κ, select highest min cohesion
    
    Parameters
    ----------
    model_metrics : Dict[str, ModelAgreementMetrics]
        Agreement metrics for each model
    min_kappa_threshold : float
        Minimum acceptable kappa (default: 0.40)
    kappa_tolerance : float
        Tolerance for considering models equivalent (default: 0.05)
    
    Returns
    -------
    best_model : str
        Name of the best model
    best_metrics : ModelAgreementMetrics
        Metrics for the best model
    """
    # Filter models meeting minimum threshold
    valid_models = {
        name: metrics for name, metrics in model_metrics.items()
        if not np.isnan(metrics.cohens_kappa) and metrics.cohens_kappa >= min_kappa_threshold
    }
    
    if not valid_models:
        warnings.warn(f"No models achieved κ ≥ {min_kappa_threshold}. "
                     "Proceeding with expert-only assignments.")
        # Return the model with highest kappa anyway
        best_name = max(model_metrics, key=lambda x: model_metrics[x].cohens_kappa 
                       if not np.isnan(model_metrics[x].cohens_kappa) else -1)
        return best_name, model_metrics[best_name]
    
    # Find highest kappa
    max_kappa = max(m.cohens_kappa for m in valid_models.values())
    
    # Find models within tolerance of highest
    top_models = {
        name: metrics for name, metrics in valid_models.items()
        if max_kappa - metrics.cohens_kappa <= kappa_tolerance
    }
    
    # If tie, use minimum cohesion as tiebreaker
    if len(top_models) > 1:
        best_name = max(top_models, key=lambda x: top_models[x].min_cohesion)
    else:
        best_name = list(top_models.keys())[0]
    
    return best_name, valid_models[best_name]


# =============================================================================
# Step 5: Combined Evidence Scores and Final Assignment
# =============================================================================

def calculate_cluster_evidence(item_id: str,
                              cluster_labels: np.ndarray,
                              item_ids: List[str],
                              expert_assignments: Dict[str, int],
                              dimensions: List[int]) -> Dict[int, float]:
    """
    Calculate cluster evidence C_ik for each dimension.
    
    C_ik = proportion of items in item i's cluster that belong to dimension k
    
    Parameters
    ----------
    item_id : str
        Item identifier
    cluster_labels : np.ndarray
        Cluster assignments
    item_ids : List[str]
        All item identifiers
    expert_assignments : Dict[str, int]
        Expert-assigned dimensions
    dimensions : List[int]
        List of possible dimensions
    
    Returns
    -------
    Dict[int, float]
        C_ik for each dimension k
    """
    # Find item's cluster
    item_idx = item_ids.index(item_id)
    item_cluster = cluster_labels[item_idx]
    
    # Get all items in this cluster
    cluster_items = [item_ids[i] for i in range(len(item_ids)) 
                    if cluster_labels[i] == item_cluster]
    
    # Count dimensions in cluster
    dim_counts = {d: 0 for d in dimensions}
    total = 0
    
    for item in cluster_items:
        if item in expert_assignments:
            dim = expert_assignments[item]
            if dim in dim_counts:
                dim_counts[dim] += 1
            total += 1
    
    # Calculate proportions
    if total > 0:
        return {d: count / total for d, count in dim_counts.items()}
    else:
        return {d: 0.0 for d in dimensions}


def calculate_combined_evidence_scores(
    item_id: str,
    construct: str,
    expert_proportions: Dict[int, float],
    cluster_evidence: Dict[int, float],
    dimension_labels: Dict[int, str],
    w_expert: float = 0.60,
    w_cluster: float = 0.40
) -> List[CombinedEvidenceScore]:
    """
    Calculate combined evidence scores S_ik for all dimensions.
    
    S_ik = (w_E × E_ik) + (w_C × C_ik)
    
    Parameters
    ----------
    item_id : str
        Item identifier
    construct : str
        Construct name
    expert_proportions : Dict[int, float]
        E_ik - proportion of experts assigning to each dimension
    cluster_evidence : Dict[int, float]
        C_ik - cluster evidence for each dimension
    dimension_labels : Dict[int, str]
        Human-readable dimension labels
    w_expert : float
        Weight for expert evidence (default: 0.60)
    w_cluster : float
        Weight for cluster evidence (default: 0.40)
    
    Returns
    -------
    List[CombinedEvidenceScore]
        Combined scores for each dimension
    """
    scores = []
    all_dims = set(expert_proportions.keys()) | set(cluster_evidence.keys())
    
    # Calculate S_ik for each dimension
    combined_scores = {}
    for dim in all_dims:
        e_ik = expert_proportions.get(dim, 0.0)
        c_ik = cluster_evidence.get(dim, 0.0)
        s_ik = (w_expert * e_ik) + (w_cluster * c_ik)
        combined_scores[dim] = s_ik
    
    # Find best dimension
    best_dim = max(combined_scores, key=combined_scores.get)
    best_score = combined_scores[best_dim]
    
    for dim in all_dims:
        s_ik = combined_scores[dim]
        
        # Determine confidence
        if s_ik >= 0.70:
            confidence = 'high'
        elif s_ik >= 0.50:
            confidence = 'moderate'
        elif s_ik >= 0.40:
            confidence = 'low'
        else:
            confidence = 'excluded'
        
        scores.append(CombinedEvidenceScore(
            item_id=item_id,
            construct=construct,
            dimension=dim,
            dimension_label=dimension_labels.get(dim, f"Dimension {dim}"),
            expert_evidence=expert_proportions.get(dim, 0.0),
            cluster_evidence=cluster_evidence.get(dim, 0.0),
            combined_score=s_ik,
            is_assigned=(dim == best_dim),
            confidence=confidence
        ))
    
    return scores


def generate_final_assignments(
    df_items: pd.DataFrame,
    item_assignments_df: pd.DataFrame,
    clustering_result: ClusteringResult,
    construct: str,
    dimension_labels: Dict[int, str],
    w_expert: float = 0.60,
    w_cluster: float = 0.40,
    exclusion_threshold: float = 0.40,
    ambiguity_threshold: float = 0.05
) -> List[FinalAssignment]:
    """
    Generate final item-to-dimension assignments using combined evidence.
    
    Parameters
    ----------
    df_items : pd.DataFrame
        Original items dataframe
    item_assignments_df : pd.DataFrame
        Expert assignments from step 3
    clustering_result : ClusteringResult
        Best model's clustering results
    construct : str
        Construct name
    dimension_labels : Dict[int, str]
        Dimension labels
    w_expert : float
        Expert weight (default: 0.60)
    w_cluster : float
        Cluster weight (default: 0.40)
    exclusion_threshold : float
        Minimum S_ik for inclusion (default: 0.40)
    ambiguity_threshold : float
        Threshold for flagging ambiguity (default: 0.05)
    
    Returns
    -------
    List[FinalAssignment]
        Final assignments for all items
    """
    df_construct = df_items[df_items['construct'] == construct]
    df_expert = item_assignments_df[item_assignments_df['Construct'] == construct]
    
    # Get expert assignments and proportions
    expert_assignments = dict(zip(df_expert['Item ID'], df_expert['Assigned Dimension']))
    
    # Parse rating distributions to get expert proportions
    # Format: "Dim 1: 80%; Dim 2: 20%"
    def parse_distribution(dist_str):
        props = {}
        if pd.isna(dist_str) or not dist_str:
            return props
        for part in dist_str.split(';'):
            part = part.strip()
            if ':' in part:
                dim_part, pct_part = part.split(':')
                dim = int(dim_part.replace('Dim', '').strip())
                pct = float(pct_part.replace('%', '').strip()) / 100
                props[dim] = pct
        return props
    
    expert_proportions_map = {}
    for _, row in df_expert.iterrows():
        expert_proportions_map[row['Item ID']] = parse_distribution(row['Rating Distribution'])
    
    # Get all dimensions for this construct
    dimensions = list(dimension_labels.keys())
    
    cluster_labels = clustering_result.cluster_labels
    item_ids = clustering_result.item_ids
    
    final_assignments = []
    
    for _, row in df_construct.iterrows():
        item_id = row['question_number']
        
        # Get expert proportions
        expert_props = expert_proportions_map.get(item_id, {})
        
        # Calculate cluster evidence
        cluster_ev = calculate_cluster_evidence(
            item_id, cluster_labels, item_ids, expert_assignments, dimensions
        )
        
        # Calculate combined scores for all dimensions
        all_scores = {}
        for dim in dimensions:
            e_ik = expert_props.get(dim, 0.0)
            c_ik = cluster_ev.get(dim, 0.0)
            s_ik = (w_expert * e_ik) + (w_cluster * c_ik)
            all_scores[dim] = s_ik
        
        # Find best dimension
        sorted_dims = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        best_dim, best_score = sorted_dims[0]
        
        # Check for exclusion
        is_excluded = best_score < exclusion_threshold
        
        # Check for ambiguity
        competing = []
        if len(sorted_dims) > 1:
            second_best_score = sorted_dims[1][1]
            if best_score - second_best_score <= ambiguity_threshold:
                competing = [d for d, s in sorted_dims if best_score - s <= ambiguity_threshold]
        
        is_ambiguous = len(competing) > 1
        
        final_assignments.append(FinalAssignment(
            item_id=item_id,
            construct=construct,
            questionnaire=row['questionnaire_name'],
            question_text=row['question_text'],
            assigned_dimension=best_dim,
            dimension_label=dimension_labels.get(best_dim, f"Dimension {best_dim}"),
            combined_score=best_score,
            expert_evidence=expert_props.get(best_dim, 0.0),
            cluster_evidence=cluster_ev.get(best_dim, 0.0),
            is_excluded=is_excluded,
            is_ambiguous=is_ambiguous,
            competing_dimensions=competing,
            all_scores=all_scores
        ))
    
    return final_assignments


def run_sensitivity_analysis(
    df_items: pd.DataFrame,
    item_assignments_df: pd.DataFrame,
    clustering_result: ClusteringResult,
    construct: str,
    dimension_labels: Dict[int, str]
) -> Dict[str, List[FinalAssignment]]:
    """
    Run sensitivity analysis with different weight schemes.
    
    Weight schemes from registered report:
    - 0.70/0.30 (more expert-weighted)
    - 0.60/0.40 (default)
    - 0.50/0.50 (equal weighting)
    - 0.80/0.20 (heavily expert-weighted)
    
    Returns
    -------
    Dict[str, List[FinalAssignment]]
        Assignments for each weight scheme
    """
    weight_schemes = {
        '0.80_0.20': (0.80, 0.20),
        '0.70_0.30': (0.70, 0.30),
        '0.60_0.40': (0.60, 0.40),
        '0.50_0.50': (0.50, 0.50),
    }
    
    results = {}
    
    for scheme_name, (w_e, w_c) in weight_schemes.items():
        assignments = generate_final_assignments(
            df_items, item_assignments_df, clustering_result,
            construct, dimension_labels, w_expert=w_e, w_cluster=w_c
        )
        results[scheme_name] = assignments
    
    return results


# =============================================================================
# Output Functions
# =============================================================================

def create_model_comparison_table(all_metrics: Dict[str, Dict[str, ModelAgreementMetrics]]) -> pd.DataFrame:
    """Create a table comparing all models across constructs."""
    rows = []
    
    for construct, model_metrics in all_metrics.items():
        for model_name, metrics in model_metrics.items():
            rows.append({
                'Construct': construct,
                'Model': model_name,
                "Cohen's κ": round(metrics.cohens_kappa, 3) if not np.isnan(metrics.cohens_kappa) else 'N/A',
                'Overall Cohesion': round(metrics.overall_cohesion, 3),
                'Min Cohesion': round(metrics.min_cohesion, 3),
                'N Items': metrics.n_items
            })
    
    return pd.DataFrame(rows)


def create_dimension_cohesion_table(
    all_metrics: Dict[str, Dict[str, ModelAgreementMetrics]],
    dimension_labels: Dict[str, Dict[int, str]]
) -> pd.DataFrame:
    """
    Create a table with dimension-level cohesion for each model.
    
    This table can be used to generate cohesion heatmaps.
    
    Returns a DataFrame with columns:
    - Construct
    - Model  
    - Dimension (numeric)
    - Dimension Label
    - Cohesion
    """
    rows = []
    
    for construct, model_metrics in all_metrics.items():
        labels = dimension_labels.get(construct, {})
        
        for model_name, metrics in model_metrics.items():
            for dim, cohesion in metrics.dimension_cohesions.items():
                rows.append({
                    'Construct': construct,
                    'Model': model_name,
                    'Dimension': dim,
                    'Dimension Label': labels.get(dim, f'Dimension {dim}'),
                    'Cohesion': round(cohesion, 3)
                })
    
    return pd.DataFrame(rows)


def save_cohesion_data_for_heatmap(
    all_metrics: Dict[str, Dict[str, ModelAgreementMetrics]],
    dimension_labels: Dict[str, Dict[int, str]],
    output_dir: Path
):
    """
    Save cohesion data in a format suitable for heatmap generation.
    
    Creates one CSV per construct with models as columns and dimensions as rows.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for construct, model_metrics in all_metrics.items():
        labels = dimension_labels.get(construct, {})
        
        # Get all dimensions and models
        all_dims = set()
        for metrics in model_metrics.values():
            all_dims.update(metrics.dimension_cohesions.keys())
        
        models = list(model_metrics.keys())
        dims = sorted(all_dims)
        
        # Build matrix
        data = {}
        for model in models:
            metrics = model_metrics[model]
            data[model] = [metrics.dimension_cohesions.get(d, 0.0) for d in dims]
        
        df = pd.DataFrame(data, index=[labels.get(d, f'Dimension {d}') for d in dims])
        df.to_csv(output_dir / f'{construct}_cohesion_by_model.csv')
        
    print(f"Saved cohesion data for heatmaps to {output_dir}/")


def create_final_assignments_table(all_assignments: Dict[str, List[FinalAssignment]]) -> pd.DataFrame:
    """Create a table of all final assignments."""
    rows = []
    
    for construct, assignments in all_assignments.items():
        for a in assignments:
            rows.append({
                'Construct': construct,
                'Questionnaire': a.questionnaire,
                'Item ID': a.item_id,
                'Question Text': a.question_text[:100] + '...' if len(a.question_text) > 100 else a.question_text,
                'Assigned Dimension': a.assigned_dimension,
                'Dimension Label': a.dimension_label,
                'Combined Score (S_ik)': round(a.combined_score, 3),
                'Expert Evidence (E_ik)': round(a.expert_evidence, 3),
                'Cluster Evidence (C_ik)': round(a.cluster_evidence, 3),
                'Is Excluded': a.is_excluded,
                'Is Ambiguous': a.is_ambiguous,
                'Competing Dimensions': str(a.competing_dimensions) if a.competing_dimensions else ''
            })
    
    return pd.DataFrame(rows)


def create_sensitivity_comparison_table(
    sensitivity_results: Dict[str, Dict[str, List[FinalAssignment]]]
) -> pd.DataFrame:
    """Create a table comparing assignment stability across weight schemes."""
    rows = []
    
    for construct, scheme_results in sensitivity_results.items():
        # Get all item IDs
        item_ids = [a.item_id for a in list(scheme_results.values())[0]]
        
        for item_id in item_ids:
            row = {'Construct': construct, 'Item ID': item_id}
            
            assignments_across_schemes = []
            for scheme, assignments in scheme_results.items():
                item_assignment = next(a for a in assignments if a.item_id == item_id)
                row[f'Dim ({scheme})'] = item_assignment.assigned_dimension
                row[f'Score ({scheme})'] = round(item_assignment.combined_score, 3)
                assignments_across_schemes.append(item_assignment.assigned_dimension)
            
            # Check if assignment is stable across schemes
            row['Stable'] = len(set(assignments_across_schemes)) == 1
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def save_embeddings_and_similarities(
    results: Dict[str, Dict[str, Dict]],
    output_dir: Path
):
    """Save embedding matrices and similarity matrices to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for construct, model_results in results.items():
        construct_dir = output_dir / construct
        construct_dir.mkdir(exist_ok=True)
        
        for model_name, data in model_results.items():
            # Save similarity matrix
            sim_result = data['similarity']
            sim_df = pd.DataFrame(
                sim_result.similarity_matrix,
                index=sim_result.item_ids,
                columns=sim_result.item_ids
            )
            sim_df.to_csv(construct_dir / f'{model_name}_similarity.csv')
            
            # Save embeddings
            emb_result = data['embeddings']
            emb_df = pd.DataFrame(
                emb_result.embeddings,
                index=emb_result.item_ids
            )
            emb_df.to_csv(construct_dir / f'{model_name}_embeddings.csv')


def print_integration_report(
    all_metrics: Dict[str, Dict[str, ModelAgreementMetrics]],
    best_models: Dict[str, str],
    all_assignments: Dict[str, List[FinalAssignment]]
):
    """Print a detailed report of the integration results."""
    print("\n" + "="*80)
    print("SEMANTIC SIMILARITY & INTEGRATION ANALYSIS - REPORT")
    print("="*80)
    
    for construct in all_metrics.keys():
        print(f"\n{'='*80}")
        print(f"CONSTRUCT: {construct.upper()}")
        print("="*80)
        
        # Model comparison
        print("\n--- Model Agreement Metrics ---")
        print(f"{'Model':<25} {'Cohen κ':>10} {'Cohesion':>10} {'Min Coh':>10}")
        print("-" * 55)
        
        for model_name, metrics in all_metrics[construct].items():
            kappa_str = f"{metrics.cohens_kappa:.3f}" if not np.isnan(metrics.cohens_kappa) else "N/A"
            marker = " *" if model_name == best_models[construct] else ""
            print(f"{model_name:<25} {kappa_str:>10} {metrics.overall_cohesion:>10.3f} {metrics.min_cohesion:>10.3f}{marker}")
        
        print(f"\n* Selected model: {best_models[construct]}")
        
        # Assignment summary
        assignments = all_assignments[construct]
        n_excluded = sum(1 for a in assignments if a.is_excluded)
        n_ambiguous = sum(1 for a in assignments if a.is_ambiguous)
        
        print(f"\n--- Final Assignments Summary ---")
        print(f"Total items: {len(assignments)}")
        print(f"Excluded (S_ik < 0.40): {n_excluded}")
        print(f"Ambiguous: {n_ambiguous}")
        
        # Dimension distribution
        dim_counts = {}
        for a in assignments:
            if not a.is_excluded:
                dim_counts[a.dimension_label] = dim_counts.get(a.dimension_label, 0) + 1
        
        print(f"\n--- Items per Dimension (excluding low-evidence items) ---")
        for dim, count in sorted(dim_counts.items()):
            print(f"  {dim}: {count}")


# =============================================================================
# Main Analysis Functions
# =============================================================================

def run_step4_analysis(
    items_filepath: Union[str, Path],
    models: Optional[Dict[str, str]] = None,
    constructs: Optional[List[str]] = None,
    n_clusters_per_construct: Optional[Dict[str, int]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, Dict]]:
    """
    Run Step 4: Semantic Similarity Analysis and Clustering.
    
    Parameters
    ----------
    items_filepath : str or Path
        Path to items CSV file
    models : Dict[str, str], optional
        Models to test (default: EMBEDDING_MODELS_LITE)
    constructs : List[str], optional
        Constructs to analyze (default: all in data)
    n_clusters_per_construct : Dict[str, int], optional
        Number of clusters per construct (default: based on dimension count)
    output_dir : str or Path, optional
        Directory to save outputs
    
    Returns
    -------
    Dict[str, Dict[str, Dict]]
        Results for each construct and model
    """
    if models is None:
        models = EMBEDDING_MODELS_LITE
    
    df = load_items_for_embedding(items_filepath)
    
    if constructs is None:
        constructs = df['construct'].unique().tolist()
    
    # Default number of clusters based on dimension definitions
    default_clusters = {
        'depression': 5,
        'anxiety': 2,
        'psychosis': 2,
        'apathy': 3,
        'impulse_control': 6,
        'sleep': 4
    }
    
    if n_clusters_per_construct is None:
        n_clusters_per_construct = default_clusters
    
    all_results = {}
    
    for construct in constructs:
        print(f"\n{'='*60}")
        print(f"Processing construct: {construct}")
        print("="*60)
        
        n_clusters = n_clusters_per_construct.get(construct, 3)
        
        results = embed_and_cluster_construct(
            df, construct, models, n_clusters
        )
        
        all_results[construct] = results
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        save_embeddings_and_similarities(all_results, output_dir / 'embeddings_similarities')
    
    return all_results


def run_step5_analysis(
    items_filepath: Union[str, Path],
    expert_assignments_filepath: Union[str, Path],
    step4_results: Dict[str, Dict[str, Dict]],
    dimension_labels: Dict[str, Dict[int, str]],
    output_dir: Optional[Union[str, Path]] = None,
    run_sensitivity: bool = True
) -> Dict[str, IntegrationResults]:
    """
    Run Step 5: Integration of Mappings.
    
    Parameters
    ----------
    items_filepath : str or Path
        Path to original items CSV
    expert_assignments_filepath : str or Path
        Path to item_assignments.csv from Step 3
    step4_results : Dict
        Results from run_step4_analysis
    dimension_labels : Dict[str, Dict[int, str]]
        Dimension labels for each construct
    output_dir : str or Path, optional
        Directory to save outputs
    run_sensitivity : bool
        Whether to run sensitivity analysis (default: True)
    
    Returns
    -------
    Dict[str, IntegrationResults]
        Integration results for each construct
    """
    df_items = pd.read_csv(items_filepath)
    df_expert = pd.read_csv(expert_assignments_filepath)
    
    all_metrics = {}
    best_models = {}
    all_assignments = {}
    all_sensitivity = {}
    integration_results = {}
    
    for construct, model_results in step4_results.items():
        print(f"\n{'='*60}")
        print(f"Integrating mappings for: {construct}")
        print("="*60)
        
        # Get expert assignments for this construct
        expert_assignments = get_expert_assignments(df_expert, construct)
        
        # Evaluate each model
        construct_metrics = {}
        
        for model_name, data in model_results.items():
            # Use embeddings-based clustering for evaluation
            clustering = data['clustering_embeddings']
            
            metrics = evaluate_model_agreement(clustering, expert_assignments)
            construct_metrics[model_name] = metrics
            
            print(f"  {model_name}: κ = {metrics.cohens_kappa:.3f}, "
                  f"cohesion = {metrics.overall_cohesion:.3f}")
        
        all_metrics[construct] = construct_metrics
        
        # Select best model
        best_model, best_metrics = select_best_model(construct_metrics)
        best_models[construct] = best_model
        print(f"\n  Selected model: {best_model}")
        
        # Generate final assignments
        best_clustering = model_results[best_model]['clustering_embeddings']
        labels = dimension_labels.get(construct, {})
        
        final_assignments = generate_final_assignments(
            df_items, df_expert, best_clustering, construct, labels
        )
        all_assignments[construct] = final_assignments
        
        # Run sensitivity analysis
        sensitivity = None
        if run_sensitivity:
            sensitivity = run_sensitivity_analysis(
                df_items, df_expert, best_clustering, construct, labels
            )
            all_sensitivity[construct] = sensitivity
        
        # Compile results
        n_excluded = sum(1 for a in final_assignments if a.is_excluded)
        n_ambiguous = sum(1 for a in final_assignments if a.is_ambiguous)
        
        integration_results[construct] = IntegrationResults(
            construct=construct,
            best_model=best_model,
            model_metrics=construct_metrics,
            final_assignments=final_assignments,
            n_excluded=n_excluded,
            n_ambiguous=n_ambiguous,
            sensitivity_results=sensitivity
        )
    
    # Print report
    print_integration_report(all_metrics, best_models, all_assignments)
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model comparison
        model_df = create_model_comparison_table(all_metrics)
        model_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        
        # Dimension-level cohesion table
        cohesion_df = create_dimension_cohesion_table(all_metrics, dimension_labels)
        cohesion_df.to_csv(output_dir / 'dimension_cohesion_detailed.csv', index=False)
        
        # Cohesion data for heatmaps (one CSV per construct)
        save_cohesion_data_for_heatmap(all_metrics, dimension_labels, output_dir / 'cohesion_heatmap_data')
        
        # Final assignments
        assign_df = create_final_assignments_table(all_assignments)
        assign_df.to_csv(output_dir / 'final_assignments_combined.csv', index=False)
        
        # Sensitivity analysis
        if run_sensitivity and all_sensitivity:
            sens_df = create_sensitivity_comparison_table(all_sensitivity)
            sens_df.to_csv(output_dir / 'sensitivity_analysis.csv', index=False)
        
        print(f"\nResults saved to {output_dir}/")
    
    return integration_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main(
    items_filepath: Union[str, Path],
    expert_assignments_filepath: Union[str, Path],
    output_dir: Union[str, Path] = './output_steps4_5',
    models: Optional[Dict[str, str]] = None,
    constructs: Optional[List[str]] = None,
    run_sensitivity: bool = True
) -> Tuple[Dict, Dict]:
    """
    Main function to run Steps 4 and 5 of the analysis.
    
    Parameters
    ----------
    items_filepath : str or Path
        Path to original items CSV (items-and-ratings.csv)
    expert_assignments_filepath : str or Path
        Path to item_assignments.csv from Step 3
    output_dir : str or Path
        Directory for outputs
    models : Dict[str, str], optional
        Models to test
    constructs : List[str], optional
        Constructs to analyze
    run_sensitivity : bool
        Whether to run sensitivity analysis
    
    Returns
    -------
    step4_results : Dict
        Embeddings and clustering results
    step5_results : Dict
        Integration results
    """
    # Dimension labels from registered report
    dimension_labels = {
        'depression': {
            1: 'Mood & affective symptoms',
            2: 'Cognitive & self-perception',
            3: 'Somatic & vegetative symptoms',
            4: 'Activity & interest deficits',
            5: 'Anxiety & distress',
            -1: 'Does not fit'
        },
        'anxiety': {
            1: 'Somatic symptoms',
            2: 'Cognitive symptoms',
            -1: 'Does not fit'
        },
        'psychosis': {
            1: 'Hallucinations',
            2: 'Delusions',
            -1: 'Does not fit'
        },
        'apathy': {
            1: 'Behavioral apathy',
            2: 'Cognitive apathy',
            3: 'Affective apathy',
            -1: 'Does not fit'
        },
        'impulse_control': {
            1: 'Pathological gambling',
            2: 'Hypersexuality',
            3: 'Compulsive buying',
            4: 'Compulsive eating',
            5: 'Punding-hobbyism',
            6: 'Dopamine dysregulation syndrome',
            -1: 'Does not fit'
        },
        'sleep': {
            1: 'Daytime sleepiness and alertness',
            2: 'Nocturnal sleep disturbances',
            3: 'REM sleep behavior and dreaming',
            4: 'Sleep-disordered breathing',
            -1: 'Does not fit'
        }
    }
    
    output_dir = Path(output_dir)
    
    print("="*80)
    print("ENIGMA-PD Questionnaire Harmonization")
    print("Step 4: Semantic Similarity Analysis & Clustering")
    print("Step 5: Integration of Mappings")
    print("="*80)
    
    # Step 4
    print("\n" + "="*80)
    print("STEP 4: SEMANTIC SIMILARITY ANALYSIS")
    print("="*80)
    
    step4_results = run_step4_analysis(
        items_filepath,
        models=models,
        constructs=constructs,
        output_dir=output_dir
    )
    
    # Step 5
    print("\n" + "="*80)
    print("STEP 5: INTEGRATION OF MAPPINGS")
    print("="*80)
    
    step5_results = run_step5_analysis(
        items_filepath,
        expert_assignments_filepath,
        step4_results,
        dimension_labels,
        output_dir=output_dir,
        run_sensitivity=run_sensitivity
    )
    
    return step4_results, step5_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Steps 4 & 5: Semantic Similarity and Integration Analysis"
    )
    parser.add_argument(
        "items_filepath",
        type=str,
        help="Path to items CSV file (items-and-ratings.csv)"
    )
    parser.add_argument(
        "expert_assignments_filepath",
        type=str,
        help="Path to item_assignments.csv from Step 3"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output_steps4_5",
        help="Output directory"
    )
    parser.add_argument(
        "--use-all-models",
        action="store_true",
        help="Use all models (slower) instead of lite subset"
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis"
    )
    parser.add_argument(
        "--constructs",
        type=str,
        nargs="+",
        help="Specific constructs to analyze"
    )
    
    args = parser.parse_args()
    
    models = EMBEDDING_MODELS if args.use_all_models else EMBEDDING_MODELS_LITE
    
    step4_results, step5_results = main(
        items_filepath=args.items_filepath,
        expert_assignments_filepath=args.expert_assignments_filepath,
        output_dir=args.output_dir,
        models=models,
        constructs=args.constructs,
        run_sensitivity=not args.no_sensitivity
    )
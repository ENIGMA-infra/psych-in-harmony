"""
Semantic Similarity Analysis and Integration of Mappings (Theory-Primed)
=========================================================================

ENIGMA-PD Questionnaire Harmonization - Steps 4 & 5

Step 4: Theory-Primed Semantic Similarity Analysis
- Compute embeddings for questionnaire items using multiple sentence transformer models
- Compute embeddings for dimension descriptions (the same information given to experts)
- Assign items to dimensions based on maximum cosine similarity to dimension descriptions
- This directly parallels the expert task: items are assigned to the dimension whose
  description they most closely match semantically

Step 5: Integration of Mappings
- Calculate agreement metrics (Cohen's kappa, dimension-level accuracy) for model selection
- Select best-performing embedding model based on agreement with expert assignments
- Compute combined evidence scores integrating expert and model-based evidence
- Generate final item-to-dimension assignments

Key Change from Original Protocol:
----------------------------------
The original protocol used unsupervised hierarchical clustering, which groups items
based on their mutual similarities without reference to the target dimensions. The
reviewer-suggested theory-primed approach instead uses dimension descriptions as
"anchor points" - each item is assigned to the dimension whose description it most
closely resembles. This:
1. Directly mirrors what experts do (read dimension descriptions, assign items)
2. Makes the method generalizable to new questionnaires (the fine-tuned model will
   learn to map items to these same dimension descriptions)
3. Is more interpretable (we know WHY an item was assigned based on similarity scores)

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
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
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
# Dimension Descriptions (same as provided to experts in the survey)
# =============================================================================

# These descriptions are taken directly from the expert survey (Galley-proof PDF)
# They provide the "anchor points" for theory-primed assignment

DIMENSION_DESCRIPTIONS = {
    'depression': {
        1: {
            'label': 'Mood & Affective Symptoms',
            'description': 'dysphoria, sadness, unhappiness, feeling low, feeling blue, hopelessness',
            'keywords': ['dysphoria', 'sadness', 'unhappiness', 'low mood', 'blue', 'hopeless']
        },
        2: {
            'label': 'Cognitive & Self-Perception',
            'description': 'self-attitudes, predictions of the future, narratives of the past, guilt, worthlessness, failure, self-criticism, pessimism',
            'keywords': ['self-attitudes', 'future', 'past', 'guilt', 'worthless', 'failure', 'self-criticism']
        },
        3: {
            'label': 'Somatic & Vegetative Symptoms',
            'description': 'physiological manifestations or consequences of depression, neurophysiological, autonomic, sleep problems, appetite changes, fatigue, weight changes, physical symptoms',
            'keywords': ['sleep', 'appetite', 'fatigue', 'weight', 'tired', 'energy', 'physical', 'somatic']
        },
        4: {
            'label': 'Activity & Interest Deficit',
            'description': 'performance, functioning, pleasure, psychomotor signs, anhedonia, loss of interest, reduced activity, difficulty doing things',
            'keywords': ['interest', 'pleasure', 'activity', 'enjoyment', 'motivation', 'psychomotor', 'functioning']
        },
        5: {
            'label': 'Anxiety & Distress',
            'description': 'irritability, withdrawal, unrest, agitation, worry, tension, nervousness',
            'keywords': ['irritability', 'withdrawal', 'unrest', 'agitation', 'worry', 'tense', 'nervous']
        }
    },
    'anxiety': {
        1: {
            'label': 'Somatic Anxiety',
            'description': 'physiological manifestations or consequences of anxiety, neurophysiological, autonomic, vegetative symptoms, physical symptoms of anxiety like racing heart, sweating, trembling, shortness of breath',
            'keywords': ['heart', 'sweating', 'trembling', 'breathing', 'physical', 'somatic', 'autonomic', 'body']
        },
        2: {
            'label': 'Cognitive Anxiety',
            'description': 'distressing thoughts, panic, obsessive thoughts, predictions of the future, worry, fear, apprehension, anticipation of negative events',
            'keywords': ['thoughts', 'worry', 'fear', 'panic', 'apprehension', 'anticipation', 'cognitive', 'mind']
        }
    },
    'psychosis': {
        1: {
            'label': 'Hallucinations',
            'description': 'hallucinations, illusions, misidentification, seeing hearing feeling or smelling things that are not there, perceptual disturbances',
            'keywords': ['hallucination', 'seeing', 'hearing', 'illusion', 'perception', 'voices', 'visions']
        },
        2: {
            'label': 'Delusions',
            'description': 'delusions, false sense of presence, paranoid beliefs, persecution, jealousy, false beliefs that are firmly held despite evidence',
            'keywords': ['delusion', 'belief', 'paranoid', 'persecution', 'jealousy', 'suspicious', 'convince']
        }
    },
    'apathy': {
        1: {
            'label': 'Cognitive Apathy',
            'description': 'goal-directed cognition, interest in the new, self-concern, curiosity, planning, thinking about goals and future',
            'keywords': ['interest', 'curious', 'planning', 'goals', 'thinking', 'new things', 'concern']
        },
        2: {
            'label': 'Behavioral Apathy',
            'description': 'goal-directed behavior, effort, dependency, initiative, getting things done, taking action, productivity',
            'keywords': ['effort', 'initiative', 'action', 'doing', 'behavior', 'activity', 'productivity', 'started']
        },
        3: {
            'label': 'Affective Apathy',
            'description': 'emotional responsivity, emotional blunting, reduced emotional reactions, indifference to positive or negative events',
            'keywords': ['emotion', 'feeling', 'responsive', 'indifferent', 'blunted', 'affective', 'reaction']
        }
    },
    'impulse_control': {
        1: {
            'label': 'Pathological Gambling',
            'description': 'thoughts occupied by gambling, difficulty controlling thoughts about gambling, time spent on gambling, financial investment in gambling',
            'keywords': ['gambling', 'betting', 'casino', 'wager']
        },
        2: {
            'label': 'Hypersexuality',
            'description': 'thoughts occupied by sex, difficulty controlling thoughts about sex, time spent on sexual activities, financial investment in sexual activities',
            'keywords': ['sex', 'sexual', 'libido']
        },
        3: {
            'label': 'Compulsive Buying',
            'description': 'thoughts occupied by buying, difficulty controlling thoughts about buying, time spent on buying things, financial investment in buying',
            'keywords': ['buying', 'shopping', 'purchase', 'spending']
        },
        4: {
            'label': 'Compulsive Eating',
            'description': 'thoughts occupied by eating, difficulty controlling thoughts about eating, time spent on eating, overeating',
            'keywords': ['eating', 'food', 'binge', 'overeating']
        },
        5: {
            'label': 'Punding-hobbyism',
            'description': 'display of stereotyped, repetitive behaviors, related or unrelated to hobbies, repetitive purposeless activities',
            'keywords': ['repetitive', 'hobby', 'stereotyped', 'punding', 'collecting', 'sorting']
        },
        6: {
            'label': 'Dopamine Dysregulation Syndrome',
            'description': 'compulsive use of dopamine medications despite adequate motor benefits and the annoying consequences',
            'keywords': ['medication', 'dopamine', 'compulsive use', 'PD medications']
        }
    },
    'sleep': {
        1: {
            'label': 'Daytime Sleepiness and Alertness',
            'description': 'excessive sleepiness during the day, difficulty staying awake, reduced awareness or vigilance, drowsiness',
            'keywords': ['sleepy', 'drowsy', 'awake', 'alertness', 'daytime', 'nap', 'vigilance']
        },
        2: {
            'label': 'Nocturnal Sleep Disturbances',
            'description': 'anything that impairs the continuity and quality of nighttime sleep, difficulty falling asleep, waking during night, insomnia',
            'keywords': ['insomnia', 'falling asleep', 'staying asleep', 'waking', 'night', 'nocturnal', 'sleep quality']
        },
        3: {
            'label': 'REM Sleep Behavior and Dreaming',
            'description': 'acting out dreams, vivid or disturbing dreams, movements during sleep, talking in sleep',
            'keywords': ['dreams', 'REM', 'acting out', 'vivid', 'nightmare', 'movement during sleep']
        },
        4: {
            'label': 'Sleep-Disordered Breathing',
            'description': 'snoring, pauses in breathing during sleep, sleep apnea, breathing difficulties at night',
            'keywords': ['snoring', 'breathing', 'apnea', 'choking', 'gasping']
        }
    }
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
class DimensionEmbeddingResult:
    """Results from embedding dimension descriptions."""
    model_name: str
    construct: str
    dimension_embeddings: np.ndarray  # Shape: (n_dimensions, embedding_dim)
    dimension_ids: List[int]
    dimension_labels: List[str]


@dataclass
class SimilarityResult:
    """Similarity matrix between items and dimension descriptions."""
    model_name: str
    construct: str
    item_to_dimension_similarity: np.ndarray  # Shape: (n_items, n_dimensions)
    item_ids: List[str]
    dimension_ids: List[int]
    dimension_labels: List[str]


@dataclass
class TheoryPrimedAssignment:
    """Model-based assignment using theory-primed similarity."""
    model_name: str
    construct: str
    item_ids: List[str]
    assigned_dimensions: np.ndarray  # Dimension ID for each item
    similarity_scores: np.ndarray  # Max similarity score for each item
    all_similarities: np.ndarray  # Full similarity matrix (items x dimensions)
    dimension_ids: List[int]
    dimension_labels: List[str]


@dataclass
class ModelAgreementMetrics:
    """Agreement metrics between expert and model-based assignments."""
    model_name: str
    construct: str
    cohens_kappa: float
    accuracy: float
    dimension_accuracies: Dict[int, float]  # Per-dimension accuracy
    n_items: int
    n_correct: int
    expert_assignments: List[int]
    model_assignments: List[int]
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class CombinedEvidenceScore:
    """Combined evidence score for an item-dimension pair."""
    item_id: str
    construct: str
    dimension: int
    dimension_label: str
    expert_evidence: float  # E_ik
    model_evidence: float  # M_ik (replaces cluster_evidence)
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
    model_evidence: float
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
# Step 4: Embedding and Theory-Primed Assignment Functions
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
    
    print(f"  Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings


def get_dimension_descriptions_for_embedding(construct: str) -> Tuple[List[int], List[str], List[str]]:
    """
    Get dimension descriptions formatted for embedding.
    
    Parameters
    ----------
    construct : str
        Name of the construct
    
    Returns
    -------
    dimension_ids : List[int]
        Numeric dimension IDs
    dimension_labels : List[str]
        Human-readable labels
    dimension_texts : List[str]
        Full text descriptions for embedding
    """
    if construct not in DIMENSION_DESCRIPTIONS:
        raise ValueError(f"Unknown construct: {construct}. Available: {list(DIMENSION_DESCRIPTIONS.keys())}")
    
    dims = DIMENSION_DESCRIPTIONS[construct]
    
    dimension_ids = []
    dimension_labels = []
    dimension_texts = []
    
    for dim_id in sorted(dims.keys()):
        dim_info = dims[dim_id]
        dimension_ids.append(dim_id)
        dimension_labels.append(dim_info['label'])
        # Combine label and description for richer embedding
        dimension_texts.append(f"{dim_info['label']}: {dim_info['description']}")
    
    return dimension_ids, dimension_labels, dimension_texts


def compute_theory_primed_assignment(
    item_embeddings: np.ndarray,
    dimension_embeddings: np.ndarray,
    item_ids: List[str],
    dimension_ids: List[int],
    dimension_labels: List[str],
    model_name: str,
    construct: str
) -> TheoryPrimedAssignment:
    """
    Assign items to dimensions based on similarity to dimension descriptions.
    
    This is the core of the theory-primed approach: each item is assigned to
    the dimension whose description embedding it is most similar to.
    
    Parameters
    ----------
    item_embeddings : np.ndarray
        Embeddings for items (n_items, embedding_dim)
    dimension_embeddings : np.ndarray
        Embeddings for dimension descriptions (n_dimensions, embedding_dim)
    item_ids : List[str]
        Item identifiers
    dimension_ids : List[int]
        Dimension numeric IDs
    dimension_labels : List[str]
        Dimension labels
    model_name : str
        Name of the embedding model
    construct : str
        Construct name
    
    Returns
    -------
    TheoryPrimedAssignment
        Assignment results including all similarity scores
    """
    # Normalize embeddings for cosine similarity
    item_emb_norm = normalize(item_embeddings, axis=1)
    dim_emb_norm = normalize(dimension_embeddings, axis=1)
    
    # Compute similarity matrix (items x dimensions)
    similarity_matrix = cosine_similarity(item_emb_norm, dim_emb_norm)
    
    # Assign each item to highest-similarity dimension
    best_dim_indices = np.argmax(similarity_matrix, axis=1)
    assigned_dimensions = np.array([dimension_ids[i] for i in best_dim_indices])
    similarity_scores = np.max(similarity_matrix, axis=1)
    
    return TheoryPrimedAssignment(
        model_name=model_name,
        construct=construct,
        item_ids=item_ids,
        assigned_dimensions=assigned_dimensions,
        similarity_scores=similarity_scores,
        all_similarities=similarity_matrix,
        dimension_ids=dimension_ids,
        dimension_labels=dimension_labels
    )


def embed_and_assign_construct(
    df: pd.DataFrame,
    construct: str,
    models: Dict[str, str]
) -> Dict[str, Dict]:
    """
    Compute embeddings and theory-primed assignments for a construct using multiple models.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with items (filtered to construct)
    construct : str
        Name of the construct
    models : Dict[str, str]
        Dictionary mapping model names to HuggingFace paths
    
    Returns
    -------
    Dict[str, Dict]
        Results for each model containing embeddings, similarities, and assignments
    """
    df_construct = df[df['construct'] == construct].copy()
    texts = df_construct['question_text'].tolist()
    item_ids = df_construct['question_number'].tolist()
    
    # Get dimension descriptions
    dimension_ids, dimension_labels, dimension_texts = get_dimension_descriptions_for_embedding(construct)
    
    print(f"\n  Construct: {construct}")
    print(f"  Items: {len(texts)}")
    print(f"  Dimensions: {len(dimension_ids)}")
    for d_id, d_label in zip(dimension_ids, dimension_labels):
        print(f"    {d_id}: {d_label}")
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\n  Processing model: {model_name}")
        
        try:
            # Set instruction for instructor-style models
            instruction = None
            if 'instructor' in model_name.lower():
                instruction = "Represent this mental health questionnaire item for semantic similarity:"
            
            # Compute item embeddings
            item_embeddings = compute_embeddings(texts, model_name, model_path, instruction)
            
            if item_embeddings is None:
                continue
            
            # Compute dimension description embeddings
            dim_instruction = None
            if 'instructor' in model_name.lower():
                dim_instruction = "Represent this mental health symptom dimension description:"
            
            # Need to reload model for dimension embeddings (or reuse)
            print(f"  Computing dimension description embeddings...")
            dimension_embeddings = compute_embeddings(dimension_texts, model_name, model_path, dim_instruction)
            
            if dimension_embeddings is None:
                continue
            
            # Compute theory-primed assignment
            assignment = compute_theory_primed_assignment(
                item_embeddings, dimension_embeddings,
                item_ids, dimension_ids, dimension_labels,
                model_name, construct
            )
            
            # Store results
            results[model_name] = {
                'item_embeddings': EmbeddingResult(
                    model_name=model_name,
                    model_path=model_path,
                    embeddings=item_embeddings,
                    item_ids=item_ids,
                    construct=construct,
                    embedding_dim=item_embeddings.shape[1]
                ),
                'dimension_embeddings': DimensionEmbeddingResult(
                    model_name=model_name,
                    construct=construct,
                    dimension_embeddings=dimension_embeddings,
                    dimension_ids=dimension_ids,
                    dimension_labels=dimension_labels
                ),
                'similarity': SimilarityResult(
                    model_name=model_name,
                    construct=construct,
                    item_to_dimension_similarity=assignment.all_similarities,
                    item_ids=item_ids,
                    dimension_ids=dimension_ids,
                    dimension_labels=dimension_labels
                ),
                'assignment': assignment
            }
            
            # Print summary
            print(f"  Assignments per dimension:")
            for d_id, d_label in zip(dimension_ids, dimension_labels):
                n_assigned = np.sum(assignment.assigned_dimensions == d_id)
                print(f"    {d_label}: {n_assigned} items")
            
        except Exception as e:
            warnings.warn(f"Error processing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
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


def evaluate_model_agreement(
    assignment: TheoryPrimedAssignment,
    expert_assignments: Dict[str, int]
) -> ModelAgreementMetrics:
    """
    Evaluate agreement between model-based and expert assignments.
    
    Parameters
    ----------
    assignment : TheoryPrimedAssignment
        Theory-primed assignment results
    expert_assignments : Dict[str, int]
        Expert-assigned dimensions
    
    Returns
    -------
    ModelAgreementMetrics
        Agreement metrics for this model
    """
    expert_labels = []
    model_labels = []
    
    for i, item_id in enumerate(assignment.item_ids):
        if item_id in expert_assignments:
            expert_labels.append(expert_assignments[item_id])
            model_labels.append(assignment.assigned_dimensions[i])
    
    if len(expert_labels) < 2:
        return ModelAgreementMetrics(
            model_name=assignment.model_name,
            construct=assignment.construct,
            cohens_kappa=np.nan,
            accuracy=np.nan,
            dimension_accuracies={},
            n_items=0,
            n_correct=0,
            expert_assignments=[],
            model_assignments=[]
        )
    
    # Calculate Cohen's kappa
    kappa = cohen_kappa_score(expert_labels, model_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(expert_labels, model_labels)
    n_correct = sum(e == m for e, m in zip(expert_labels, model_labels))
    
    # Calculate per-dimension accuracy
    dimension_accuracies = {}
    for dim in assignment.dimension_ids:
        dim_mask = [e == dim for e in expert_labels]
        if sum(dim_mask) > 0:
            dim_correct = sum(e == m for e, m, mask in zip(expert_labels, model_labels, dim_mask) if mask)
            dimension_accuracies[dim] = dim_correct / sum(dim_mask)
        else:
            dimension_accuracies[dim] = np.nan
    
    return ModelAgreementMetrics(
        model_name=assignment.model_name,
        construct=assignment.construct,
        cohens_kappa=kappa,
        accuracy=accuracy,
        dimension_accuracies=dimension_accuracies,
        n_items=len(expert_labels),
        n_correct=n_correct,
        expert_assignments=expert_labels,
        model_assignments=model_labels
    )


def select_best_model(
    model_metrics: Dict[str, ModelAgreementMetrics],
    min_kappa_threshold: float = 0.40,
    kappa_tolerance: float = 0.05
) -> Tuple[str, ModelAgreementMetrics]:
    """
    Select the best-performing embedding model based on agreement metrics.
    
    Selection criteria (from registered report):
    1. Primary: Highest Cohen's kappa (minimum threshold κ ≥ 0.40)
    2. Secondary: If multiple models within 0.05 of highest κ, select highest accuracy
    
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
                     "Proceeding with best available model.")
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
    
    # If tie, use accuracy as tiebreaker
    if len(top_models) > 1:
        best_name = max(top_models, key=lambda x: top_models[x].accuracy)
    else:
        best_name = list(top_models.keys())[0]
    
    return best_name, valid_models[best_name]


# =============================================================================
# Step 5: Combined Evidence Scores and Final Assignment
# =============================================================================

def calculate_model_evidence(
    item_id: str,
    assignment: TheoryPrimedAssignment,
    dimensions: List[int]
) -> Dict[int, float]:
    """
    Calculate model evidence M_ik for each dimension.
    
    M_ik = normalized similarity score for item i to dimension k
    
    We normalize the similarity scores to sum to 1 (like a probability distribution)
    so they're comparable to expert proportions.
    
    Parameters
    ----------
    item_id : str
        Item identifier
    assignment : TheoryPrimedAssignment
        Theory-primed assignment results
    dimensions : List[int]
        List of possible dimensions
    
    Returns
    -------
    Dict[int, float]
        M_ik for each dimension k
    """
    # Find item index
    item_idx = assignment.item_ids.index(item_id)
    
    # Get similarity scores for this item
    similarities = assignment.all_similarities[item_idx]
    
    # Convert to softmax-like probabilities for better comparability with expert proportions
    # Using temperature=1 for standard softmax
    exp_sims = np.exp(similarities - np.max(similarities))  # Subtract max for numerical stability
    probs = exp_sims / exp_sims.sum()
    
    # Map to dimension IDs
    model_evidence = {}
    for i, dim_id in enumerate(assignment.dimension_ids):
        model_evidence[dim_id] = probs[i]
    
    return model_evidence


def calculate_combined_evidence_scores(
    item_id: str,
    construct: str,
    expert_proportions: Dict[int, float],
    model_evidence: Dict[int, float],
    dimension_labels: Dict[int, str],
    w_expert: float = 0.60,
    w_model: float = 0.40
) -> List[CombinedEvidenceScore]:
    """
    Calculate combined evidence scores S_ik for all dimensions.
    
    S_ik = (w_E × E_ik) + (w_M × M_ik)
    
    Where E_ik is expert evidence and M_ik is model evidence (replacing cluster evidence).
    
    Parameters
    ----------
    item_id : str
        Item identifier
    construct : str
        Construct name
    expert_proportions : Dict[int, float]
        E_ik - proportion of experts assigning to each dimension
    model_evidence : Dict[int, float]
        M_ik - model evidence for each dimension
    dimension_labels : Dict[int, str]
        Human-readable dimension labels
    w_expert : float
        Weight for expert evidence (default: 0.60)
    w_model : float
        Weight for model evidence (default: 0.40)
    
    Returns
    -------
    List[CombinedEvidenceScore]
        Combined scores for each dimension
    """
    scores = []
    all_dims = set(expert_proportions.keys()) | set(model_evidence.keys())
    
    # Calculate S_ik for each dimension
    combined_scores = {}
    for dim in all_dims:
        e_ik = expert_proportions.get(dim, 0.0)
        m_ik = model_evidence.get(dim, 0.0)
        s_ik = (w_expert * e_ik) + (w_model * m_ik)
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
            model_evidence=model_evidence.get(dim, 0.0),
            combined_score=s_ik,
            is_assigned=(dim == best_dim),
            confidence=confidence
        ))
    
    return scores


def generate_final_assignments(
    df_items: pd.DataFrame,
    item_assignments_df: pd.DataFrame,
    assignment: TheoryPrimedAssignment,
    construct: str,
    dimension_labels: Dict[int, str],
    w_expert: float = 0.60,
    w_model: float = 0.40,
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
    assignment : TheoryPrimedAssignment
        Best model's assignment results
    construct : str
        Construct name
    dimension_labels : Dict[int, str]
        Dimension labels
    w_expert : float
        Expert weight (default: 0.60)
    w_model : float
        Model weight (default: 0.40)
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
    
    # Get expert assignments
    expert_assignments = dict(zip(df_expert['Item ID'], df_expert['Assigned Dimension']))
    
    # Parse rating distributions to get expert proportions
    # Format: "Dim 1: 80%; Dim 2: 20%"
    def parse_distribution(dist_str):
        props = {}
        if pd.isna(dist_str) or not dist_str:
            return props
        for part in str(dist_str).split(';'):
            part = part.strip()
            if ':' in part:
                try:
                    dim_part, pct_part = part.split(':')
                    dim = int(dim_part.replace('Dim', '').strip())
                    pct = float(pct_part.replace('%', '').strip()) / 100
                    props[dim] = pct
                except (ValueError, AttributeError):
                    continue
        return props
    
    expert_proportions_map = {}
    for _, row in df_expert.iterrows():
        if 'Rating Distribution' in df_expert.columns:
            expert_proportions_map[row['Item ID']] = parse_distribution(row['Rating Distribution'])
        else:
            # If no distribution available, use binary assignment
            expert_proportions_map[row['Item ID']] = {row['Assigned Dimension']: 1.0}
    
    # Get all dimensions for this construct
    dimensions = list(dimension_labels.keys())
    
    final_assignments = []
    
    for _, row in df_construct.iterrows():
        item_id = row['question_number']
        
        # Get expert proportions
        expert_props = expert_proportions_map.get(item_id, {})
        
        # Calculate model evidence
        model_ev = calculate_model_evidence(item_id, assignment, dimensions)
        
        # Calculate combined scores for all dimensions
        all_scores = {}
        for dim in dimensions:
            e_ik = expert_props.get(dim, 0.0)
            m_ik = model_ev.get(dim, 0.0)
            s_ik = (w_expert * e_ik) + (w_model * m_ik)
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
            model_evidence=model_ev.get(best_dim, 0.0),
            is_excluded=is_excluded,
            is_ambiguous=is_ambiguous,
            competing_dimensions=competing,
            all_scores=all_scores
        ))
    
    return final_assignments


def run_sensitivity_analysis(
    df_items: pd.DataFrame,
    item_assignments_df: pd.DataFrame,
    assignment: TheoryPrimedAssignment,
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
    
    for scheme_name, (w_e, w_m) in weight_schemes.items():
        assignments = generate_final_assignments(
            df_items, item_assignments_df, assignment,
            construct, dimension_labels, w_expert=w_e, w_model=w_m
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
                'Accuracy': round(metrics.accuracy, 3) if not np.isnan(metrics.accuracy) else 'N/A',
                'N Items': metrics.n_items,
                'N Correct': metrics.n_correct
            })
    
    return pd.DataFrame(rows)


def create_dimension_accuracy_table(
    all_metrics: Dict[str, Dict[str, ModelAgreementMetrics]],
    dimension_labels: Dict[str, Dict[int, str]]
) -> pd.DataFrame:
    """
    Create a table with dimension-level accuracy for each model.
    
    Returns a DataFrame with columns:
    - Construct
    - Model  
    - Dimension (numeric)
    - Dimension Label
    - Accuracy
    """
    rows = []
    
    for construct, model_metrics in all_metrics.items():
        labels = dimension_labels.get(construct, {})
        
        for model_name, metrics in model_metrics.items():
            for dim, accuracy in metrics.dimension_accuracies.items():
                rows.append({
                    'Construct': construct,
                    'Model': model_name,
                    'Dimension': dim,
                    'Dimension Label': labels.get(dim, f'Dimension {dim}'),
                    'Accuracy': round(accuracy, 3) if not np.isnan(accuracy) else 'N/A'
                })
    
    return pd.DataFrame(rows)


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
                'Model Evidence (M_ik)': round(a.model_evidence, 3),
                'Is Excluded': a.is_excluded,
                'Is Ambiguous': a.is_ambiguous,
                'Competing Dimensions': str(a.competing_dimensions) if a.competing_dimensions else ''
            })
    
    return pd.DataFrame(rows)


def create_similarity_matrix_table(
    results: Dict[str, Dict[str, Dict]],
    construct: str,
    model_name: str
) -> pd.DataFrame:
    """
    Create a table showing item-to-dimension similarity scores.
    
    This is useful for understanding why items were assigned to specific dimensions.
    """
    if construct not in results or model_name not in results[construct]:
        return pd.DataFrame()
    
    similarity_result = results[construct][model_name]['similarity']
    assignment = results[construct][model_name]['assignment']
    
    rows = []
    for i, item_id in enumerate(similarity_result.item_ids):
        row = {'Item ID': item_id}
        for j, dim_label in enumerate(similarity_result.dimension_labels):
            row[dim_label] = round(similarity_result.item_to_dimension_similarity[i, j], 3)
        row['Assigned To'] = assignment.dimension_labels[
            assignment.dimension_ids.index(assignment.assigned_dimensions[i])
        ]
        rows.append(row)
    
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
            # Save item-dimension similarity matrix
            sim_result = data['similarity']
            sim_df = pd.DataFrame(
                sim_result.item_to_dimension_similarity,
                index=sim_result.item_ids,
                columns=sim_result.dimension_labels
            )
            sim_df.to_csv(construct_dir / f'{model_name}_item_dimension_similarity.csv')
            
            # Save item embeddings
            emb_result = data['item_embeddings']
            emb_df = pd.DataFrame(
                emb_result.embeddings,
                index=emb_result.item_ids
            )
            emb_df.to_csv(construct_dir / f'{model_name}_item_embeddings.csv')
            
            # Save dimension embeddings
            dim_emb_result = data['dimension_embeddings']
            dim_emb_df = pd.DataFrame(
                dim_emb_result.dimension_embeddings,
                index=dim_emb_result.dimension_labels
            )
            dim_emb_df.to_csv(construct_dir / f'{model_name}_dimension_embeddings.csv')


def print_integration_report(
    all_metrics: Dict[str, Dict[str, ModelAgreementMetrics]],
    best_models: Dict[str, str],
    all_assignments: Dict[str, List[FinalAssignment]]
):
    """Print a summary report of the integration results."""
    print("\n" + "="*80)
    print("INTEGRATION SUMMARY REPORT")
    print("="*80)
    
    for construct in all_metrics.keys():
        print(f"\n{construct.upper()}")
        print("-" * 40)
        
        # Best model
        best_model = best_models[construct]
        best_metrics = all_metrics[construct][best_model]
        print(f"Best model: {best_model}")
        print(f"  Cohen's κ: {best_metrics.cohens_kappa:.3f}")
        print(f"  Accuracy: {best_metrics.accuracy:.3f} ({best_metrics.n_correct}/{best_metrics.n_items})")
        
        # Dimension-level accuracy
        print(f"  Dimension-level accuracy:")
        for dim, acc in best_metrics.dimension_accuracies.items():
            if not np.isnan(acc):
                print(f"    Dim {dim}: {acc:.3f}")
        
        # Assignment summary
        assignments = all_assignments[construct]
        n_excluded = sum(1 for a in assignments if a.is_excluded)
        n_ambiguous = sum(1 for a in assignments if a.is_ambiguous)
        print(f"  Total items: {len(assignments)}")
        print(f"  Excluded (S_ik < 0.40): {n_excluded}")
        print(f"  Ambiguous: {n_ambiguous}")


# =============================================================================
# Main Pipeline Functions
# =============================================================================

def run_step4_analysis(
    items_filepath: Union[str, Path],
    models: Optional[Dict[str, str]] = None,
    constructs: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, Dict]]:
    """
    Run Step 4: Theory-Primed Semantic Similarity Analysis.
    
    Parameters
    ----------
    items_filepath : str or Path
        Path to items CSV file
    models : Dict[str, str], optional
        Models to test (default: EMBEDDING_MODELS_LITE)
    constructs : List[str], optional
        Constructs to analyze (default: all)
    output_dir : str or Path, optional
        Directory to save outputs
    
    Returns
    -------
    Dict[str, Dict[str, Dict]]
        Results organized by construct, then model
    """
    if models is None:
        models = EMBEDDING_MODELS_LITE
    
    df_items = pd.read_csv(items_filepath)
    
    if constructs is None:
        constructs = df_items['construct'].unique().tolist()
    
    all_results = {}
    
    for construct in constructs:
        print(f"\n{'='*60}")
        print(f"Processing construct: {construct}")
        print('='*60)
        
        results = embed_and_assign_construct(df_items, construct, models)
        all_results[construct] = results
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_embeddings_and_similarities(all_results, output_dir / 'embeddings_and_similarities')
        print(f"\nStep 4 outputs saved to {output_dir}/")
    
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
        Path to items CSV file
    expert_assignments_filepath : str or Path
        Path to item_assignments.csv from Step 3
    step4_results : Dict
        Results from Step 4
    dimension_labels : Dict
        Dimension labels for each construct
    output_dir : str or Path, optional
        Directory to save outputs
    run_sensitivity : bool
        Whether to run sensitivity analysis
    
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
        print(f"\n{'-'*40}")
        print(f"Evaluating models for: {construct}")
        print('-'*40)
        
        # Get expert assignments
        expert_assignments = get_expert_assignments(df_expert, construct)
        
        # Evaluate each model
        construct_metrics = {}
        for model_name, data in model_results.items():
            assignment = data['assignment']
            
            metrics = evaluate_model_agreement(assignment, expert_assignments)
            construct_metrics[model_name] = metrics
            
            print(f"  {model_name}: κ = {metrics.cohens_kappa:.3f}, "
                  f"accuracy = {metrics.accuracy:.3f}")
        
        all_metrics[construct] = construct_metrics
        
        # Select best model
        best_model, best_metrics = select_best_model(construct_metrics)
        best_models[construct] = best_model
        print(f"\n  Selected model: {best_model}")
        
        # Generate final assignments
        best_assignment = model_results[best_model]['assignment']
        labels = dimension_labels.get(construct, {})
        
        final_assignments = generate_final_assignments(
            df_items, df_expert, best_assignment, construct, labels
        )
        all_assignments[construct] = final_assignments
        
        # Run sensitivity analysis
        sensitivity = None
        if run_sensitivity:
            sensitivity = run_sensitivity_analysis(
                df_items, df_expert, best_assignment, construct, labels
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
        
        # Dimension-level accuracy table
        accuracy_df = create_dimension_accuracy_table(all_metrics, dimension_labels)
        accuracy_df.to_csv(output_dir / 'dimension_accuracy_detailed.csv', index=False)
        
        # Final assignments
        assign_df = create_final_assignments_table(all_assignments)
        assign_df.to_csv(output_dir / 'final_assignments_combined.csv', index=False)
        
        # Sensitivity analysis
        if run_sensitivity and all_sensitivity:
            sens_df = create_sensitivity_comparison_table(all_sensitivity)
            sens_df.to_csv(output_dir / 'sensitivity_analysis.csv', index=False)
        
        # Similarity matrices for best models
        for construct, model_name in best_models.items():
            sim_df = create_similarity_matrix_table(step4_results, construct, model_name)
            sim_df.to_csv(output_dir / f'{construct}_similarity_scores.csv', index=False)
        
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
        Embeddings and assignment results
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
            1: 'Cognitive apathy',
            2: 'Behavioral apathy',
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
    print("Step 4: Theory-Primed Semantic Similarity Analysis")
    print("Step 5: Integration of Mappings")
    print("="*80)
    print("\nThis analysis uses THEORY-PRIMED assignment:")
    print("  - Items are assigned to dimensions based on semantic similarity")
    print("    to dimension descriptions (the same info given to experts)")
    print("  - This directly mirrors the expert task and enables generalization")
    print("    to new questionnaires via fine-tuning")
    
    # Step 4
    print("\n" + "="*80)
    print("STEP 4: THEORY-PRIMED SEMANTIC SIMILARITY ANALYSIS")
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
        description="Steps 4 & 5: Theory-Primed Semantic Similarity and Integration Analysis"
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
"""
P_i Calculation
"""

import pandas as pd
import numpy as np
from collections import Counter

def calculate_Pi_for_item(ratings):
    """
    Calculate P_i for a single item given ratings array.
    
    P_i = [1 / (n(n-1))] × [Σ(n_ij²) - n]
    
    handles:
    - Missing data (NaN, -9)
    - Variable number of raters per item
    
    Parameters:
    -----------
    ratings : array-like
        Ratings from all raters (may contain NaN or -9)
    
    Returns:
    --------
    dict with P_i, modal_dimension, modal_count, n_raters
    """
    # Remove invalid ratings (-9 and NaN)
    valid_ratings = []
    for r in ratings:
        if pd.notna(r) and r != -9:
            valid_ratings.append(int(r))
    
    valid_ratings = np.array(valid_ratings)
    n = len(valid_ratings)  # Number of raters who provided valid ratings
    
    if n <= 1:
        # Not enough raters to calculate agreement
        return {
            'P_i': np.nan,
            'modal_dimension': np.nan,
            'modal_count': np.nan,
            'modal_percentage': np.nan,
            'n_raters': n,
            'all_ratings': str(list(valid_ratings))
        }
    
    # Count ratings per category (n_ij)
    counts = Counter(valid_ratings)
    n_ij = list(counts.values())
    
    # Calculate P_i 
    sum_n_ij_squared = sum(count**2 for count in n_ij)
    P_i = (sum_n_ij_squared - n) / (n * (n - 1))
    
    # Calculate modal assignment
    modal_dimension = max(counts, key=counts.get)
    modal_count = counts[modal_dimension]
    modal_percentage = modal_count / n
    
    return {
        'P_i': P_i,
        'modal_dimension': modal_dimension,
        'modal_count': modal_count,
        'modal_percentage': modal_percentage,
        'n_raters': n,
        'all_ratings': str(list(valid_ratings))
    }


def calculate_Pi_for_dataframe(df, ratings_columns=None):
    """
    Calculate P_i for all items in a dataframe.
    
    Parameters:
    -----------
    df : DataFrame
        Data with ratings columns
    ratings_columns : list, optional
        List of column names containing ratings
        If None, automatically detects columns starting with 'rater_'
    
    Returns:
    --------
    DataFrame with P_i results
    """
    # Auto-detect ratings columns if not specified
    if ratings_columns is None:
        ratings_columns = [col for col in df.columns if col.startswith('rater_')]
    
    if not ratings_columns:
        raise ValueError("No ratings columns found! Expected columns starting with 'rater_'")
    
    print(f"Found {len(ratings_columns)} rating columns: {ratings_columns}")
    
    # Calculate P_i for each item
    results = []
    
    for idx, row in df.iterrows():
        # Get item information
        item_info = {
            'construct': row.get('construct', ''),
            'item_id': row.get('question_number', row.get('item_id', f'item_{idx}')),
            'item_text': row.get('question_text', '')[:100],  # Truncate to 100 chars
        }
        
        # Get ratings for this item
        ratings = row[ratings_columns].values
        
        # Calculate P_i
        pi_results = calculate_Pi_for_item(ratings)
        
        # Combine all information
        results.append({**item_info, **pi_results})
    
    return pd.DataFrame(results)


def analyze_agreement_by_construct(results_df):
    """
    Create summary statistics by construct.
    """
    if 'construct' not in results_df.columns or results_df['construct'].isna().all():
        print("No construct column found, analyzing all items together...")
        constructs = ['all']
        results_df['construct'] = 'all'
    else:
        constructs = results_df['construct'].unique()
    
    summary_data = []
    
    for construct in constructs:
        construct_df = results_df[results_df['construct'] == construct]
        
        # Calculate statistics (excluding NaN values)
        valid_df = construct_df[construct_df['P_i'].notna()]
        
        if len(valid_df) == 0:
            continue
        
        summary_data.append({
            'Construct': construct,
            'N_items': len(construct_df),
            'Mean_n_raters': construct_df['n_raters'].mean(),
            'Mean_P_i': valid_df['P_i'].mean(),
            'Mean_modal_%': valid_df['modal_percentage'].mean(),
            'Perfect_agreement': (valid_df['P_i'] == 1.0).sum(),
            'P_i_≥_0.60': (valid_df['P_i'] >= 0.60).sum(),
            'Modal_%_≥_0.60': (valid_df['modal_percentage'] >= 0.60).sum(),
        })
    
    return pd.DataFrame(summary_data)


def main():
    """
    Main function - loads data and calculates P_i for all items.
    """
    print("="*80)
    print("AUTOMATIC P_i CALCULATION")
    print("="*80)
    
    # Load data
    input_file = 'items-and-ratings.csv'
    print(f"\nLoading: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} items")
        print(f"  Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        print("Please ensure the file is in the current directory")
        return
    
    # Calculate P_i for all items
    print("\nCalculating P_i for all items...")
    results_df = calculate_Pi_for_dataframe(df)
    
    # Check for items with insufficient raters
    insufficient = results_df[results_df['n_raters'] < 2]
    if len(insufficient) > 0:
        print(f"\n⚠️  Warning: {len(insufficient)} items have < 2 raters (P_i cannot be calculated)")
    
    # Create summary by construct
    print("\n" + "="*80)
    print("SUMMARY BY CONSTRUCT")
    print("="*80)
    
    summary_df = analyze_agreement_by_construct(results_df)
    print("\n" + summary_df.to_string(index=False))
    
    # Show distribution of number of raters
    print("\n" + "="*80)
    print("DISTRIBUTION OF NUMBER OF RATERS")
    print("="*80)
    
    rater_counts = results_df['n_raters'].value_counts().sort_index()
    print("\nNumber of raters per item:")
    for n_raters, count in rater_counts.items():
        print(f"  {n_raters} raters: {count} items ({count/len(results_df)*100:.1f}%)")
    
    # Show examples
    print("\n" + "="*80)
    print("EXAMPLES (First 20 items)")
    print("="*80)
    
    example_cols = ['item_id', 'n_raters', 'all_ratings', 'P_i', 'modal_dimension', 'modal_percentage']
    print("\n" + results_df[example_cols].head(20).to_string(index=False))
    
    # Show items where P_i and modal % differ most
    print("\n" + "="*80)
    print("ITEMS WHERE P_i AND MODAL % DIFFER MOST")
    print("="*80)
    
    results_df['difference'] = abs(results_df['P_i'] - results_df['modal_percentage'])
    top_diff = results_df.nlargest(15, 'difference')
    
    diff_cols = ['item_id', 'construct', 'n_raters', 'all_ratings', 'P_i', 'modal_percentage', 'difference']
    print("\n" + top_diff[diff_cols].to_string(index=False))
    
    # Save results
    output_csv = '/code/expert-survey/outputs/expert_ratings_Pi_analysis.csv'
    
    results_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"✓ CSV: {output_csv}")
    print(f"✓ Excel: {output_xlsx}")
    
    # Save summary
    summary_csv = '/code/expert-survey/outputs/expert_ratings_Pi_analysis.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary: {summary_csv}")


if __name__ == "__main__":
    main()
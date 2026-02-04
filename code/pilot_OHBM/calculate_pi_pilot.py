import pandas as pd
import numpy as np
from collections import Counter

# Load the expert ratings data
ratings_file = 'items-and-ratings.csv'
df = pd.read_csv(ratings_file)

print("="*80)
print("CALCULATING P_i FOR EACH ITEM")
print("P_i = [1 / (n(n-1))] × [Σ(n_ij²) - n]")
print("="*80)

# Replace -9 (not answered) with NaN
rater_cols = [col for col in df.columns if col.startswith('rater_')]
for col in rater_cols:
    df[col] = df[col].replace(-9, np.nan)

# Process each construct
constructs = df['construct'].unique()

all_results = []

for construct in constructs:
    print(f"\n{'='*80}")
    print(f"CONSTRUCT: {construct.upper()}")
    print(f"{'='*80}")
    
    # Filter for construct
    construct_df = df[df['construct'] == construct].copy()
    
    # Determine number of raters
    if construct == 'sleep':
        n_raters = 3
        true_rater_cols = rater_cols[:3]
    else:
        n_raters = 5
        true_rater_cols = rater_cols[:5]
    
    n = n_raters
    
    print(f"\nNumber of raters (n): {n}")
    print(f"Number of items: {len(construct_df)}")
    
    # Calculate P_i for each item
    for idx, row in construct_df.iterrows():
        item_id = row['question_number']
        item_text = row['question_text']
        
        # Get ratings for this item
        ratings = row[true_rater_cols].values
        valid_ratings = ratings[~pd.isna(ratings)].astype(int)
        
        # Get all categories
        all_categories = sorted(list(set(valid_ratings)))
        
        # Count n_ij (number of raters who chose category j)
        counts = Counter(valid_ratings)
        n_ij = [counts.get(cat, 0) for cat in all_categories]
        
        # Calculate P_i using the formula:
        # P_i = [1 / (n(n-1))] × [Σ(n_ij²) - n]
        sum_n_ij_squared = sum(count**2 for count in n_ij)
        P_i = (sum_n_ij_squared - n) / (n * (n - 1))
        
        # Calculate modal assignment for comparison
        modal_dimension = max(counts, key=counts.get)
        modal_count = counts[modal_dimension]
        modal_percentage = modal_count / n
        
        all_results.append({
            'construct': construct,
            'item_id': item_id,
            'item_text': item_text[:50] + '...' if len(item_text) > 50 else item_text,
            'n_raters': n,
            'ratings': str(list(valid_ratings)),
            'P_i': P_i,
            'modal_dimension': modal_dimension,
            'modal_count': modal_count,
            'modal_percentage': modal_percentage
        })

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

print(f"\n{'='*80}")
print("SUMMARY BY CONSTRUCT")
print(f"{'='*80}")

for construct in constructs:
    construct_results = results_df[results_df['construct'] == construct]
    
    print(f"\n{construct.upper()}:")
    print(f"  Mean P_i: {construct_results['P_i'].mean():.4f}")
    print(f"  Mean modal %: {construct_results['modal_percentage'].mean():.4f}")
    print(f"  Items with P_i = 1.0 (perfect): {(construct_results['P_i'] == 1.0).sum()}")
    print(f"  Items with P_i ≥ 0.60: {(construct_results['P_i'] >= 0.60).sum()}")
    print(f"  Items with modal % ≥ 0.60: {(construct_results['modal_percentage'] >= 0.60).sum()}")

# Show some examples to illustrate difference
print(f"\n{'='*80}")
print("EXAMPLES: P_i vs Modal % (Depression, first 20 items)")
print(f"{'='*80}")

depression_results = results_df[results_df['construct'] == 'depression'].head(20)
print(f"\n{'Item ID':<10} {'Ratings':<20} {'P_i':>8} {'Modal Dim':>10} {'Modal %':>10} {'Difference':>12}")
print("-" * 85)

for _, row in depression_results.iterrows():
    diff = abs(row['P_i'] - row['modal_percentage'])
    print(f"{row['item_id']:<10} {row['ratings']:<20} {row['P_i']:>8.3f} {row['modal_dimension']:>10} {row['modal_percentage']:>10.1%} {diff:>12.3f}")

# Save full results
output_file = '/mnt/user-data/outputs/expert_ratings_Pi_analysis.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Full results saved to: {output_file}")

# Also create Excel with formatting
excel_file = '/mnt/user-data/outputs/expert_ratings_Pi_analysis.xlsx'
results_df.to_excel(excel_file, index=False, engine='openpyxl')
print(f"✓ Excel version saved to: {excel_file}")

# Show interesting cases where P_i and modal % differ significantly
print(f"\n{'='*80}")
print("CASES WHERE P_i AND MODAL % DIFFER MOST")
print("(These show the difference between pairwise agreement vs modal agreement)")
print(f"{'='*80}")

results_df['difference'] = abs(results_df['P_i'] - results_df['modal_percentage'])
top_differences = results_df.nlargest(15, 'difference')

print(f"\n{'Item ID':<10} {'Construct':<12} {'Ratings':<20} {'P_i':>8} {'Modal %':>10} {'Diff':>8}")
print("-" * 85)

for _, row in top_differences.iterrows():
    print(f"{row['item_id']:<10} {row['construct']:<12} {row['ratings']:<20} {row['P_i']:>8.3f} {row['modal_percentage']:>10.1%} {row['difference']:>8.3f}")

print(f"\n{'='*80}")
print("KEY INSIGHT:")
print("When P_i < Modal %, it means there's disagreement among minority raters")
print("Example: [1,1,2,2,2] → P_i=0.40 (only 40% of pairs agree)")
print("                      → Modal %=0.60 (60% chose dimension 2)")
print(f"{'='*80}")
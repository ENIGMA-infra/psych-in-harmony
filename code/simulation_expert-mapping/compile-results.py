#!/usr/bin/env python
"""
Compile and summarize results from multi-kappa analysis
Run after all HPC jobs complete to generate summary report
"""

import pandas as pd
from pathlib import Path
import sys

def compile_results():
    """Compile results from all construct analyses."""
    
    constructs = ['depression', 'anxiety', 'psychosis', 'apathy', 'icd', 'sleep']
    
    print("="*70)
    print("COMPILING MULTI-KAPPA ANALYSIS RESULTS")
    print("="*70)
    print()
    
    all_recommendations = []
    missing_constructs = []
    
    # Collect recommendations from each construct
    for construct in constructs:
        analysis_dir = Path(f'analysis_multikappa_{construct}')
        rec_file = analysis_dir / 'recommendations_by_kappa.csv'
        
        if rec_file.exists():
            df = pd.read_csv(rec_file)
            all_recommendations.append(df)
            print(f"✓ Found results for {construct.capitalize()}")
        else:
            missing_constructs.append(construct)
            print(f"✗ Missing results for {construct.capitalize()}")
    
    print()
    
    if missing_constructs:
        print(f"Warning: Missing results for: {', '.join(missing_constructs)}")
        print()
    
    if not all_recommendations:
        print("No results found! Check that jobs completed successfully.")
        return None
    
    # Combine all recommendations
    combined_df = pd.concat(all_recommendations, ignore_index=True)
    
    # Save combined results
    output_dir = Path('simulation_results_multikappa')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'all_constructs_all_kappas.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"✓ Saved combined results to: {output_file}")
    print()
    
    return combined_df

def print_summary(df):
    """Print summary statistics."""
    
    print("="*70)
    print("SUMMARY: RECOMMENDED SAMPLE SIZES")
    print("="*70)
    print()
    
    # Overall recommendations by construct
    print("Overall Recommendations by Construct (worst-case κ):")
    print("-"*70)
    
    for construct in df['construct'].unique():
        construct_df = df[df['construct'] == construct]
        max_rec = construct_df['overall_recommendation'].max()
        max_conservative = construct_df['conservative'].max()
        worst_kappa = construct_df.loc[construct_df['overall_recommendation'].idxmax(), 'true_kappa']
        
        print(f"{construct:12s}: {max_rec:2.0f} raters (conservative: {max_conservative:2.0f}) "
              f"[worst case: κ={worst_kappa:.2f}]")
    
    print()
    print("-"*70)
    print()
    
    # Recommendations by kappa level (averaged across constructs)
    print("Average Recommendations by Agreement Level:")
    print("-"*70)
    print(f"{'True κ':>10s} {'Stability':>12s} {'Precision':>12s} {'Replication':>12s} {'Overall':>10s}")
    print("-"*70)
    
    for kappa in sorted(df['true_kappa'].unique()):
        kappa_df = df[df['true_kappa'] == kappa]
        print(f"{kappa:10.2f} {kappa_df['stability_n'].mean():12.1f} "
              f"{kappa_df['precision_n'].mean():12.1f} "
              f"{kappa_df['replication_range_n'].mean():12.1f} "
              f"{kappa_df['overall_recommendation'].mean():10.1f}")
    
    print()
    
    # Detailed table
    print("="*70)
    print("DETAILED RECOMMENDATIONS TABLE")
    print("="*70)
    print()
    
    # Pivot table: constructs as rows, kappa as columns
    pivot = df.pivot_table(
        values='overall_recommendation',
        index='construct',
        columns='true_kappa',
        aggfunc='first'
    )
    
    print("Overall Recommended N (by construct and true κ):")
    print(pivot.to_string())
    print()
    
    # Conservative recommendations
    pivot_conservative = df.pivot_table(
        values='conservative',
        index='construct',
        columns='true_kappa',
        aggfunc='first'
    )
    
    print("Conservative N (+3 buffer, by construct and true κ):")
    print(pivot_conservative.to_string())
    print()
    
    # Criteria breakdown
    print("="*70)
    print("SAMPLE SIZE BY CRITERION")
    print("="*70)
    print()
    
    for construct in df['construct'].unique():
        print(f"\n{construct.upper()}")
        print("-"*70)
        construct_df = df[df['construct'] == construct]
        print(construct_df[['true_kappa', 'stability_n', 'precision_n', 
                           'replication_range_n', 'overall_recommendation']].to_string(index=False))
    
    print()

def check_files():
    """Check that all expected output files exist."""
    
    print("="*70)
    print("FILE VERIFICATION")
    print("="*70)
    print()
    
    constructs = ['depression', 'anxiety', 'psychosis', 'apathy', 'icd', 'sleep']
    expected_files = [
        'stability_all_kappas.csv',
        'precision_all_kappas.csv',
        'replication_all_kappas.csv',
        'recommendations_by_kappa.csv',
        'stability_comparison.png',
        'precision_comparison.png',
        'replication_comparison.png'
    ]
    
    all_found = True
    
    for construct in constructs:
        analysis_dir = Path(f'analysis_multikappa_{construct}')
        
        if not analysis_dir.exists():
            print(f"✗ Directory missing: {analysis_dir}")
            all_found = False
            continue
        
        missing = []
        for filename in expected_files:
            filepath = analysis_dir / filename
            if not filepath.exists():
                missing.append(filename)
        
        if missing:
            print(f"✗ {construct.capitalize()}: Missing {len(missing)} files")
            for f in missing:
                print(f"    - {f}")
            all_found = False
        else:
            print(f"✓ {construct.capitalize()}: All files present ({len(expected_files)} files)")
    
    print()
    
    if all_found:
        print("All expected output files found!")
    else:
        print("Some files are missing. Check job logs for errors.")
    
    print()
    
    return all_found

def main():
    """Main execution."""
    
    print()
    
    # Check files
    files_ok = check_files()
    
    if not files_ok:
        print("Note: Continuing with available results...")
        print()
    
    # Compile results
    df = compile_results()
    
    if df is None:
        print("No results to summarize. Exiting.")
        sys.exit(1)
    
    # Print summary
    print_summary(df)
    
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print("Results saved to: simulation_results_multikappa/")
    print("Individual construct results in: analysis_multikappa_*/")
    print()

if __name__ == "__main__":
    main()
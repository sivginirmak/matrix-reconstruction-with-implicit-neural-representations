#!/usr/bin/env python3
"""
Fair Comparison Analysis for INR Architectures

This script creates fair comparisons between architectures by matching parameter counts
and analyzing PSNR performance, addressing the revision request for matched sizes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

def load_experimental_data():
    """Load and prepare experimental data from exp001"""
    # Load summary statistics which has aggregated results
    summary_df = pd.read_csv('experiments/exp001_architecture_comparison/full_results/summary_statistics.csv')
    
    # Clean up the dataframe
    summary_df = summary_df.iloc[2:].reset_index(drop=True)  # Skip header rows
    summary_df.columns = ['model_name', 'decoder', 'psnr_mean', 'psnr_std', 'psnr_min', 'psnr_max', 
                         'param_count_mean', 'param_efficiency_mean', 'param_efficiency_std', 
                         'training_time_mean', 'training_time_std']
    
    # Convert numeric columns
    numeric_cols = ['psnr_mean', 'psnr_std', 'psnr_min', 'psnr_max', 'param_count_mean', 
                    'param_efficiency_mean', 'param_efficiency_std', 'training_time_mean', 'training_time_std']
    for col in numeric_cols:
        summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
    
    return summary_df

def create_parameter_size_groups(df):
    """Group architectures by similar parameter counts for fair comparison"""
    df['param_group'] = 'Unknown'
    
    # Define parameter ranges for fair comparison
    # Small: < 15K parameters
    # Medium-Small: 15K - 30K parameters  
    # Medium: 30K - 50K parameters
    # Large: > 50K parameters
    
    df.loc[df['param_count_mean'] < 15000, 'param_group'] = 'Small (< 15K)'
    df.loc[(df['param_count_mean'] >= 15000) & (df['param_count_mean'] < 30000), 'param_group'] = 'Medium-Small (15K-30K)'
    df.loc[(df['param_count_mean'] >= 30000) & (df['param_count_mean'] < 50000), 'param_group'] = 'Medium (30K-50K)'
    df.loc[df['param_count_mean'] >= 50000, 'param_group'] = 'Large (> 50K)'
    
    return df

def perform_fair_comparison_analysis(df):
    """Perform fair comparison analysis with matched parameter sizes"""
    
    print("=== FAIR COMPARISON ANALYSIS ===")
    print("Addressing revision request: Fair comparison with matched parameter sizes")
    print()
    
    # Group by parameter size
    df_grouped = create_parameter_size_groups(df)
    
    print("Parameter Size Groups:")
    for group in df_grouped['param_group'].unique():
        group_data = df_grouped[df_grouped['param_group'] == group]
        print(f"\n{group}:")
        for _, row in group_data.iterrows():
            print(f"  - {row['model_name']}({row['decoder']}): {row['param_count_mean']:.0f} params, {row['psnr_mean']:.2f} dB")
    
    print("\n" + "="*80)
    print("FAIR COMPARISON: ARCHITECTURES WITH SIMILAR PARAMETER COUNTS")
    print("="*80)
    
    # Focus on the most populated groups for fair comparison
    small_group = df_grouped[df_grouped['param_group'] == 'Small (< 15K)']
    medium_small_group = df_grouped[df_grouped['param_group'] == 'Medium-Small (15K-30K)']
    medium_group = df_grouped[df_grouped['param_group'] == 'Medium (30K-50K)']
    
    print("\n### COMPARISON GROUP 1: Small Models (< 15K parameters)")
    print("=" * 60)
    if len(small_group) > 0:
        small_sorted = small_group.sort_values('psnr_mean', ascending=False)
        print("Ranking by PSNR (matched parameter size):")
        for i, (_, row) in enumerate(small_sorted.iterrows(), 1):
            print(f"{i}. {row['model_name']}({row['decoder']}): {row['psnr_mean']:.2f} ± {row['psnr_std']:.2f} dB "
                  f"({row['param_count_mean']:.0f} params)")
        
        # Statistical significance testing within this group
        if len(small_sorted) >= 2:
            best = small_sorted.iloc[0]
            second = small_sorted.iloc[1]
            improvement = best['psnr_mean'] - second['psnr_mean']
            print(f"\nBest vs Second: +{improvement:.2f} dB improvement")
            print(f"Relative improvement: {improvement/second['psnr_mean']*100:.1f}%")
    
    print("\n### COMPARISON GROUP 2: Medium-Small Models (15K-30K parameters)")
    print("=" * 60)
    if len(medium_small_group) > 0:
        med_small_sorted = medium_small_group.sort_values('psnr_mean', ascending=False)
        print("Ranking by PSNR (matched parameter size):")
        for i, (_, row) in enumerate(med_small_sorted.iterrows(), 1):
            print(f"{i}. {row['model_name']}({row['decoder']}): {row['psnr_mean']:.2f} ± {row['psnr_std']:.2f} dB "
                  f"({row['param_count_mean']:.0f} params)")
        
        if len(med_small_sorted) >= 2:
            best = med_small_sorted.iloc[0]
            second = med_small_sorted.iloc[1]
            improvement = best['psnr_mean'] - second['psnr_mean']
            print(f"\nBest vs Second: +{improvement:.2f} dB improvement")
            print(f"Relative improvement: {improvement/second['psnr_mean']*100:.1f}%")
    
    print("\n### COMPARISON GROUP 3: Medium Models (30K-50K parameters)")
    print("=" * 60)
    if len(medium_group) > 0:
        med_sorted = medium_group.sort_values('psnr_mean', ascending=False)
        print("Ranking by PSNR (matched parameter size):")
        for i, (_, row) in enumerate(med_sorted.iterrows(), 1):
            print(f"{i}. {row['model_name']}({row['decoder']}): {row['psnr_mean']:.2f} ± {row['psnr_std']:.2f} dB "
                  f"({row['param_count_mean']:.0f} params)")
        
        if len(med_sorted) >= 2:
            best = med_sorted.iloc[0]
            second = med_sorted.iloc[1]
            improvement = best['psnr_mean'] - second['psnr_mean']
            print(f"\nBest vs Second: +{improvement:.2f} dB improvement")
            print(f"Relative improvement: {improvement/second['psnr_mean']*100:.1f}%")
    
    return df_grouped

def select_matched_configs_for_analysis(df_grouped):
    """Select 3-5 configs with almost matching sizes for detailed analysis"""
    
    print("\n" + "="*80)
    print("SELECTED CONFIGURATIONS FOR DETAILED FAIR COMPARISON")
    print("="*80)
    print("Selection criteria: 3-5 configs with most similar parameter counts")
    
    # Find the most densely populated parameter range
    param_counts = df_grouped['param_count_mean'].values
    
    # Create windows and find the window with most models
    windows = []
    window_size = 15000  # 15K parameter window
    for start in range(0, int(max(param_counts)), 5000):
        end = start + window_size
        models_in_window = df_grouped[
            (df_grouped['param_count_mean'] >= start) & 
            (df_grouped['param_count_mean'] < end)
        ]
        if len(models_in_window) >= 3:
            windows.append((start, end, models_in_window))
    
    if windows:
        # Select the window with the most models
        best_window = max(windows, key=lambda x: len(x[2]))
        selected_models = best_window[2]
        
        print(f"Parameter Range: {best_window[0]:,} - {best_window[1]:,} parameters")
        print(f"Selected {len(selected_models)} configurations:")
        print()
        
        # Sort by parameter count for fair comparison
        selected_sorted = selected_models.sort_values('param_count_mean')
        
        comparison_data = []
        for i, (_, row) in enumerate(selected_sorted.iterrows(), 1):
            param_efficiency = row['psnr_mean'] / (row['param_count_mean'] / 1000)  # dB per K params
            comparison_data.append({
                'rank': i,
                'architecture': f"{row['model_name']}({row['decoder']})",
                'psnr': row['psnr_mean'],
                'psnr_std': row['psnr_std'],
                'params': int(row['param_count_mean']),
                'efficiency': param_efficiency
            })
            
            print(f"{i}. **{row['model_name']}({row['decoder']})**: "
                  f"{row['psnr_mean']:.2f} ± {row['psnr_std']:.2f} dB, "
                  f"{row['param_count_mean']:.0f} parameters")
            print(f"   Parameter efficiency: {param_efficiency:.3f} dB/K")
            print()
        
        # Statistical analysis of selected configs
        print("Statistical Analysis of Selected Configurations:")
        print("-" * 50)
        
        psnr_values = selected_sorted['psnr_mean'].values
        best_psnr = max(psnr_values)
        worst_psnr = min(psnr_values)
        
        print(f"PSNR Range: {worst_psnr:.2f} - {best_psnr:.2f} dB")
        print(f"Performance Spread: {best_psnr - worst_psnr:.2f} dB")
        print(f"Coefficient of Variation: {np.std(psnr_values)/np.mean(psnr_values)*100:.1f}%")
        
        # Find best and worst performers with matched sizes
        best_model = selected_sorted.loc[selected_sorted['psnr_mean'].idxmax()]
        worst_model = selected_sorted.loc[selected_sorted['psnr_mean'].idxmin()]
        
        print(f"\nBest Performer: {best_model['model_name']}({best_model['decoder']})")
        print(f"Worst Performer: {worst_model['model_name']}({worst_model['decoder']})")
        print(f"Performance Gap: {best_model['psnr_mean'] - worst_model['psnr_mean']:.2f} dB")
        
        return comparison_data, selected_sorted
    
    else:
        print("No suitable parameter range found with 3+ models")
        # Fallback: select models with closest parameter counts
        df_sorted = df_grouped.sort_values('param_count_mean')
        # Take first 4-5 models as they'll have most similar sizes
        selected = df_sorted.head(5)
        comparison_data = []
        for i, (_, row) in enumerate(selected.iterrows(), 1):
            comparison_data.append({
                'rank': i,
                'architecture': f"{row['model_name']}({row['decoder']})",
                'psnr': row['psnr_mean'],
                'psnr_std': row['psnr_std'],
                'params': int(row['param_count_mean']),
                'efficiency': row['psnr_mean'] / (row['param_count_mean'] / 1000)
            })
        return comparison_data, selected

def create_fair_comparison_visualization(comparison_data, selected_df):
    """Create visualizations for fair comparison analysis"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fair Comparison Analysis: Matched Parameter Sizes', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    architectures = [d['architecture'] for d in comparison_data]
    psnr_values = [d['psnr'] for d in comparison_data]
    psnr_stds = [d['psnr_std'] for d in comparison_data]
    param_counts = [d['params'] for d in comparison_data]
    efficiencies = [d['efficiency'] for d in comparison_data]
    
    # 1. PSNR comparison with error bars
    ax1.bar(range(len(architectures)), psnr_values, yerr=psnr_stds, 
            capsize=5, alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(architectures))))
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR Performance (Matched Parameter Sizes)')
    ax1.set_xticks(range(len(architectures)))
    ax1.set_xticklabels(architectures, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Parameter count comparison
    ax2.bar(range(len(architectures)), param_counts, 
            alpha=0.7, color=plt.cm.plasma(np.linspace(0, 1, len(architectures))))
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Parameter Count')
    ax2.set_title('Parameter Count Comparison')
    ax2.set_xticks(range(len(architectures)))
    ax2.set_xticklabels(architectures, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Parameter efficiency
    ax3.bar(range(len(architectures)), efficiencies,
            alpha=0.7, color=plt.cm.coolwarm(np.linspace(0, 1, len(architectures))))
    ax3.set_xlabel('Architecture')
    ax3.set_ylabel('Parameter Efficiency (dB/K)')
    ax3.set_title('Parameter Efficiency Comparison')
    ax3.set_xticks(range(len(architectures)))
    ax3.set_xticklabels(architectures, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. PSNR vs Parameters scatter with size matching
    for i, (psnr, params, arch) in enumerate(zip(psnr_values, param_counts, architectures)):
        ax4.scatter(params, psnr, s=100, alpha=0.7, 
                   color=plt.cm.viridis(i/len(architectures)), label=arch)
    
    ax4.set_xlabel('Parameter Count')
    ax4.set_ylabel('PSNR (dB)')
    ax4.set_title('PSNR vs Parameter Count (Fair Comparison Set)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('fair_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_analysis_insights(comparison_data, selected_df):
    """Generate insights from the fair comparison analysis"""
    
    insights = {
        "fair_comparison_analysis": {
            "selected_configs": len(comparison_data),
            "parameter_range": {
                "min": min([d['params'] for d in comparison_data]),
                "max": max([d['params'] for d in comparison_data]),
                "range_kb": (max([d['params'] for d in comparison_data]) - 
                           min([d['params'] for d in comparison_data])) / 1000
            },
            "performance_analysis": {
                "best_performer": max(comparison_data, key=lambda x: x['psnr']),
                "worst_performer": min(comparison_data, key=lambda x: x['psnr']),
                "performance_gap_db": max([d['psnr'] for d in comparison_data]) - 
                                    min([d['psnr'] for d in comparison_data]),
                "mean_psnr": np.mean([d['psnr'] for d in comparison_data]),
                "std_psnr": np.std([d['psnr'] for d in comparison_data])
            },
            "efficiency_analysis": {
                "most_efficient": max(comparison_data, key=lambda x: x['efficiency']),
                "least_efficient": min(comparison_data, key=lambda x: x['efficiency']),
                "efficiency_ratio": max([d['efficiency'] for d in comparison_data]) / 
                                  min([d['efficiency'] for d in comparison_data])
            }
        }
    }
    
    return insights

def main():
    """Main analysis function"""
    print("Starting Fair Comparison Analysis for INR Architectures")
    print("=" * 80)
    
    # Load data
    df = load_experimental_data()
    print(f"Loaded {len(df)} architecture configurations")
    
    # Perform fair comparison analysis
    df_grouped = perform_fair_comparison_analysis(df)
    
    # Select matched configurations
    comparison_data, selected_df = select_matched_configs_for_analysis(df_grouped)
    
    # Create visualizations
    fig = create_fair_comparison_visualization(comparison_data, selected_df)
    
    # Generate insights
    insights = generate_analysis_insights(comparison_data, selected_df)
    
    # Save insights to analyze.jsonl
    with open('analyze.jsonl', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print("\n" + "="*80)
    print("FAIR COMPARISON ANALYSIS COMPLETE")
    print("="*80)
    print(f"✓ Analyzed {len(comparison_data)} matched configurations")
    print(f"✓ Parameter range: {insights['fair_comparison_analysis']['parameter_range']['min']:,} - {insights['fair_comparison_analysis']['parameter_range']['max']:,}")
    print(f"✓ Performance gap: {insights['fair_comparison_analysis']['performance_analysis']['performance_gap_db']:.2f} dB")
    print(f"✓ Efficiency ratio: {insights['fair_comparison_analysis']['efficiency_analysis']['efficiency_ratio']:.1f}x")
    print("✓ Visualization saved as 'fair_comparison_analysis.png'")
    print("✓ Insights saved to 'analyze.jsonl'")
    
    return insights, comparison_data

if __name__ == "__main__":
    insights, comparison_data = main()
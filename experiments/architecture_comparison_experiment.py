#!/usr/bin/env python3
"""
Architecture Comparison Experiment for 2D Matrix Reconstruction with INRs

This experiment systematically compares different INR architectures (K-Planes, NeRF, GA-Planes) 
for 2D matrix reconstruction tasks, following rigorous scientific methodology.

Primary Hypothesis: K-Planes with planar factorization will outperform NeRF for 2D matrix 
reconstruction by >5dB PSNR due to explicit geometric bias toward planar structures.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Statistical testing
import scipy.stats as stats
import pingouin as pg

# Import the base models from some_examples.py
sys.path.append(str(Path(__file__).parent))
from some_examples import (
    CustomModel, experiment, experiment_quantized, 
    psnr_normalized, find_pareto, model_size, model_size_quantized
)

import torch
import torch.nn as nn
import torch.optim as optim
import skimage.data
from PIL import Image

# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class INRArchitectureComparisonExperiment:
    """
    Systematic comparison of INR architectures for 2D matrix reconstruction.
    
    Tests the hypothesis that K-Planes with explicit geometric bias will achieve
    superior parameter efficiency and reconstruction quality compared to traditional
    MLP-based approaches (NeRF) for 2D tasks.
    """
    
    def __init__(self, output_dir, device='cpu', seeds=[42, 123, 456, 789, 101112]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.seeds = seeds
        
        # Setup logging
        self.setup_logging()
        
        # Experimental parameters from proposal.jsonl
        self.line_resolutions = [32, 64, 128, 192, 256]
        self.plane_resolutions = [8, 16, 32, 64] 
        self.feature_dims = [32, 64, 128]
        self.hidden_dims = [32, 64, 128, 256]
        
        # Architecture configurations to test
        self.architectures = [
            # K-Planes variants
            {'name': 'K-Planes-Multiply-Linear', 'operation': 'multiply', 'decoder': 'linear', 'mode': 'lowres'},
            {'name': 'K-Planes-Add-Linear', 'operation': 'add', 'decoder': 'linear', 'mode': 'lowres'},
            {'name': 'K-Planes-Multiply-NonConvex', 'operation': 'multiply', 'decoder': 'nonconvex', 'mode': 'lowres'},
            {'name': 'K-Planes-Add-NonConvex', 'operation': 'add', 'decoder': 'nonconvex', 'mode': 'lowres'},
            
            # NeRF-style variants (no plane features, just line interactions)
            {'name': 'NeRF-Multiply-NonConvex', 'operation': 'multiply', 'decoder': 'nonconvex', 'mode': 'lines_only'},
            {'name': 'NeRF-Add-NonConvex', 'operation': 'add', 'decoder': 'nonconvex', 'mode': 'lines_only'},
            
            # GA-Planes style (additive + plane features)
            {'name': 'GA-Planes-Add-NonConvex', 'operation': 'add', 'decoder': 'nonconvex', 'mode': 'lowres'},
            
            # Convex decoders
            {'name': 'K-Planes-Multiply-Convex', 'operation': 'multiply', 'decoder': 'convex', 'mode': 'lowres'},
        ]
        
        # Results storage
        self.results = []
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_test_images(self):
        """Load and prepare test images for evaluation."""
        images = {}
        
        # Primary: Astronaut image (as specified in proposal)
        img = skimage.data.astronaut() / 255.0
        if len(img.shape) > 2:
            img = np.mean(img, axis=-1)  # Convert to grayscale
        images['astronaut'] = img
        
        # Secondary: Load from available datasets
        data_dir = Path(__file__).parent.parent / 'data'
        
        # Try to load a few CIFAR-10 images for diversity
        try:
            import pickle
            cifar_path = data_dir / 'processed/cifar10/cifar-10-batches-py/data_batch_1'
            if cifar_path.exists():
                with open(cifar_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    # Get first image and convert to grayscale
                    cifar_img = batch[b'data'][0].reshape(3, 32, 32).transpose(1, 2, 0) / 255.0
                    cifar_gray = np.mean(cifar_img, axis=-1)
                    images['cifar_sample'] = cifar_gray
        except Exception as e:
            self.logger.warning(f"Could not load CIFAR-10 sample: {e}")
            
        # Synthetic patterns for controlled testing
        try:
            synthetic_path = data_dir / 'processed/synthetic/checkerboards.npy'
            if synthetic_path.exists():
                checkers = np.load(synthetic_path)[0]  # Get first pattern
                if checkers.max() > 1:
                    checkers = checkers / 255.0
                images['checkerboard'] = checkers
        except Exception as e:
            self.logger.warning(f"Could not load synthetic patterns: {e}")
            
        self.logger.info(f"Loaded {len(images)} test images: {list(images.keys())}")
        return images
        
    def run_single_experiment(self, img_name, img, arch_config, line_res, plane_res, 
                            feature_dim, hidden_dim, seed):
        """Run a single experiment configuration."""
        set_seeds(seed)
        
        try:
            start_time = time.time()
            
            # For NeRF-style (lines_only), don't use plane features
            mode = 'lowres' if arch_config['mode'] != 'lines_only' else 'lowres'
            actual_plane_res = plane_res if arch_config['mode'] != 'lines_only' else 8  # minimal
            
            # Run experiment
            if arch_config.get('quantized', False):
                psnr, param_count = experiment_quantized(
                    img=img,
                    lineres=line_res,
                    planeres=actual_plane_res, 
                    dim_features=feature_dim,
                    m=hidden_dim,
                    operation=arch_config['operation'],
                    decoder=arch_config['decoder'],
                    interpolation='bilinear',
                    bias=True,
                    mode=mode
                )
            else:
                psnr, param_count = experiment(
                    lineres=line_res,
                    planeres=actual_plane_res,
                    dim_features=feature_dim, 
                    m=hidden_dim,
                    operation=arch_config['operation'],
                    decoder=arch_config['decoder'],
                    interpolation='bilinear',
                    bias=True,
                    mode=mode
                )
            
            elapsed_time = time.time() - start_time
            
            # Calculate parameter efficiency 
            param_efficiency = psnr / param_count if param_count > 0 else 0
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'image_name': img_name,
                'image_shape': img.shape,
                'architecture': arch_config['name'],
                'operation': arch_config['operation'],
                'decoder': arch_config['decoder'],
                'mode': arch_config['mode'],
                'line_resolution': line_res,
                'plane_resolution': actual_plane_res,
                'feature_dim': feature_dim,
                'hidden_dim': hidden_dim,
                'seed': seed,
                'psnr_db': psnr,
                'param_count': param_count,
                'param_efficiency': param_efficiency,
                'training_time_sec': elapsed_time,
                'success': True
            }
            
            self.logger.info(f"✓ {arch_config['name']} | {img_name} | PSNR: {psnr:.2f}dB | Params: {param_count} | Time: {elapsed_time:.1f}s")
            
        except Exception as e:
            self.logger.error(f"✗ {arch_config['name']} | {img_name} | Error: {str(e)}")
            result = {
                'timestamp': datetime.now().isoformat(),
                'image_name': img_name,
                'architecture': arch_config['name'],
                'seed': seed,
                'success': False,
                'error': str(e)
            }
            
        return result
        
    def run_architecture_comparison(self, max_configs=50):
        """
        Run Phase 1: Systematic architecture comparison.
        
        Tests primary hypothesis H1: K-Planes will achieve >5dB PSNR improvement 
        over NeRF at equivalent parameter counts.
        """
        self.logger.info("=== Phase 1: Architecture Comparison Study ===")
        
        images = self.load_test_images()
        
        # Focus on astronaut image for primary hypothesis testing
        primary_image = 'astronaut'
        if primary_image not in images:
            raise ValueError(f"Primary test image '{primary_image}' not available")
            
        img = images[primary_image]
        self.logger.info(f"Primary test image: {primary_image} {img.shape}")
        
        # Generate experimental configurations
        configs = []
        for arch in self.architectures:
            for line_res in self.line_resolutions:
                for feature_dim in self.feature_dims:
                    for hidden_dim in self.hidden_dims:
                        # Use fixed plane resolution to control parameter count
                        plane_res = 32  
                        configs.append({
                            'arch': arch,
                            'line_res': line_res,
                            'plane_res': plane_res,
                            'feature_dim': feature_dim,
                            'hidden_dim': hidden_dim
                        })
        
        # Limit configs for reasonable runtime
        if len(configs) > max_configs:
            configs = configs[:max_configs]
            self.logger.warning(f"Limited to {max_configs} configurations for runtime constraints")
            
        self.logger.info(f"Running {len(configs)} configurations x {len(self.seeds)} seeds = {len(configs) * len(self.seeds)} total experiments")
        
        # Run experiments
        for i, config in enumerate(tqdm(configs, desc="Configurations")):
            for seed in self.seeds:
                result = self.run_single_experiment(
                    img_name=primary_image,
                    img=img,
                    arch_config=config['arch'],
                    line_res=config['line_res'],
                    plane_res=config['plane_res'],
                    feature_dim=config['feature_dim'],
                    hidden_dim=config['hidden_dim'],
                    seed=seed
                )
                self.results.append(result)
                
        # Save intermediate results
        self.save_results()
        self.logger.info(f"Phase 1 completed. {len(self.results)} experiments run.")
        
    def analyze_results(self):
        """Analyze experimental results with statistical testing."""
        self.logger.info("=== Statistical Analysis ===")
        
        # Convert to DataFrame
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            self.logger.error("No successful experiments to analyze")
            return
            
        df = pd.DataFrame(successful_results)
        
        # Group architectures for comparison
        df['architecture_family'] = df['architecture'].apply(self.classify_architecture_family)
        
        # Statistical summary
        summary_stats = df.groupby('architecture').agg({
            'psnr_db': ['mean', 'std', 'count'],
            'param_count': 'mean',
            'param_efficiency': 'mean',
            'training_time_sec': 'mean'
        }).round(3)
        
        self.logger.info("Summary Statistics by Architecture:")
        self.logger.info(f"\n{summary_stats}")
        
        # Test primary hypothesis H1: K-Planes > NeRF by >5dB
        self.test_primary_hypothesis(df)
        
        # ANOVA across architecture families
        self.test_architecture_families(df)
        
        # Parameter efficiency analysis
        self.analyze_parameter_efficiency(df)
        
        # Generate plots
        self.generate_plots(df)
        
        return df
        
    def classify_architecture_family(self, arch_name):
        """Classify architecture into family for statistical testing."""
        if 'K-Planes' in arch_name:
            return 'K-Planes'
        elif 'NeRF' in arch_name:
            return 'NeRF'
        elif 'GA-Planes' in arch_name:
            return 'GA-Planes'
        else:
            return 'Other'
            
    def test_primary_hypothesis(self, df):
        """Test H1: K-Planes will achieve >5dB PSNR improvement over NeRF."""
        self.logger.info("Testing Primary Hypothesis H1...")
        
        kplanes_results = df[df['architecture_family'] == 'K-Planes']['psnr_db']
        nerf_results = df[df['architecture_family'] == 'NeRF']['psnr_db']
        
        if len(kplanes_results) == 0 or len(nerf_results) == 0:
            self.logger.warning("Insufficient data for K-Planes vs NeRF comparison")
            return
            
        # Statistical test
        statistic, p_value = stats.ttest_ind(kplanes_results, nerf_results)
        effect_size = (kplanes_results.mean() - nerf_results.mean()) / np.sqrt(
            ((len(kplanes_results) - 1) * kplanes_results.var() + 
             (len(nerf_results) - 1) * nerf_results.var()) / 
            (len(kplanes_results) + len(nerf_results) - 2)
        )
        
        improvement_db = kplanes_results.mean() - nerf_results.mean()
        
        self.logger.info(f"H1 Results:")
        self.logger.info(f"  K-Planes mean PSNR: {kplanes_results.mean():.2f} ± {kplanes_results.std():.2f} dB")
        self.logger.info(f"  NeRF mean PSNR: {nerf_results.mean():.2f} ± {nerf_results.std():.2f} dB")
        self.logger.info(f"  Improvement: {improvement_db:.2f} dB")
        self.logger.info(f"  Statistical significance: p = {p_value:.6f}")
        self.logger.info(f"  Effect size (Cohen's d): {effect_size:.3f}")
        
        # Test hypothesis: improvement > 5dB and statistically significant
        hypothesis_supported = improvement_db > 5.0 and p_value < 0.05
        
        self.logger.info(f"  H1 RESULT: {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}")
        if hypothesis_supported:
            self.logger.info(f"  ✓ K-Planes achieved {improvement_db:.2f}dB > 5dB improvement with p < 0.05")
        else:
            if improvement_db <= 5.0:
                self.logger.info(f"  ✗ Improvement {improvement_db:.2f}dB ≤ 5dB threshold")
            if p_value >= 0.05:
                self.logger.info(f"  ✗ Not statistically significant (p = {p_value:.6f} ≥ 0.05)")
        
    def test_architecture_families(self, df):
        """Test differences across architecture families using ANOVA."""
        self.logger.info("Testing Architecture Family Differences...")
        
        families = df['architecture_family'].unique()
        if len(families) < 2:
            self.logger.warning("Need at least 2 architecture families for comparison")
            return
            
        # ANOVA
        family_groups = [df[df['architecture_family'] == family]['psnr_db'].values 
                        for family in families]
        
        f_stat, p_value = stats.f_oneway(*family_groups)
        
        self.logger.info(f"ANOVA Results:")
        self.logger.info(f"  F-statistic: {f_stat:.3f}")
        self.logger.info(f"  p-value: {p_value:.6f}")
        self.logger.info(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Post-hoc pairwise comparisons if significant
        if p_value < 0.05:
            self.logger.info("Post-hoc pairwise comparisons:")
            for i, fam1 in enumerate(families):
                for fam2 in families[i+1:]:
                    group1 = df[df['architecture_family'] == fam1]['psnr_db']
                    group2 = df[df['architecture_family'] == fam2]['psnr_db']
                    
                    if len(group1) > 0 and len(group2) > 0:
                        t_stat, t_p = stats.ttest_ind(group1, group2)
                        # Bonferroni correction
                        n_comparisons = len(families) * (len(families) - 1) // 2
                        corrected_p = t_p * n_comparisons
                        
                        self.logger.info(f"  {fam1} vs {fam2}: p = {corrected_p:.6f} {'*' if corrected_p < 0.05 else ''}")
        
    def analyze_parameter_efficiency(self, df):
        """Analyze parameter efficiency (PSNR per parameter)."""
        self.logger.info("Parameter Efficiency Analysis...")
        
        efficiency_stats = df.groupby('architecture_family')['param_efficiency'].agg([
            'mean', 'std', 'count'
        ]).round(6)
        
        self.logger.info(f"Parameter Efficiency by Family:")
        self.logger.info(f"\n{efficiency_stats}")
        
        # Find Pareto optimal solutions
        psnr_vals = df['psnr_db'].values
        param_vals = df['param_count'].values
        
        pareto_psnr, pareto_params, pareto_idx = find_pareto(param_vals, psnr_vals)
        
        self.logger.info(f"Pareto Optimal Solutions:")
        for i, idx in enumerate(pareto_idx):
            result = df.iloc[idx]
            self.logger.info(f"  {result['architecture']}: {result['psnr_db']:.2f}dB @ {result['param_count']} params")
            
    def generate_plots(self, df):
        """Generate visualization plots."""
        self.logger.info("Generating plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig_dir = self.output_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)
        
        # 1. PSNR comparison by architecture
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='architecture_family', y='psnr_db')
        plt.title('PSNR Performance by Architecture Family')
        plt.ylabel('PSNR (dB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fig_dir / 'psnr_by_architecture.png', dpi=300)
        plt.close()
        
        # 2. Parameter efficiency scatter plot
        plt.figure(figsize=(10, 8))
        
        families = df['architecture_family'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(families)))
        
        for i, family in enumerate(families):
            family_df = df[df['architecture_family'] == family]
            plt.scatter(family_df['param_count'], family_df['psnr_db'], 
                       label=family, alpha=0.7, c=[colors[i]], s=60)
        
        # Add Pareto frontier
        psnr_vals = df['psnr_db'].values
        param_vals = df['param_count'].values
        pareto_psnr, pareto_params, pareto_idx = find_pareto(param_vals, psnr_vals)
        
        # Sort for proper line plotting
        sort_idx = np.argsort(pareto_params)
        plt.plot(pareto_params[sort_idx], pareto_psnr[sort_idx], 
                'k--', linewidth=2, alpha=0.8, label='Pareto Frontier')
        
        plt.xlabel('Parameter Count')
        plt.ylabel('PSNR (dB)')
        plt.title('Parameter Efficiency: PSNR vs Model Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / 'parameter_efficiency.png', dpi=300)
        plt.close()
        
        # 3. Training time comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='architecture_family', y='training_time_sec')
        plt.title('Training Time by Architecture Family')
        plt.ylabel('Training Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fig_dir / 'training_time.png', dpi=300)
        plt.close()
        
        self.logger.info(f"Plots saved to {fig_dir}")
        
    def save_results(self):
        """Save experimental results."""
        # Save raw results as JSON
        results_file = self.output_dir / 'experiment_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Save as CSV for easy analysis
        successful_results = [r for r in self.results if r.get('success', False)]
        if successful_results:
            df = pd.DataFrame(successful_results)
            csv_file = self.output_dir / 'experiment_results.csv'
            df.to_csv(csv_file, index=False)
            
        self.logger.info(f"Results saved to {self.output_dir}")
        
    def run_full_experiment(self):
        """Run the complete experimental protocol."""
        self.logger.info("Starting INR Architecture Comparison Experiment")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Seeds: {self.seeds}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        try:
            # Phase 1: Architecture comparison
            self.run_architecture_comparison(max_configs=30)  # Limited for CI/CD
            
            # Analysis and statistical testing
            df = self.analyze_results()
            
            self.logger.info("=== EXPERIMENT COMPLETED SUCCESSFULLY ===")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='INR Architecture Comparison Experiment')
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='Device to run experiments on')
    parser.add_argument('--max-configs', type=int, default=30,
                       help='Maximum number of configurations to test')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                       help='Random seeds for reproducibility')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = INRArchitectureComparisonExperiment(
        output_dir=args.output_dir,
        device=args.device,
        seeds=args.seeds
    )
    
    df = experiment.run_full_experiment()
    
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    if df is not None and len(df) > 0:
        print(f"Total successful experiments: {len(df)}")
        print(f"Architecture families tested: {df['architecture_family'].nunique()}")
        print(f"Best PSNR: {df['psnr_db'].max():.2f} dB ({df.loc[df['psnr_db'].idxmax(), 'architecture']})")
        print(f"Most efficient: {df.loc[df['param_efficiency'].idxmax(), 'architecture']} "
              f"({df['param_efficiency'].max():.2e} PSNR/param)")
        
        # Check primary hypothesis result
        kplanes_mean = df[df['architecture_family'] == 'K-Planes']['psnr_db'].mean()
        nerf_mean = df[df['architecture_family'] == 'NeRF']['psnr_db'].mean()
        
        if not pd.isna(kplanes_mean) and not pd.isna(nerf_mean):
            improvement = kplanes_mean - nerf_mean
            print(f"\nPrimary Hypothesis (H1): K-Planes vs NeRF")
            print(f"Improvement: {improvement:.2f} dB")
            print(f"Hypothesis {'SUPPORTED' if improvement > 5.0 else 'NOT SUPPORTED'}")
    
    print(f"\nFull results saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
INR Architecture Comparison Experiment
=====================================

Systematic comparison of K-Planes vs NeRF architectures for 2D matrix reconstruction.
Tests the primary research hypothesis using rigorous statistical methodology.

Model Naming Conventions:
------------------------
- K-planes: Models using ONLY line features (line_feat_x * line_feat_y or line_feat_x + line_feat_y)
  * K-planes(multiply): decoder(line_feat_x * line_feat_y)
  * K-planes(add): decoder(line_feat_x + line_feat_y)

- GA-Planes: Models using line features + low-resolution plane features
  * GA-Planes(multiply+plane): decoder(line_feat_x * line_feat_y + plane_feat)
  * GA-Planes(add+plane): decoder(line_feat_x + line_feat_y + plane_feat)

- NeRF: Coordinate-based models with deeper MLPs
  * NeRF(nonconvex): Standard ReLU-based deep network
  * NeRF(siren): SIREN-based with sinusoidal activations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import skimage
from tqdm import tqdm
import json
import argparse
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis imports
from scipy import stats
import pingouin as pg
from sklearn.metrics import r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f"Using device: {device}")

class CustomModel(nn.Module):
    """
    Unified INR model supporting K-Planes and NeRF architectures.
    Adapted from notes/some_examples.py with architectural variants.
    """
    def __init__(self, dim1: int, dim2: int, dim_features: int, m: int, 
                 resolution: Tuple[int, int], operation: str, decoder: str,
                 interpolation: str = 'bilinear', bias: bool = True, 
                 mode: str = 'lowres', architecture: str = 'kplanes'):
        super().__init__()

        self.resx, self.resy = resolution
        self.operation = operation
        self.decoder = decoder
        self.interpolation = interpolation
        self.bias = bias
        self.mode = mode
        self.architecture = architecture
        
        # Initialize features based on architecture
        torch.manual_seed(42)  # Reproducible initialization
        
        if architecture == 'kplanes':
            # K-Planes: explicit line and plane features
            if operation == 'multiply':
                self.line_feature_x = nn.Parameter(torch.rand(dim_features, dim1) * 0.15 + 0.1)
                self.line_feature_y = nn.Parameter(torch.rand(dim_features, dim1) * 0.15 + 0.1)
            else:  # add
                self.line_feature_x = nn.Parameter(torch.rand(dim_features, dim1) * 0.03 + 0.005)
                self.line_feature_y = nn.Parameter(torch.rand(dim_features, dim1) * 0.03 + 0.005)
            
            if "lowres" in mode:
                self.plane_feature = nn.Parameter(torch.randn(dim_features, dim2, dim2) * 0.01)
        
        elif architecture == 'nerf':
            # NeRF: coordinate encoding + MLP
            # For fair comparison, use similar parameter count
            if decoder == 'siren':
                # SIREN initialization with deeper network
                self.coord_mlp = nn.Sequential(
                    nn.Linear(2, dim_features, bias=bias),
                    nn.Linear(dim_features, dim_features, bias=bias),
                    nn.Linear(dim_features, dim_features, bias=bias),
                    nn.Linear(dim_features, dim_features, bias=bias),
                )
                # Initialize for SIREN
                for layer in self.coord_mlp:
                    if isinstance(layer, nn.Linear):
                        with torch.no_grad():
                            layer.weight.uniform_(-1 / layer.in_features, 1 / layer.in_features)
            else:
                # Standard ReLU NeRF with deeper network (typical for NeRF)
                self.coord_mlp = nn.Sequential(
                    nn.Linear(2, dim_features, bias=bias),
                    nn.ReLU(),
                    nn.Linear(dim_features, dim_features, bias=bias),
                    nn.ReLU(),
                    nn.Linear(dim_features, dim_features, bias=bias),
                    nn.ReLU(),
                    nn.Linear(dim_features, dim_features, bias=bias),
                    nn.ReLU(),
                )
        
        # Decoder network
        if decoder == 'linear':
            self.decoder_net = nn.Sequential(
                nn.Linear(dim_features, 1, bias=bias),
            )
        elif decoder == 'nonconvex':
            self.decoder_net = nn.Sequential(
                nn.Linear(dim_features, m, bias=bias),
                nn.ReLU(),
                nn.Linear(m, 1, bias=bias)
            )
        elif decoder == 'convex':
            self.fc1 = nn.Linear(dim_features, m, bias=bias)
            self.fc2 = nn.Linear(dim_features, m, bias=bias)
            self.fc2.weight.requires_grad = False
            if bias:
                self.fc2.bias.requires_grad = False
        elif decoder == 'siren':
            # SIREN decoder with sine activation
            self.decoder_net = nn.Sequential(
                nn.Linear(dim_features, 1, bias=bias),
            )
        else:
            raise ValueError(f"Invalid decoder {decoder}")

    def forward(self, coords):
        """Forward pass through the INR model."""
        if self.architecture == 'kplanes':
            return self._forward_kplanes(coords)
        elif self.architecture == 'nerf':
            return self._forward_nerf(coords)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def _forward_kplanes(self, coords):
        """K-Planes forward pass with explicit factorization."""
        # Prepare coordinates for grid sampling
        x_coords = coords[..., 0].unsqueeze(-1)
        y_coords = coords[..., 1].unsqueeze(-1)

        # Scale to [-1, 1] for grid_sample
        x_coords = (x_coords * 2 / self.resx) - 1
        y_coords = (y_coords * 2 / self.resy) - 1
        
        # Grid sampling for line features
        gridx = torch.cat((x_coords, x_coords), dim=-1).unsqueeze(0)
        gridy = torch.cat((y_coords, y_coords), dim=-1).unsqueeze(0)

        line_features_x = self.line_feature_x.unsqueeze(0).unsqueeze(-1)
        line_features_y = self.line_feature_y.unsqueeze(0).unsqueeze(-1)

        feature_x = F.grid_sample(line_features_x, gridx, mode=self.interpolation, 
                                 padding_mode='border', align_corners=True)
        feature_y = F.grid_sample(line_features_y, gridy, mode=self.interpolation, 
                                 padding_mode='border', align_corners=True)

        # Plane feature sampling
        if "lowres" in self.mode:
            plane_features = self.plane_feature.unsqueeze(0)
            plane_grid = torch.cat((x_coords, y_coords), dim=-1).unsqueeze(0)
            sampled_plane_features = F.grid_sample(plane_features, plane_grid, 
                                                  mode=self.interpolation, align_corners=True)
        else:
            sampled_plane_features = 0

        # Combine features based on operation
        if self.operation == 'add':
            combined_features = feature_x + feature_y + sampled_plane_features
        elif self.operation == 'multiply':
            combined_features = feature_x * feature_y + sampled_plane_features
        else:
            raise ValueError(f"Invalid operation {self.operation}")

        combined_features = combined_features.squeeze().permute(1, 2, 0)
        
        # Decoder
        if self.decoder == 'convex':
            output = self.fc1(combined_features) * (self.fc2(combined_features) > 0)
            output = torch.mean(output, dim=-1)
        else:
            output = self.decoder_net(combined_features).squeeze()
        
        return output
    
    def _forward_nerf(self, coords):
        """NeRF forward pass with coordinate encoding."""
        # Normalize coordinates to [-1, 1]
        normalized_coords = coords.clone()
        normalized_coords[..., 0] = (coords[..., 0] / self.resx) * 2 - 1
        normalized_coords[..., 1] = (coords[..., 1] / self.resy) * 2 - 1
        
        # Coordinate encoding through MLP
        if self.decoder == 'siren':
            # SIREN activation
            features = normalized_coords
            for i, layer in enumerate(self.coord_mlp):
                features = layer(features)
                features = torch.sin(features)  # SIREN uses sine activations
        else:
            # Standard ReLU encoding
            features = self.coord_mlp(normalized_coords)
        
        # Decoder
        if self.decoder == 'convex':
            output = self.fc1(features) * (self.fc2(features) > 0)
            output = torch.mean(output, dim=-1)
        elif self.decoder == 'siren':
            output = torch.sin(self.decoder_net(features)).squeeze()
        else:
            output = self.decoder_net(features).squeeze()
        
        return output

def psnr_normalized(gt: np.ndarray, recon: np.ndarray) -> float:
    """Compute normalized PSNR between ground truth and reconstruction."""
    scale = 1 if np.max(gt) <= 1 else 255
    residual = (gt - recon) / scale
    mse = np.mean(np.square(residual))
    if mse == 0:
        return float('inf')
    return -10 * np.log10(mse)

def model_parameter_count(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_reconstruction_visualization(reconstruction_history: List[Dict], 
                                    ground_truth: np.ndarray,
                                    output_path: Path,
                                    model_name: str):
    """Save visualization of reconstruction progress over training epochs."""
    if not reconstruction_history:
        return
    
    # Select up to 6 checkpoints for visualization
    num_checkpoints = min(6, len(reconstruction_history))
    indices = np.linspace(0, len(reconstruction_history) - 1, num_checkpoints, dtype=int)
    
    fig, axes = plt.subplots(2, num_checkpoints + 1, figsize=(3 * (num_checkpoints + 1), 6))
    
    # Ground truth
    axes[0, 0].imshow(ground_truth, cmap='gray')
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Reconstructions at different epochs
    for i, idx in enumerate(indices):
        checkpoint = reconstruction_history[idx]
        epoch = checkpoint['epoch']
        recon = checkpoint['reconstruction']
        psnr = checkpoint['psnr']
        
        axes[0, i + 1].imshow(recon, cmap='gray')
        axes[0, i + 1].set_title(f'Epoch {epoch}')
        axes[0, i + 1].axis('off')
        
        # Difference image
        diff = np.abs(ground_truth - recon)
        axes[1, i + 1].imshow(diff, cmap='hot')
        axes[1, i + 1].set_title(f'Error (PSNR: {psnr:.1f}dB)')
        axes[1, i + 1].axis('off')
    
    plt.suptitle(f'{model_name} Reconstruction Progress', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_single_experiment(config: Dict[str, Any], seed: int = 42, visualize_every: int = 200) -> Dict[str, Any]:
    """
    Run a single experiment configuration with statistical tracking.
    
    Args:
        config: Experiment configuration dictionary
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with experiment results
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load and preprocess image
    img = skimage.data.astronaut() / 255.0
    if len(img.shape) > 2:
        img = np.mean(img, axis=-1)  # Convert to grayscale
    
    xdim, ydim = img.shape
    
    # Generate coordinates
    y_indices, x_indices = np.indices((xdim, ydim))
    coords = np.stack((x_indices, y_indices), axis=-1)
    coords = torch.from_numpy(coords).float().to(device)
    targets = torch.from_numpy(img).float().to(device)
    
    # Initialize model
    model = CustomModel(
        dim1=config['line_resolution'],
        dim2=config['plane_resolution'], 
        dim_features=config['feature_dim'],
        m=config.get('hidden_dim', 64),
        resolution=img.shape,
        operation=config['operation'],
        decoder=config['decoder'],
        interpolation='bilinear',
        bias=True,
        mode=config.get('mode', 'no_plane'),  # Use mode from config
        architecture=config['architecture']
    ).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    learning_rate = config.get('learning_rate', 0.05)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    num_epochs = config.get('num_epochs', 1000)
    psnr_history = []
    reconstruction_history = []  # Store reconstructions for visualization
    
    start_time = time.time()
    
    for epoch in tqdm(range(num_epochs), desc=f"Training {config['model_name']}-{config['decoder']}"):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(coords)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        # Track PSNR every 100 epochs
        if epoch % 100 == 0:
            current_psnr = -10 * np.log10(loss.item())
            psnr_history.append(current_psnr)
        
        # Visualize reconstruction every visualize_every epochs
        if visualize_every > 0 and epoch % visualize_every == 0:
            model.eval()
            with torch.no_grad():
                vis_outputs = model(coords)
                reconstruction_history.append({
                    'epoch': epoch,
                    'reconstruction': vis_outputs.cpu().numpy(),
                    'psnr': -10 * np.log10(criterion(vis_outputs, targets).item())
                })
            model.train()
    
    training_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(coords)
        final_loss = criterion(final_outputs, targets)
        final_psnr = -10 * np.log10(final_loss.item())
    
    # Compute parameter efficiency
    param_count = model_parameter_count(model)
    param_efficiency = final_psnr / param_count if param_count > 0 else 0
    
    # Store reconstruction for analysis
    reconstruction = final_outputs.cpu().numpy()
    
    return {
        'architecture': config['architecture'],
        'model_name': config['model_name'],  # Use the proper naming convention
        'operation': config['operation'],
        'decoder': config['decoder'],
        'feature_dim': config['feature_dim'],
        'line_resolution': config['line_resolution'],
        'plane_resolution': config['plane_resolution'],
        'seed': seed,
        'psnr': final_psnr,
        'param_count': param_count,
        'param_efficiency': param_efficiency,
        'training_time': training_time,
        'psnr_history': psnr_history,
        'reconstruction_history': reconstruction_history,  # Include visualization history
        'final_loss': final_loss.item(),
        'reconstruction': reconstruction
    }

def generate_experiment_configs() -> List[Dict[str, Any]]:
    """Generate systematic experiment configuration matrix."""
    configs = []
    
    # Base parameter ranges
    feature_dims = [32, 64, 128]
    resolutions = [32, 64, 128]
    
    # K-Planes configurations (line features only)
    for feature_dim in feature_dims:
        for resolution in resolutions:
            for operation in ['multiply', 'add']:
                for decoder in ['linear', 'nonconvex']:
                    configs.append({
                        'architecture': 'kplanes',
                        'model_name': f'K-planes({operation})',  # Proper naming
                        'operation': operation,
                        'decoder': decoder,
                        'feature_dim': feature_dim,
                        'line_resolution': resolution,
                        'plane_resolution': 0,  # No plane features for K-planes
                        'mode': 'no_plane',  # Line features only
                        'learning_rate': 0.15 if operation == 'multiply' and decoder == 'linear' else 0.05,
                        'num_epochs': 1000
                    })
    
    # GA-Planes configurations (line features + low-res plane)
    for feature_dim in feature_dims:
        for resolution in resolutions:
            for operation in ['multiply', 'add']:
                for decoder in ['linear', 'nonconvex']:
                    configs.append({
                        'architecture': 'kplanes',
                        'model_name': f'GA-Planes({operation}+plane)',  # Proper naming
                        'operation': operation,
                        'decoder': decoder,
                        'feature_dim': feature_dim,
                        'line_resolution': resolution,
                        'plane_resolution': resolution // 4,  # Lower res plane for efficiency
                        'mode': 'lowres',  # Include plane features
                        'learning_rate': 0.15 if operation == 'multiply' and decoder == 'linear' else 0.05,
                        'num_epochs': 1000
                    })
    
    # NeRF configurations  
    for feature_dim in feature_dims:
        for decoder in ['nonconvex', 'siren']:
            configs.append({
                'architecture': 'nerf',
                'model_name': f'NeRF({decoder})',  # Proper naming
                'operation': 'none',  # Not applicable for NeRF
                'decoder': decoder,
                'feature_dim': feature_dim,
                'line_resolution': 0,  # Not applicable
                'plane_resolution': 0,  # Not applicable
                'mode': 'none',  # Not applicable
                'learning_rate': 0.001 if decoder == 'siren' else 0.05,
                'num_epochs': 1000
            })
    
    logger.info(f"Generated {len(configs)} experiment configurations")
    return configs

def run_experiment_suite(output_dir: Path, num_seeds: int = 3) -> pd.DataFrame:
    """
    Run complete experimental suite with multiple random seeds.
    
    Args:
        output_dir: Directory to save results
        num_seeds: Number of random seeds per configuration
    
    Returns:
        DataFrame with all experimental results
    """
    configs = generate_experiment_configs()
    results = []
    
    logger.info(f"Starting experimental suite: {len(configs)} configs × {num_seeds} seeds = {len(configs) * num_seeds} total experiments")
    
    for i, config in enumerate(configs):
        logger.info(f"Running configuration {i+1}/{len(configs)}: {config['model_name']}-{config['decoder']}")
        
        for seed in range(num_seeds):
            try:
                result = run_single_experiment(config, seed=seed + 42, visualize_every=200)
                results.append(result)
                
                # Save reconstruction visualization
                vis_dir = output_dir / 'visualizations'
                vis_dir.mkdir(exist_ok=True)
                
                # Load ground truth image for visualization
                img = skimage.data.astronaut() / 255.0
                if len(img.shape) > 2:
                    img = np.mean(img, axis=-1)
                
                # Save visualization if we have reconstruction history
                if result.get('reconstruction_history'):
                    vis_filename = f"{config['model_name']}_{config['decoder']}_seed{seed}.png".replace(' ', '_').replace('(', '').replace(')', '')
                    save_reconstruction_visualization(
                        result['reconstruction_history'],
                        img,
                        vis_dir / vis_filename,
                        f"{config['model_name']} ({config['decoder']})"
                    )
                
                # Save intermediate results
                if len(results) % 10 == 0:
                    df_temp = pd.DataFrame(results)
                    # Remove reconstruction_history from CSV to save space
                    df_temp = df_temp.drop(columns=['reconstruction_history'], errors='ignore')
                    df_temp.to_csv(output_dir / 'intermediate_results.csv', index=False)
                    
            except Exception as e:
                logger.error(f"Failed experiment {config} with seed {seed}: {str(e)}")
                continue
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save complete results (excluding reconstruction history to save space)
    df_save = df_results.drop(columns=['reconstruction_history'], errors='ignore')
    df_save.to_csv(output_dir / 'complete_results.csv', index=False)
    
    # Save summary statistics
    summary_stats = df_results.groupby(['model_name', 'decoder']).agg({
        'psnr': ['mean', 'std', 'min', 'max'],
        'param_count': ['mean'],
        'param_efficiency': ['mean', 'std'],
        'training_time': ['mean', 'std']
    }).round(4)
    
    summary_stats.to_csv(output_dir / 'summary_statistics.csv')
    
    logger.info(f"Completed {len(results)} experiments")
    return df_results

def statistical_analysis(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis of experimental results.
    
    Args:
        df: DataFrame with experimental results
        output_dir: Directory to save analysis outputs
    
    Returns:
        Dictionary with statistical test results
    """
    logger.info("Performing statistical analysis...")
    
    stats_results = {}
    
    # 1. Descriptive Statistics
    desc_stats = df.groupby(['model_name', 'decoder']).agg({
        'psnr': ['count', 'mean', 'std', 'min', 'max'],
        'param_efficiency': ['mean', 'std']
    }).round(4)
    
    desc_stats.to_csv(output_dir / 'descriptive_statistics.csv')
    stats_results['descriptive'] = desc_stats.to_dict()
    
    # 2. Primary Hypothesis Testing: K-Planes vs NeRF
    kplanes_data = df[df['architecture'] == 'kplanes']['psnr']
    nerf_data = df[df['architecture'] == 'nerf']['psnr']
    
    if len(kplanes_data) > 0 and len(nerf_data) > 0:
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(kplanes_data, nerf_data)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(kplanes_data) - 1) * kplanes_data.var() + 
                             (len(nerf_data) - 1) * nerf_data.var()) / 
                            (len(kplanes_data) + len(nerf_data) - 2))
        cohens_d = (kplanes_data.mean() - nerf_data.mean()) / pooled_std
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(kplanes_data, nerf_data, alternative='greater')
        
        primary_results = {
            'kplanes_mean_psnr': float(kplanes_data.mean()),
            'nerf_mean_psnr': float(nerf_data.mean()),
            'psnr_difference': float(kplanes_data.mean() - nerf_data.mean()),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'u_statistic': float(u_stat),
            'u_p_value': float(u_p_value),
            'significant': p_value < 0.05
        }
        
        stats_results['primary_hypothesis'] = primary_results
        
        # 3. ANOVA for multiple group comparison
        if len(df['decoder'].unique()) > 2:
            decoder_groups = [df[df['decoder'] == decoder]['psnr'].values 
                             for decoder in df['decoder'].unique()]
            f_stat, anova_p = stats.f_oneway(*decoder_groups)
            
            stats_results['anova_decoders'] = {
                'f_statistic': float(f_stat),
                'p_value': float(anova_p),
                'significant': anova_p < 0.05
            }
        
        # 4. Parameter Efficiency Analysis
        efficiency_corr, eff_p = stats.pearsonr(df['param_count'], df['psnr'])
        stats_results['parameter_efficiency'] = {
            'correlation_params_psnr': float(efficiency_corr),
            'correlation_p_value': float(eff_p)
        }
    
    # Save statistical results
    with open(output_dir / 'statistical_analysis.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    
    logger.info("Statistical analysis completed")
    return stats_results

def generate_visualizations(df: pd.DataFrame, stats_results: Dict[str, Any], output_dir: Path):
    """Generate publication-quality visualizations."""
    logger.info("Generating visualizations...")
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Architecture Comparison Box Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PSNR comparison
    sns.boxplot(data=df, x='model_name', y='psnr', hue='decoder', ax=ax1)
    ax1.set_title('PSNR by Model Type and Decoder', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_xlabel('Model Type', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Parameter efficiency
    sns.scatterplot(data=df, x='param_count', y='psnr', hue='architecture', 
                   style='decoder', s=100, ax=ax2)
    ax2.set_title('Parameter Efficiency Analysis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Parameter Count', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Statistical Significance Visualization
    if 'primary_hypothesis' in stats_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Mean PSNR with error bars
        grouped_stats = df.groupby('architecture')['psnr'].agg(['mean', 'std', 'count'])
        architectures = grouped_stats.index
        means = grouped_stats['mean']
        stds = grouped_stats['std']
        n_samples = grouped_stats['count']
        
        # Calculate 95% confidence intervals
        ci_95 = 1.96 * stds / np.sqrt(n_samples)
        
        bars = ax.bar(architectures, means, yerr=ci_95, capsize=10, alpha=0.8)
        ax.set_title('Mean PSNR by Architecture (95% CI)', fontsize=14, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_xlabel('Architecture', fontsize=12)
        
        # Add significance annotation
        primary = stats_results['primary_hypothesis']
        if primary['significant']:
            ax.text(0.5, max(means) + max(ci_95) * 0.1, 
                   f'p = {primary["p_value"]:.4f} (significant)', 
                   ha='center', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Detailed Architecture Performance Heatmap
    if len(df) > 10:  # Only if sufficient data
        pivot_data = df.groupby(['model_name', 'decoder'])['psnr'].mean().unstack(fill_value=0)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis', 
                   ax=ax, cbar_kws={'label': 'Mean PSNR (dB)'})
        ax.set_title('Architecture Performance Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("Visualizations generated successfully")

def generate_naming_convention_documentation(output_dir: Path):
    """Generate documentation explaining the model naming conventions."""
    doc_content = """# Model Naming Conventions

## K-Planes (Line Features Only)
- **K-planes(multiply)**: Uses decoder(line_feat_x * line_feat_y)
- **K-planes(add)**: Uses decoder(line_feat_x + line_feat_y)
- These models use ONLY line features without any plane features
- The operation (multiply/add) determines how line features are combined

## GA-Planes (Line + Plane Features)
- **GA-Planes(multiply+plane)**: Uses decoder(line_feat_x * line_feat_y + plane_feat)
- **GA-Planes(add+plane)**: Uses decoder(line_feat_x + line_feat_y + plane_feat)
- These models include both line features AND low-resolution plane features
- The plane features add a global context to the line-based factorization

## NeRF (Coordinate-based)
- **NeRF(nonconvex)**: Standard ReLU-based MLP with deeper networks
- **NeRF(siren)**: SIREN-based continuous representation with sine activations
- These models encode coordinates directly through MLPs without explicit factorization

## Key Differences
1. **K-planes**: Pure factorized representation using only 1D line features
2. **GA-Planes**: Hybrid approach combining line features with 2D plane features
3. **NeRF**: Implicit representation learning features from coordinates

This naming convention accurately reflects the architectural differences and helps
distinguish between models that use only line-based factorization (K-planes) versus
those that incorporate additional planar features (GA-Planes).
"""
    
    with open(output_dir / 'model_naming_conventions.md', 'w') as f:
        f.write(doc_content)
    
    logger.info("Generated model naming convention documentation")

def main():
    """Main experimental execution function."""
    parser = argparse.ArgumentParser(description='INR Architecture Comparison Experiment')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Directory to save experimental results')
    parser.add_argument('--num_seeds', type=int, default=3,
                       help='Number of random seeds per configuration')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with minimal configurations')
    parser.add_argument('--test_naming', action='store_true',
                       help='Generate naming convention documentation only')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configure logging to file
    file_handler = logging.FileHandler(output_dir / 'experiment.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("INR ARCHITECTURE COMPARISON EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Device: {device}")
    logger.info(f"Number of seeds: {args.num_seeds}")
    
    # Generate naming convention documentation
    generate_naming_convention_documentation(output_dir)
    
    # If only testing naming, exit early
    if args.test_naming:
        logger.info("Generated naming convention documentation. Exiting.")
        return
    
    # Quick test mode for debugging
    if args.quick_test:
        logger.info("Running in quick test mode")
        args.num_seeds = 2
    
    try:
        # Run experimental suite
        df_results = run_experiment_suite(output_dir, num_seeds=args.num_seeds)
        
        if len(df_results) == 0:
            logger.error("No experimental results obtained")
            return
        
        # Perform statistical analysis
        stats_results = statistical_analysis(df_results, output_dir)
        
        # Generate visualizations
        generate_visualizations(df_results, stats_results, output_dir)
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 50)
        
        if 'primary_hypothesis' in stats_results:
            primary = stats_results['primary_hypothesis']
            logger.info(f"K-Planes mean PSNR: {primary['kplanes_mean_psnr']:.2f} dB")
            logger.info(f"NeRF mean PSNR: {primary['nerf_mean_psnr']:.2f} dB")
            logger.info(f"Difference: {primary['psnr_difference']:.2f} dB")
            logger.info(f"Statistical significance: {'YES' if primary['significant'] else 'NO'} (p={primary['p_value']:.4f})")
            logger.info(f"Effect size (Cohen's d): {primary['cohens_d']:.3f}")
            
            if primary['significant'] and primary['psnr_difference'] > 5.0:
                logger.info("🎉 PRIMARY HYPOTHESIS CONFIRMED: K-Planes > 5dB improvement over NeRF!")
            elif primary['significant']:
                logger.info("✅ STATISTICAL SIGNIFICANCE ACHIEVED")
            else:
                logger.info("❌ No significant difference found")
        
        logger.info(f"\nResults saved to: {output_dir.absolute()}")
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
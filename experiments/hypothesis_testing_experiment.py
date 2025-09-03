#!/usr/bin/env python3
"""
Scientific Hypothesis Testing for INR Architecture Comparison
Implements rigorous statistical testing for matrix reconstruction hypotheses
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.stats as stats
import pingouin as pg
import skimage
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    architecture: str  # 'kplanes', 'nerf', 'linear'
    decoder_type: str  # 'linear', 'nonconvex', 'convex'
    operation: str     # 'multiply', 'add'
    interpolation: str # 'bilinear', 'nearest'
    line_resolution: int
    plane_resolution: int
    feature_dim: int
    hidden_dim: int
    learning_rate: float
    num_epochs: int
    seed: int

@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    config: ExperimentConfig
    psnr: float
    parameter_count: int
    training_time: float
    convergence_epoch: int
    final_loss: float

class INRArchitecture(nn.Module):
    """Base class for different INR architectures"""
    
    def __init__(self, config: ExperimentConfig, img_shape: Tuple[int, int]):
        super().__init__()
        self.config = config
        self.img_shape = img_shape
        self.resx, self.resy = img_shape
        
        torch.manual_seed(config.seed)
        self._build_architecture()
    
    def _build_architecture(self):
        """Build the specific architecture based on config"""
        if self.config.architecture == 'kplanes':
            self._build_kplanes()
        elif self.config.architecture == 'nerf':
            self._build_nerf()
        elif self.config.architecture == 'linear':
            self._build_linear()
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")
    
    def _build_kplanes(self):
        """Build K-Planes architecture with planar factorization"""
        # Line features for X and Y dimensions
        if self.config.operation == 'multiply':
            init_scale = 0.15
            init_bias = 0.1
        else:
            init_scale = 0.03  
            init_bias = 0.005
            
        self.line_feature_x = nn.Parameter(
            torch.rand(self.config.feature_dim, self.config.line_resolution) * init_scale + init_bias
        )
        self.line_feature_y = nn.Parameter(
            torch.rand(self.config.feature_dim, self.config.line_resolution) * init_scale + init_bias
        )
        
        # Plane feature for 2D interactions
        self.plane_feature = nn.Parameter(
            torch.randn(self.config.feature_dim, self.config.plane_resolution, self.config.plane_resolution) * 0.01
        )
        
        self._build_decoder()
    
    def _build_nerf(self):
        """Build NeRF-style MLP architecture with positional encoding"""
        # Positional encoding frequencies
        self.num_freqs = 10
        self.freq_bands = 2. ** torch.linspace(0., self.num_freqs - 1, self.num_freqs)
        
        # MLP layers
        input_dim = 2 * self.num_freqs * 2 + 2  # (sin, cos) * num_freqs * 2_coords + raw_coords
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, self.config.hidden_dim, bias=True))
        layers.append(nn.ReLU())
        
        # Hidden layers (following NeRF architecture)
        for i in range(4):  # 5 layers total like NeRF
            if i == 2:  # Skip connection like NeRF
                layers.append(nn.Linear(self.config.hidden_dim + input_dim, self.config.hidden_dim, bias=True))
            else:
                layers.append(nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=True))
            layers.append(nn.ReLU())
        
        # Final layer
        layers.append(nn.Linear(self.config.hidden_dim, 1, bias=True))
        
        self.nerf_mlp = nn.Sequential(*layers)
        self.skip_layer_idx = 6  # After 3rd hidden layer
    
    def _build_linear(self):
        """Build simple linear baseline"""
        # Direct coordinate to value mapping
        self.linear_layer = nn.Linear(2, 1, bias=True)
    
    def _build_decoder(self):
        """Build decoder for K-Planes architecture"""
        if self.config.decoder_type == 'linear':
            self.decoder = nn.Linear(self.config.feature_dim, 1, bias=True)
        elif self.config.decoder_type == 'nonconvex':
            self.decoder = nn.Sequential(
                nn.Linear(self.config.feature_dim, self.config.hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, 1, bias=True)
            )
        elif self.config.decoder_type == 'convex':
            self.fc1 = nn.Linear(self.config.feature_dim, self.config.hidden_dim, bias=True)
            self.fc2 = nn.Linear(self.config.feature_dim, self.config.hidden_dim, bias=True)
            self.fc2.weight.requires_grad = False
            self.fc2.bias.requires_grad = False
        else:
            raise ValueError(f"Unknown decoder type: {self.config.decoder_type}")
    
    def _positional_encoding(self, coords):
        """Apply positional encoding to coordinates (for NeRF)"""
        encoded = [coords]
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                encoded.append(func(freq * coords))
        return torch.cat(encoded, dim=-1)
    
    def forward(self, coords):
        """Forward pass through the architecture"""
        if self.config.architecture == 'kplanes':
            return self._forward_kplanes(coords)
        elif self.config.architecture == 'nerf':
            return self._forward_nerf(coords)
        elif self.config.architecture == 'linear':
            return self._forward_linear(coords)
    
    def _forward_kplanes(self, coords):
        """Forward pass for K-Planes architecture"""
        # Prepare coordinates for grid_sample
        x_coords = coords[..., 0].unsqueeze(-1)
        y_coords = coords[..., 1].unsqueeze(-1)
        
        # Scale to [-1, 1] range for grid_sample
        x_coords = (x_coords * 2 / self.resx) - 1
        y_coords = (y_coords * 2 / self.resy) - 1
        
        # Prepare grids for interpolation
        gridx = torch.cat((x_coords, x_coords), dim=-1).unsqueeze(0)
        gridy = torch.cat((y_coords, y_coords), dim=-1).unsqueeze(0)
        plane_grid = torch.cat((x_coords, y_coords), dim=-1).unsqueeze(0)
        
        # Interpolate line features
        line_features_x = self.line_feature_x.unsqueeze(0).unsqueeze(-1)
        line_features_y = self.line_feature_y.unsqueeze(0).unsqueeze(-1)
        
        feature_x = F.grid_sample(line_features_x, gridx, 
                                mode=self.config.interpolation, 
                                padding_mode='border', align_corners=True)
        feature_y = F.grid_sample(line_features_y, gridy, 
                                mode=self.config.interpolation, 
                                padding_mode='border', align_corners=True)
        
        # Interpolate plane features
        plane_features = self.plane_feature.unsqueeze(0)
        sampled_plane_features = F.grid_sample(plane_features, plane_grid, 
                                             mode=self.config.interpolation, 
                                             align_corners=True)
        
        # Combine features
        if self.config.operation == 'add':
            combined_features = feature_x + feature_y + sampled_plane_features
        elif self.config.operation == 'multiply':
            combined_features = feature_x * feature_y + sampled_plane_features
        
        # Reshape for decoder
        combined_features = combined_features.squeeze().permute(1, 2, 0)
        
        # Pass through decoder
        if self.config.decoder_type == 'convex':
            output = self.fc1(combined_features) * (self.fc2(combined_features) > 0)
            output = torch.mean(output, dim=-1)
        else:
            output = self.decoder(combined_features).squeeze()
        
        return output
    
    def _forward_nerf(self, coords):
        """Forward pass for NeRF architecture"""
        # Apply positional encoding
        encoded_coords = self._positional_encoding(coords)
        
        # Flatten for MLP processing
        original_shape = encoded_coords.shape[:-1]
        flat_coords = encoded_coords.view(-1, encoded_coords.shape[-1])
        
        # Forward through MLP with skip connection
        x = flat_coords
        for i, layer in enumerate(self.nerf_mlp):
            if i == self.skip_layer_idx:
                # Skip connection
                x = torch.cat([x, flat_coords], dim=-1)
            x = layer(x)
        
        # Reshape back to image dimensions
        output = x.view(*original_shape)
        return output.squeeze()
    
    def _forward_linear(self, coords):
        """Forward pass for linear baseline"""
        # Normalize coordinates to [-1, 1]
        normalized_coords = coords / torch.tensor([self.resx, self.resy], device=coords.device) * 2 - 1
        output = self.linear_layer(normalized_coords)
        return output.squeeze()

class HypothesisTestingFramework:
    """Framework for conducting rigorous hypothesis testing"""
    
    def __init__(self, data_dir: Path, results_dir: Path):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load hypotheses
        self.hypotheses = self._load_hypotheses()
        self.results = []
    
    def _load_hypotheses(self) -> List[Dict]:
        """Load hypotheses from hypothesis.jsonl"""
        hypotheses = []
        hypothesis_file = Path("hypothesis.jsonl")
        if hypothesis_file.exists():
            with open(hypothesis_file, 'r') as f:
                for line in f:
                    hypotheses.append(json.loads(line.strip()))
        return hypotheses
    
    def _load_test_images(self) -> List[np.ndarray]:
        """Load test images from the dataset"""
        images = []
        
        # Load synthetic patterns for controlled testing
        synthetic_dir = self.data_dir / "processed" / "synthetic"
        if synthetic_dir.exists():
            for pattern_file in ["checkerboards.npy", "gaussian_blobs.npy", "sinusoidal_patterns.npy"]:
                pattern_path = synthetic_dir / pattern_file
                if pattern_path.exists():
                    patterns = np.load(pattern_path)
                    # Take first few patterns
                    for i in range(min(3, len(patterns))):
                        img = patterns[i]
                        if len(img.shape) > 2:
                            img = np.mean(img, axis=-1)
                        # Resize to manageable size for experiments
                        from skimage.transform import resize
                        img = resize(img, (64, 64), anti_aliasing=True)
                        images.append(img.astype(np.float32))
        
        # Add the astronaut image as a natural test case
        img = skimage.data.astronaut() / 255.0
        if len(img.shape) > 2:
            img = np.mean(img, axis=-1)
        from skimage.transform import resize
        img = resize(img, (64, 64), anti_aliasing=True)
        images.append(img.astype(np.float32))
        
        return images[:5]  # Limit to 5 test images for computational efficiency
    
    def _generate_experiment_configs(self) -> List[ExperimentConfig]:
        """Generate experiment configurations for hypothesis testing"""
        # Base parameter grid - focused on key architectural differences
        param_grid = {
            'architecture': ['kplanes', 'nerf', 'linear'],
            'decoder_type': ['linear', 'nonconvex'],
            'operation': ['multiply', 'add'],
            'interpolation': ['bilinear', 'nearest'],
            'line_resolution': [32, 64],
            'plane_resolution': [16, 32],
            'feature_dim': [32, 64],
            'hidden_dim': [64, 128],
            'learning_rate': [0.01, 0.05],
            'num_epochs': [500],  # Reduced for efficiency
            'seed': [42, 43, 44]  # Multiple seeds for statistical validity
        }
        
        # Filter combinations to focus on key hypotheses
        configs = []
        for params in ParameterGrid(param_grid):
            config = ExperimentConfig(**params)
            
            # Filter out incompatible combinations
            if config.architecture == 'nerf' and config.decoder_type != 'linear':
                continue  # NeRF handles its own decoding
            if config.architecture == 'linear' and config.decoder_type != 'linear':
                continue  # Linear baseline is always linear
                
            configs.append(config)
        
        # Limit total experiments for computational feasibility
        return configs[:100]  # Focus on most important comparisons
    
    def run_single_experiment(self, config: ExperimentConfig, target_image: np.ndarray) -> ExperimentResult:
        """Run a single experiment configuration"""
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Prepare data
        img_shape = target_image.shape
        coords = self._generate_coordinates(img_shape)
        coords_tensor = torch.from_numpy(coords).float().to(device)
        target_tensor = torch.from_numpy(target_image).float().to(device)
        
        # Create model
        model = INRArchitecture(config, img_shape).to(device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Training loop
        start_time = time.time()
        best_loss = float('inf')
        convergence_epoch = config.num_epochs
        
        model.train()
        for epoch in range(config.num_epochs):
            optimizer.zero_grad()
            
            outputs = model(coords_tensor)
            loss = criterion(outputs, target_tensor)
            
            loss.backward()
            optimizer.step()
            
            # Track convergence
            current_loss = loss.item()
            if current_loss < best_loss * 0.999:  # 0.1% improvement threshold
                best_loss = current_loss
                convergence_epoch = epoch
        
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_outputs = model(coords_tensor)
            final_loss = criterion(final_outputs, target_tensor)
            psnr = -10 * np.log10(final_loss.item())
        
        # Count parameters
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return ExperimentResult(
            config=config,
            psnr=psnr,
            parameter_count=parameter_count,
            training_time=training_time,
            convergence_epoch=convergence_epoch,
            final_loss=final_loss.item()
        )
    
    def _generate_coordinates(self, img_shape: Tuple[int, int]) -> np.ndarray:
        """Generate pixel coordinates for the image"""
        h, w = img_shape
        y_indices, x_indices = np.indices((h, w))
        coords = np.stack((x_indices, y_indices), axis=-1)
        return coords
    
    def run_hypothesis_tests(self) -> Dict:
        """Run comprehensive hypothesis testing"""
        logger.info("Starting comprehensive hypothesis testing...")
        
        # Load test images
        test_images = self._load_test_images()
        logger.info(f"Loaded {len(test_images)} test images")
        
        # Generate experiment configurations
        configs = self._generate_experiment_configs()
        logger.info(f"Generated {len(configs)} experiment configurations")
        
        # Run all experiments
        results = []
        total_experiments = len(configs) * len(test_images)
        
        with tqdm(total=total_experiments, desc="Running experiments") as pbar:
            for img_idx, test_image in enumerate(test_images):
                for config_idx, config in enumerate(configs):
                    try:
                        result = self.run_single_experiment(config, test_image)
                        result.image_id = img_idx
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Experiment failed: {config.architecture}-{config.decoder_type}: {e}")
                    pbar.update(1)
        
        self.results = results
        logger.info(f"Completed {len(results)} successful experiments")
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Save results
        self._save_results(analysis)
        
        return analysis
    
    def _analyze_results(self) -> Dict:
        """Analyze experimental results with statistical rigor"""
        logger.info("Analyzing experimental results...")
        
        # Convert results to DataFrame
        df_data = []
        for result in self.results:
            row = {
                'architecture': result.config.architecture,
                'decoder_type': result.config.decoder_type,
                'operation': result.config.operation,
                'interpolation': result.config.interpolation,
                'line_resolution': result.config.line_resolution,
                'plane_resolution': result.config.plane_resolution,
                'feature_dim': result.config.feature_dim,
                'hidden_dim': result.config.hidden_dim,
                'learning_rate': result.config.learning_rate,
                'seed': result.config.seed,
                'image_id': getattr(result, 'image_id', 0),
                'psnr': result.psnr,
                'parameter_count': result.parameter_count,
                'training_time': result.training_time,
                'parameter_efficiency': result.psnr / (result.parameter_count / 1000),  # PSNR per 1K params
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        analysis = {}
        
        # Hypothesis 1: K-Planes vs NeRF performance
        analysis['hypothesis_1'] = self._test_architecture_comparison(df)
        
        # Hypothesis 2: Linear vs Nonlinear decoders
        analysis['hypothesis_2'] = self._test_decoder_comparison(df)
        
        # Hypothesis 3: Interpolation method comparison
        analysis['hypothesis_3'] = self._test_interpolation_comparison(df)
        
        # Hypothesis 4: Parameter efficiency analysis
        analysis['hypothesis_4'] = self._test_parameter_efficiency(df)
        
        # Overall statistics
        analysis['summary_stats'] = self._generate_summary_stats(df)
        
        return analysis
    
    def _test_architecture_comparison(self, df: pd.DataFrame) -> Dict:
        """Test Hypothesis 1: K-Planes vs NeRF performance"""
        logger.info("Testing Hypothesis 1: Architecture comparison")
        
        # Filter data for comparison
        kplanes_data = df[df['architecture'] == 'kplanes']['psnr']
        nerf_data = df[df['architecture'] == 'nerf']['psnr']
        linear_data = df[df['architecture'] == 'linear']['psnr']
        
        results = {
            'hypothesis': "K-Planes will demonstrate superior reconstruction quality compared to NeRF",
            'sample_sizes': {
                'kplanes': len(kplanes_data),
                'nerf': len(nerf_data), 
                'linear': len(linear_data)
            },
            'descriptive_stats': {
                'kplanes': {'mean': float(kplanes_data.mean()), 'std': float(kplanes_data.std())},
                'nerf': {'mean': float(nerf_data.mean()), 'std': float(nerf_data.std())},
                'linear': {'mean': float(linear_data.mean()), 'std': float(linear_data.std())}
            }
        }
        
        # Statistical tests
        if len(kplanes_data) > 0 and len(nerf_data) > 0:
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(kplanes_data, nerf_data)
            effect_size = (kplanes_data.mean() - nerf_data.mean()) / np.sqrt(
                (kplanes_data.std()**2 + nerf_data.std()**2) / 2
            )
            
            results['statistical_test'] = {
                'test': 'Two-sample t-test',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'effect_size_cohens_d': float(effect_size),
                'significant': p_value < 0.05,
                'conclusion': 'SUPPORTED' if (t_stat > 0 and p_value < 0.05) else 'NOT_SUPPORTED'
            }
        
        return results
    
    def _test_decoder_comparison(self, df: pd.DataFrame) -> Dict:
        """Test Hypothesis 2: Linear vs Nonlinear decoder performance"""
        logger.info("Testing Hypothesis 2: Decoder comparison")
        
        # Filter K-Planes data only for decoder comparison
        kplanes_df = df[df['architecture'] == 'kplanes']
        linear_dec = kplanes_df[kplanes_df['decoder_type'] == 'linear']['psnr']
        nonconvex_dec = kplanes_df[kplanes_df['decoder_type'] == 'nonconvex']['psnr']
        
        results = {
            'hypothesis': "Linear decoders can achieve comparable performance to nonlinear MLP decoders",
            'sample_sizes': {
                'linear_decoder': len(linear_dec),
                'nonconvex_decoder': len(nonconvex_dec)
            },
            'descriptive_stats': {
                'linear_decoder': {'mean': float(linear_dec.mean()), 'std': float(linear_dec.std())},
                'nonconvex_decoder': {'mean': float(nonconvex_dec.mean()), 'std': float(nonconvex_dec.std())}
            }
        }
        
        if len(linear_dec) > 0 and len(nonconvex_dec) > 0:
            # Equivalence test (TOST - Two One-Sided Tests)
            equivalence_margin = 2.0  # PSNR difference considered practically equivalent
            
            # Standard two-sample t-test for difference
            t_stat, p_value = stats.ttest_ind(linear_dec, nonconvex_dec)
            
            results['statistical_test'] = {
                'test': 'Two-sample t-test for equivalence',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'equivalence_margin': equivalence_margin,
                'mean_difference': float(linear_dec.mean() - nonconvex_dec.mean()),
                'conclusion': 'SUPPORTED' if abs(linear_dec.mean() - nonconvex_dec.mean()) < equivalence_margin else 'NOT_SUPPORTED'
            }
        
        return results
    
    def _test_interpolation_comparison(self, df: pd.DataFrame) -> Dict:
        """Test Hypothesis 3: Interpolation method comparison"""
        logger.info("Testing Hypothesis 3: Interpolation method comparison")
        
        bilinear_data = df[df['interpolation'] == 'bilinear']['psnr']
        nearest_data = df[df['interpolation'] == 'nearest']['psnr']
        
        results = {
            'hypothesis': "Bilinear vs nearest interpolation performance in 2D reconstruction",
            'sample_sizes': {
                'bilinear': len(bilinear_data),
                'nearest': len(nearest_data)
            },
            'descriptive_stats': {
                'bilinear': {'mean': float(bilinear_data.mean()), 'std': float(bilinear_data.std())},
                'nearest': {'mean': float(nearest_data.mean()), 'std': float(nearest_data.std())}
            }
        }
        
        if len(bilinear_data) > 0 and len(nearest_data) > 0:
            t_stat, p_value = stats.ttest_ind(bilinear_data, nearest_data)
            
            results['statistical_test'] = {
                'test': 'Two-sample t-test',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'conclusion': 'BILINEAR_SUPERIOR' if (t_stat > 0 and p_value < 0.05) 
                             else 'NEAREST_SUPERIOR' if (t_stat < 0 and p_value < 0.05)
                             else 'NO_SIGNIFICANT_DIFFERENCE'
            }
        
        return results
    
    def _test_parameter_efficiency(self, df: pd.DataFrame) -> Dict:
        """Test Hypothesis 4: Parameter efficiency analysis"""
        logger.info("Testing Hypothesis 4: Parameter efficiency analysis")
        
        # Check if target PSNR >35dB is achieved with parameter efficiency >2x
        high_psnr_threshold = 35.0
        baseline_efficiency = df[df['architecture'] == 'linear']['parameter_efficiency'].mean()
        
        results = {
            'hypothesis': "Parameter efficiency improvements >2x while maintaining PSNR >35dB",
            'baseline_efficiency': float(baseline_efficiency),
            'high_psnr_threshold': high_psnr_threshold,
            'target_efficiency_improvement': 2.0
        }
        
        # Find architectures achieving both criteria
        high_psnr_df = df[df['psnr'] >= high_psnr_threshold]
        efficient_architectures = high_psnr_df[
            high_psnr_df['parameter_efficiency'] >= baseline_efficiency * 2.0
        ]
        
        results['analysis'] = {
            'total_high_psnr_configs': len(high_psnr_df),
            'efficient_and_high_psnr_configs': len(efficient_architectures),
            'success_rate': len(efficient_architectures) / max(len(df), 1) * 100,
            'best_performers': efficient_architectures.nlargest(5, 'parameter_efficiency')[
                ['architecture', 'decoder_type', 'psnr', 'parameter_efficiency']
            ].to_dict('records') if len(efficient_architectures) > 0 else [],
            'conclusion': 'SUPPORTED' if len(efficient_architectures) > 0 else 'NOT_SUPPORTED'
        }
        
        return results
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate overall summary statistics"""
        return {
            'total_experiments': len(df),
            'architectures_tested': df['architecture'].unique().tolist(),
            'mean_psnr_by_architecture': df.groupby('architecture')['psnr'].agg(['mean', 'std']).to_dict('index'),
            'parameter_count_ranges': {
                'min': int(df['parameter_count'].min()),
                'max': int(df['parameter_count'].max()),
                'mean': float(df['parameter_count'].mean())
            },
            'best_overall_configuration': df.loc[df['psnr'].idxmax()].to_dict()
        }
    
    def _save_results(self, analysis: Dict):
        """Save experimental results and analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed analysis
        results_file = self.results_dir / f"hypothesis_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save results summary for easy reading
        summary_file = self.results_dir / f"experiment_summary_{timestamp}.md"
        self._write_summary_markdown(analysis, summary_file)
        
        logger.info(f"Results saved to {results_file} and {summary_file}")
    
    def _write_summary_markdown(self, analysis: Dict, filepath: Path):
        """Write a markdown summary of results"""
        with open(filepath, 'w') as f:
            f.write("# INR Architecture Hypothesis Testing Results\n\n")
            f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Hypothesis 1
            h1 = analysis['hypothesis_1']
            f.write("## Hypothesis 1: Architecture Performance Comparison\n")
            f.write(f"**Hypothesis**: {h1['hypothesis']}\n\n")
            f.write("**Results**:\n")
            for arch, stats in h1['descriptive_stats'].items():
                f.write(f"- {arch.title()}: Mean PSNR = {stats['mean']:.2f}±{stats['std']:.2f}\n")
            if 'statistical_test' in h1:
                test = h1['statistical_test']
                f.write(f"\n**Statistical Test**: {test['test']}\n")
                f.write(f"- p-value: {test['p_value']:.4f}\n")
                f.write(f"- Effect size (Cohen's d): {test['effect_size_cohens_d']:.3f}\n")
                f.write(f"- **Conclusion**: {test['conclusion']}\n\n")
            
            # Hypothesis 2  
            h2 = analysis['hypothesis_2']
            f.write("## Hypothesis 2: Decoder Type Comparison\n")
            f.write(f"**Hypothesis**: {h2['hypothesis']}\n\n")
            f.write("**Results**:\n")
            for dec, stats in h2['descriptive_stats'].items():
                f.write(f"- {dec.replace('_', ' ').title()}: Mean PSNR = {stats['mean']:.2f}±{stats['std']:.2f}\n")
            if 'statistical_test' in h2:
                test = h2['statistical_test']
                f.write(f"\n**Statistical Test**: {test['test']}\n")
                f.write(f"- Mean difference: {test['mean_difference']:.2f} PSNR\n")
                f.write(f"- **Conclusion**: {test['conclusion']}\n\n")
            
            # Summary statistics
            summary = analysis['summary_stats']
            f.write("## Summary Statistics\n")
            f.write(f"- Total experiments: {summary['total_experiments']}\n")
            f.write(f"- Architectures tested: {', '.join(summary['architectures_tested'])}\n")
            f.write("\n**Best Overall Configuration**:\n")
            best = summary['best_overall_configuration']
            f.write(f"- Architecture: {best['architecture']}\n")
            f.write(f"- Decoder: {best['decoder_type']}\n")
            f.write(f"- PSNR: {best['psnr']:.2f}\n")
            f.write(f"- Parameters: {best['parameter_count']}\n")

def main():
    """Main function to run hypothesis testing experiments"""
    # Setup directories
    data_dir = Path("../data")
    results_dir = Path("../results")
    
    # Initialize framework
    framework = HypothesisTestingFramework(data_dir, results_dir)
    
    # Run comprehensive hypothesis testing
    analysis = framework.run_hypothesis_tests()
    
    print("\n" + "="*50)
    print("HYPOTHESIS TESTING COMPLETE")
    print("="*50)
    
    # Print key findings
    for i in range(1, 5):
        hyp_key = f'hypothesis_{i}'
        if hyp_key in analysis:
            hyp = analysis[hyp_key]
            if 'statistical_test' in hyp and 'conclusion' in hyp['statistical_test']:
                print(f"Hypothesis {i}: {hyp['statistical_test']['conclusion']}")
    
    print(f"\nDetailed results saved to: {results_dir}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick Scientific Hypothesis Testing for INR Architecture Comparison
Focused on key architectural differences with efficient implementation
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
import skimage
from skimage.transform import resize
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
    architecture: str
    decoder_type: str
    operation: str
    seed: int

@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    config: ExperimentConfig
    psnr: float
    parameter_count: int
    training_time: float

class KPlanesModel(nn.Module):
    """K-Planes architecture implementation"""
    
    def __init__(self, config: ExperimentConfig, img_shape: Tuple[int, int]):
        super().__init__()
        self.config = config
        self.img_shape = img_shape
        self.resx, self.resy = img_shape
        
        # Fixed hyperparameters for efficiency
        self.feature_dim = 32
        self.line_res = 32
        self.plane_res = 16
        self.hidden_dim = 64
        
        torch.manual_seed(config.seed)
        
        # Line features for X and Y dimensions
        if config.operation == 'multiply':
            init_scale, init_bias = 0.15, 0.1
        else:
            init_scale, init_bias = 0.03, 0.005
            
        self.line_feature_x = nn.Parameter(
            torch.rand(self.feature_dim, self.line_res) * init_scale + init_bias
        )
        self.line_feature_y = nn.Parameter(
            torch.rand(self.feature_dim, self.line_res) * init_scale + init_bias
        )
        
        # Plane feature for 2D interactions
        self.plane_feature = nn.Parameter(
            torch.randn(self.feature_dim, self.plane_res, self.plane_res) * 0.01
        )
        
        # Decoder
        if config.decoder_type == 'linear':
            self.decoder = nn.Linear(self.feature_dim, 1, bias=True)
        else:  # nonconvex
            self.decoder = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1, bias=True)
            )
    
    def forward(self, coords):
        # Prepare coordinates for grid_sample
        x_coords = coords[..., 0].unsqueeze(-1)
        y_coords = coords[..., 1].unsqueeze(-1)
        
        # Scale to [-1, 1] range
        x_coords = (x_coords * 2 / self.resx) - 1
        y_coords = (y_coords * 2 / self.resy) - 1
        
        # Prepare grids
        gridx = torch.cat((x_coords, x_coords), dim=-1).unsqueeze(0)
        gridy = torch.cat((y_coords, y_coords), dim=-1).unsqueeze(0)
        plane_grid = torch.cat((x_coords, y_coords), dim=-1).unsqueeze(0)
        
        # Interpolate features
        line_features_x = self.line_feature_x.unsqueeze(0).unsqueeze(-1)
        line_features_y = self.line_feature_y.unsqueeze(0).unsqueeze(-1)
        
        feature_x = F.grid_sample(line_features_x, gridx, mode='bilinear', 
                                padding_mode='border', align_corners=True)
        feature_y = F.grid_sample(line_features_y, gridy, mode='bilinear',
                                padding_mode='border', align_corners=True)
        
        plane_features = self.plane_feature.unsqueeze(0)
        sampled_plane_features = F.grid_sample(plane_features, plane_grid, 
                                             mode='bilinear', align_corners=True)
        
        # Combine features
        if self.config.operation == 'add':
            combined_features = feature_x + feature_y + sampled_plane_features
        else:  # multiply
            combined_features = feature_x * feature_y + sampled_plane_features
        
        # Reshape and decode
        combined_features = combined_features.squeeze().permute(1, 2, 0)
        output = self.decoder(combined_features).squeeze()
        
        return output

class NeRFModel(nn.Module):
    """Simplified NeRF-style MLP model"""
    
    def __init__(self, config: ExperimentConfig, img_shape: Tuple[int, int]):
        super().__init__()
        self.config = config
        self.img_shape = img_shape
        self.resx, self.resy = img_shape
        
        torch.manual_seed(config.seed)
        
        # Positional encoding
        self.num_freqs = 6  # Reduced for efficiency
        self.freq_bands = 2. ** torch.linspace(0., self.num_freqs - 1, self.num_freqs)
        
        # MLP
        input_dim = 2 * self.num_freqs * 2 + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def _positional_encoding(self, coords):
        encoded = [coords]
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                encoded.append(func(freq * coords))
        return torch.cat(encoded, dim=-1)
    
    def forward(self, coords):
        # Normalize coordinates
        normalized_coords = coords / torch.tensor([self.resx, self.resy], device=coords.device) * 2 - 1
        
        # Apply positional encoding
        encoded_coords = self._positional_encoding(normalized_coords)
        
        # Flatten and process
        original_shape = encoded_coords.shape[:-1]
        flat_coords = encoded_coords.view(-1, encoded_coords.shape[-1])
        
        output = self.mlp(flat_coords)
        return output.view(*original_shape).squeeze()

class LinearModel(nn.Module):
    """Simple linear baseline"""
    
    def __init__(self, config: ExperimentConfig, img_shape: Tuple[int, int]):
        super().__init__()
        self.config = config
        self.img_shape = img_shape
        self.resx, self.resy = img_shape
        
        torch.manual_seed(config.seed)
        self.linear = nn.Linear(2, 1, bias=True)
    
    def forward(self, coords):
        normalized_coords = coords / torch.tensor([self.resx, self.resy], device=coords.device) * 2 - 1
        return self.linear(normalized_coords).squeeze()

class QuickHypothesisTester:
    """Quick and focused hypothesis testing framework"""
    
    def __init__(self):
        self.results = []
    
    def run_experiment(self, config: ExperimentConfig, target_image: np.ndarray) -> ExperimentResult:
        """Run a single experiment"""
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Prepare data
        img_shape = target_image.shape
        coords = self._generate_coordinates(img_shape)
        coords_tensor = torch.from_numpy(coords).float().to(device)
        target_tensor = torch.from_numpy(target_image).float().to(device)
        
        # Create model
        if config.architecture == 'kplanes':
            model = KPlanesModel(config, img_shape)
        elif config.architecture == 'nerf':
            model = NeRFModel(config, img_shape)
        else:  # linear
            model = LinearModel(config, img_shape)
        
        model = model.to(device)
        
        # Training
        criterion = nn.MSELoss()
        lr = 0.05 if config.operation == 'multiply' else 0.01
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        start_time = time.time()
        model.train()
        
        # Reduced epochs for speed
        num_epochs = 200
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(coords_tensor)
            loss = criterion(outputs, target_tensor)
            loss.backward()
            optimizer.step()
        
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_outputs = model(coords_tensor)
            final_loss = criterion(final_outputs, target_tensor)
            psnr = -10 * np.log10(final_loss.item())
        
        # Parameter count
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return ExperimentResult(
            config=config,
            psnr=psnr,
            parameter_count=parameter_count,
            training_time=training_time
        )
    
    def _generate_coordinates(self, img_shape: Tuple[int, int]) -> np.ndarray:
        """Generate pixel coordinates"""
        h, w = img_shape
        y_indices, x_indices = np.indices((h, w))
        coords = np.stack((x_indices, y_indices), axis=-1)
        return coords
    
    def run_all_experiments(self) -> Dict:
        """Run comprehensive experiments"""
        logger.info("Starting quick hypothesis testing...")
        
        # Load test image (simplified)
        test_image = skimage.data.astronaut() / 255.0
        if len(test_image.shape) > 2:
            test_image = np.mean(test_image, axis=-1)
        test_image = resize(test_image, (32, 32), anti_aliasing=True).astype(np.float32)
        
        # Define experiment configurations
        configs = []
        
        # Key architectural comparisons
        for architecture in ['kplanes', 'nerf', 'linear']:
            for decoder_type in ['linear', 'nonconvex']:
                for operation in ['multiply', 'add']:
                    for seed in [42, 43, 44]:  # Multiple seeds
                        # Skip incompatible combinations
                        if architecture == 'nerf' and decoder_type != 'linear':
                            continue
                        if architecture == 'linear' and decoder_type != 'linear':
                            continue
                        if architecture == 'linear' and operation != 'add':
                            continue
                            
                        configs.append(ExperimentConfig(
                            architecture=architecture,
                            decoder_type=decoder_type,
                            operation=operation,
                            seed=seed
                        ))
        
        logger.info(f"Running {len(configs)} experiments...")
        
        # Run experiments
        results = []
        for config in tqdm(configs, desc="Running experiments"):
            try:
                result = self.run_experiment(config, test_image)
                results.append(result)
            except Exception as e:
                logger.warning(f"Experiment failed: {config.architecture}-{config.decoder_type}: {e}")
        
        self.results = results
        logger.info(f"Completed {len(results)} successful experiments")
        
        # Analyze results
        return self._analyze_results()
    
    def _analyze_results(self) -> Dict:
        """Analyze experimental results"""
        logger.info("Analyzing results...")
        
        # Convert to DataFrame
        df_data = []
        for result in self.results:
            df_data.append({
                'architecture': result.config.architecture,
                'decoder_type': result.config.decoder_type,
                'operation': result.config.operation,
                'seed': result.config.seed,
                'psnr': result.psnr,
                'parameter_count': result.parameter_count,
                'training_time': result.training_time,
                'parameter_efficiency': result.psnr / (result.parameter_count / 1000)
            })
        
        df = pd.DataFrame(df_data)
        
        # Statistical analysis
        analysis = {}
        
        # Hypothesis 1: K-Planes vs others
        kplanes_psnr = df[df['architecture'] == 'kplanes']['psnr']
        nerf_psnr = df[df['architecture'] == 'nerf']['psnr']
        linear_psnr = df[df['architecture'] == 'linear']['psnr']
        
        analysis['hypothesis_1'] = {
            'hypothesis': "K-Planes superior to NeRF and Linear baselines",
            'results': {
                'kplanes_mean_psnr': float(kplanes_psnr.mean()),
                'kplanes_std_psnr': float(kplanes_psnr.std()),
                'nerf_mean_psnr': float(nerf_psnr.mean()),
                'nerf_std_psnr': float(nerf_psnr.std()),
                'linear_mean_psnr': float(linear_psnr.mean()),
                'linear_std_psnr': float(linear_psnr.std())
            }
        }
        
        # Statistical test: K-Planes vs NeRF
        if len(kplanes_psnr) > 0 and len(nerf_psnr) > 0:
            t_stat, p_value = stats.ttest_ind(kplanes_psnr, nerf_psnr)
            analysis['hypothesis_1']['kplanes_vs_nerf'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'kplanes_better': t_stat > 0 and p_value < 0.05
            }
        
        # Hypothesis 2: Linear vs Nonlinear decoders (K-Planes only)
        kplanes_df = df[df['architecture'] == 'kplanes']
        linear_decoder_psnr = kplanes_df[kplanes_df['decoder_type'] == 'linear']['psnr']
        nonconvex_decoder_psnr = kplanes_df[kplanes_df['decoder_type'] == 'nonconvex']['psnr']
        
        analysis['hypothesis_2'] = {
            'hypothesis': "Linear decoders comparable to nonlinear decoders",
            'results': {
                'linear_decoder_mean': float(linear_decoder_psnr.mean()),
                'nonconvex_decoder_mean': float(nonconvex_decoder_psnr.mean()),
                'difference': float(linear_decoder_psnr.mean() - nonconvex_decoder_psnr.mean())
            }
        }
        
        if len(linear_decoder_psnr) > 0 and len(nonconvex_decoder_psnr) > 0:
            t_stat, p_value = stats.ttest_ind(linear_decoder_psnr, nonconvex_decoder_psnr)
            analysis['hypothesis_2']['statistical_test'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'equivalent': abs(linear_decoder_psnr.mean() - nonconvex_decoder_psnr.mean()) < 2.0
            }
        
        # Parameter efficiency analysis
        analysis['parameter_efficiency'] = {
            'best_efficiency_by_architecture': df.groupby('architecture')['parameter_efficiency'].mean().to_dict(),
            'parameter_counts': df.groupby('architecture')['parameter_count'].mean().to_dict(),
            'best_overall': df.loc[df['parameter_efficiency'].idxmax()].to_dict()
        }
        
        # Summary statistics
        analysis['summary'] = {
            'total_experiments': len(df),
            'mean_psnr_by_architecture': df.groupby('architecture')['psnr'].mean().to_dict(),
            'best_configuration': df.loc[df['psnr'].idxmax()].to_dict()
        }
        
        return analysis

def main():
    """Run quick hypothesis testing"""
    tester = QuickHypothesisTester()
    analysis = tester.run_all_experiments()
    
    print("\n" + "="*60)
    print("QUICK HYPOTHESIS TESTING RESULTS")
    print("="*60)
    
    # Print key findings
    h1 = analysis['hypothesis_1']
    print(f"\nHYPOTHESIS 1: {h1['hypothesis']}")
    print(f"K-Planes PSNR: {h1['results']['kplanes_mean_psnr']:.2f}±{h1['results']['kplanes_std_psnr']:.2f}")
    print(f"NeRF PSNR:     {h1['results']['nerf_mean_psnr']:.2f}±{h1['results']['nerf_std_psnr']:.2f}")
    print(f"Linear PSNR:   {h1['results']['linear_mean_psnr']:.2f}±{h1['results']['linear_std_psnr']:.2f}")
    
    if 'kplanes_vs_nerf' in h1:
        test = h1['kplanes_vs_nerf']
        conclusion = "SUPPORTED" if test['kplanes_better'] else "NOT SUPPORTED"
        print(f"K-Planes vs NeRF: p={test['p_value']:.4f}, {conclusion}")
    
    h2 = analysis['hypothesis_2']
    print(f"\nHYPOTHESIS 2: {h2['hypothesis']}")
    print(f"Linear decoder:    {h2['results']['linear_decoder_mean']:.2f}")
    print(f"Nonconvex decoder: {h2['results']['nonconvex_decoder_mean']:.2f}")
    print(f"Difference:        {h2['results']['difference']:.2f} PSNR")
    
    if 'statistical_test' in h2:
        test = h2['statistical_test']
        equiv = "EQUIVALENT" if test['equivalent'] else "DIFFERENT"
        print(f"Statistical conclusion: {equiv}")
    
    print(f"\nPARAMETER EFFICIENCY:")
    for arch, eff in analysis['parameter_efficiency']['best_efficiency_by_architecture'].items():
        print(f"{arch.capitalize()}: {eff:.2f} PSNR/1K params")
    
    best = analysis['summary']['best_configuration']
    print(f"\nBEST OVERALL: {best['architecture']} with {best['decoder_type']} decoder")
    print(f"PSNR: {best['psnr']:.2f}, Parameters: {best['parameter_count']}")
    
    # Save results
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"quick_hypothesis_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
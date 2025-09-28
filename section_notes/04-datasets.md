

# Datasets for 2D Matrix Reconstruction Research

## Research Framework

**Research Question**: How do different INR architectures originally designed for 3D radiance fields perform when repurposed for 2D matrix reconstruction, and what architectural components drive superior performance in the 2D domain?

**Primary Hypothesis**: K-Planes will demonstrate superior reconstruction quality compared to traditional MLP-based approaches (NeRF) for 2D matrix reconstruction, due to their explicit geometric bias toward planar structures inherent in 2D data.

**Target Metrics**: >35dB PSNR, >2x parameter efficiency improvement, >30% training time reduction

**Research Vectoring**: The biggest risk dimension is whether architectural differences will be significant enough to detect in the "simpler" 2D domain compared to 3D scenes.

## Dataset Selection Strategy

### Research-Driven Dataset Categories

Our dataset selection follows a systematic approach to validate the core hypothesis across multiple dimensions:

1. **Complexity Gradient**: From simple geometric patterns (MNIST) to complex natural scenes (BSD100)
2. **Frequency Content**: Low-frequency smooth regions (CelebA faces) to high-frequency textures (DTD)
3. **Structural Bias**: Datasets that favor planar representations (CIFAR) vs. those requiring complex modeling
4. **Scale Diversity**: Multiple resolutions to test architectural scalability
5. **Domain Specificity**: General objects, faces, textures, and synthetic patterns

### Experimental Design Rationale

Each dataset serves a specific role in testing our central hypothesis:
- **Baseline Performance**: MNIST, CIFAR-10 establish fundamental reconstruction capabilities
- **Architectural Stress-Testing**: High-resolution datasets (BSD100, CelebA) reveal efficiency differences
- **Controlled Analysis**: Synthetic datasets isolate specific architectural components
- **Real-World Validation**: Natural image datasets provide practical performance metrics

## Primary Datasets

### Computer Vision Benchmarks

**CIFAR-10** ⭐ *Primary Benchmark*

* **Dataset Characteristics**: 50,000 train + 10,000 test images (32×32×3 RGB)
* **Source**: [CIFAR Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
* **Research Rationale**: 
  - Standard CV benchmark with balanced complexity for INR comparison
  - 32×32 resolution optimal for testing K-Planes' planar factorization efficiency
  - 10 distinct object categories test generalization across visual domains
* **Hypothesis Testing**: K-Planes should excel due to natural planar structure in 2D images
* **Expected PSNR Baseline**: ~28-32dB for standard reconstruction methods
* **Computational Profile**: Moderate complexity, efficient for architectural comparison
* **Split Strategy**: Official train/test (5:1) with additional 10% validation subset

**MNIST** ⭐ *Baseline Control*

* **Dataset Characteristics**: 60,000 train + 10,000 test images (28×28×1 grayscale)
* **Source**: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
* **Research Rationale**:
  - Simple geometric patterns ideal for isolating architectural differences
  - Single-channel reduces complexity, focusing on spatial representation
  - Well-established baselines for reconstruction quality comparison
* **Hypothesis Testing**: Should reveal minimal architectural differences due to simplicity
* **Expected PSNR Baseline**: >35dB achievable, testing efficiency metrics
* **Computational Profile**: Lightweight, ideal for rapid experimentation
* **Research Value**: Establishes lower-bound performance and computational baselines

### High-Resolution Benchmarks

**CelebA-subset** ⭐ *High-Frequency Detail Testing*

* **Dataset Characteristics**: 1,000 face images (64×64×3 RGB, cropped and aligned)
* **Source**: [HuggingFace CelebA](https://huggingface.co/datasets/nielsr/CelebA-faces)
* **Research Rationale**:
  - Human faces contain structured high-frequency details (hair, skin texture)
  - 64×64 resolution challenges architectural efficiency
  - Aligned faces reduce positional variation, focusing on textural reconstruction
* **Hypothesis Testing**: K-Planes' planar bias may struggle with fine facial features vs. NeRF's flexibility
* **Expected PSNR Baseline**: 30-35dB for high-quality face reconstruction
* **Critical Analysis**: Tests whether geometric bias helps or hinders detailed texture reconstruction
* **Split Strategy**: 70/15/15 train/validation/test with stratified sampling across demographic features

**BSD100** ⭐ *Industry Benchmark*

* **Dataset Characteristics**: 100 natural images (variable size, typically 256×256+)
* **Source**: [Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
* **Research Rationale**:
  - Industry-standard super-resolution benchmark with established baselines
  - High resolution tests architectural scalability and memory efficiency
  - Diverse natural scenes challenge both geometric and textural modeling
* **Hypothesis Testing**: Critical test for K-Planes' scalability to high-resolution reconstruction
* **Expected PSNR Baseline**: 25-30dB for challenging high-resolution reconstruction
* **Research Impact**: Results directly comparable to published super-resolution literature
* **Computational Challenge**: Large images test memory efficiency and training stability

## Specialized Testing Datasets

### Controlled Ablation Studies

**Synthetic-Sinusoidal** 🧪 *Frequency Analysis*
* **Purpose**: 4 patterns (64×64) with varying frequencies (0.5, 1.0, 2.0, 4.0 cycles/image)
* **Research Value**: Tests smooth function approximation capabilities across architectures
* **Hypothesis Testing**: K-Planes should excel at low-frequency reconstruction due to planar bias
* **Expected Outcome**: Clear architectural differences in frequency-dependent performance

**Synthetic-Gaussian** 🧪 *Spatial Localization*
* **Purpose**: 10 Gaussian blob patterns (64×64) with varying scales and positions
* **Research Value**: Tests spatial representation efficiency and overfitting tendencies
* **Critical Insight**: Isolates spatial encoding effectiveness between architectures

**Synthetic-Checkerboard** 🧪 *Edge Reconstruction*
* **Purpose**: 4 checkerboard patterns (64×64) with tile sizes: 2×2, 4×4, 8×8, 16×16
* **Research Value**: Stress-tests high-frequency edge reconstruction capabilities
* **Hypothesis Testing**: NeRF's nonlinear capacity may outperform K-Planes on sharp edges

### Texture and Pattern Analysis

**DTD-Textures** 📊 *Repetitive Pattern Testing*
* **Dataset Characteristics**: 5,640 texture images across 47 categories
* **Source**: [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
* **Research Rationale**: 
  - Tests architectural performance on repetitive, structured patterns
  - Challenges both spatial and textural representation capacity
  - Provides diverse texture types: cracked, dotted, fibrous, lined, etc.
* **Hypothesis Testing**: K-Planes' factorization may be ideal for repetitive texture patterns
* **Expected Discovery**: Architecture-dependent performance across texture categories

### Secondary Validation Benchmarks

**CIFAR-100** 🔄 *Fine-Grained Complexity*
* **Dataset Characteristics**: 50,000 train + 10,000 test (32×32×3)
* **Research Value**: 100 fine-grained classes vs CIFAR-10's 10 broad classes
* **Complexity Analysis**: Tests architectural generalization across detailed visual categories
* **Expected Challenge**: Higher intra-class variation may favor NeRF's representational flexibility

**Fashion-MNIST** 👕 *Structured Alternative*
* **Dataset Characteristics**: 60,000 train + 10,000 test (28×28×1)
* **Research Value**: More complex than MNIST while maintaining single-channel simplicity
* **Comparative Analysis**: Direct complexity bridge between MNIST and CIFAR datasets
* **Architectural Insight**: Tests whether increased complexity favors specific INR approaches

## Experimental Data Pipeline

### Standardized Preprocessing Protocol

**Phase 1: Data Normalization**
1. **Pixel Value Scaling**: 
   - Convert to [0,1] range for consistent INR input domain
   - Alternative [-1,1] scaling for SIREN activation compatibility
   - Document scaling choice impact on reconstruction quality

2. **Resolution Standardization**:
   - Maintain native resolutions where possible (28×28 MNIST, 32×32 CIFAR)
   - Standardize high-res images to 256×256 for computational efficiency
   - Create multi-scale versions for resolution-dependent analysis

3. **Channel Normalization**:
   - RGB datasets: Per-channel mean/std normalization using ImageNet statistics
   - Grayscale: Zero-mean, unit-variance normalization
   - Document normalization impact on convergence rates

### Controlled Augmentation Strategy

**Research-Focused Augmentation** (Applied selectively for specific experiments):

1. **Geometric Variations**:
   - Rotation (±15°): Tests architectural rotation invariance
   - Horizontal flipping: Validates spatial representation consistency
   - **Research Question**: Do K-Planes maintain efficiency under geometric transforms?

2. **Intensity Modulation**:
   - Brightness/contrast (±10%): Tests photometric robustness
   - **Architectural Comparison**: How do different INRs handle intensity variations?

3. **Noise Robustness Testing**:
   - Gaussian noise (σ=0.05, 0.1, 0.15): Progressive corruption analysis
   - **Research Value**: Determines architectural sensitivity to input corruption
   - **Expected Finding**: Overparameterized models may show different noise handling

### Data Quality and Validation Framework

**Comprehensive Quality Assurance**:

1. **Integrity Verification**:
   - Automated outlier detection using pixel statistics
   - Manual inspection of edge cases and anomalies
   - Corruption detection for downloaded datasets

2. **Statistical Distribution Analysis**:
   - Class balance verification across all categorical datasets
   - Pixel intensity distribution analysis by dataset
   - Frequency domain analysis for texture and pattern datasets

3. **Synthetic Data Validation**:
   - Mathematical verification of generated patterns (sinusoidal, Gaussian)
   - Visual inspection of checkerboard tile accuracy
   - Comparison against analytical ground truth

4. **Reproducibility Standards**:
   - Fixed random seeds (42) for all data sampling
   - Deterministic train/validation/test splits
   - Documented data loading procedures with checksums

## Comprehensive Evaluation Framework

### Multi-Dimensional Assessment Metrics

**Primary Quality Metrics**:
* **Peak Signal-to-Noise Ratio (PSNR)**:
  - Target: >35dB for high-quality reconstruction
  - Dataset-specific baselines: MNIST (>40dB), CIFAR-10 (>30dB), BSD100 (>28dB)
  - Research Focus: Architecture comparison at equivalent parameter counts

* **Structural Similarity Index (SSIM)**:
  - Perceptual quality assessment beyond pixel-wise error
  - Critical for face reconstruction (CelebA) and texture analysis (DTD)
  - Expected Range: 0.85-0.95 for successful 2D reconstruction

**Efficiency and Scalability Metrics**:
* **Parameter Efficiency Ratio**: Quality/Parameters (PSNR per 1K parameters)
* **Training Time Efficiency**: PSNR improvement per training hour
* **Memory Footprint**: Peak GPU memory during training and inference
* **Convergence Rate**: Steps to achieve 90% of final PSNR

### Rigorous Validation Methodology

**Standardized Split Strategy**:
* **Established Benchmarks**: Maintain official splits (CIFAR, MNIST, Fashion-MNIST)
  - Enables direct comparison with published baselines
  - Preserves research reproducibility standards

* **Custom Dataset Partitioning**: 70/15/15 train/validation/test
  - Stratified sampling preserves class/pattern distribution
  - Validation set for hyperparameter optimization
  - Held-out test set for final architectural comparison

* **Synthetic Pattern Evaluation**: 5-fold cross-validation
  - Accounts for limited synthetic sample sizes
  - Statistical robustness across pattern variations
  - Enables reliable architectural difference detection

**Statistical Rigor**:
* **Multiple Random Seeds**: 5 independent runs per architecture/dataset pair
* **Confidence Intervals**: 95% CI for all reported metrics
* **Significance Testing**: Paired t-tests for architectural comparisons
* **Effect Size Analysis**: Cohen's d for practical significance assessment

### Systematic Experimental Design

**Controlled Architecture Comparison**:
* **Identical Conditions**: Same data splits, hyperparameter search space, compute resources
* **Fair Comparison Protocol**: Equivalent parameter budgets across architectures
* **Standardized Training**: Adam optimizer, cosine annealing, early stopping

**Multi-Level Ablation Framework**:
* **Architectural Components**: 
  - Decoder types (linear vs. MLP)
  - Interpolation methods (bilinear vs. learned)
  - Positional encodings (Fourier features vs. coordinate grids)

* **Dataset-Specific Analysis**:
  - Synthetic patterns: Isolate frequency response and edge handling
  - Natural images: Test generalization and scaling properties
  - Texture datasets: Evaluate repetitive pattern modeling

**Computational Profiling**:
* **Training Efficiency**: Time-to-convergence across dataset complexity
* **Memory Scaling**: Peak usage vs. input resolution
* **Inference Speed**: Forward pass timing for deployment considerations
* **Hardware Utilization**: GPU/CPU usage patterns by architecture

### Research Success Validation

**Primary Hypothesis Validation Criteria**:

1. **Reconstruction Quality Superiority**:
   - PSNR improvement >3dB over NeRF baseline (statistically significant)
   - Consistent advantage across at least 6/8 primary datasets
   - Maintained quality at 50% parameter reduction

2. **Parameter Efficiency Advantage**:
   - >2x parameter efficiency (PSNR per parameter) improvement
   - Scalability advantage increases with input resolution
   - Efficiency maintained across diverse visual domains

3. **Training and Inference Speed**:
   - >30% training time reduction vs. 3D-optimized NeRF implementations
   - Inference speed advantage for deployment applications
   - Memory efficiency enabling larger batch sizes

4. **Architectural Robustness**:
   - Performance advantage holds across texture categories (DTD)
   - Maintained efficiency under data augmentation
   - Consistent behavior across synthetic pattern complexities

**Research Impact Thresholds**:
* **Literature-Level Contribution**: Results generalize beyond specific datasets
* **Practical Significance**: Improvements matter for real applications
* **Theoretical Insight**: Findings explain when/why K-Planes excel in 2D domain

## Dataset Management and Access

**Current Repository Status**:
* **Storage Method**: Git LFS for large dataset files
* **Total Collection Size**: ~2.2GB across 8 primary dataset categories
* **Repository Structure**: 
  - `data/processed/` - Ready-to-use dataset files
  - `data/README.md` - Complete dataset documentation
  - `data/dataset_catalog.json` - Structured metadata

**Access and Loading Protocol**:

```bash
# Repository setup
git clone <repository-url>
git lfs pull  # Downloads actual dataset files

# Verify dataset integrity
python scripts/verify_datasets.py
```

**Programmatic Access**:

```python
# Standard dataset loading
from utils.data_loader import load_dataset

# Load with automatic preprocessing
cifar10_train, cifar10_test = load_dataset('cifar10', 
                                          preprocess=True,
                                          normalize=True)

# Custom loading for experiments
mnist_data = load_dataset('mnist', 
                         resolution=32,  # upsampled
                         format='neural_field')  # coordinate-value pairs
```

**Research Reproducibility**:
* **Dataset Checksums**: MD5 verification for all downloaded files
* **Version Control**: Dataset versions tracked with experimental runs
* **Loading Scripts**: Standardized preprocessing for consistent results
* **Documentation**: Complete metadata including source URLs, preprocessing steps

**Quality Assurance Pipeline**:
1. Automated integrity checks on dataset loading
2. Statistical validation of preprocessing operations
3. Visual inspection tools for manual verification
4. Comparison against published dataset statistics

## Critical Dataset Analysis and Research Gaps

### Current Dataset Collection Strengths

**Comprehensive Coverage Across Key Dimensions**:
- ✅ **Resolution Spectrum**: 28×28 (MNIST) → 32×32 (CIFAR) → 64×64 (CelebA) → 256×256+ (BSD100)
- ✅ **Complexity Gradient**: Simple patterns → Natural objects → Fine-grained textures
- ✅ **Domain Diversity**: Handwritten digits, natural objects, faces, textures, synthetic patterns
- ✅ **Established Baselines**: All datasets have published reconstruction benchmarks

**Research-Aligned Selection**:
- **Hypothesis Testing**: Datasets chosen to reveal K-Planes vs NeRF architectural differences
- **Controlled Experiments**: Synthetic patterns enable isolated component analysis
- **Practical Validation**: Industry-standard benchmarks (BSD100, CIFAR) ensure relevance

### Identified Gaps and Recommendations

**Gap 1: Limited Medical/Scientific Imaging**
- **Missing**: Medical images (MRI, CT, X-ray) with different frequency characteristics
- **Research Impact**: Medical imaging often requires precise reconstruction of subtle features
- **Recommendation**: Add chest X-ray dataset (NIH) or brain MRI slices for domain-specific validation

**Gap 2: Insufficient High-Resolution Natural Images**
- **Current**: Only BSD100 for high-res testing (100 images)
- **Missing**: Larger-scale high-resolution benchmarks
- **Recommendation**: Consider DIV2K (2K resolution) subset for scalability testing

**Gap 3: Limited Temporal/Sequential Data**
- **Missing**: Video frames or sequential image data
- **Research Value**: Could test architectural efficiency on temporal patterns
- **Future Work**: Consider moving MNIST sequences for dynamic reconstruction

### Dataset Selection Rationale Summary

Our final dataset collection represents a **systematic, hypothesis-driven approach** to INR architecture comparison:

1. **Breadth**: Covers major visual domains and complexity levels
2. **Depth**: Multiple datasets per complexity tier enable robust comparison  
3. **Control**: Synthetic datasets isolate architectural components
4. **Practicality**: Standard benchmarks ensure reproducible, comparable results
5. **Efficiency**: Manageable computational requirements while maintaining research rigor

**Total Experimental Scope**: 8 primary datasets × 3 architectures × 5 random seeds = 120 comprehensive experiments, designed to definitively validate or refute our core hypothesis about K-Planes' advantages in 2D reconstruction tasks.

### Research Implementation Priority

**Phase 1 (Immediate)**: MNIST, CIFAR-10, synthetic patterns - establish baseline architectural differences
**Phase 2 (Core Research)**: CelebA, BSD100, DTD textures - test scalability and domain generalization  
**Phase 3 (Validation)**: CIFAR-100, Fashion-MNIST - confirm findings across additional domains

This prioritization ensures early detection of null results while building toward comprehensive validation of our research hypothesis.


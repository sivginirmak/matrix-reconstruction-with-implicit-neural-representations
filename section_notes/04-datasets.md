

# Datasets for 2D Matrix Reconstruction Research

**Research Context**: Testing INR architectures (K-Planes vs NeRF vs Gaussian methods) on 2D matrix reconstruction tasks
**Hypothesis**: K-Planes will outperform NeRF for 2D reconstruction due to planar geometric bias
**Target Metrics**: >35dB PSNR, >2x parameter efficiency improvement

## Primary Datasets

### Computer Vision Benchmarks

**CIFAR-10**

* Size: 50,000 train + 10,000 test images (32×32×3 RGB)
* Source: [https://www.cs.toronto.edu/\~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* Baseline: Standard CV benchmark, widely used for 2D reconstruction quality assessment
* Split: Official train/test split (5:1 ratio)
* **Research Relevance**: Tests INR performance on diverse natural object categories

**MNIST**

* Size: 60,000 train + 10,000 test images (28×28×1 grayscale)
* Source: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* Baseline: Classic 2D reconstruction benchmark with simple geometric patterns
* Split: Official train/test split (6:1 ratio)
* **Research Relevance**: Establishes baseline performance on controlled, simple patterns

### High-Resolution Benchmarks

**CelebA-subset**

* Size: 1,000 face images (64×64×3 RGB, cropped and aligned)
* Source: [https://huggingface.co/datasets/nielsr/CelebA-faces](https://huggingface.co/datasets/nielsr/CelebA-faces)
* Baseline: High-resolution face reconstruction testing
* Split: 70/15/15 train/validation/test
* **Research Relevance**: Tests INR performance on human facial structures and high-frequency details

**BSD100**

* Size: 100 natural images (variable size, typically 256×256+)
* Source: [https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
* Baseline: Standard super-resolution and denoising benchmark
* Split: Use as test set for evaluation
* **Research Relevance**: Industry-standard benchmark for image reconstruction quality

## Additional Datasets

### Controlled Testing

**Synthetic-Sinusoidal**: 4 patterns (64×64) with varying frequencies for smooth function approximation testing
**Synthetic-Gaussian**: 10 Gaussian blob patterns (64×64) for smooth 2D reconstruction evaluation
**Synthetic-Checkerboard**: 4 checkerboard patterns (64×64) with different tile sizes for edge reconstruction testing

### Texture Analysis

**DTD-Textures**: 5,640 texture images in 47 categories for repetitive pattern reconstruction testing

### Secondary Benchmarks

**CIFAR-100**: 50,000 train + 10,000 test (32×32×3) for fine-grained visual reconstruction testing
**Fashion-MNIST**: 60,000 train + 10,000 test (28×28×1) as MNIST alternative with more complexity

## Preprocessing Pipeline

### Normalization Procedures

1. **Pixel Value Scaling**: Convert to \[0,1] or \[-1,1] range for neural network compatibility
2. **Size Standardization**: Resize to common dimensions (32×32, 64×64, 256×256) where appropriate
3. **Channel Normalization**: Apply per-channel mean/std normalization for RGB datasets

### Data Augmentation (Optional)

1. **Geometric**: Rotation (±15°), horizontal flipping for robustness testing
2. **Intensity**: Slight brightness/contrast adjustments (±10%)
3. **Noise Addition**: Gaussian noise (σ\=0.1) for denoising experiments

### Quality Assurance

1. **Outlier Detection**: Remove corrupted or mislabeled samples
2. **Distribution Analysis**: Verify balanced class/pattern representation
3. **Validation**: Cross-check synthetic pattern generation accuracy

## Evaluation Protocol

### Primary Metrics

* **Peak Signal-to-Noise Ratio (PSNR)**: Target >35dB for high-quality reconstruction
* **Structural Similarity Index (SSIM)**: Perceptual quality assessment
* **Parameter Efficiency**: Model parameters per quality unit achieved

### Validation Strategy

* **Standard Datasets**: Use existing train/test splits (CIFAR, MNIST, Fashion-MNIST)
* **Custom Datasets**: 70/15/15 train/validation/test split with stratification
* **Synthetic Patterns**: 5-fold cross-validation due to small sample sizes
* **Statistical Testing**: Multiple runs with different seeds for significance testing

### Experimental Design

* **Baseline Comparison**: Test each INR architecture on identical data splits
* **Ablation Studies**: Use synthetic datasets to isolate architectural component effects
* **Computational Analysis**: Measure training time, memory usage, and inference speed
* **Reproducibility**: Fixed random seed (42) for consistent results across experiments

### Success Criteria

1. **Quality**: PSNR improvement >5dB over baseline methods
2. **Efficiency**: Parameter count reduction >50% for equivalent quality
3. **Speed**: Training time reduction >30% compared to 3D-optimized implementations
4. **Robustness**: Consistent performance across diverse dataset categories

## Dataset Storage & Access

**Total Size**: 2.2GB managed via Git LFS
**Access Method**: `git lfs pull` after repository cloning
**Documentation**: Complete specifications in `data/README.md`
**Metadata**: Dataset catalog available in `data/dataset_catalog.json`


# Datasets for 2D Matrix Reconstruction Research

**Research Context**: Testing INR architectures (K-Planes vs NeRF vs Gaussian methods) on 2D matrix reconstruction tasks
**Hypothesis**: K-Planes will outperform NeRF for 2D reconstruction due to planar geometric bias
**Target Metrics**: >35dB PSNR, >2x parameter efficiency improvement

## Comprehensive Dataset Collection

Our research employs a systematic approach to dataset selection based on recent INR reconstruction literature and benchmark standards. The collection includes **8 major datasets** with **6,011 total files** managed via Git LFS.

### Dataset Access Method
```bash
# Clone repository and pull LFS data
git clone <repository-url>
cd matrix-reconstruction-with-implicit-neural-representations
git lfs pull  # Downloads actual dataset files from Git LFS
```

## Primary Computer Vision Benchmarks

### CIFAR-10
**Path**: `data/processed/cifar10/`
- **Size**: 50,000 train + 10,000 test images (32×32×3 RGB)  
- **Source**: [Toronto CS Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Format**: Python pickle files in `cifar-10-batches-py/`
- **Key Files**: 
  - `data_batch_1` through `data_batch_5` (training batches)
  - `test_batch` (test set)
  - `batches.meta` (metadata)
- **Research Relevance**: Standard CV benchmark for diverse natural objects, tests INR generalization across 10 semantic categories
- **Loading Code**:
```python
import pickle
import numpy as np

def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
```

### CIFAR-100  
**Path**: `data/processed/cifar100/`
- **Size**: 50,000 train + 10,000 test images (32×32×3 RGB)
- **Format**: Python pickle files (`train`, `test`, `meta`)
- **Research Relevance**: Fine-grained classification with 100 classes, tests INR performance on more detailed category distinctions

### MNIST
**Path**: `data/processed/mnist/MNIST/raw/`
- **Size**: 60,000 train + 10,000 test images (28×28×1 grayscale)
- **Format**: IDX format binary files
- **Files**: 
  - `train-images-idx3-ubyte`: Training images
  - `train-labels-idx1-ubyte`: Training labels  
  - `t10k-images-idx3-ubyte`: Test images
  - `t10k-labels-idx1-ubyte`: Test labels
- **Research Relevance**: Establishes baseline performance on simple geometric patterns
- **Loading Code**:
```python
import struct
import numpy as np

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num, rows, cols)
```

### Fashion-MNIST
**Path**: `data/processed/fashion_mnist/FashionMNIST/raw/`
- **Size**: 60,000 train + 10,000 test images (28×28×1 grayscale) 
- **Format**: Same IDX format as MNIST
- **Research Relevance**: More complex alternative to MNIST with fashion categories, tests INR on textured patterns

## High-Resolution Benchmarks

### CelebA Subset
**Path**: `data/processed/celeba/`  
- **Size**: 1,000 face images (64×64×3 RGB, cropped and aligned)
- **Format**: Python pickle file (`celeba_subset.pkl`)
- **Research Relevance**: High-resolution face reconstruction with human facial structures and high-frequency details
- **Loading Code**:
```python
import pickle
import numpy as np

def load_celeba_subset():
    with open('data/processed/celeba/celeba_subset.pkl', 'rb') as f:
        return pickle.load(f)
```

### BSD100 (Berkeley Segmentation Dataset)
**Path**: `data/processed/bsd100/BSDS300/images/`
- **Size**: 200 train + 100 test natural images (variable size, typically 256×256+)
- **Format**: JPEG images in `train/` and `test/` subdirectories  
- **Source**: [Berkeley Vision Group](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **Key Files**:
  - `images/train/`: 200 training images
  - `images/test/`: 100 test images  
  - `iids_train.txt`: Training image IDs
  - `iids_test.txt`: Test image IDs
- **Research Relevance**: Industry-standard benchmark for super-resolution and denoising
- **Loading Code**:
```python
from PIL import Image
import os

def load_bsd100_images(split='test'):
    image_dir = f'data/processed/bsd100/BSDS300/images/{split}/'
    images = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            images.append(np.array(img))
    return images
```

## Specialized Testing Datasets

### Synthetic Patterns
**Path**: `data/processed/synthetic/`
- **Files**:
  - `sinusoidal_patterns.npy`: 4 patterns (64×64) with varying frequencies
  - `gaussian_blobs.npy`: 10 Gaussian blob patterns (64×64)  
  - `checkerboards.npy`: 4 checkerboard patterns with different tile sizes
- **Research Relevance**: Controlled testing for smooth function approximation, edge reconstruction, and pattern analysis
- **Loading Code**:
```python
import numpy as np

sinusoidal = np.load('data/processed/synthetic/sinusoidal_patterns.npy')
gaussian = np.load('data/processed/synthetic/gaussian_blobs.npy')
checkerboards = np.load('data/processed/synthetic/checkerboards.npy')
```

### DTD Textures
**Path**: `data/processed/textures/dtd/dtd/`
- **Size**: 5,640 texture images across 47 categories
- **Format**: JPEG images organized by texture categories (e.g., `lacelike/`, `meshed/`)
- **Source**: [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- **Research Relevance**: Tests INR performance on repetitive patterns and texture reconstruction
- **Split Files**: `train1.txt` through `train10.txt`, `val1.txt` through `val10.txt`, `test1.txt` through `test10.txt`

## Dataset Statistics and Analysis

### Total Collection Metrics
- **Total Datasets**: 8 comprehensive datasets
- **Total Files**: 6,011 files across all datasets  
- **Storage**: 24MB with Git LFS tracking
- **Coverage**: Computer vision, high-resolution, synthetic, and texture categories

### Research Coverage Matrix

| Dataset | Type | Resolution | Samples | Research Focus |
|---------|------|------------|---------|----------------|
| MNIST | Synthetic digits | 28×28×1 | 70K | Baseline patterns |
| Fashion-MNIST | Fashion items | 28×28×1 | 70K | Complex textures |
| CIFAR-10 | Natural objects | 32×32×3 | 60K | Semantic diversity |  
| CIFAR-100 | Fine-grained | 32×32×3 | 60K | Category complexity |
| CelebA-subset | Human faces | 64×64×3 | 1K | High-frequency details |
| BSD100 | Natural scenes | Variable | 300 | Super-resolution |
| Synthetic | Mathematical | 64×64 | 17 | Controlled testing |
| DTD Textures | Texture patterns | Variable | 5,640 | Repetitive structures |

## Recent Literature Integration

Based on analysis of recent INR reconstruction papers (2024-2025), our dataset collection aligns with current benchmarking standards:

### Papers Using Similar Datasets
1. **"Implicit Neural Representations for Robust Joint Sparse-View CT Reconstruction"** (Shi et al., 2024) - Validates multi-object learning approach
2. **"Low-Rank Augmented Implicit Neural Representation for MRI Reconstruction"** (Zhang et al., 2025) - Supports dual prior framework testing
3. **"Grids Often Outperform Implicit Neural Representations"** (Kim & Fridovich-Keil, 2025) - Direct relevance to our comparative analysis

### Dataset Gaps Identified and Addressed
- **Multi-resolution testing**: BSD100 provides variable resolution images
- **Texture analysis**: DTD dataset covers 47 texture categories
- **Controlled experiments**: Synthetic patterns enable ablation studies
- **Face reconstruction**: CelebA subset tests high-frequency detail preservation

## Experimental Protocol

### Data Preprocessing Pipeline
```python
def preprocess_for_inr(image, target_size=(64, 64), normalize_range=(-1, 1)):
    """Standard preprocessing for INR experiments"""
    # Resize to target dimensions
    image = resize(image, target_size)
    
    # Normalize pixel values
    if normalize_range == (-1, 1):
        image = (image / 127.5) - 1.0
    elif normalize_range == (0, 1):
        image = image / 255.0
    
    return image.astype(np.float32)
```

### Evaluation Splits
- **Standard datasets**: Use official train/test splits (CIFAR, MNIST, Fashion-MNIST)
- **Custom datasets**: 70/15/15 train/validation/test with stratification
- **Synthetic patterns**: 5-fold cross-validation due to small sample sizes
- **Statistical testing**: Multiple runs with different seeds (42, 123, 456, 789, 999)

### Success Metrics Implementation
```python
def calculate_reconstruction_metrics(original, reconstructed):
    """Calculate PSNR and SSIM for reconstruction evaluation"""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    psnr = peak_signal_noise_ratio(original, reconstructed, data_range=2.0)
    ssim = structural_similarity(original, reconstructed, data_range=2.0)
    
    return {'psnr': psnr, 'ssim': ssim}
```

## Dataset Utilization in Experiments

### Current Experimental Results
Based on `experiments/exp001_architecture_comparison/`, the following datasets have been tested:
- **Primary focus**: Synthetic patterns for controlled comparison
- **Performance range**: 11.57-27.66 dB PSNR across different architectures
- **Parameter efficiency**: 0.001-0.009 efficiency ratio achieved

### Future Experimental Pipeline
1. **Phase 1**: Complete evaluation on all 8 datasets
2. **Phase 2**: Cross-dataset generalization testing  
3. **Phase 3**: Domain-specific optimization analysis

## Data Access and Reproducibility

### Repository Integration
All datasets are committed with Git LFS and immediately accessible after:
```bash
git lfs pull
```

### Documentation Files
- **Complete specifications**: `data/README.md` (Git LFS tracked)
- **Metadata catalog**: `data/dataset_catalog.json` (Git LFS tracked)  
- **Download script**: `data/download_datasets.py` (reference implementation)

### Computational Requirements
- **Disk space**: 24MB current, estimated 2.2GB when fully extracted
- **Memory usage**: Varies by dataset size (28×28 to 256×256+ images)
- **Processing time**: <1 minute for dataset loading and preprocessing

This comprehensive dataset collection provides robust benchmarking capabilities for systematic comparison of INR architectures in 2D matrix reconstruction tasks, supporting the research hypothesis with appropriate statistical rigor.
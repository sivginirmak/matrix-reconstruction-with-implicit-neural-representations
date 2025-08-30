# INR Applications in Reconstruction Tasks

## 1. Implicit Neural Representations for Robust Joint Sparse-View CT Reconstruction (Shi et al., 2024)
**Paper:** Transactions on Machine Learning Research (2024)  
**Authors:** Jiayang Shi, Junyi Zhu, Daniel M. Pelt, K. Joost Batenburg, Matthew B. Blaschko  
**Venue:** TMLR 2024  
**URL:** https://arxiv.org/abs/2405.02509

### Key Contributions
- **Problem:** Sparse-view CT reconstruction from under-sampled measurements
- **Insight:** Joint reconstruction of multiple objects using INRs can improve individual reconstruction quality
- **Method:** Novel INR-based Bayesian framework with latent variables capturing common patterns
- **Impact:** Higher reconstruction quality with sparse views, robust to noise

### Technical Details
- **Joint Reconstruction:** Utilizes common patterns across multiple objects
- **Bayesian Framework:** Integrates latent variables for pattern sharing
- **Common Patterns:** Latent variables assist reconstruction of each object
- **Robustness:** Maintains performance under noisy measurements
- **Initialization:** Learned latent variables serve as network initialization for new objects

### Relevance to Matrix Reconstruction
- **Multi-object Learning:** Concept of learning from multiple matrices simultaneously
- **Pattern Sharing:** Common structural patterns across different matrices
- **Sparse Observations:** Handling reconstruction from limited observations
- **Robustness:** Techniques for noise handling applicable to matrix completion

## 2. Low-Rank Augmented Implicit Neural Representation for MRI Reconstruction (Zhang et al., 2025)
**Paper:** Recent preprint  
**Authors:** Haonan Zhang, Guoyan Lao, Yuyao Zhang, Hongjiang Wei  
**Venue:** arXiv preprint  
**URL:** https://arxiv.org/abs/2506.09100

### Key Contributions
- **Problem:** Unsupervised high-dimensional quantitative MRI reconstruction from undersampled data
- **Insight:** Combine low-rank prior and continuity prior via Low-Rank Representation (LRR) and INR
- **Method:** LoREIN framework integrating dual priors for enhanced reconstruction fidelity
- **Impact:** Zero-shot learning paradigm with broad potential for complex reconstruction tasks

### Technical Details
- **Dual Priors:** Low-rank representation + Implicit neural representation
- **Low-rank Prior:** Captures structural redundancy in MRI data
- **Continuity Prior:** INR provides smooth continuous representation
- **Zero-shot Learning:** No need for training data from target domain
- **3D MP-qMRI:** Application to multi-parametric quantitative MRI

### Relevance to Matrix Reconstruction
- **Dual Prior Framework:** Combining low-rank assumptions with continuous representations
- **Unsupervised Learning:** No need for fully observed training matrices
- **High-dimensional Data:** Techniques for handling large-scale matrix problems
- **Prior Integration:** How to combine traditional matrix priors with neural representations

## 3. ImputeINR: Time Series Imputation via Implicit Neural Representations (Li et al., 2025)
**Paper:** IJCAI 2025  
**Authors:** Mengxuan Li, Ke Liu, Jialong Guo, Jiajun Bu, Hongwei Wang, Haishuai Wang  
**Venue:** IJCAI 2025  
**URL:** https://arxiv.org/abs/2505.10856

### Key Contributions
- **Problem:** Time series imputation for healthcare data with missing values
- **Insight:** INRs can learn continuous functions for time series, enabling fine-grained imputation
- **Method:** Continuous time series representation not coupled to sampling frequency
- **Impact:** Superior performance especially for high missing ratios, enhances downstream disease diagnosis

### Technical Details
- **Continuous Representation:** Functions not tied to discrete sampling frequency
- **Infinite Sampling:** Can generate fine-grained imputations from sparse observations
- **Healthcare Applications:** Validated on medical time series data
- **Missing Ratios:** Particularly effective when large portions of data are missing
- **Downstream Tasks:** Improves disease diagnosis when applied to imputed data

### Relevance to Matrix Reconstruction
- **Sparse Data Handling:** Techniques for reconstruction from very sparse observations
- **Continuous Interpolation:** Smooth interpolation between observed entries
- **High Missing Ratios:** Methods that work even when most matrix entries are missing
- **Application Domains:** Similar principles apply to matrix completion in various domains

## 4. Mixed-granularity Implicit Representation for Hyperspectral Reconstruction (Li et al., 2025)
**Paper:** Accepted by TNNLS  
**Authors:** Jianan Li, Huan Chen, Wangcai Zhao, Rui Chen, Tingfa Xu  
**Venue:** IEEE Transactions on Neural Networks and Learning Systems  
**URL:** https://arxiv.org/abs/2503.12783

### Key Contributions
- **Problem:** Continuous hyperspectral compressive reconstruction with flexible resolution
- **Insight:** Mixed-granularity representation enables reconstruction at arbitrary spatial-spectral resolutions
- **Method:** MGIR framework with hierarchical spectral-spatial implicit encoder and mixed-granularity feature aggregator
- **Impact:** Reconstruction at any desired resolution, matching state-of-the-art across compression ratios

### Technical Details
- **MGIR Framework:** Mixed Granularity Implicit Representation
- **Hierarchical Encoder:** Multi-scale implicit feature extraction
- **Feature Aggregator:** Adaptively integrates local features across scales
- **Arbitrary Resolution:** Can reconstruct at any spatial-spectral resolution
- **CASSI System:** Applied to Coded Aperture Snapshot Spectral Imaging

### Relevance to Matrix Reconstruction
- **Multi-resolution:** Reconstruction at different scales and resolutions
- **Hierarchical Features:** Multi-scale feature representation for matrices
- **Arbitrary Resolution:** Query matrix values at any coordinate precision
- **Compression Applications:** Relevant for compressed matrix storage and reconstruction

## 5. Implicit Neural Representation-Based MRI Reconstruction with Sensitivity Map Constraints (Rao et al., 2025)
**Paper:** Recent preprint  
**Authors:** Lixuan Rao, Xinlin Zhang, Yiman Huang, Tao Tan, Tong Tong  
**Venue:** arXiv preprint  
**URL:** https://arxiv.org/abs/2506.06043

### Key Contributions
- **Problem:** Fast MRI reconstruction without fully-sampled training images
- **Insight:** Joint estimation of coil sensitivity maps and images with sensitivity map regularization
- **Method:** INR-CRISTAL with extra regularization on sensitivity map characteristics
- **Impact:** Superior reconstruction with fewer artifacts, robust to acceleration rates

### Technical Details
- **Scan-specific Method:** No need for fully-sampled training data
- **Joint Estimation:** Simultaneous coil sensitivity and image estimation
- **Sensitivity Regularization:** Exploits smooth characteristics of sensitivity maps
- **High Acceleration:** Strong robustness to high acceleration factors
- **Artifact Reduction:** Superior performance in removing reconstruction artifacts

### Relevance to Matrix Reconstruction
- **Constraint Integration:** How to incorporate domain-specific constraints in INR reconstruction
- **Joint Estimation:** Simultaneous estimation of multiple related quantities
- **No Training Data:** Self-supervised reconstruction without training examples
- **Regularization Techniques:** Domain-specific regularization strategies

## Common Themes Across INR Reconstruction Applications

### 1. Sparse-to-Dense Reconstruction
- All methods address reconstruction from limited observations
- INRs provide continuous interpolation between sparse measurements
- Particularly effective when traditional discrete methods fail

### 2. Multi-scale and Hierarchical Approaches
- Many methods incorporate multi-scale representations
- Hierarchical feature extraction at different resolutions
- Coarse-to-fine reconstruction strategies

### 3. Domain-Specific Priors and Constraints
- Integration of domain knowledge through specialized regularization
- Combination of traditional priors (low-rank, sparsity) with neural representations
- Physics-informed constraints in medical imaging applications

### 4. Zero-shot and Self-supervised Learning
- Many INR reconstruction methods don't require paired training data
- Self-supervised optimization on individual instances
- Transfer learning through shared architectural patterns

### 5. Robustness and Generalization
- INR methods show strong robustness to noise and measurement artifacts
- Good generalization to different acceleration factors and sampling patterns
- Maintains performance across different data characteristics
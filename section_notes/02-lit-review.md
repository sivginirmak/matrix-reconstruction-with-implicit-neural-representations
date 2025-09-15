Remove all pre-print referenc

# Literature Review

## Overview

This literature review examines the intersection of **Implicit Neural Representations (INRs)** and **matrix reconstruction/completion**, focusing on adapting 3D INR methods (NeRF, K-Planes, TensoRF) to 2D matrix problems. We systematically analyze foundational INR work, tensor factorization methods, traditional matrix completion approaches, and recent reconstruction applications to identify research gaps and opportunities.

**Research Focus:** Comparing 2D fitting performances of (normally 3D) INRs such as K-Planes, GA-Planes, NeRF etc. Since everything is based on 2D reconstruction, any open-source image dataset can be used & the approach is CPU-friendly.

## Key Research Areas

### 1. Foundational INR Methods

#### Positional Encoding Approaches

**Tancik et al. (2020)** established that standard MLPs suffer from **spectral bias** - inability to learn high-frequency functions in low-dimensional domains. Their **Fourier feature mapping** γ(v) \= \[cos(2πBv), sin(2πBv)]^T enables MLPs to overcome this limitation by transforming the Neural Tangent Kernel into a stationary kernel with tunable bandwidth.

**Sitzmann et al. (2020)** proposed **SIREN** networks using sine activation functions sin(ω₀ · Wx + b), providing an alternative that enables access to all derivatives and smooth signal representation without explicit positional encoding.

**Key Insight for Matrix Reconstruction:** Both approaches address the fundamental challenge of representing continuous functions from discrete coordinates, directly applicable to matrix completion where we want smooth interpolation between observed entries.

#### Neural Radiance Fields Foundation

**Mildenhall et al. (2020)** demonstrated how MLPs with positional encoding can represent complex 3D scenes as continuous 5D radiance fields, establishing the paradigm of coordinate-based neural representations.

### 2. Tensor Factorization Methods for INRs

#### TensoRF: Tensorial Decomposition

**Chen et al. (2022)** revolutionized INR efficiency by modeling radiance fields as **4D tensors** and applying factorization:

* **CP Decomposition:** Factorizes tensors into rank-one components with compact vectors
* **Vector-Matrix (VM) Decomposition:** Novel factorization into vector and matrix factors
* **Performance:** 10-30 minute training vs. hours/days for NeRF, with <4-75MB models

**Direct Relevance to Matrix Problems:** TensoRF's tensor factorization techniques are immediately applicable to 2D matrix decomposition, providing efficient low-rank representations.

#### K-Planes: Planar Factorization

**Fridovich-Keil et al. (2023)** introduced elegant **planar factorization** using (d choose 2) planes for d-dimensional scenes:

* **4D Scenes:** 6 planes (3 spatial: xy,xz,yz + 3 spatio-temporal: xt,yt,zt)
* **Interpretability:** Static objects appear only in spatial planes, dynamic in spatio-temporal
* **Efficiency:** 1000x compression over full 4D grid

**Matrix Application:** For 2D matrices, this reduces to single plane representation, but the factorization principles and multi-scale approaches remain applicable.

### 3. Traditional Matrix Completion Foundations

#### Nuclear Norm Minimization

**Candès & Recht (2009)** and **Recht (2011)** established theoretical foundations for matrix completion via **nuclear norm minimization**:

* **Problem:** Recover low-rank matrix from small fraction of entries
* **Method:** min ||X||\* subject to observed entries constraints
* **Guarantees:** Exact recovery under incoherence conditions with O(nr polylog(n)) samples

**Limitations for INR Context:**

1. **Discrete representation** - no interpolation between entries
2. **Storage requirements** - must explicitly store matrix dimensions
3. **Complex pattern capture** - struggles with non-linear relationships
4. **Fixed resolution** - cannot query non-integer coordinates

#### Collaborative Filtering Applications

**Netflix Prize** context demonstrated matrix completion challenges:

* **Sparse matrices** with users >> movies, most entries missing
* **SVD-based factorization** M \= UΣV^T with regularization
* **Scalability issues** and cold-start problems

### 4. Recent INR Reconstruction Applications

#### Medical Imaging and Sparse Reconstruction

**Zhang et al. (2025)** combined **low-rank priors with INR continuity priors** in LoREIN framework:

* **Dual priors:** Traditional low-rank + neural continuity
* **Zero-shot learning:** No external training data required
* **High-dimensional capability:** 3D multi-parametric quantitative MRI

**Shi et al. (2024)** demonstrated **joint reconstruction** using INRs:

* **Multi-object learning:** Common patterns improve individual reconstruction
* **Bayesian framework:** Latent variables capture shared structure
* **Sparse-view robustness:** Superior performance with limited observations

#### Time Series and Imputation

**Li et al. (2025)** showed INRs excel at **imputation tasks**:

* **Continuous representation** not coupled to sampling frequency
* **High missing ratios:** Particularly effective when most data missing
* **Fine-grained interpolation** from extremely sparse observations

### 5. Theoretical Advances

#### Convex INR Formulations

**Sivgin et al. (2024)** introduced **GA-Planes** - first **convex optimization** framework for INRs:

* **Convex training:** Avoids local minima issues of standard non-convex INR training
* **Theoretical guarantees:** Global optimality vs. sensitivity to initialization
* **Generalization:** Framework encompasses many existing representations

**Impact for Matrix Reconstruction:** Potential to combine INR benefits with theoretical guarantees of convex matrix completion.

## Key Papers Analysis

### TensoRF (Chen et al., 2022)

* **Contribution:** Revolutionary tensor factorization approach for radiance fields achieving 10-100x speedup over NeRF
* **Points:** CP and VM decomposition, compact representations (<4-75MB), fast training (<10-30min)
* **Gap:** Limited to 3D/4D domains, no direct 2D matrix application

### K-Planes (Fridovich-Keil et al., 2023)

* **Contribution:** Interpretable planar factorization with natural space-time decomposition
* **Points:** (d choose 2) planes, 1000x compression, explicit representation options
* **Gap:** Single plane for 2D case limits factorization benefits

### SIREN (Sitzmann et al., 2020)

* **Contribution:** Periodic activation functions enable derivative access and smooth representations
* **Points:** sin(ω₀ · Wx + b) activation, special initialization, works across domains
* **Gap:** Sensitive to initialization, requires careful hyperparameter tuning

### Fourier Features (Tancik et al., 2020)

* **Contribution:** Overcomes spectral bias in coordinate-based MLPs via positional encoding
* **Points:** Simple Fourier mapping, NTK theoretical foundation, enables high-frequency learning
* **Gap:** Frequency selection remains empirical, may not capture all signal patterns

### GA-Planes (Sivgin et al., 2024)

* **Contribution:** First convex optimization framework for implicit neural volumes
* **Points:** Theoretical guarantees, generalizes existing representations, avoids local minima
* **Gap:** Recent work with limited empirical validation, potential expressiveness trade-offs

## Common Points Across Literature

### 1. Low-Rank Structure Assumption

* **Traditional methods:** Explicit low-rank via nuclear norm or SVD factorization
* **INR methods:** Implicit low-rank through network architecture and tensor factorization
* **Convergence:** Both approaches assume underlying low-dimensional structure in data

### 2. Sparse-to-Dense Reconstruction Challenge

* **Matrix completion:** Recover full matrix from sparse entry observations
* **INR reconstruction:** Learn continuous functions from sparse coordinate-value pairs
* **Common solution:** Regularization (nuclear norm vs. neural architecture) to ensure generalization

### 3. Interpolation and Smoothness

* **Traditional limitation:** No natural interpolation between observed entries
* **INR advantage:** Inherent smooth interpolation through neural function approximation
* **Applications:** Critical for applications requiring non-integer coordinate queries

### 4. Computational Trade-offs

* **Traditional methods:** Fast inference, complex optimization (semidefinite programming)
* **INR methods:** Fast optimization, potentially slower inference (network evaluation)
* **Factorization benefits:** TensoRF/K-Planes enable fast training and inference

### 5. Theoretical Understanding Gap

* **Traditional methods:** Strong theoretical foundations (incoherence conditions, sample complexity)
* **INR methods:** Limited theoretical analysis, mostly empirical validation
* **Recent progress:** GA-Planes provides convex framework, bridging this gap

## Research Gaps and Opportunities

### 1. Direct 2D Matrix Applications

**Gap:** Most INR work focuses on 3D/4D radiance fields. Limited exploration of direct 2D matrix reconstruction applications.

**Opportunity:** Systematic evaluation of 3D INR methods (K-Planes, TensoRF) adapted to 2D matrix problems with standard image datasets.

### 2. Theoretical Analysis for Matrix INRs

**Gap:** Lack of theoretical analysis comparing INR matrix completion with traditional methods.

**Opportunity:** Establish sample complexity, approximation bounds, and convergence guarantees for INR-based matrix completion.

### 3. Convex vs. Non-convex Decoder Comparison

**Gap:** Limited systematic comparison of different decoder architectures for matrix reconstruction.

**Opportunity:** Compare linear decoders (convex), MLP decoders (non-convex), and hybrid approaches on standardized matrix completion benchmarks.

### 4. Interpolation Method Effects

**Gap:** Insufficient analysis of how different interpolation methods (bilinear, bicubic, neural) affect reconstruction quality.

**Opportunity:** Systematic study of interpolation strategies in tensor-factorized INRs for matrix reconstruction.

### 5. Hybrid Prior Integration

**Gap:** Limited exploration of combining traditional matrix priors (nuclear norm, sparsity) with neural representations.

**Opportunity:** Develop frameworks like LoREIN but specifically for matrix completion, integrating classical priors with INR benefits.

## Our Position

### Research Questions Addressed

1. **How do 3D INR methods perform when adapted to 2D matrix reconstruction?**
2. **What are the trade-offs between different decoder architectures (linear vs. MLP)?**
3. **How does interpolation method choice affect reconstruction quality?**
4. **Can we achieve better parameter efficiency than traditional matrix completion methods?**

### Challenges We Address

1. **3D-to-2D Adaptation Gap:** Adapting successful 3D INR methods to 2D matrix problems
2. **Theoretical Understanding:** Providing analysis of when and why INRs outperform traditional matrix completion
3. **Practical Applicability:** Demonstrating CPU-friendly, dataset-agnostic approaches using standard image datasets
4. **Architecture Design:** Systematic comparison of decoder choices and their trade-offs

### Extensions of Prior Work

* **Builds on TensoRF:** Adapts tensor factorization principles to 2D matrix decomposition
* **Extends K-Planes:** Applies planar factorization concepts to matrix completion
* **Leverages SIREN/Fourier features:** Compares positional encoding strategies for matrix coordinate representation
* **Incorporates GA-Planes insights:** Explores convex formulations for theoretical guarantees

### Hypothesis Validation Framework

Our research tests the **core hypothesis** that INRs can achieve superior matrix reconstruction accuracy with fewer parameters than traditional methods, particularly for matrices with underlying continuous structure. We validate through:

1. **Systematic benchmarking** on standard image datasets (CPU-friendly)
2. **Comparative analysis** with nuclear norm minimization and SVD-based methods
3. **Ablation studies** on architecture choices (decoder types, positional encoding)
4. **Effect of interpolation** strategies on reconstruction quality
5. **Nonconvex vs convex decoder** performance comparison
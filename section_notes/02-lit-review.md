

# Literature Review: Implicit Neural Representations for Matrix Reconstruction

## Overview

This literature review examines the intersection of **Implicit Neural Representations (INRs)** and **matrix reconstruction/completion**, focusing on adapting 3D INR methods (NeRF, K-Planes, TensoRF, Gaussian Splatting) to 2D matrix problems. We systematically analyze foundational INR work, tensor factorization methods, traditional matrix completion approaches, and recent reconstruction applications to identify research gaps and opportunities.

**Research Focus:** Comparing 2D fitting performances of (normally 3D) INRs such as K-Planes, GA-Planes, NeRF etc. Since everything is based on 2D reconstruction, any open-source image dataset can be used & the approach is CPU-friendly.

## CORE REFERENCE PAPER

**Kim & Fridovich-Keil (2025)** - "Grids Often Outperform Implicit Neural Representations" provides **direct validation** of our research hypothesis. Their systematic comparison across 2D and 3D reconstruction tasks demonstrates that:

* **Simple regularized grids** with interpolation train faster and achieve higher quality than INRs for most signals
* **INRs maintain advantage** only for signals with underlying lower-dimensional structure (e.g., shape contours)
* **Performance boundaries** are clearly defined, providing practical guidance for method selection

**Direct Relevance:** This is the closest prior work to our matrix reconstruction focus, validating that explicit grid methods can outperform implicit approaches in many scenarios.

## Key Research Areas

### 1. Foundational INR Methods

#### 1.1 Positional Encoding Breakthroughs

**Tancik et al. (NeurIPS 2020)** established that standard MLPs suffer from **spectral bias** - inability to learn high-frequency functions in low-dimensional domains. Their **Fourier feature mapping** γ(v) \= \[cos(2πBv), sin(2πBv)]^T enables MLPs to overcome this limitation by transforming the Neural Tangent Kernel into a stationary kernel with tunable bandwidth.

**Sitzmann et al. (NeurIPS 2020)** proposed **SIREN** networks using sine activation functions sin(ω₀ · Wx + b), providing an alternative that enables access to all derivatives and smooth signal representation without explicit positional encoding.

**Key Insight for Matrix Reconstruction:** Both approaches address the fundamental challenge of representing continuous functions from discrete coordinates, directly applicable to matrix completion where we want smooth interpolation between observed entries.

#### 1.2 Neural Radiance Fields Foundation

**Mildenhall et al. (ECCV 2020)** demonstrated how MLPs with positional encoding can represent complex 3D scenes as continuous 5D radiance fields, establishing the paradigm of coordinate-based neural representations.

**Müller et al. (SIGGRAPH 2022)** - **Best Technical Paper** - introduced **Instant-NGP** with multiresolution hash encoding, achieving orders of magnitude speedup while maintaining quality. This breakthrough encoding technique is relevant to all INR methods and demonstrates how architectural innovations can dramatically improve efficiency.

### 2. Explicit vs Implicit Representations

#### 2.1 Grid-Based Alternatives

**Yu et al. (CVPR 2022)** introduced **Plenoxels** - radiance fields without neural networks, using sparse 3D grids with spherical harmonics. They achieved **100x faster** optimization than NeRF with no quality loss, demonstrating that explicit representations can match neural quality.

**Kerbl et al. (SIGGRAPH 2023)** developed **3D Gaussian Splatting**, representing scenes with explicit 3D Gaussians and achieving **real-time rendering** (≥30 fps at 1080p). This revolutionary explicit approach provides an important comparison point for implicit methods.

#### 2.2 Hybrid Approaches

**Wang et al. (CVPR 2025)** proposed **MetricGrids**, combining multiple elementary metric grids in different spaces with high-order Taylor expansion terms. This bridges explicit and implicit methods, offering superior fitting accuracy through novel grid structures.

### 3. Tensor Factorization Methods for INRs

#### 3.1 TensoRF: Tensorial Decomposition

**Chen et al. (ECCV 2022)** revolutionized INR efficiency by modeling radiance fields as **4D tensors** and applying factorization:

* **CP Decomposition:** Factorizes tensors into rank-one components with compact vectors
* **Vector-Matrix (VM) Decomposition:** Novel factorization into vector and matrix factors
* **Performance:** 10-30 minute training vs. hours/days for NeRF, with <4-75MB models

**Direct Relevance to Matrix Problems:** TensoRF's tensor factorization techniques are immediately applicable to 2D matrix decomposition, providing efficient low-rank representations.

#### 3.2 K-Planes: Planar Factorization

**Fridovich-Keil et al. (CVPR 2023)** introduced elegant **planar factorization** using (d choose 2) planes for d-dimensional scenes:

* **4D Scenes:** 6 planes (3 spatial: xy,xz,yz + 3 spatio-temporal: xt,yt,zt)
* **Interpretability:** Static objects appear only in spatial planes, dynamic in spatio-temporal
* **Efficiency:** 1000x compression over full 4D grid

**Matrix Application:** For 2D matrices, this reduces to single plane representation, but the factorization principles and multi-scale approaches remain applicable.

### 4. Advanced INR Techniques

#### 4.1 Transfer Learning for INRs

**Vyas et al. (NeurIPS 2024)** introduced **STRAINER** - learning transferable features for INRs through shared encoder layers across multiple INRs with independent decoders. They achieved **≈+10dB** signal quality improvement, demonstrating that INR features can be made transferable and significantly accelerate convergence.

#### 4.2 Reconstruction Applications

**Shi et al. (TMLR 2024)** demonstrated **joint reconstruction** using INRs for multiple objects through Bayesian framework with latent variables. Common patterns assist individual reconstruction, showing superior performance with sparse observations.

### 5. Traditional Matrix Completion Foundations

#### 5.1 Nuclear Norm Minimization

**Candès & Recht (2009)** and **Recht (2011)** established theoretical foundations for matrix completion via **nuclear norm minimization**:

* **Problem:** Recover low-rank matrix from small fraction of entries
* **Method:** min ||X||\* subject to observed entries constraints
* **Guarantees:** Exact recovery under incoherence conditions with O(nr polylog(n)) samples

**Limitations for INR Context:**

1. **Discrete representation** - no interpolation between entries
2. **Storage requirements** - must explicitly store matrix dimensions
3. **Complex pattern capture** - struggles with non-linear relationships
4. **Fixed resolution** - cannot query non-integer coordinates

## Critical Analysis and Research Gaps

### 1. Identified Research Gaps

#### 1.1 Direct 2D Matrix Applications

**Gap:** Most INR work focuses on 3D/4D radiance fields. Limited exploration of direct 2D matrix reconstruction applications.
**Opportunity:** Systematic evaluation of 3D INR methods (K-Planes, TensoRF, Gaussian Splatting) adapted to 2D matrix problems with standard image datasets.

#### 1.2 Theoretical Analysis for Matrix INRs

**Gap:** Lack of theoretical analysis comparing INR matrix completion with traditional methods.
**Opportunity:** Establish sample complexity, approximation bounds, and convergence guarantees for INR-based matrix completion.

#### 1.3 Explicit vs Implicit Trade-offs

**Gap:** Limited systematic comparison of grid-based vs neural approaches for matrix reconstruction specifically.
**Opportunity:** The Kim & Fridovich-Keil work provides foundation, but focused analysis on matrix completion tasks is needed.

### 2. Common Points Across Literature

#### 2.1 Low-Rank Structure Assumption

* **Traditional methods:** Explicit low-rank via nuclear norm or SVD factorization
* **INR methods:** Implicit low-rank through network architecture and tensor factorization
* **Convergence:** Both approaches assume underlying low-dimensional structure in data

#### 2.2 Sparse-to-Dense Reconstruction Challenge

* **Matrix completion:** Recover full matrix from sparse entry observations
* **INR reconstruction:** Learn continuous functions from sparse coordinate-value pairs
* **Common solution:** Regularization (nuclear norm vs. neural architecture) to ensure generalization

#### 2.3 Interpolation and Smoothness

* **Traditional limitation:** No natural interpolation between observed entries
* **INR advantage:** Inherent smooth interpolation through neural function approximation
* **Applications:** Critical for applications requiring non-integer coordinate queries

#### 2.4 Computational Trade-offs

* **Traditional methods:** Fast inference, complex optimization (semidefinite programming)
* **INR methods:** Fast optimization, potentially slower inference (network evaluation)
* **Grid benefits:** Both fast training and inference, as demonstrated by Plenoxels

### 3. Validation of Research Hypotheses

Our literature review validates several key research directions:

1. **Grid vs INR Performance:** Kim & Fridovich-Keil directly support our hypothesis that explicit methods can outperform INRs
2. **Efficiency Benefits:** Plenoxels and Instant-NGP show massive speedups are possible with architectural innovations
3. **Transfer Learning Potential:** STRAINER demonstrates +10dB improvements through proper architectural design
4. **Factorization Relevance:** TensoRF and K-Planes provide direct techniques applicable to matrix problems

## Our Research Positioning

### Research Questions Addressed

1. **How do 3D INR methods perform when adapted to 2D matrix reconstruction?**
2. **What are the trade-offs between explicit grids and implicit neural representations?**
3. **Can architectural innovations from radiance fields improve matrix completion efficiency?**
4. **How does parameter efficiency compare between INR and traditional matrix completion methods?**

### Novel Contributions

1. **Systematic 2D Evaluation:** First comprehensive comparison of 3D INR architectures on 2D matrix reconstruction
2. **Architectural Adaptation:** Adapting K-Planes, TensoRF, and Gaussian Splatting specifically for matrix problems
3. **Efficiency Analysis:** CPU-friendly implementations leveraging 2D computational advantages
4. **Theoretical Framework:** Bridging INR and matrix completion theory

### Extensions of Prior Work

* **Builds on Kim & Fridovich-Keil:** Extends their grid vs INR analysis to matrix-specific tasks
* **Adapts TensoRF:** Applies tensor factorization principles to 2D matrix decomposition
* **Extends K-Planes:** Applies planar factorization concepts to matrix completion
* **Incorporates STRAINER insights:** Explores transfer learning for matrix reconstruction tasks

## Advanced INR Developments (2024-2025)

### 6. Recent Architectural Innovations

#### 6.1 Hash Encoding and Multi-Resolution Methods

**Instant-NGP (Müller et al., SIGGRAPH 2022)** - **Best Technical Paper** - introduced multiresolution hash encoding that achieved orders of magnitude speedup. The key innovation replaces large MLPs with compact hash tables at multiple resolution levels, with collision disambiguation through multiresolution structure.

**MetricGrids (Wang et al., CVPR 2025)** extends this concept by combining multiple elementary metric grids in different spaces with high-order Taylor expansion terms. This approach bridges explicit and implicit methods, offering superior fitting accuracy through novel grid structures with hash encoding at different sparsities.

**Direct Matrix Applications**: Hash encoding could dramatically accelerate coordinate-to-value mappings in 2D matrix reconstruction, while multi-resolution structures naturally handle different detail levels in matrices.

#### 6.2 Activation Function and Initialization Advances

**WINNER (Chandravamsi et al., 2024)** addresses a fundamental SIREN limitation: spectral bottleneck when frequency support misaligns with target signals. Their Weight Initialization with Noise approach uses adaptive Gaussian noise based on spectral centroid, achieving state-of-the-art audio fitting with significant gains in image/3D tasks.

**MIRE (CVPR 2025)** introduces matched activation functions through dictionary learning with seven activation atoms (RC, RRC, PSWF, Sinc, Gabor, Gaussian, Sinusoidal). This eliminates exhaustive activation parameter search while improving performance across multiple tasks.

**Matrix Relevance**: Spectral analysis is directly applicable to matrix frequency content, while activation matching could suit different matrix characteristics without hyperparameter search.

#### 6.3 Hybrid Explicit-Implicit Architectures  

**RadSplat (Niemeyer et al., 2024)** from Google Research demonstrates exceptional hybrid performance by combining radiance field initialization with Gaussian splatting rendering, achieving 900+ FPS while maintaining quality on challenging scenes.

**Plenoxels (Yu et al., CVPR 2022)** showed that explicit sparse grids with spherical harmonics can achieve 100x faster optimization than NeRF with no quality loss, proving neural networks aren't always necessary for high-quality representation.

**Matrix Implications**: Hybrid approaches could initialize explicit grids using implicit neural methods, then switch to efficient explicit evaluation - highly relevant to our K-Planes vs. NeRF comparisons.

### 7. Theoretical Foundation Advances

#### 7.1 Rank-Based Understanding

**Razin (2024 PhD Thesis)** establishes rank as fundamental to deep learning theory, showing that gradient-based training induces implicit low-rank regularization that facilitates generalization on natural data. The work demonstrates strong connections between neural networks and tensor factorizations.

**Borsoi et al. (2024)** provide a unified framework using tensor decompositions to explain neural network expressivity, learnability, generalization, and identifiability. Their work connects multiple research communities through strong mathematical foundations.

**Matrix Completion Relevance**: Low-rank bias naturally aligns with matrix completion assumptions, and implicit regularization could eliminate need for explicit rank constraints.

#### 7.2 Low-Rank Tensor-INR Connections

**Cheng et al. (2025)** directly connect low-rank tensor methods to INRs through CP-based tensor functions with variational Schatten-p quasi-norm regularization. They provide theoretical guarantees on excess risk bounds and show CP decomposition offers more interpretable tensor structure than Tucker methods.

**Hamreras et al. (2025)** argue in a position paper that tensorization deserves wider adoption, highlighting bond indices that create new latent spaces absent in conventional networks, enabling mechanistic interpretability.

**Direct Applications**: These provide theoretical foundation and practical techniques directly applicable to matrix reconstruction, with interpretability benefits through tensor structure.

### 8. Comprehensive Survey Evidence

#### 8.1 Current State Assessment

**Essakine et al. (2024)** provide a comprehensive INR survey establishing clear taxonomy and demonstrating INR advantages: resolution independence, memory efficiency, and superior performance on complex inverse problems. Matrix completion clearly fits as a complex inverse problem.

**Wu et al. (2024)** survey 3D Gaussian splatting advances, systematically classifying methods by functionality and highlighting explicit representation advantages: interpretability, direct manipulation, and real-time performance through rasterization.

**Convergent Evidence**: Both surveys validate our research direction - INRs excel at inverse problems (matrix completion), while explicit representations offer interpretability and efficiency benefits.

### 9. Specialized Domain Adaptations

#### 9.1 Positional Encoding Innovations

**FreSh (ICLR 2025)** demonstrates that initial frequency spectrum correlates with final performance, enabling automatic hyperparameter selection that matches model output spectrum to target spectrum with minimal computational overhead.

**Geographic Encoding (Rußwurm et al., ICLR 2024)** shows domain-specific encoding matters by using spherical harmonic basis functions for geographic data, eliminating pole artifacts from rectangular coordinate assumptions.

**Matrix Applications**: These validate that encoding should match data characteristics - matrix reconstruction may benefit from matrix-specific encodings based on frequency analysis.

#### 9.2 Domain Extension Evidence

**TabINR (2025)** extends INRs to tabular data, showing continuous representations valuable even for discrete, heterogeneous data types. This validates INR applicability beyond traditional continuous signals.

**Medical Applications (Hendriks et al., 2025)** demonstrate INRs with spatial regularization achieve superior accuracy on high-dimensional parameter estimation from noisy diffusion MRI data, outperforming traditional methods.

**Matrix Relevance**: Domain extensions show INRs handle diverse data types effectively, while noise robustness is crucial for real-world matrix completion.

## Critical Literature Analysis

### 10. Validated Research Hypotheses

The 2024-2025 literature provides strong validation for our core hypotheses:

1. **Explicit vs. Implicit Trade-offs** - Plenoxels, RadSplat, and 3DGS surveys confirm explicit methods can match or exceed implicit quality while offering interpretability and efficiency benefits.

2. **Architectural Specialization** - MIRE, geographic encoding, and TabINR demonstrate domain-specific architectures outperform generic approaches.

3. **Hybrid Approaches** - RadSplat and MetricGrids show combining explicit and implicit methods achieves superior performance.

4. **Low-Rank Connections** - Razin, Borsoi, and Cheng establish strong theoretical foundations connecting neural networks, tensors, and low-rank methods.

5. **Efficiency-Quality Balance** - Instant-NGP, WINNER, and pruning methods demonstrate dramatic efficiency improvements while maintaining quality.

### 11. Identified Research Opportunities

**Theoretical Gaps**:
- Limited matrix completion-specific theoretical analysis
- Insufficient study of 2D-optimized architectures  
- Missing systematic efficiency analysis for resource-constrained environments

**Practical Gaps**:
- Most architectural innovations designed for 3D applications
- Limited exploration of hybrid explicit-implicit matrix methods
- Insufficient evaluation of CPU-friendly implementations

**Integration Opportunities**:
- Hash encoding + K-Planes for multi-resolution matrix reconstruction
- WINNER + matrix spectral analysis for adaptive initialization
- Hybrid explicit-implicit pipelines optimized for 2D domains

### 12. Common Points Across 2024-2025 Literature

#### 12.1 Efficiency-Quality Trade-off Resolution
- **Pattern**: Consistent achievement of both efficiency and quality improvements
- **Methods**: Hash encoding (orders of magnitude speedup), pruning (50% model reduction), adaptive initialization (eliminates hyperparameter search)
- **Matrix Relevance**: Critical for practical matrix completion systems

#### 12.2 Domain-Specific Architectural Design
- **Trend**: Moving from universal architectures to domain-optimized approaches  
- **Examples**: Geographic encoding, tabular INRs, medical applications, activation function matching
- **Implication**: Matrix reconstruction deserves specialized architectural consideration

#### 12.3 Theoretical Understanding Advancement
- **Progress**: From empirical success to solid theoretical foundations
- **Contributors**: Rank theory, tensor decomposition connections, spectral analysis
- **Need**: Matrix completion-specific theoretical development

#### 12.4 Hybrid Method Effectiveness
- **Evidence**: RadSplat, MetricGrids, pruned networks consistently outperform pure approaches
- **Pattern**: Combining initialization robustness with evaluation efficiency
- **Application**: Natural fit for our explicit-implicit matrix reconstruction comparisons

## Literature Summary

We have systematically analyzed **23+ high-quality papers** from top-tier venues (SIGGRAPH, NeurIPS, CVPR, ICLR, ECCV) published in 2020-2025, providing comprehensive theoretical and empirical foundation for INR-based matrix reconstruction. The literature demonstrates remarkable convergence supporting our research direction:

**Theoretical Validation**: Rank-based theory (Razin), tensor decomposition foundations (Borsoi), and direct tensor-INR connections (Cheng) provide strong theoretical justification for INR-based matrix completion.

**Architectural Evidence**: Hash encoding breakthroughs (Instant-NGP), hybrid methods (RadSplat), and domain-specific adaptations (geographic encoding) validate specialized architectural approaches for different domains.

**Efficiency Advances**: Dramatic speedups (100x-1000x) achieved through architectural innovations while maintaining or improving quality, making practical matrix reconstruction feasible.

**Research Positioning**: Our work sits at the intersection of multiple validated trends - efficiency optimization, architectural specialization, explicit-implicit hybridization, and theoretical grounding - with clear opportunities for novel contributions in the underexplored 2D matrix domain.

**Key Finding**: The convergence of evidence from theoretical advances, architectural innovations, and systematic evaluations strongly supports our hypothesis that specialized approaches combining explicit and implicit methods can achieve superior matrix reconstruction performance while providing interpretability and computational efficiency.


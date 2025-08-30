# Tensor Factorization Methods for Neural Radiance Fields

## 1. TensoRF: Tensorial Radiance Fields (Chen et al., 2022)
**Paper:** ECCV 2022 (Most influential ECCV'22 papers #2)  
**Authors:** Anpei Chen*, Zexiang Xu*, Andreas Geiger, Jingyi Yu, Hao Su  
**Venue:** ECCV 2022  
**URL:** https://arxiv.org/abs/2203.09517  
**Project:** https://apchenstu.github.io/TensoRF/

### Key Contributions
- **Problem:** NeRF requires extensive computation due to MLP evaluations
- **Insight:** Model radiance field as 4D tensor and factorize into compact low-rank components
- **Method:** CP decomposition and novel Vector-Matrix (VM) decomposition for tensor factorization
- **Impact:** Fast reconstruction (<30min CP, <10min VM) with better quality and smaller models (<4MB CP, <75MB VM)

### Technical Details
- **CP Decomposition:** Factorizes tensor into rank-one components with compact vectors
- **VM Decomposition:** Relaxes low-rank constraints for two modes, factorizes into vector and matrix factors
- **Performance:** TensoRF-CP: <30min training, <4MB model; TensoRF-VM: <10min training, <75MB model
- **Quality:** Outperforms NeRF in PSNR while being much faster

### Relevance to Matrix Reconstruction
- **Direct relevance:** Tensor factorization techniques directly applicable to 2D matrix decomposition
- **Low-rank structure:** CP and VM decompositions are fundamental matrix completion techniques
- **Efficiency:** Shows how factorization enables compact, fast representations
- **2D adaptation:** Methods can be adapted from 4D (3D space + time) to 2D (matrix entries)

## 2. K-Planes: Explicit Radiance Fields in Space, Time, and Appearance (Fridovich-Keil et al., 2023)
**Paper:** CVPR 2023  
**Authors:** Sara Fridovich-Keil*, Giacomo Meanti*, Frederik Rahbæk Warburg, Benjamin Recht, Angjoo Kanazawa  
**Venue:** CVPR 2023  
**URL:** https://arxiv.org/abs/2301.10241  
**Project:** https://sarafridov.github.io/K-Planes/

### Key Contributions
- **Problem:** Extending tensor factorization to arbitrary dimensions while maintaining interpretability
- **Insight:** Use (d choose 2) planes to represent d-dimensional scenes with natural space-time decomposition
- **Method:** Planar factorization using 6 planes for 4D volumes (3 space + 3 space-time planes)
- **Impact:** 1000x compression over full 4D grid, interpretable space-time decomposition, fast optimization

### Technical Details
- **Planar Factorization:** For d dimensions, use (d choose 2) 2D planes
- **4D Dynamic Volumes:** 6 planes total - 3 for space (xy, xz, yz) + 3 for space-time (xt, yt, zt)
- **Linear Decoder:** Can use linear decoder instead of MLP for fully explicit representation
- **Interpretability:** Static objects appear only in space planes, dynamic objects in space-time planes

### Relevance to Matrix Reconstruction
- **2D Application:** For 2D matrices, would use (2 choose 2) = 1 plane (the matrix itself) - but concept extends to higher-order tensors
- **Factorization Principles:** Planar decomposition principles applicable to matrix factorization
- **Explicit Representation:** Linear decoder approach relevant for matrix completion without MLPs
- **Multi-scale:** Can incorporate multi-resolution structure

## 3. Multiscale Tensor Decomposition and Rendering Equation Encoding (Han & Xiang, 2023)
**Paper:** Published in conference proceedings  
**Authors:** Kang Han, Wei Xiang  
**Venue:** Conference publication  
**URL:** https://arxiv.org/abs/2303.03808

### Key Contributions
- **Problem:** Improving view synthesis quality through better tensor decomposition
- **Insight:** Multiscale tensor decomposition represents scenes from coarse to fine scales
- **Method:** Neural Radiance Feature Field (NRFF) with multiscale representation and anisotropic spherical Gaussian mixture
- **Impact:** >1dB PSNR improvement on NeRF and NSVF datasets, significant improvement on Tanks & Temples

### Technical Details
- **Multiscale Representation:** Organizes learnable features across multiple scales
- **Benefits:** More accurate scene reconstruction, faster convergence vs single-scale
- **Rendering Equation Encoding:** Encodes rendering equation in feature space using anisotropic spherical Gaussians
- **Performance:** Substantial improvements on both synthetic and real-world datasets

### Relevance to Matrix Reconstruction
- **Multiscale Structure:** Directly applicable to hierarchical matrix decomposition
- **Feature Organization:** Coarse-to-fine representation relevant for matrix completion at different resolutions
- **Accuracy Improvements:** Techniques for improving reconstruction quality applicable to matrix settings
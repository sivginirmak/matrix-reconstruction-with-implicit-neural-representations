# Implicit Neural Representations (INRs) - Fundamental Papers

## 1. Fourier Features Let Networks Learn High Frequency Functions (Tancik et al., 2020)
**Paper:** NeurIPS 2020 (spotlight)  
**Authors:** Matthew Tancik*, Pratul P. Srinivasan*, Ben Mildenhall*, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, Ren Ng  
**Venue:** NeurIPS 2020  
**URL:** https://arxiv.org/abs/2006.10739

### Key Contributions
- **Problem:** Standard MLPs fail to learn high-frequency functions in low-dimensional domains due to spectral bias
- **Insight:** Fourier feature mapping can overcome spectral bias by transforming the Neural Tangent Kernel (NTK) into a stationary kernel with tunable bandwidth  
- **Method:** Simple Fourier feature mapping: γ(v) = [cos(2πBv), sin(2πBv)]^T where B is sampled from Gaussian N(0,σ²)
- **Impact:** Fundamental technique enabling coordinate-based MLPs for computer vision and graphics tasks

### Technical Details
- Uses Neural Tangent Kernel theory to explain why standard MLPs fail on high-frequency functions
- Fourier features transform effective NTK into stationary kernel with tunable bandwidth
- Frequency selection approach greatly improves MLP performance on low-dimensional regression

### Relevance to Matrix Reconstruction
- **Direct relevance:** Provides theoretical foundation for positional encoding in INRs used for 2D signal representation
- **Matrix connection:** 2D images can be viewed as matrices, and this work shows how to represent them continuously
- **Performance implications:** Critical for capturing high-frequency details in matrix reconstruction tasks

## 2. SIREN: Implicit Neural Representations with Periodic Activation Functions (Sitzmann et al., 2020)
**Paper:** NeurIPS 2020  
**Authors:** Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, Gordon Wetzstein  
**Venue:** NeurIPS 2020  
**URL:** https://arxiv.org/abs/2006.09661

### Key Contributions
- **Problem:** ReLU networks struggle to represent complex natural signals and their derivatives
- **Insight:** Sine activation functions enable networks to represent complex signals and all their derivatives
- **Method:** Uses sine activations: sin(ω₀ · Wx + b) with special initialization scheme
- **Impact:** Enables fitting of complex signals including images, audio, and video with a single network

### Technical Details
- All derivatives of sine are sine functions → derivatives accessible at any order
- Special initialization: first layer weights U(-1/n, 1/n), hidden layers U(-√6/n_in/ω₀, √6/n_in/ω₀)
- Can fit natural images, 3D shapes, signed distance functions, and time-varying phenomena

### Relevance to Matrix Reconstruction  
- **Direct relevance:** Alternative to Fourier features for representing 2D matrices/images continuously
- **Smooth representations:** Sine activations provide smooth interpolation between matrix entries
- **Derivative access:** Important for incorporating smoothness constraints in matrix completion

## 3. Neural Radiance Fields (NeRF) - Mildenhall et al., 2020
**Paper:** ECCV 2020  
**Authors:** Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng  
**Venue:** ECCV 2020  
**URL:** https://arxiv.org/abs/2003.08934

### Key Contributions
- **Problem:** Novel view synthesis from sparse input views
- **Insight:** Represent scenes as continuous 5D radiance fields using MLPs
- **Method:** MLP F_Θ: (x,y,z,θ,φ) → (c,σ) with positional encoding and volume rendering
- **Impact:** Revolutionary approach to 3D scene representation and novel view synthesis

### Technical Details
- Uses positional encoding for spatial coordinates and viewing directions
- Volume rendering equation integrates radiance and density along rays
- Hierarchical sampling with coarse and fine networks

### Relevance to Matrix Reconstruction
- **Conceptual relevance:** Demonstrates power of INRs for continuous signal representation
- **Positional encoding:** Similar techniques applicable to 2D matrix coordinates
- **Continuous representation:** Shows how to move from discrete to continuous representations

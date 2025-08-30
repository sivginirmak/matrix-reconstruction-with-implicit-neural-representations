Please update based on this main idea and references:

The main idea is comparing 2D fitting performances of (normally 3D) INRs such as K-Planes, GA-Planes, NeRF etc. I had done some initial experiments in my paper (mostly to complement the theory). Since everything is based on 2D reconstruction, any open-source image dataset can be used & it is cpu-friendly.

 [https://arxiv.org/pdf/2506.11139](https://arxiv.org/pdf/2506.11139)

[https://openaccess.thecvf.com/content/CVPR2023/papers/Fridovich-Keil\_K-Planes\_Explicit\_Radiance\_Fields\_in\_Space\_Time\_and\_Appearance\_CVPR\_2023\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Fridovich-Keil_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance_CVPR_2023_paper.pdf)

[https://dl.acm.org/doi/abs/10.1145/3503250](https://dl.acm.org/doi/abs/10.1145/3503250) (classic nerf paper; can also compare Siren vs Fourier features)

* Nonconvex vs convex decoders
* effect of interpolation



# Research Concept & Direction

## Research Question

This research explores the application of Implicit Neural Representations (INRs) to the fundamental problem of 2D matrix reconstruction and fitting. INRs have shown remarkable success in representing continuous signals and complex spatial data, but their potential for matrix completion and reconstruction tasks remains underexplored.

The project aims to develop novel neural architectures that can learn continuous representations of matrices using only partial observations, leveraging the inherent structure and patterns within the data. By representing matrices as continuous functions through INRs, we hypothesize that we can achieve more efficient and accurate reconstruction compared to traditional discrete approaches.

A key focus will be on understanding how INRs capture and interpolate matrix patterns, particularly in scenarios with missing or corrupted data. The research will investigate various neural architectures, positional encodings, and training strategies specifically optimized for matrix reconstruction tasks.

## Core Hypothesis

Implicit Neural Representations can achieve superior matrix reconstruction accuracy while requiring fewer parameters compared to traditional matrix completion methods, particularly for matrices with underlying continuous structure.

### Your Hypothesis

Implicit Neural Representations can achieve superior matrix reconstruction accuracy while requiring fewer parameters compared to traditional matrix completion methods, particularly for matrices with underlying continuous structure.

### Assumptions

What existing assumptions in the literature are you addressing?

### Expected Impact

How will proving this hypothesis change the field?

## Approach

Describe how you will test your hypothesis and validate your claims.

## Success Metrics

* Primary metric and target threshold
* Secondary metric and target threshold
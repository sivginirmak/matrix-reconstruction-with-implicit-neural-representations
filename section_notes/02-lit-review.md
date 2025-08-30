Refine based on this

The main idea is comparing 2D fitting performances of (normally 3D) INRs such as K-Planes, GA-Planes, NeRF etc. I had done some initial experiments in my paper (mostly to complement the theory). Since everything is based on 2D reconstruction, any open-source image dataset can be used & it is cpu-friendly.

 

These resourc[https://arxiv.org/pdf/2506.11139](https://arxiv.org/pdf/2506.11139)

[https://openaccess.thecvf.com/content/CVPR2023/papers/Fridovich-Keil\_K-Planes\_Explicit\_Radiance\_Fields\_in\_Space\_Time\_and\_Appearance\_CVPR\_2023\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Fridovich-Keil_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance_CVPR_2023_paper.pdf)

[https://dl.acm.org/doi/abs/10.1145/3503250](https://dl.acm.org/doi/abs/10.1145/3503250) (classic nerf paper; can also compare Siren vs Fourier features)

* Nonconvex vs convex decoders
* effect of interpolation



\---



# Literature Review

## Initial Literature Notes

Recent work on INRs has demonstrated their effectiveness in representing complex signals like images, 3D shapes, and audio. Studies by Tancik et al. (2020) and Sitzmann et al. (2020) have established fundamental principles for frequency encoding and periodic activation functions in INRs, though primarily focusing on spatial and temporal signals.

Traditional matrix completion methods, including nuclear norm minimization and collaborative filtering, provide important baselines. However, these approaches often struggle with capturing complex patterns and typically require storing the full matrix dimensions explicitly.

## Key Papers

### Paper Title 1

* **Contribution:** Main finding and contribution
* **Points:** Core points made by this work
* **Gap:** What is missing or could be improved

### Paper Title 2

* **Contribution:** Main finding and contribution
* **Points:** Core points made by this work
* **Gap:** What is missing or could be improved

## Common Points Across Literature

1. Most common point across papers
2. Another prevalent point in the field

## Our Position

* **Challenges:** Which assumptions we are questioning, if any
* **Builds on:** Which prior work we extend
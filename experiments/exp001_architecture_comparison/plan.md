# Experiment Plan: INR Architecture Comparison for 2D Matrix Reconstruction

## Research Question
Does the choice of INR architecture (K-Planes vs NeRF) significantly impact reconstruction quality for 2D matrix fitting tasks, and which architectural components drive superior performance in the 2D domain?

## Background & Motivation
While Implicit Neural Representations (INRs) like NeRF and K-Planes have shown success in 3D scene reconstruction, their architectural principles may offer different advantages when adapted for 2D matrix fitting. K-Planes uses explicit planar factorization which may align better with 2D data structures, while NeRF relies on implicit MLP architectures. This experiment addresses the fundamental gap in understanding architectural transferability between 3D and 2D domains.

## Hypothesis

**Primary Hypothesis:** K-Planes architectures will demonstrate superior reconstruction quality compared to NeRF for 2D matrix reconstruction, achieving >5dB PSNR improvement due to explicit geometric bias toward planar structures inherent in 2D data.

**Secondary Hypotheses:**
- Linear decoders with K-Planes will match or exceed nonlinear MLP performance
- Multiplicative operations in K-Planes will outperform additive operations for 2D reconstruction
- Architecture differences will be statistically significant across multiple test images

## Experimental Design

### Variables
- **Independent Variables:** 
  - Architecture type (K-Planes multiplicative, K-Planes additive, NeRF ReLU, NeRF SIREN)
  - Decoder type (linear, nonconvex, convex)
  - Feature dimensions (32, 64, 128)
  - Resolution (32, 64, 128, 192, 256)
- **Dependent Variables:** 
  - PSNR (Peak Signal-to-Noise Ratio)
  - Parameter efficiency (params per PSNR unit)
  - Training time
- **Control Variables:** 
  - Training epochs (1000)
  - Loss function (MSE)
  - Optimizer (Adam)
  - Bias settings
  - Random seeds
- **Confounding Factors:** 
  - Hyperparameter differences between architectures
  - Implementation differences
  - Dataset characteristics

### Methodology
1. **Baseline Setup:** NeRF with ReLU activations, nonconvex decoder, 128 features
2. **Treatment Conditions:** 
   - K-Planes multiplicative with linear decoder
   - K-Planes additive with linear decoder  
   - NeRF with SIREN activations
   - Various decoder combinations
3. **Data Collection:** Automated experimental pipeline with multiple runs per configuration
4. **Sample Size:** 5 random seeds per configuration for statistical validity (total ~200 experiments)

## Evaluation Metrics

### Primary Metrics
- **PSNR:** Peak Signal-to-Noise Ratio, measuring reconstruction fidelity
- **Parameter Efficiency:** PSNR per parameter count ratio

### Secondary Metrics
- **Training Time:** Wall-clock time to convergence
- **Model Size:** Total trainable parameters
- **Convergence Rate:** Loss reduction over training epochs

### Statistical Tests
- **Paired t-test:** Compare PSNR between architectures
- **ANOVA:** Multi-group comparison across all conditions
- **Effect Size:** Cohen's d for practical significance
- **Significance Level:** α = 0.05
- **Power Analysis:** n=5 provides 80% power to detect 2dB PSNR differences

## Implementation Milestones

### Phase 1: Setup (Day 1)
- [x] Load and verify datasets (astronaut image, BSD100 samples)
- [ ] Implement experiment framework with statistical tracking
- [ ] Validate baseline NeRF reproduction

### Phase 2: Development (Day 1-2)
- [ ] Implement K-Planes architecture variants
- [ ] Implement systematic parameter sweeps
- [ ] Add proper statistical analysis pipeline

### Phase 3: Evaluation (Day 2)
- [ ] Run full experimental matrix (4 architectures × 3 decoders × 3 features × 5 seeds)
- [ ] Collect comprehensive metrics
- [ ] Statistical analysis with hypothesis testing
- [ ] Generate publication-quality visualizations

## Success Criteria

### Minimum Success
- ✅ Baseline NeRF reproduces expected PSNR performance  
- All architecture variants run without errors
- Results show statistical significance (p < 0.05)

### Target Success
- K-Planes achieves >5dB PSNR improvement over NeRF baseline
- Parameter efficiency demonstrates >2x improvement
- Consistent results across multiple test images

### Stretch Goals
- >10dB PSNR improvement with optimized configurations
- Clear architectural insights about 2D vs 3D domain differences
- Generalizable findings across diverse image types

## Risk Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|--------------------|
| Hyperparameter mismatch between architectures | Medium | High | Systematic grid search with fair comparison |
| Insufficient statistical power | Low | Medium | Use multiple random seeds and test images |
| Implementation bugs in architectures | Medium | High | Unit testing, comparison with reference code |
| Computational limitations | Low | Medium | Prioritize core comparisons, use smaller test images |

## Resources Required

### Computational
- **Hardware:** CPU-only (GitHub Actions runner)
- **Estimated Runtime:** 2-4 hours for full experimental matrix
- **Storage:** ~1GB for results and visualizations

### Data
- **Primary Dataset:** Scikit-image astronaut (512×512)
- **Secondary Dataset:** BSD100 test images, synthetic patterns

### Dependencies
- **Core:** PyTorch, NumPy, Matplotlib, Scikit-image
- **Statistics:** Pingouin, SciPy, Seaborn  
- **Utilities:** tqdm, Pandas

## Timeline

| Phase | Task | Deliverable |
|-------|------|-------------|
| 1 | Setup experiment framework | Working baseline comparison |
| 2 | Implement architecture variants | All models running correctly |
| 3 | Execute systematic evaluation | Complete results matrix |
| 4 | Statistical analysis | Hypothesis testing results |
| 5 | Documentation | Enhanced experiment runs section |

## Statistical Analysis Plan

### Primary Analysis
1. **Descriptive Statistics:** Mean PSNR ± std for each architecture
2. **Hypothesis Testing:** Paired t-tests between K-Planes and NeRF variants
3. **Effect Sizes:** Cohen's d to quantify practical significance
4. **Confidence Intervals:** 95% CI for mean PSNR differences

### Secondary Analysis
1. **Parameter Efficiency:** Scatter plots of PSNR vs model size
2. **Convergence Analysis:** Training curves and convergence rates
3. **Pareto Frontiers:** Efficiency analysis using existing `find_pareto` function

### Null Hypothesis Testing
- **H0:** No significant difference in PSNR between K-Planes and NeRF architectures
- **H1:** K-Planes achieves significantly higher PSNR than NeRF (one-tailed test)

## Notes
This experiment directly tests the core research hypothesis using systematic methodology. The implementation leverages existing code from `notes/some_examples.py` while adding proper statistical rigor through multiple runs, hypothesis testing, and effect size analysis.
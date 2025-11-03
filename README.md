# Documentation for Latent Space Constrained Bayesian Optimization

## Overview

This documentation provides comprehensive explanations of the implementation of "High-Dimensional Bayesian Optimisation with Large-Scale Constraints via Latent Space Gaussian Processes" (Maathuis et al., 2025).

The implementation includes three variants:
1. **SCBO**: Baseline method (Eriksson & Poloczek, 2021) - models each constraint independently
2. **PCA-GP SCBO**: Linear dimensionality reduction via Principal Component Analysis
3. **kPCA-GP SCBO**: Nonlinear dimensionality reduction via Kernel PCA

## Documentation Files

### 1. IMPLEMENTATION_GUIDE.md (Full Documentation)
**Purpose**: Complete technical documentation with detailed explanations

**Contents**:
- Cell-by-cell breakdown of implementation
- Mathematical foundations with equation references
- Paper citations for every major component
- Design rationale and implementation choices
- Complexity analysis
- Hyperparameter guidance

**Use when**: 
- Understanding the complete algorithm
- Learning how paper concepts translate to code
- Debugging implementation issues
- Extending the codebase

**Key sections**:
- GP Infrastructure (Section 2.1 → Cell 3)
- Trust Regions (Section 2.4 → Cell 4)
- PCA/kPCA Theory (Sections 3.1-3.2 → Cell 5)
- Thompson Sampling (Algorithm 1 → Cell 5)
- Main Loop (Algorithm 2 → Cell 6)

---

### 2. CODE_BLOCK_COMMENTS.md (Inline Comments)
**Purpose**: Ready-to-use comments for Jupyter notebook cells

**Contents**:
- Pre-written docstrings for each cell
- Concise summaries with paper references
- Key equation numbers and algorithm lines
- Implementation notes and warnings

**Use when**:
- Adding documentation to your notebook
- Creating presentation materials
- Quick reference while coding
- Teaching/explaining the implementation

**Format**: Copy-paste blocks directly before each notebook cell

**Example**:
```python
"""
Cell 3: Gaussian Process Infrastructure

Purpose: Implements GP surrogate models...
References: Section 2.1, Equations (2-8)
Key methods: fit_gp(), posterior_mean_std()
"""
```

---

### 3. QUICK_REFERENCE.md (Lookup Tables)
**Purpose**: Fast lookup of code-to-paper mappings

**Contents**:
- Function → Paper reference table
- Equation → Code implementation table  
- Algorithm pseudocode → Code location
- Section → Implementation mapping
- Hyperparameter decision guides
- Debugging checklists
- Performance optimization tips

**Use when**:
- Finding where specific equations are implemented
- Locating algorithm lines in code
- Choosing hyperparameters
- Troubleshooting convergence issues
- Optimizing performance

**Key tables**:
- Function/Class → Paper Reference (100+ entries)
- Algorithm 1 & 2 line-by-line mapping
- Equation (1-26) implementation locations
- Hyperparameter defaults with justifications

---

## Quick Start

### For Understanding the Implementation:
1. Read IMPLEMENTATION_GUIDE.md sequentially (Cells 1-7)
2. Follow along with the Jupyter notebook
3. Refer to QUICK_REFERENCE.md for specific questions

### For Documenting Your Own Notebook:
1. Open CODE_BLOCK_COMMENTS.md
2. Copy the comment block for each cell
3. Paste before the corresponding cell in your notebook
4. Customize as needed for your specific use case

### For Debugging/Optimization:
1. Check "Debugging Guide" in QUICK_REFERENCE.md
2. Follow the decision trees for your specific issue
3. Refer to "Performance Optimization Tips"
4. Consult IMPLEMENTATION_GUIDE.md for detailed context

---

## Code Structure Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Cell 1: Initialization                                      │
│ ├─ Seeds for reproducibility                                │
│ └─ Device selection (CPU/GPU)                               │
├─────────────────────────────────────────────────────────────┤
│ Cell 2: Problem Definition (Speed Reducer)                  │
│ ├─ 7 design variables                                       │
│ ├─ 11 black-box constraints                                 │
│ ├─ Integer handling (x3)                                    │
│ └─ LHS sampling                                             │
├─────────────────────────────────────────────────────────────┤
│ Cell 3: Gaussian Process Infrastructure                     │
│ ├─ GP model class (RBF kernel + ARD)                        │
│ ├─ Training via marginal likelihood                         │
│ ├─ Posterior predictions                                    │
│ └─ Feasibility utilities                                    │
├─────────────────────────────────────────────────────────────┤
│ Cell 4: Trust Region + Feasibility-First                    │
│ ├─ TrustRegion class (adaptive sizing)                      │
│ ├─ Feasible seed finding                                    │
│ └─ Feasibility-first acquisition (Phase 1)                  │
├─────────────────────────────────────────────────────────────┤
│ Cell 5: Model Builders + Thompson Sampling                  │
│ ├─ build_models_scbo() - baseline                           │
│ ├─ build_models_pca() - PCA reduction                       │
│ ├─ build_models_kpca() - kernel PCA                         │
│ └─ pick_candidate_cts() - CTS acquisition (Phase 2)         │
├─────────────────────────────────────────────────────────────┤
│ Cell 6: Main Optimization Loop                              │
│ ├─ Unified run_bo() function                                │
│ ├─ Switches between 3 methods                               │
│ ├─ Two-phase acquisition strategy                           │
│ └─ Trust region adaptation                                  │
├─────────────────────────────────────────────────────────────┤
│ Cell 7: Experiments & Visualization                         │
│ ├─ Run all 3 methods                                        │
│ ├─ Convergence plots                                        │
│ └─ Performance comparison                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts Explained

### 1. Two-Phase Acquisition Strategy

**Phase 1** (No feasible points yet):
- **Function**: `pick_candidate_feasibility_first()`
- **Goal**: Maximize probability of finding ANY feasible point
- **Method**: argmax ∏_{j=1}^G P(c_j(x) ≤ 0)
- **Reference**: Section 2.3, Equation (12)

**Phase 2** (≥1 feasible point exists):
- **Function**: `pick_candidate_cts()`  
- **Goal**: Optimize objective while maintaining feasibility
- **Method**: Constrained Thompson Sampling
- **Reference**: Algorithm 1

**Why this works**:
- Finding feasible region is often the first challenge
- Once found, can use more sophisticated acquisition
- Prevents wasting iterations on infeasible regions

### 2. Dimensionality Reduction for Constraints

**Problem**: Modeling G constraints requires G independent GPs
- Computational cost: O((G+1)N³)
- Memory: O((G+1)N²)
- Infeasible when G > 1000

**Solution**: Project to g-dimensional latent space where g ≪ G
- **PCA (Linear)**: Find principal directions of constraint variation
- **kPCA (Nonlinear)**: Use kernel trick for nonlinear manifolds

**Key Insight**: Constraints often correlated
- Multiple loadcases → similar failure modes
- Many constraints → few underlying physics principles
- Can capture with much lower dimension

**Implementation**:
```python
# Original: G GPs
for i in range(G):
    model_c[i] = fit_gp(X, C[:, i])

# Reduced: g GPs  
Z = PCA(n_components=g).fit_transform(C)  # C ∈ R^(N×G) → Z ∈ R^(N×g)
for i in range(g):
    model_z[i] = fit_gp(X, Z[:, i])
```

**Critical**: Must project back before feasibility check!
```python
Z_new = sample_from_latent_GPs()
C_new = reducer.inverse_transform(Z_new)  # Back to original space
feasible = all(C_new[:, j] <= 0)  # Check in original space
```

**Reference**: Section 3, Algorithm 2

### 3. Trust Regions for High Dimensions

**Challenge**: Global GP struggles when D > 10
- Curse of dimensionality
- Need too much data for global model

**Solution**: Local optimization within adaptive trust region
- Focus on promising subregion
- Expand when making progress
- Contract when stuck

**Mechanics**:
```python
# Sample candidates from hyperrectangle
halfspan = 0.5 * frac * (ub - lb)
candidates = center ± halfspan

# Update size based on success
if found_better:
    succ_count += 1
    if succ_count >= 3:  # 3 consecutive improvements
        frac *= 1.6  # Expand by 60%
else:
    fail_count += 1  
    if fail_count >= 3:  # 3 consecutive failures
        frac *= 0.5  # Contract by 50%
```

**Re-centering**: Always center at current best (not initial center)

**Reference**: Section 2.4, Eriksson et al. (2020)

---

## Common Questions

### Q: When should I use SCBO vs PCA-GP vs kPCA-GP?

**Use SCBO when**:
- G ≤ 20 (few constraints)
- Want exact constraint modeling
- Have sufficient computational resources
- Need fastest convergence in # iterations

**Use PCA-GP when**:
- G > 20 (many constraints)  
- Constraints have linear/near-linear structure
- Eigenvalue decay is rapid (Figure 6 in paper)
- Want interpretable reduction

**Use kPCA-GP when**:
- Constraints lie on nonlinear manifold
- PCA eigenvalue decay is slow
- Willing to tune γ hyperparameter
- G is moderate (pre-image approximation accurate)

**Paper findings**:
- Speed Reducer (G=11): All converge to same optimum, PCA 60% faster
- Aeroelastic (G=1786): Only PCA/kPCA feasible, SCBO runs out of memory

---

### Q: How do I choose the number of components (g)?

**Method 1: Eigenvalue threshold**
```python
from sklearn.decomposition import PCA
pca = PCA().fit(C)
eigenvalues = pca.explained_variance_
g = np.sum(eigenvalues > 0.01)  # Components with λ > 0.01
```

**Method 2: Variance explained**
```python
cumvar = np.cumsum(pca.explained_variance_ratio_)
g = np.argmax(cumvar >= 0.95) + 1  # 95% variance
```

**Method 3: Problem-specific**
- Speed Reducer (G=11): g=4 captures 99%+ variance
- Aeroelastic (G=1786): g=35 for 99% variance
- Rule of thumb: g ≈ G/10 to G/50 depending on correlation

**Paper guidance**: Figure 4 shows impact of varying g
- g too small (e.g., g=1): Misses feasible region
- g just right (g=4 for Speed Reducer): Good performance
- g too large (e.g., g=6): Marginal improvement, added cost

---

### Q: What if my problem has discrete/integer variables?

**Current implementation**: Handles via rounding
```python
if enforce_integer:
    X[:, integer_indices] = np.rint(X[:, integer_indices])
```

**When applied**:
1. After sampling from trust region
2. Before evaluation
3. After projection (if using latent space)

**Limitations**: 
- Naive rounding may miss optimal integer points
- Better: Mixed-integer BO methods (not implemented)

**For Speed Reducer**: x3 (number of teeth) must be integer
- Range: [17, 28] integers
- Rounding works well since range is small

---

### Q: How do I know if my GPs are trained properly?

**Check 1: Training loss**
```python
# During fit_gp(), monitor:
for i in range(iters):
    loss = -mll(out, y)
    if i % 20 == 0:
        print(f"Iter {i}: loss = {loss.item()}")
```
Should decrease consistently

**Check 2: Hyperparameters**
```python
# After training:
print("Length scales:", model.covar_module.base_kernel.lengthscale)
print("Noise:", model.likelihood.noise)
```
- Length scales: Should be between 0.01 and 10 (roughly)
- Noise: Should be small (< 0.1) for deterministic functions

**Check 3: Posterior predictions**
```python
# On training data:
with torch.no_grad():
    pred = model(X_train)
    mse = ((pred.mean - y_train) ** 2).mean()
print(f"Training MSE: {mse}")
```
Should be small (< 0.01 after standardization)

---

### Q: My optimization isn't converging. What should I check?

**Checklist**:

1. **Are any feasible points found?**
   ```python
   feas = np.all(C <= 0.0, axis=1)
   print(f"Feasible: {feas.sum()} / {len(feas)}")
   ```
   - If 0: Problem may be over-constrained
   - Solution: Relax constraints, try more initial samples

2. **Is trust region too small?**
   - Add tracking: `tr_sizes.append(tr.frac)` in run_bo()
   - If frac < 0.1: May have contracted too much
   - Solution: Increase min_frac or adjust shrink rate

3. **Is PCA reduction too aggressive?**
   ```python
   error = np.linalg.norm(C - C_reconstructed) / np.linalg.norm(C)
   print(f"Reconstruction error: {error:.1%}")
   ```
   - If > 10%: g too small
   - Solution: Increase n_components

4. **Are GPs underfitting?**
   - Very smooth predictions → may need more flexibility
   - Solution: Try different kernel, more training iterations

5. **Is acquisition function appropriate?**
   - Stuck in local optimum → may need more exploration
   - Solution: Increase n_cand, adjust TR parameters

---

### Q: How can I speed up the optimization?

**1. Use GPU acceleration**
```python
device = torch.device("cuda")  # Force GPU
```
Speedup: 2-5× for GP training

**2. Reduce GP training iterations**
```python
# Default: 100 iterations (conservative)
# Try: 50 iterations for faster experiments
mf, _ = fit_gp(X, f, iters=50)
```
Speedup: 2× with minor accuracy loss

**3. Use dimensionality reduction**
```python
# Instead of SCBO (G GPs):
run_bo(method="pca", n_components=min(10, G//10))
```
Speedup: 60%+ when G > 20

**4. Reduce candidate count**
```python
# Default: 4096
# Try: 1024 for faster iterations  
run_bo(n_cand=1024)
```
Speedup: 4× per iteration, may hurt convergence

**5. Batch parallelization** (not implemented)
- Evaluate multiple points per iteration
- Requires batch acquisition function
- Potential speedup: 5-10× if parallel evaluation possible

---

## Extending the Code

### Adding a New Test Problem

1. **Create problem class** (similar to Cell 2):
```python
class MyProblem:
    def __init__(self):
        self.dim = D
        self.lb = np.array([...])  # Lower bounds
        self.ub = np.array([...])  # Upper bounds
        self.n_constraints = G
    
    def evaluate(self, X):
        # Return (f, c) where f is objective, c is constraints
        # Constraints: c[:, j] <= 0 for feasibility
        pass
```

2. **Modify run_bo()** in Cell 6:
```python
# Change line:
problem = SpeedReducerProblem(...)
# To:
problem = MyProblem(...)
```

3. **Adjust hyperparameters** as needed

### Adding a New Dimensionality Reduction Method

1. **Create reducer class** (similar to Cell 5):
```python
class MyReducer:
    def fit_transform(self, C):
        # C ∈ R^(N×G) → Z ∈ R^(N×g)
        pass
    
    def transform(self, C):
        # Project new C to latent space
        pass
    
    def inverse_transform(self, Z):
        # Reconstruct C from Z
        pass
```

2. **Add builder function** in Cell 5:
```python
def build_models_mymethod(X, f, C, n_components=4):
    reducer = MyReducer(n_components)
    Z = reducer.fit_transform(C)
    mf, _ = fit_gp(X, f)
    models_z = [fit_gp(X, Z[:, i])[0] for i in range(g)]
    return mf, models_z, reducer
```

3. **Modify run_bo()** to include new method:
```python
elif method == "mymethod":
    mf, mcs, reducer = build_models_mymethod(X, f_vals, C_vals, ...)
```

### Adding a New Acquisition Function

1. **Implement function** (similar to Cell 5):
```python
@torch.no_grad()
def my_acquisition(model_f, models_c, cand_t, reducer=None):
    # Compute acquisition value for each candidate
    # Return index of best candidate
    pass
```

2. **Integrate into run_bo()** in Cell 6:
```python
# In acquisition section:
if use_my_acquisition:
    best_idx = my_acquisition(mf, mcs, cand_t, reducer)
```

---

## Troubleshooting

### Common Errors and Solutions

**Error: "CUDA out of memory"**
- **Cause**: Too many GPs or too much data
- **Solution**: Use PCA/kPCA to reduce GPs, or reduce n_init/iters

**Error: "Matrix not positive definite"**
- **Cause**: Ill-conditioned kernel matrix
- **Solution**: Increase cholesky_jitter in fit_gp(), check for duplicate points

**Error: "No feasible points found"**
- **Cause**: Problem may be over-constrained, or TR stuck in infeasible region
- **Solution**: Try feasible seeding, increase n_cand, relax constraints

**Warning: "Reconstruction error high"**
- **Cause**: g too small for PCA/kPCA
- **Solution**: Increase n_components

**Issue: Very slow convergence**
- **Cause**: Could be many reasons (see Q&A above)
- **Solution**: Follow debugging checklist in QUICK_REFERENCE.md

---

## Performance Benchmarks

### Speed Reducer (D=7, G=11, N=20, iters=100)

| Method | Best f | Time (s) | Speedup | GPs Trained |
|--------|--------|----------|---------|-------------|
| SCBO | 2996.35 | 501 | 1.0× | 12 per iter |
| PCA-GP (g=4) | 2996.35 | 201 | 2.5× | 5 per iter |
| kPCA-GP (g=4) | 2996.35 | 217 | 2.3× | 5 per iter |

**Reference**: Table 1 in paper (20 runs average)

### Aeroelastic Tailoring (D=108, G=1786, N=416, iters=2000)

| Method | Best f | Time | Memory | Status |
|--------|--------|------|--------|--------|
| SCBO | - | - | >64GB | Out of memory |
| PCA-GP (g=35) | 494.5 | ~45min | ~8GB | Success |
| kPCA-GP (g=35) | 507.1 | ~52min | ~10GB | Success |

**Reference**: Section 4.2, Figure 7 in paper

**Key Insight**: PCA/kPCA enables problems impossible with SCBO

---

## Further Reading

### Primary Papers
1. **Maathuis et al. (2025)**: Main paper - latent space method
2. **Eriksson & Poloczek (2021)**: SCBO - baseline constrained BO
3. **Eriksson et al. (2020)**: TuRBO - trust region method
4. **Rasmussen & Williams (2006)**: GPs for Machine Learning - foundational theory

### Related Work
- **Hernández-Lobato et al. (2017)**: Thompson Sampling for BO
- **Schölkopf et al. (1998)**: Kernel PCA theory
- **Lemonge et al. (2010)**: Speed Reducer problem formulation

### Recommended Learning Path

**Beginner** (No BO background):
1. Rasmussen & Williams Ch 1-2 (GP basics)
2. Frazier (2018) tutorial on BO
3. This IMPLEMENTATION_GUIDE.md sections 2.1-2.3

**Intermediate** (Know BO basics):
1. Eriksson et al. (2020) - TuRBO paper
2. Eriksson & Poloczek (2021) - SCBO paper
3. This IMPLEMENTATION_GUIDE.md sections 2.4-3.4

**Advanced** (Want to extend):
1. Maathuis et al. (2025) - full paper
2. This QUICK_REFERENCE.md debugging/optimization sections
3. Experiment with your own problems

---

## Contact and Contributions

**For questions about this implementation**:
- Check QUICK_REFERENCE.md first
- Review relevant section in IMPLEMENTATION_GUIDE.md
- Consult debugging guide

**For questions about the paper**:
- Contact: h.f.maathuis@tudelft.nl (first author)
- arXiv: 2412.15679

**For bug reports or improvements**:
- Document the issue with minimal reproducible example
- Check if already addressed in documentation
- Reference specific section/cell in your report

---

## Acknowledgments

This documentation was created to accompany the implementation of:

> Maathuis, H.F., De Breuker, R., & Castro, S.G.P. (2025). 
> High-Dimensional Bayesian Optimisation with Large-Scale Constraints 
> via Latent Space Gaussian Processes. arXiv preprint arXiv:2412.15679.

Implementation uses:
- **GPyTorch**: Efficient GP library (Gardner et al., 2018)
- **PyTorch**: Deep learning framework (Paszke et al., 2019)
- **Scikit-learn**: PCA/kPCA implementations (Pedregosa et al., 2011)

---

## File Organization

```
latent_bo_docs/
├── README.md (this file)
│   └── Overview and navigation guide
│
├── IMPLEMENTATION_GUIDE.md
│   ├── Complete technical documentation
│   ├── Cell-by-cell breakdown
│   ├── Mathematical foundations
│   └── ~1000 lines, highly detailed
│
├── CODE_BLOCK_COMMENTS.md
│   ├── Ready-to-use docstrings
│   ├── Concise cell summaries  
│   ├── Copy-paste into notebook
│   └── ~750 lines, practical focus
│
└── QUICK_REFERENCE.md
    ├── Lookup tables
    ├── Debugging guides
    ├── Hyperparameter decisions
    └── ~500 lines, reference format
```

**Total documentation**: ~2,250 lines across 4 files

**Estimated reading time**:
- README: 15 minutes (overview)
- QUICK_REFERENCE: 30 minutes (skim tables)
- CODE_BLOCK_COMMENTS: 1 hour (with code)
- IMPLEMENTATION_GUIDE: 3-4 hours (deep dive)

---

## Version History

- **v1.0** (Nov 2025): Initial comprehensive documentation
  - All 7 cells documented
  - 3 methods covered (SCBO, PCA-GP, kPCA-GP)
  - Complete paper-to-code tracing
  - Debugging and optimization guides

---

*End of README*

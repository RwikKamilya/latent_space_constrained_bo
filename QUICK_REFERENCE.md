# Quick Reference: Code to Paper Mapping

## Function/Class → Paper Reference Table

| Code Element | Location | Paper Reference | Page/Section | Key Equation/Algorithm |
|--------------|----------|-----------------|--------------|------------------------|
| **CELL 1: Initialization** |
| `set_global_seed()` | Cell 1 | Standard practice | Multiple runs (Sec 4.2) | - |
| `device` selection | Cell 1 | Computational efficiency | Section 3.4 | Complexity O((g+1)N³) |
| **CELL 2: Problem Definition** |
| `SpeedReducerProblem` | Cell 2 | Lemonge et al. (2010) | Section 4.1, Table 3 | Full formulation |
| `.evaluate()` | Cell 2 | Problem definition | Section 4.1 | Equation (1): c_j(x) ≤ 0 |
| `.sample_lhs()` | Cell 2 | Initial sampling | Section 4.2 | "via Latin Hypercube Sampling" |
| `.enforce_integer_teeth` | Cell 2 | Mixed-integer problem | Section 4.1 | "x3 is integer" |
| **CELL 3: GP Infrastructure** |
| `standardize()` | Cell 3 | Numerical stability | Rasmussen & Williams (2006) | Standard preprocessing |
| `ExactGP` class | Cell 3 | GP framework | Section 2.1 | Equation (2): f(x) ~ GP(m,k) |
| RBF Kernel | Cell 3 | Kernel choice | Section 2.1, Equation (3) | k(x,x') = σ²exp(-½Σ(x_i-x'_i)²/l_i²) |
| ARD (length scales) | Cell 3 | Feature relevance | Equation (3) | Different l_i per dimension |
| `fit_gp()` | Cell 3 | Training | Section 2.1, Equations (4-5) | Marginal likelihood maximization |
| Adam optimizer | Cell 3 | Hyperparameter learning | Equation (5) | ∂/∂θ_j log p(f\|D,θ) |
| `best_feasible_value()` | Cell 3 | Progress tracking | Throughout | min f(x) s.t. c_j(x) ≤ 0 |
| `least_violation_index()` | Cell 3 | Infeasible handling | Section 2.3 | argmin Σ max(c_j,0) |
| `posterior_mean_std()` | Cell 3 | Prediction | Section 2.1, Equations (7-8) | μ(x), σ(x) |
| **CELL 4: Trust Region** |
| `TrustRegion` class | Cell 4 | High-dim BO | Section 2.4 | TuRBO (Eriksson 2020) |
| `.sample()` | Cell 4 | Local candidates | Algorithm 2, line 7 | Sample from TR |
| `.step()` | Cell 4 | Adaptive sizing | Eriksson 2020 | Expand/contract rules |
| Hyperparameters | Cell 4 | Default values | TuRBO paper | init=0.8, grow=1.6, shrink=0.5 |
| `cheap_filter()` | Cell 4 | Efficient filtering | Not explicit | Problem-specific speedup |
| `find_feasible_seed()` | Cell 4 | Bootstrap | Section 4.2 | "efficiently identified" |
| `pick_candidate_feasibility_first()` | Cell 4 | Phase 1 acquisition | Section 2.3, Equation (12) | argmax ∏ P(c_j ≤ 0) |
| Probability of feasibility | Cell 4 | PoF computation | Equation (12) | Φ(-μ/σ) |
| **CELL 5: Model Builders** |
| `build_models_scbo()` | Cell 5 | Baseline method | Eriksson & Poloczek (2021) | G independent GPs |
| **PCA Components** |
| `PCAReducer` class | Cell 5 | Linear reduction | Section 3.1 | Full PCA algorithm |
| `.fit_transform()` | Cell 5 | Projection | Equation (20) | Z = CΨ_g |
| `.inverse_transform()` | Cell 5 | Reconstruction | Section 3.1 | C̃ = ZΨ_g^T |
| Covariance matrix | Cell 5 | PCA step | Equation (17) | C = (1/(N-1))C̄^T C̄ |
| Eigendecomposition | Cell 5 | PCA step | Equations (18-19) | C = ΨΛΨ^T |
| **kPCA Components** |
| `KPCAReducer` class | Cell 5 | Nonlinear reduction | Section 3.2 | Kernel PCA algorithm |
| Kernel function | Cell 5 | RBF kernel | Equations (23, 28) | k(c,c') = exp(-γ\\|c-c'\\|²) |
| Kernel matrix | Cell 5 | kPCA step | Equation (24) | K_ij = ⟨φ(c_i),φ(c_j)⟩ |
| Projection | Cell 5 | kPCA step | Equation (26) | v^T φ(c) = Σ α_i k(c_i,c) |
| `build_models_pca()` | Cell 5 | Main contribution | Algorithm 2 | Latent GP construction |
| `build_models_kpca()` | Cell 5 | Nonlinear variant | Section 3.2 | Kernel version |
| **Thompson Sampling** |
| `pick_candidate_cts()` | Cell 5 | Phase 2 acquisition | Algorithm 1 | Constrained TS |
| Sampling from posterior | Cell 5 | TS theory | Hernández-Lobato 2017 | Draw from p(θ\|D) |
| Feasibility check | Cell 5 | Constraint satisfaction | Algorithm 1 | c̃_j(x) ≤ 0 ∀j |
| Objective minimization | Cell 5 | Optimization | Algorithm 1 | argmin f̃ if feasible |
| Violation minimization | Cell 5 | Infeasible handling | Algorithm 1 | argmin Σ max(c̃_j,0) |
| Inverse projection | Cell 5 | Critical step | Section 3.3 | "checked in original space" |
| **CELL 6: Main Loop** |
| `run_bo()` | Cell 6 | Complete algorithm | Algorithm 2 | Full pseudocode |
| DoE generation | Cell 6 | Initialization | Algorithm 2, line 2 | Compute DoE D_0 |
| Feasible seeding | Cell 6 | Bootstrap | Section 4.2 | Optional seed finding |
| TR initialization | Cell 6 | Setup | Eriksson 2020 | Center at best/least-infeas |
| Model building loop | Cell 6 | Core iteration | Algorithm 2, lines 5-6 | Compute P_k, fit GPs |
| Candidate generation | Cell 6 | Sampling | Algorithm 2, line 7 | Sample from TR |
| Acquisition switch | Cell 6 | Two-phase strategy | Section 3.3 | Feasibility-first vs CTS |
| Evaluation | Cell 6 | Function call | Algorithm 2, line 8 | Evaluate x_+ |
| Dataset update | Cell 6 | Learning | Algorithm 2, line 9 | D_{k+1} = D_k ∪ {...} |
| TR update | Cell 6 | Adaptation | Algorithm 2, line 10 | Update TuRBO state |
| Success detection | Cell 6 | Progress metric | Section 4.1 | Better feasible found? |
| TR re-centering | Cell 6 | Focus search | Eriksson 2020 | Center at incumbent |
| **CELL 7: Experiments** |
| `plot_convergence()` | Cell 7 | Visualization | Figure 3 | Convergence curves |
| Final comparisons | Cell 7 | Results | Table 1 | Best/median/avg/worst |
| f* reference line | Cell 7 | Known optimum | Section 4.1 | f* = 2996.3482 |

---

## Algorithm Mapping

| Algorithm in Paper | Code Implementation | File Location |
|--------------------|---------------------|---------------|
| **Algorithm 1: Constrained Thompson Sampling** | `pick_candidate_cts()` | Cell 5 |
| Line 1: Compute posterior | `model_f(cand_t)`, `models_c_or_z[i](cand_t)` | Cell 5 |
| Line 2: Sample θ | `.sample()` on GP distributions | Cell 5 |
| Line 3: Get realizations | `f_samp`, `Z_samp` or `C_samp` | Cell 5 |
| Line 4: Evaluate candidates | Loop over cand_t | Cell 5 |
| Line 5: Identify feasible | `feas = torch.all(C_samp <= 0.0)` | Cell 5 |
| Line 6-8: Select best | `torch.argmin(f_samp[feas])` or violation | Cell 5 |
| **Algorithm 2: SCBO with Latent GPs** | `run_bo()` | Cell 6 |
| Line 1: Input space X | Problem definition | Cell 2 |
| Line 2: Compute DoE | `problem.sample_lhs()` | Cell 6 |
| Line 3: k = 0 | Loop initialization | Cell 6 |
| Line 4: While not exhausted | `for t in range(iters)` | Cell 6 |
| Line 5: Compute P_k | `reducer.fit_transform()` | Cell 6 (via Cell 5) |
| Line 6: Project & fit GPs | `build_models_pca/kpca()` | Cell 6 |
| Line 7: Acquisition | `pick_candidate_cts()` or feasibility-first | Cell 6 |
| Line 8: Evaluate | `problem.evaluate(x_next)` | Cell 6 |
| Line 9: Update D | `X = np.vstack([X, x_next])` etc. | Cell 6 |
| Line 10: Update TuRBO | `tr.step(success)`, `tr.set_center()` | Cell 6 |
| Line 11: k ← k+1 | Loop increment | Cell 6 |

---

## Equation Mapping

| Equation # | Description | Code Implementation | Location |
|------------|-------------|---------------------|----------|
| (1) | Constrained optimization | `problem.evaluate()` returns (f, c) | Cell 2 |
| (2) | GP definition | `ExactGP` class | Cell 3 |
| (3) | RBF kernel | `gpytorch.kernels.RBFKernel(ard_num_dims=D)` | Cell 3 |
| (4) | Marginal likelihood | `gpytorch.mlls.ExactMarginalLogLikelihood` | Cell 3 |
| (5) | Gradient for learning | Automatic (PyTorch autograd) | Cell 3 |
| (7) | Posterior mean | `post.mean` in `posterior_mean_std()` | Cell 3 |
| (8) | Posterior variance | `post.variance.sqrt()` | Cell 3 |
| (12) | Constrained acquisition | `pick_candidate_feasibility_first()` | Cell 4 |
| (14) | Trust region size | Implicit in `tr.sample()` | Cell 4 |
| (16) | Eigenvalue problem | `Cv = λv` | Implicit in PCA |
| (17) | Covariance matrix | `sklearn.decomposition.PCA` | Cell 5 |
| (18-19) | Eigendecomposition | `PCA.fit_transform()` | Cell 5 |
| (20) | PCA projection | `Z = CΨ_g` | `PCAReducer.fit_transform()` |
| (21-22) | kPCA covariance | `KernelPCA` internals | Cell 5 |
| (23) | Kernel function | `kernel='rbf'` in KernelPCA | Cell 5 |
| (24) | Kernel matrix | KernelPCA internals | Cell 5 |
| (25) | kPCA eigenvalues | KernelPCA internals | Cell 5 |
| (26) | kPCA projection | `KPCAReducer.fit_transform()` | Cell 5 |

---

## Section-by-Section Code Mapping

### Section 2.1: Gaussian Processes
- **Full implementation**: Cell 3
- **Key classes**: `ExactGP`, `fit_gp()`, `posterior_mean_std()`
- **Equations used**: (2), (3), (4), (5), (7), (8)

### Section 2.3: Constrained Bayesian Optimisation  
- **Baseline (SCBO)**: `build_models_scbo()` in Cell 5
- **Feasibility probability**: `pick_candidate_feasibility_first()` in Cell 4
- **Equation used**: (12)

### Section 2.4: High-Dimensional BO
- **Trust Region**: Cell 4, `TrustRegion` class
- **Challenges discussed**: Addressed by TuRBO approach
- **Key papers**: Eriksson et al. (2020, 2021)

### Section 3.1: Principal Component Analysis
- **Implementation**: Cell 5, `PCAReducer` class
- **Key method**: `.fit_transform()` computes projection
- **Equations used**: (16), (17), (18), (19), (20)
- **When used**: `build_models_pca()`

### Section 3.2: Kernel Principal Component Analysis
- **Implementation**: Cell 5, `KPCAReducer` class  
- **Key method**: `.fit_transform()` with kernel trick
- **Equations used**: (21), (22), (23), (24), (25), (26)
- **When used**: `build_models_kpca()`

### Section 3.3: Dimensionality Reduction for Constraints
- **Integration**: Cell 6, `run_bo()` function
- **Critical quote**: "validity of a feasible design is checked in the original space"
- **Implementation**: `reducer.inverse_transform()` before feasibility check

### Section 3.4: Complexity Considerations
- **Analyzed in**: Comments throughout, especially Cells 3, 5
- **Key insight**: O((g+1)N³) vs O((G+1)N³)
- **Demonstrated**: Speed comparisons in Cell 7

### Section 4.1: Speed Reducer Problem
- **Problem class**: Cell 2, `SpeedReducerProblem`
- **Experiments**: Cell 7, `run_bo()` calls
- **Results format**: Table 1 comparisons
- **Hyperparameters**: Listed in Cell 7 comments

### Section 4.2: Aeroelastic Tailoring
- **Not implemented**: This is mentioned in paper but not in this notebook
- **Would use**: Same `run_bo()` framework
- **Key difference**: G=1786, g=35 (vs G=11, g=4 for Speed Reducer)

---

## Figure Mapping

| Figure # | Description | Code to Generate |
|----------|-------------|------------------|
| Figure 1 | PCA/kPCA graphical interpretation | Not generated (conceptual) |
| Figure 2 | Schematic of (k)PCA-GP | Not generated (conceptual) |
| Figure 3 | Speed Reducer convergence | Cell 7, `plot_convergence()` |
| Figure 4 | Influence of g on results | Could modify Cell 7 to vary n_components |
| Figure 6 | Eigenvalue decay | Could add after PCA: `plt.plot(reducer.pca.explained_variance_)` |

---

## Table Mapping

| Table # | Description | Code to Generate |
|---------|-------------|------------------|
| Table 1 | Speed Reducer results | Cell 7 print statements (single run) |
|         |             | Would need 20 runs for full statistics |
| Table 2 | Aeroelastic problem definition | Not in this notebook |

---

## Common Workflows

### 1. Running Standard SCBO
```python
results = run_bo(method="scbo", seed=42, n_init=20, iters=100)
```
**Paper reference**: Baseline method (Eriksson & Poloczek 2021)

### 2. Running PCA-GP SCBO
```python
results = run_bo(method="pca", seed=42, n_init=20, iters=100, n_components=4)
```
**Paper reference**: Section 3.1, Algorithm 2

### 3. Running kPCA-GP SCBO
```python
results = run_bo(method="kpca", seed=42, n_init=20, iters=100, 
                 n_components=4, kpca_gamma=0.2)
```
**Paper reference**: Section 3.2, Algorithm 2 with kernel projection

### 4. Analyzing Results
```python
# Best feasible value found
best = np.nanmin(results["best_hist"])

# Convergence history
plt.plot(results["best_hist"])

# Final incumbent
feas = np.all(results["C"] <= 0.0, axis=1)
x_best = results["X"][feas][np.argmin(results["f"][feas])]
```

### 5. Checking Dimensionality Reduction
```python
# After running PCA method
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(results["C"])

# Eigenvalue decay
plt.plot(pca.explained_variance_)
plt.yscale('log')
plt.xlabel('Component')
plt.ylabel('Eigenvalue')

# Cumulative variance explained
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(0.99, ls='--', label='99% variance')
```
**Paper reference**: Figure 6 (eigenvalue analysis)

---

## Hyperparameter Decision Guide

### Number of Components (g)

**From eigenvalue analysis** (Figure 6):
```python
# Rule of thumb: keep components with λ > 0.01
eigenvalues = pca.explained_variance_
g = np.sum(eigenvalues > 0.01)
```

**From variance explained**:
```python
# Keep enough to explain 95-99% variance
cumvar = np.cumsum(pca.explained_variance_ratio_)
g = np.argmax(cumvar >= 0.95) + 1
```

**From paper examples**:
- Speed Reducer (G=11): g=4 works well
- Aeroelastic (G=1786): g=35 captures 99% variance

### kPCA Gamma (γ)

**Default heuristic**:
```python
gamma = 1.0 / G  # Inverse of constraint count
```

**Paper value** (Section 4.1):
```python
gamma = 0.2  # Used for Speed Reducer
```

**Tuning principle**:
- Smaller γ → smoother, more global features
- Larger γ → more local, complex features
- Check reconstruction error on held-out data

### Trust Region Parameters

**From TuRBO paper** (used in implementation):
```python
init_frac = 0.8   # Start with 80% of domain
min_frac = 0.05   # Don't shrink below 5%
max_frac = 1.0    # Can expand to full domain
grow = 1.6        # Expand by 60% on success
shrink = 0.5      # Contract by 50% on failure
succ_tol = 3      # Need 3 successes to expand
fail_tol = 3      # Need 3 failures to contract
```

### Candidate Count (N_c)

**Standard**:
```python
n_cand = 4096  # Used in paper
```

**Adaptive** (in code):
```python
if no_feasible_yet:
    n_cand = max(4096, 8192)  # More when searching for feasibility
else:
    n_cand = 4096  # Standard when optimizing
```

---

## Debugging Guide

### Issue: Method not converging

**Check 1**: Are any feasible points found?
```python
feas = np.all(results["C"] <= 0.0, axis=1)
print(f"Feasible points: {feas.sum()} / {len(feas)}")
```
- If 0: Problem may be over-constrained or TR is stuck in infeasible region
- Solution: Increase n_cand, try feasible seeding

**Check 2**: Is TR too small?
```python
# Would need to modify code to track TR size
# Add to run_bo(): tr_sizes.append(tr.frac)
```
- If frac < 0.1: TR may have contracted too much
- Solution: Increase min_frac, adjust shrink rate

**Check 3**: Are GPs trained properly?
```python
# Check if GP hyperparameters are reasonable
# After fit_gp(), examine:
print("Length scales:", model.covar_module.base_kernel.lengthscale)
print("Noise:", model.likelihood.noise)
```
- Very large/small length scales → training issue
- Very large noise → underfitting

### Issue: PCA/kPCA not helping

**Check 1**: Eigenvalue decay
```python
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(results["C"])
plt.semilogy(pca.explained_variance_)
```
- Slow decay → many components needed, PCA not effective
- Solution: Increase g or use kPCA

**Check 2**: Reconstruction error
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
Z = pca.fit_transform(C)
C_recon = pca.inverse_transform(Z)
error = np.linalg.norm(C - C_recon, 'fro') / np.linalg.norm(C, 'fro')
print(f"Reconstruction error: {error:.2%}")
```
- Error > 10% → g too small
- Solution: Increase n_components

**Check 3**: Constraint linearity
```python
# If kPCA doesn't help, constraints may be linear
# Try both and compare convergence curves
```

### Issue: Out of memory

**Likely cause**: Too many GPs or too much data

**Solution 1**: Use PCA/kPCA to reduce GP count
```python
# Instead of:
results = run_bo(method="scbo")  # G GPs

# Use:
results = run_bo(method="pca", n_components=min(10, G//10))  # Much fewer
```

**Solution 2**: Reduce population size
```python
results = run_bo(method="pca", n_init=10, iters=50, n_cand=1024)
```

### Issue: Integer variables not handled correctly

**Check**: Is rounding happening?
```python
# In Speed Reducer, x3 should be integer
print("x3 values:", results["X"][:, 2])
print("All integer?", np.all(results["X"][:, 2] == np.rint(results["X"][:, 2])))
```

**In code**: 
```python
if problem.enforce_integer_teeth:
    cand_t[:, 2] = torch.round(cand_t[:, 2])
```

---

## Performance Optimization Tips

### 1. GPU Acceleration
```python
# Check GPU is being used
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 2. Batch Size Tuning
```python
# Larger batches → more GPU utilization but more memory
n_cand = 8192  # vs default 4096
```

### 3. GP Training Iterations
```python
# Fewer iterations → faster but less accurate
# Current: 100 iterations (conservative)
# Can reduce to 50 for faster experiments
```

### 4. Parallel Evaluation
```python
# If problem allows parallel evaluation (not implemented):
# Select Q points per iteration instead of 1
# Modify to return top-Q from acquisition
```

---

## Citation Information

**If using this code, cite:**

Primary paper:
```bibtex
@article{maathuis2024highdimensional,
  title={High-Dimensional Bayesian Optimisation with Large-Scale Constraints 
         via Latent Space Gaussian Processes},
  author={Maathuis, Hauke F. and De Breuker, Roeland and Castro, Saullo G.P.},
  journal={arXiv preprint arXiv:2412.15679},
  year={2024}
}
```

Baseline method (SCBO):
```bibtex
@article{eriksson2021scalable,
  title={Scalable Constrained Bayesian Optimization},
  author={Eriksson, David and Poloczek, Matthias},
  journal={arXiv preprint arXiv:2002.08526},
  year={2021}
}
```

Trust regions (TuRBO):
```bibtex
@inproceedings{eriksson2019scalable,
  title={Scalable global optimization via local Bayesian optimization},
  author={Eriksson, David and Pearce, Michael and Gardner, Jacob R and 
          Turner, Ryan and Poloczek, Matthias},
  booktitle={NeurIPS},
  year={2019}
}
```

---

*End of Quick Reference*

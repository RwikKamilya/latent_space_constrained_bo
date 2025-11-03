# Per-Block Code Comments for Jupyter Notebook

## How to Use This File
Copy the relevant block comment before each cell in your Jupyter notebook for inline documentation.

---

## BLOCK 1: Initialization and Seed Management

```python
"""
Cell 1: Initialization and Seed Management

Purpose:
- Sets up reproducible experiments across all random number generators
- Configures GPU acceleration for efficient GP training
- Establishes deterministic behavior for fair method comparison

Key Components:
1. Device selection: Uses CUDA when available (critical for large constraint sets)
2. set_global_seed(): Ensures reproducibility across NumPy, PyTorch, and Python random
3. Deterministic settings: Makes CUDA operations reproducible (may reduce performance)

References:
- Section 3.4 (Maathuis et al.): Discusses computational complexity where GPU acceleration helps
- Paper reports results over multiple runs - reproducibility is essential

Notes:
- torch.float64 is used for numerical stability in GP training
- Deterministic CUDA may be slower but ensures exact reproducibility
"""
```

---

## BLOCK 2: Speed Reducer Problem

```python
"""
Cell 2: Speed Reducer Problem Definition

Problem: 7D mechanical design optimization with 11 black-box constraints
Objective: Minimize weight of speed reducer gearbox
Known optimum: f* = 2996.3482 (used for convergence assessment)

Variables:
- x1 (b): face width [2.6, 3.6]
- x2 (m): module of teeth [0.7, 0.8]  
- x3 (n): number of teeth [17, 28] - INTEGER
- x4, x5 (l1, l2): shaft lengths [7.3, 8.3]
- x6, x7 (d1, d2): shaft diameters [2.9, 3.9], [5.0, 5.5]

Constraints (G=11):
- g1, g2: Bending and contact stress on gear teeth
- g3, g4: Transverse deflection of shafts
- g5, g6: Stress in shafts under loading
- g7-g11: Geometric and manufacturing constraints

Key Methods:
1. evaluate(X): Returns (f, g) where g[:, i] ≤ 0 for feasibility
2. is_feasible(X): Boolean mask for constraint satisfaction
3. sample_lhs(n): Latin Hypercube Sampling for space-filling DoE
4. project_to_bounds(X): Enforces variable bounds

Integer Handling:
- x3 (number of teeth) must be integer → rounded to nearest int
- All other variables are continuous

References:
- Problem source: Lemonge et al. (2010)  
- Section 4.1 (Maathuis et al.): Details this as benchmark problem
- Table 3: Shows comparative results across methods

Notes:
- enforce_integer_teeth=True ensures discrete design variable
- Constraint formulation uses implicit functions (numerically evaluated)
- f_star_ref stored for convergence plots
"""
```

---

## BLOCK 3: Gaussian Process Infrastructure

```python
"""
Cell 3: Gaussian Process Surrogate Models

Purpose:
- Implements GP regression for probabilistic modeling of f(x) and c_j(x)
- Provides utilities for training, prediction, and feasibility assessment
- Uses GPyTorch for efficient GPU-accelerated GP operations

Theoretical Foundation:
A GP is defined by mean m(x) and covariance k(x, x'):
    f(x) ~ GP(m(x), k(x, x'))

Posterior predictions (Equations 7-8):
    μ(x) = k(x,X)K^(-1)f  
    σ²(x) = k(x,x) - k(x,X)K^(-1)k(X,x)

Key Components:

1. standardize(y):
   - Centers and scales to N(0,1) for numerical stability
   - Stores (μ, σ) for inverse transformation during prediction
   - Reference: Standard practice (Rasmussen & Williams 2006)

2. ExactGP class:
   - Mean: ConstantMean() → assumes m(x) = c (learned)
   - Kernel: ScaleKernel(RBFKernel) with ARD
     * RBF: k(x,x') = σ² exp(-½ Σ(x_i - x'_i)²/l_i²)  [Equation 3]
     * ARD: Separate length scale l_i per dimension
     * Identifies relevant dimensions (small l_i = important feature)

3. fit_gp(X, y, iters=100, lr=0.05):
   - Trains GP by maximizing marginal likelihood [Equation 4]:
     log p(f|D,θ) = -½f^T K^(-1)f - ½log|K| - (n/2)log(2π)
   - Uses Adam optimizer for hyperparameter learning
   - Constraints: σ²_noise ≥ 1e-6 (prevents overfitting)
   - Cholesky jitter (1e-5): Ensures K positive definite
   - Returns trained model + likelihood

4. best_feasible_value(f_hist, C_hist):
   - Finds minimum f among feasible points
   - Returns NaN if no feasible points exist
   - Used for: Convergence tracking, success detection

5. least_violation_index(C_hist):
   - Finds point with minimum Σ max(c_j, 0)
   - Used for: Trust region centering when no feasible points

6. posterior_mean_std(model, X):
   - Computes μ(x), σ(x) from trained GP
   - Uses fast_pred_var for batch efficiency
   - Clamps σ² ≥ 1e-12 for numerical stability

Training Details:
- Device: Moves all data to GPU if available
- Iterations: 100 (sufficient for most problems)
- Learning rate: 0.05 (conservative, stable)
- No early stopping (may waste compute but ensures convergence)

References:
- Section 2.1 (Maathuis et al.): GP theory and equations
- Section 3.4: Discusses GP computational complexity O(N³)
- Rasmussen & Williams (2006): Canonical GP reference

Implementation Notes:
- GPyTorch used for GPU acceleration (critical when G or N is large)
- Exact GP used (vs sparse/variational) since N ≤ 200 typically
- Each constraint gets independent GP (SCBO) or latent GPs (PCA/kPCA)
"""
```

---

## BLOCK 4: Trust Region and Feasibility-First

```python
"""
Cell 4: Trust Region Management and Feasibility-First Acquisition

Purpose:
- Implements TuRBO (Trust Region Bayesian Optimization) for high-dimensional problems
- Provides feasibility-first acquisition when no feasible points exist
- Includes utilities for efficient feasible seed finding

Why Trust Regions?
Problem: Standard BO struggles in high dimensions (curse of dimensionality)
- Collecting enough data for global surrogate is expensive
- GP uncertainty grows rapidly away from evaluated points

Solution: Local optimization within adaptive trust regions
- Focus computational budget on promising subregions
- Expand region when making progress, contract when struggling
- Re-center at current best solution

Trust Region Class:
Manages hyperrectangle [c - r, c + r] where:
- c: center (current incumbent)
- r: radius (adaptive, based on success/failure)

Hyperparameters (from TuRBO paper):
- init_frac=0.8: Start with 80% of full domain
- min_frac=0.05: Minimum 5% of domain (prevents over-contraction)  
- max_frac=1.0: Can expand to full domain
- grow=1.6: Expand by 60% after successes
- shrink=0.5: Contract by 50% after failures
- succ_tol=3: Require 3 consecutive improvements to expand
- fail_tol=3: Require 3 consecutive failures to contract

Key Methods:

1. set_center(x): Move TR center to new incumbent

2. sample(n_cand): Generate candidates uniformly in TR
   - Clips to respect variable bounds
   - Returns n_cand points in [lb, ub] ∩ TR

3. step(success): Update TR size based on iteration outcome
   - Success: increment succ counter, reset fail
   - Failure: increment fail counter, reset succ  
   - Expand when succ ≥ succ_tol
   - Contract when fail ≥ fail_tol

Cheap Filtering (Speed Reducer specific):
- Evaluates subset of "easy" constraints without full model
- g7: x2*x3 ≤ 40 (linear)
- g8: x1/x2 ≥ 5 (simple ratio)
- g9: x1/x2 ≤ 12 (simple ratio)
- g10: 1.5*x6 + 1.9 ≤ x4 (linear)
- Purpose: Quickly filter infeasible candidates
- Used in: find_feasible_seed() for bootstrap

Find Feasible Seed:
- Goal: Ensure ≥1 feasible point in initial DoE
- Method: 
  1. Generate many candidates (50k per batch)
  2. Apply cheap filter
  3. Fully evaluate survivors
  4. Return first feasible point found
- Why: Enables immediate use of CTS (vs feasibility-first)
- Reference: Section 4.2 notes efficient feasible point identification

Feasibility-First Acquisition:
Used when: No feasible points exist yet (X_f = ∅)
Goal: Maximize probability of finding any feasible point

Computes: argmax_x ∏_{j=1}^G P(c_j(x) ≤ 0)

For each constraint GP c_j ~ N(μ_j, σ_j²):
- P(c_j(x) ≤ 0) = Φ(-μ_j(x) / σ_j(x))
- Φ: standard normal CDF
- z = -μ/σ: standardized distance to feasibility

Implementation:
- Compute log probabilities: log Φ(z_j) for all j
- Sum logs: log P = Σ log P_j (numerically stable vs product)
- Return: argmax log P

Key Insight: Ignores objective, focuses purely on feasibility
- Rapidly explores to find feasible region
- Switches to CTS once ≥1 feasible point found

References:
- Section 2.4 (Maathuis et al.): High-dimensional BO challenges
- Eriksson et al. (2020): TuRBO algorithm
- Eriksson & Poloczek (2021): Constrained extension (SCBO)
- Section 2.3, Equation (12): Probability of feasibility term

Design Rationale:
- Two-phase strategy:
  1. Phase 1 (no feasible): Maximize P(feasibility)
  2. Phase 2 (≥1 feasible): Optimize objective via CTS
- Why effective: Most problems have small feasible regions
  - Finding ANY feasible point is first challenge
  - Optimizing comes after feasible region discovered
"""
```

---

## BLOCK 5: Model Builders and Constrained Thompson Sampling

```python
"""
Cell 5: Model Building and Constrained Thompson Sampling (CTS)

Purpose:
- Implements three BO variants: SCBO, PCA-GP SCBO, kPCA-GP SCBO
- Provides Constrained Thompson Sampling acquisition function
- Core contribution: Latent space modeling for large-scale constraints

Three Methods:

1. SCBO (Baseline):
   - Models each constraint independently
   - G+1 GPs total: 1 for f, G for c_1, ..., c_G
   - Complexity: O((G+1)N³) training, O((G+1)N²) memory
   - Reference: Eriksson & Poloczek (2021)
   - Use when: G ≤ 20, want exact constraint modeling

2. PCA-GP SCBO (Linear reduction):
   Mathematical steps (Section 3.1):
   a) Center constraint matrix: C̄ = C - mean  
   b) Covariance: Σ = (1/(N-1))C̄^T C̄ ∈ R^(G×G)
   c) Eigendecomposition: Σ = ΨΛΨ^T
      - Λ = diag(λ_1, ..., λ_G) with λ_1 ≥ λ_2 ≥ ... ≥ λ_G
   d) Truncate: Keep top g eigenvectors → Ψ_g ∈ R^(G×g)
   e) Project: Z = CΨ_g  [Equation 20]

   - g+1 GPs total: 1 for f, g for z_1, ..., z_g
   - Complexity: O((g+1)N³ + G³) training, O((g+1)N²) memory
   - Speedup: ~60% faster when g=4, G=11 (Table 1)
   - Reference: Section 3.1, Equations (16-20)
   - Use when: G > 20, constraints approximately linear

3. kPCA-GP SCBO (Nonlinear reduction):
   Mathematical steps (Section 3.2):
   a) Define kernel: k(c_i, c_j) = exp(-γ||c_i - c_j||²)  [Eq 23, 28]
      - Maps to infinite-dimensional feature space F
      - φ: R^G → F (implicit via kernel trick)
   b) Kernel matrix: K_ij = k(c_i, c_j) ∈ R^(N×N)  [Eq 24]
   c) Eigendecomposition: Kα = λα  [Eq 25]
   d) Project: z_q(c) = Σ_i α_i^q k(c_i, c)  [Eq 26]
   e) Inverse (pre-image): c̃ = Ψ_g^(-1) z (approximate)

   - g+1 GPs total: 1 for f, g for z_1, ..., z_g  
   - Complexity: Similar to PCA
   - Handles nonlinear constraint manifolds
   - Reference: Section 3.2, Equations (23-26)
   - Use when: Slow PCA eigenvalue decay, nonlinear constraints

PCAReducer Class:
- fit_transform(C): C ∈ R^(N×G) → Z ∈ R^(N×g)
- transform(C): Project new C to latent space
- inverse_transform(Z): Reconstruct C̃ ≈ C from Z
- Linear, exact inverse

KPCAReducer Class:
- fit_transform(C): Nonlinear projection via kernel
- inverse_transform(Z): Approximate (pre-image problem)
- Hyperparameter: gamma (RBF kernel width)
  * Small γ → smooth, global features
  * Large γ → local, complex features
  * Default γ=0.1, paper uses γ=0.2

Constrained Thompson Sampling:
Acquisition function for constrained BO (Algorithm 1)

Theory (Hernández-Lobato et al. 2017):
- Sample from GP posteriors → get realization of f̃, c̃_1, ..., c̃_G
- Solve deterministic problem on samples:
  * If feasible samples exist: min_{x: c̃_j(x)≤0} f̃(x)
  * Else: min_x Σ_j max(c̃_j(x), 0)

Why Thompson Sampling?
1. Naturally balances exploration vs exploitation
2. Scales well to high dimensions (vs EI)
3. Parallelizable (can select batches)
4. No hyperparameters (vs UCB's β)

Implementation Steps:
1. Sample objective: f̃(x) from f(x) | D
2. Sample constraints: c̃_j(x) from c_j(x) | D for all j
   - If using latent models: sample z̃_j, then project back
3. Identify feasible samples: c̃_j(x) ≤ 0 ∀j
4. Select candidate:
   - If feasible: argmin f̃(x)
   - Else: argmin Σ max(c̃_j(x), 0)

Critical Detail - Latent Space Handling:
```python
if reducer is not None:
    Z_np = Z_samp.cpu().numpy()
    C_np = reducer.inverse_transform(Z_np)  # MUST project back!
```
Why necessary?
- Feasibility check (c_j ≤ 0) only meaningful in original G-dimensional space
- Latent space Z has no interpretable feasibility boundaries
- Must reconstruct C̃ from Z̃ before checking constraints

Reference: Section 3.3 emphasizes "validity of a feasible design is checked 
in the original space rather than within the low-dimensional subspace"

Comparison of Acquisitions:
┌─────────────────────┬──────────────────────┬────────────────────┐
│ Situation           │ Acquisition          │ Objective          │
├─────────────────────┼──────────────────────┼────────────────────┤
│ No feasible pts yet │ Feasibility-first    │ Maximize P(feas)   │
│ ≥1 feasible pt      │ Constrained TS       │ Minimize f s.t. c  │
└─────────────────────┴──────────────────────┴────────────────────┘

Complexity Comparison:
┌──────────┬─────────────────┬─────────────────┬──────────────┐
│ Method   │ # GPs           │ Training        │ Memory       │
├──────────┼─────────────────┼─────────────────┼──────────────┤
│ SCBO     │ G+1             │ O((G+1)N³)      │ O((G+1)N²)   │
│ PCA-GP   │ g+1             │ O((g+1)N³ + G³) │ O((g+1)N²)   │
│ kPCA-GP  │ g+1             │ O((g+1)N³ + G³) │ O((g+1)N²)   │
└──────────┴─────────────────┴─────────────────┴──────────────┘

When G=1786, g=35, N=416 (aeroelastic case):
- SCBO: 1787 GPs, infeasible due to memory
- PCA/kPCA: 36 GPs, ~50× reduction in GPs

References:
- Section 2.3: Constrained BO theory
- Section 3: Dimensionality reduction methods
- Section 3.4: Complexity analysis
- Algorithm 1: Constrained Thompson Sampling
- Hernández-Lobato et al. (2017): Thompson Sampling for BO
"""
```

---

## BLOCK 6: Unified BO Runner

```python
"""
Cell 6: Main Optimization Loop (Algorithm 2)

Purpose:
- Unified framework for SCBO, PCA-GP SCBO, and kPCA-GP SCBO
- Implements complete Bayesian Optimization with Trust Regions
- Handles feasibility bootstrapping and adaptive model updating

Algorithm Flow (Algorithm 2 from paper):

INITIALIZATION:
1. Set random seed (reproducibility)
2. Generate initial DoE via LHS
3. Evaluate all initial points
4. [Optional] Find feasible seed if DoE is infeasible
5. Initialize trust region at best/least-infeasible point

MAIN LOOP (for t = 1 to iters):
┌────────────────────────────────────────────────────────────┐
│ 1. BUILD MODELS                                            │
│    ├─ SCBO: Fit G+1 GPs (1 for f, G for constraints)      │
│    ├─ PCA: Project C→Z, fit g+1 GPs (1 for f, g for Z)    │
│    └─ kPCA: Kernel project C→Z, fit g+1 GPs               │
├────────────────────────────────────────────────────────────┤
│ 2. GENERATE CANDIDATES                                     │
│    ├─ Sample N_cand points from trust region              │
│    ├─ If no feasible yet: use more candidates (8192)      │
│    └─ Round integer variables                              │
├────────────────────────────────────────────────────────────┤
│ 3. SELECT NEXT POINT (Acquisition)                         │
│    ├─ IF no feasible points exist:                        │
│    │  └─ Use feasibility-first: max P(all c_j ≤ 0)        │
│    └─ ELSE (≥1 feasible):                                  │
│       ├─ Sample from GP posteriors                         │
│       ├─ [If latent] Project Z̃ → C̃                        │
│       └─ Use CTS: min f̃ among feasible samples            │
├────────────────────────────────────────────────────────────┤
│ 4. EVALUATE                                                │
│    ├─ Compute f(x_next), c(x_next)                         │
│    └─ Add to dataset: D_{k+1} = D_k ∪ {x_next, f, c}      │
├────────────────────────────────────────────────────────────┤
│ 5. UPDATE TRUST REGION                                     │
│    ├─ Determine success: found better feasible point?     │
│    ├─ Update TR size: expand if success, contract if fail │
│    └─ Re-center at current incumbent                       │
└────────────────────────────────────────────────────────────┘

RETURN: All evaluated points, objectives, constraints, best history

Key Design Decisions:

1. Success Definition:
   success = (found feasible when none before) OR 
             (found better feasible than previous best)
   - Uses tolerance 1e-12 to handle floating point
   - Reference: Trust region papers use similar criteria

2. Trust Region Re-centering:
   - Always center at current incumbent (not initial center)
   - Incumbent = best feasible if any, else least infeasible
   - Why: Keeps search focused on promising region
   - Happens every iteration (not just on success)

3. Adaptive Candidate Count:
   - Standard: n_cand (typically 4096)
   - No feasible yet: max(n_cand, 8192)
   - Reason: Finding feasibility is harder, needs more exploration

4. Model Rebuilding:
   - Rebuild all GPs every iteration with full dataset
   - Alternative: Incremental updates (not implemented)
   - Trade-off: Simpler code, ensures consistency, more compute

5. Projection Matrix Updating:
   - PCA/kPCA: Recompute P_k every iteration
   - Uses all data collected so far
   - Critical: Latent space adapts as we explore
   - Reference: Algorithm 2, line 5 "compute projection P_k"

Computational Details:

Dataset Growth:
- Initial: N samples
- After t iterations: N + t samples
- GP training time grows as O(N³) → can become expensive
- Speed Reducer: N=20, iters=100 → final N=120 (manageable)
- Aeroelastic: N=416, iters=2000 → final N=2416 (significant)

Memory Usage:
- Store X (N × D), f (N), C (N × G)
- For latent methods: Projection matrix (G × g)
- GPs: Kernel matrices (g+1 GPs of size N × N)
- Speed Reducer: ~10 MB
- Aeroelastic: ~500 MB (G=1786 constraint values)

Hyperparameters:
┌────────────────────┬──────────────┬──────────────────────┐
│ Parameter          │ Default      │ Reference            │
├────────────────────┼──────────────┼──────────────────────┤
│ n_init             │ 20           │ Section 4.1          │
│ iters              │ 100          │ Section 4.1          │
│ n_cand             │ 4096         │ Common in TuRBO      │
│ n_components (g)   │ 4            │ Speed Reducer        │
│                    │ 35           │ Aeroelastic (Sec 4.2)│
│ kpca_gamma         │ 0.1          │ Default              │
│                    │ 0.2          │ Used in paper        │
│ TR init_frac       │ 0.8          │ TuRBO defaults       │
│ TR grow/shrink     │ 1.6 / 0.5    │ TuRBO defaults       │
│ TR succ/fail_tol   │ 3 / 3        │ TuRBO defaults       │
└────────────────────┴──────────────┴──────────────────────┘

Output Structure:
{
    "X": (N+iters) × D array of evaluated points,
    "f": (N+iters) array of objective values,
    "C": (N+iters) × G array of constraint values,
    "best_hist": iters array of best feasible per iteration,
    "problem": Problem instance for reference
}

best_hist interpretation:
- Contains best feasible objective found up to each iteration
- NaN if no feasible point exists at that iteration
- Use for convergence plots and performance metrics
- Final value: Overall best found

Comparison to Standard BO:
┌────────────────────┬─────────────────┬──────────────────┐
│ Aspect             │ Standard BO     │ TuRBO + CTS      │
├────────────────────┼─────────────────┼──────────────────┤
│ Search region      │ Global          │ Local (adaptive) │
│ Acquisition        │ EI/UCB          │ Thompson Sampling│
│ Constraints        │ Penalty/barrier │ Probabilistic GP │
│ Dimensionality     │ D ≤ 10          │ D ≤ 100s         │
│ Constraint count   │ G ≤ 10          │ G ≤ 1000s (PCA)  │
└────────────────────┴─────────────────┴──────────────────┘

References:
- Algorithm 2 (Maathuis et al.): Complete pseudocode
- Eriksson et al. (2020): TuRBO trust region mechanics
- Eriksson & Poloczek (2021): SCBO constraint handling
- Section 4: Experimental setup and hyperparameters

Potential Extensions:
1. Batch parallelization: Select Q points per iteration
2. Multi-fidelity: Use cheap approximations initially
3. Transfer learning: Initialize from previous optimizations
4. Adaptive g: Vary latent dimensions based on eigenvalue decay
5. Early stopping: Terminate if no improvement after K iterations
"""
```

---

## BLOCK 7: Experiments and Visualization

```python
"""
Cell 7: Comparative Experiments and Visualization

Purpose:
- Run SCBO, PCA-GP SCBO, and kPCA-GP SCBO with identical settings
- Visualize convergence behavior
- Compare final solutions and computational efficiency

Experimental Design:

Fixed Parameters (for fair comparison):
- seed=12345: Same random initialization across methods
- n_init=20: Same initial DoE size
- iters=100: Same optimization budget
- n_cand=4096: Same candidate count per iteration

Method-Specific Parameters:
- SCBO: None (baseline)
- PCA-GP: n_components=4 (g=4)
- kPCA-GP: n_components=4, gamma=0.2

Why These Settings?
- Figure 6 in paper shows eigenvalue decay for Speed Reducer
- Top 4 eigenvalues capture most variance → g=4 sufficient
- gamma=0.2 from Section 4.1 experiments

Execution:
1. Run SCBO (baseline, full constraint modeling)
2. Run PCA-GP SCBO (linear dimensionality reduction)
3. Run kPCA-GP SCBO (nonlinear dimensionality reduction)

Same seed ensures:
- Identical initial LHS samples
- Fair comparison (no randomness advantage)
- Reproducible results

Visualization Function: plot_convergence()

Plots best feasible objective vs. function evaluations:
- X-axis: Cumulative evaluations (n_init + current iteration)
- Y-axis: Best feasible f found up to that point
- Horizontal line: Known optimum f* = 2996.3482

Features:
- NaN masking: Removes points where no feasible solution exists yet
- Multiple curves: Compare all methods on same plot
- Legend: Method names and optimal value
- Grid: Easier to read values

Interpretation Guide:

1. Convergence Speed:
   - Steeper initial drop → faster initial progress
   - Earlier plateau → quicker convergence

2. Final Solution Quality:
   - Lower final value → better optimum found
   - Closer to f* line → near-optimal

3. Feasibility:
   - Gaps in curves → iterations without feasible points
   - Continuous curve → maintained feasibility throughout

4. Robustness:
   - Smooth curve → consistent progress
   - Jagged curve → unstable (may be too aggressive)

Expected Observations (from Figure 3, Table 1):

SCBO:
- Fastest initial convergence
- Reaches near-optimal quickly (~20-30 iterations)
- Most expensive per iteration (G=11 GPs to train)
- Final: ~2996.35 (optimal)

PCA-GP SCBO:
- Slightly slower initial convergence
- Reaches near-optimal in ~40-50 iterations
- 60% faster per iteration (g=4 vs G=11 GPs)
- Final: ~2996.35 (optimal)
- Overall runtime: ~201s vs ~501s (SCBO)

kPCA-GP SCBO:
- Similar to PCA (constraints approximately linear)
- Reaches near-optimal in ~40-50 iterations
- 57% faster per iteration
- Final: ~2996.35 (optimal)
- Overall runtime: ~217s

Key Findings:

1. All methods converge to same optimum:
   - Problem is well-behaved (single global optimum)
   - Dimensionality reduction doesn't hurt solution quality

2. PCA/kPCA are faster:
   - Despite more iterations needed
   - Per-iteration cost dominates total time
   - Savings: ~60% for Speed Reducer (G=11)
   - Would be ~95% for Aeroelastic (G=1786)

3. PCA ≈ kPCA for this problem:
   - Constraints have approximately linear structure
   - Eigenvalue decay is rapid (Figure 6)
   - Nonlinear kernel doesn't help much

When to Expect Differences:

PCA better when:
- Constraints are linear or near-linear
- Eigenvalue decay is rapid
- G is very large (faster than kPCA)

kPCA better when:
- Constraints lie on nonlinear manifold
- Eigenvalue decay is slow (need many components with PCA)
- G is moderate (pre-image approximation is accurate)

SCBO better when:
- G is small (≤10)
- Want fastest convergence in # iterations
- Have sufficient computational resources
- Need exact constraint modeling (no approximation)

Metrics Printed:

Final best values:
- Minimum of best_hist array (best feasible found)
- All should be ≈ 2996.35 (known optimum)
- Larger values indicate method struggled

For Full Benchmarking (as in paper):
- Run 20+ replications with different seeds
- Compute mean, median, std dev across runs
- Report success rate (% runs finding feasible solution)
- Measure total runtime and # function evaluations
- See Table 1 in paper for complete statistics

References:
- Section 4.1: Speed Reducer experiments
- Figure 3: Convergence plots (similar to this visualization)
- Table 1: Comprehensive comparison including 20 runs per method
- Section 4.2: Aeroelastic experiments (g=35, G=1786)

Reproducibility Note:
- Results should be identical across runs with same seed
- Different platforms may have slight numerical differences
- GPU vs CPU may differ in floating point rounding
- set_global_seed() ensures consistency within platform
"""
```

---

## Summary: How to Use These Comments

1. **For Inline Documentation**: Copy the block comment before each corresponding cell in your Jupyter notebook.

2. **For Understanding Flow**: Read comments sequentially to understand how methods build on each other:
   - Blocks 1-2: Setup and problem definition
   - Block 3: GP infrastructure (used by all methods)
   - Block 4: Trust regions and feasibility handling
   - Block 5: The three BO variants (core contribution)
   - Block 6: Integration into unified optimization loop
   - Block 7: Experimental validation

3. **For Method Selection**: Use the decision guides in Block 5 and 7 to choose appropriate method for your problem.

4. **For Extension**: Each block includes "Potential Extensions" sections suggesting improvements.

5. **Key References Quick Lookup**:
   - Algorithm 2 (paper): Complete pseudocode for latent GP BO
   - Section 2.1: GP theory
   - Section 3.1-3.2: PCA/kPCA details
   - Section 3.3: Latent space methodology
   - Section 4.1-4.2: Experimental results

---

*End of Per-Block Comments*

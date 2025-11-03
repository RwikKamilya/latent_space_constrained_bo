# Implementation Guide: Latent Space Constrained Bayesian Optimization

## Overview
This document provides detailed explanations of each code block in the implementation, with references to the source papers:
- **Primary Paper**: Maathuis et al. (2025) - "High-Dimensional Bayesian Optimisation with Large-Scale Constraints via Latent Space Gaussian Processes"
- **Baseline Method**: Eriksson & Poloczek (2021) - "Scalable Constrained Bayesian Optimization" (SCBO)
- **Trust Region**: Eriksson et al. (2020) - TuRBO algorithm
- **Test Problem**: Lemonge et al. (2010) - Speed Reducer benchmark

---

## Cell 1: Initialization and Seed Management

```python
# Cell 1 — seeds + imports (run once)
```

### Purpose
Sets up reproducible experiments with deterministic behavior across NumPy, PyTorch, and random modules.

### Key Components

**Device Selection** (Line 10):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- Uses GPU acceleration when available
- Critical for GP training efficiency with large constraint sets
- **Reference**: Section 3.4 (Maathuis et al.) discusses computational complexity O((g+1)N³) where GPU acceleration is beneficial

**Seed Function** (Lines 13-24):
```python
def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # ... deterministic settings
```
- Ensures reproducibility across all experiments
- **Reference**: Paper reports results over 5 runs (Section 4.2) - reproducibility is essential for fair comparison

---

## Cell 2: Speed Reducer Problem Definition

### Purpose
Implements the 7D Speed Reducer mechanical design optimization problem with 11 black-box constraints.

### Paper References
- **Problem Source**: Lemonge et al. (2010) - detailed in Section 4.1 of Maathuis et al.
- **Known Optimum**: f* = 2996.3482 (used for convergence assessment)
- **Constraint Count**: G = 11 (demonstrates method's ability to handle multiple constraints)

### Key Methods

**1. Integer Variable Handling** (Lines 17-18):
```python
if self.enforce_integer_teeth:
    X[:, 2] = np.rint(X[:, 2])  # x3 = number of teeth (integer)
```
- **Reference**: Section 4.1 states "The variable x3 is integer and all others are continuous"
- Rounds x3 to nearest integer, representing discrete design choice

**2. Objective Function** (Lines 29-33):
```python
f = (0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
     - 1.508 * x1 * (x6 ** 2 + x7 ** 2)
     + 7.4777 * (x6 ** 3 + x7 ** 3)
     + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))
```
- Minimizes weight of speed reducer
- **Reference**: Complete formulation in Lemonge et al. (2010), Table 3 of Maathuis et al.

**3. Constraint Evaluation** (Lines 35-46):
```python
g = np.empty((X.shape[0], 11), dtype=np.float64)
g[:, 0] = 27.0 / (x1 * x2 ** 2 * x3) - 1.0  # Bending stress
g[:, 1] = 397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1.0  # Contact stress
# ... 9 more constraints
```
- Returns constraint violations: c_i(x) where feasibility requires c_i(x) ≤ 0
- **Reference**: Equation (1) in Maathuis et al. defines the constrained optimization problem

**4. Latin Hypercube Sampling** (Lines 55-67):
```python
def sample_lhs(self, n, rng=None):
    # ... stratified sampling implementation
```
- Generates space-filling Design of Experiments (DoE)
- **Reference**: Section 4.2 mentions "sampling was performed via Latin Hypercube Sampling (LHS)" with N=416 initial samples

---

## Cell 3: Gaussian Process Infrastructure

### Purpose
Implements GP surrogate models using GPyTorch for efficient probabilistic modeling of objectives and constraints.

### Paper References
- **GP Framework**: Section 2.1 (Maathuis et al.) - "Gaussian Processes"
- **Kernel Choice**: Equation (3) - Squared Exponential (RBF) kernel
- **Training**: Marginal likelihood maximization (Equation 4)

### Key Components

**1. Standardization** (Lines 7-12):
```python
def standardize(y_np):
    y_np = np.asarray(y_np, dtype=np.float64).reshape(-1, 1)
    mu = float(y_np.mean())
    sd = float(y_np.std() + 1e-12)
    y_std = (y_np - mu) / sd
    return y_std.ravel(), mu, sd
```
- **Why**: Improves numerical stability and optimization convergence
- Stores μ, σ for inverse transformation during prediction
- **Reference**: Standard practice mentioned in Rasmussen & Williams (2006), cited in Section 2.1

**2. GP Model Class** (Lines 15-25):
```python
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood):
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[-1])
        )
```
- **Constant Mean**: Assumes zero prior mean (standard in BO)
- **RBF Kernel with ARD**: Automatic Relevance Determination
  - **Reference**: Equation (3) - uses length scales l_i for each dimension i
  - Different length scales per dimension identify important features

**3. GP Training** (Lines 28-48):
```python
def fit_gp(X_np, y_np, iters=100, lr=0.05):
    # ... standardization and device transfer
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.noise_covar.register_constraint("raw_noise", 
                                               gpytorch.constraints.GreaterThan(1e-6))
    model = ExactGP(X_t, y_t, likelihood).to(device)
    
    # Training loop
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for _ in range(iters):
        opt.zero_grad()
        with gpytorch.settings.cholesky_jitter(1e-5):
            out = model(X_t)
            loss = -mll(out, y_t)
        loss.backward()
        opt.step()
```

**Key Training Details**:
- **Noise Constraint**: Prevents numerical instability by ensuring σ² ≥ 1e-6
- **Marginal Likelihood**: Maximizes Equation (4) from paper
  ```
  log p(f | D, θ) = -½f^T K^(-1) f - ½log|K| - (n/2)log(2π)
  ```
- **Adam Optimizer**: Efficient for GP hyperparameter optimization
- **Cholesky Jitter**: Stabilizes matrix inversion (K must be positive definite)
- **Reference**: Section 2.1 describes hyperparameter learning via gradient-based optimization (Equation 5)

**4. Feasibility Utilities** (Lines 51-59):
```python
def best_feasible_value(f_hist, C_hist):
    feas = np.all(C_hist <= 0.0, axis=1)
    return np.min(f_hist[feas]) if np.any(feas) else np.nan

def least_violation_index(C_hist):
    viol = np.clip(C_hist, 0.0, None).sum(axis=1)
    return int(np.argmin(viol))
```
- **best_feasible_value**: Tracks optimization progress
- **least_violation_index**: Used when no feasible point exists
  - Finds point with minimum total constraint violation
  - **Reference**: Section 2.3 (Constrained BO) - when X_f = ∅, minimize total violation

**5. Posterior Computation** (Lines 62-68):
```python
def posterior_mean_std(model, X_t):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = model(X_t)
        mean = post.mean
        std = post.variance.clamp_min(1e-12).sqrt()
    return mean, std
```
- Computes μ(x), σ(x) from Equations (7-8):
  ```
  μ(x) = k(x, X)K(X,X)^(-1)f
  σ²(x) = k(x,x) - k(x,X)K(X,X)^(-1)k(X,x)
  ```
- **fast_pred_var**: GPyTorch optimization for batch predictions
- **Reference**: Section 2.1, Equations (7-8)

---

## Cell 4: Trust Region and Feasibility-First Acquisition

### Purpose
Implements TuRBO trust region management and feasibility-focused acquisition when no feasible points exist.

### Paper References
- **Trust Region**: Section 2.4 - "High-Dimensional Bayesian Optimisation: Challenges and Advances"
- **Base Algorithm**: Eriksson et al. (2020) - TuRBO
- **Constrained Extension**: Eriksson & Poloczek (2021) - SCBO

### Trust Region Class (Lines 7-46)

**Initialization** (Lines 8-18):
```python
class TrustRegion:
    def __init__(self, lb, ub, init_frac=0.8, min_frac=0.05, max_frac=1.0,
                 grow=1.6, shrink=0.5, succ_tol=3, fail_tol=3, rng=None):
        self.lb = lb.astype(np.float32)
        self.ub = ub.astype(np.float32)
        self.center = (self.lb + self.ub) / 2.0
        self.frac = init_frac  # Current size fraction
        # ... other hyperparameters
```

**Why Trust Regions?**
- **Problem**: Standard BO struggles in high dimensions (curse of dimensionality)
- **Solution**: Focus search on local regions, expand/contract based on success
- **Reference**: Section 2.4 explains that TuRBO addresses high-dimensional challenges by:
  1. Partitioning design space into trust regions
  2. Adjusting region size dynamically
  3. Building local surrogate models

**Key Parameters** (from Section 2.4 and Eriksson et al. 2020):
- `init_frac=0.8`: Start with 80% of full space
- `grow=1.6`: Expand by 60% after successes
- `shrink=0.5`: Contract by 50% after failures
- `succ_tol=3`: Require 3 consecutive improvements to expand
- `fail_tol=3`: Require 3 consecutive failures to contract

**Sampling** (Lines 25-30):
```python
def sample(self, n_cand):
    halfspan = 0.5 * self.frac * (self.ub - self.lb)
    lo = np.maximum(self.center - halfspan, self.lb)
    hi = np.minimum(self.center + halfspan, self.ub)
    U = self.rng.random((n_cand, len(self.lb))).astype(np.float32)
    return lo + U * (hi - lo)
```
- Samples uniformly within hyperrectangle centered at current best
- **Reference**: Equation (14) in paper defines trust region size L_TR

**Adaptive Step** (Lines 32-44):
```python
def step(self, success):
    if success:
        self.succ += 1; self.fail = 0
        if self.succ >= self.succ_tol:
            self.frac = min(self.max_frac, self.frac * self.grow)
            self.succ = 0
    else:
        self.fail += 1; self.succ = 0
        if self.fail >= self.fail_tol:
            self.frac = max(self.min_frac, self.frac * self.shrink)
            self.fail = 0
```
- **Success**: Found better feasible point → potentially expand
- **Failure**: No improvement → potentially contract
- **Reference**: Algorithm described in Eriksson & Poloczek (2021) for SCBO

### Cheap Filtering (Lines 49-62)

```python
def cheap_filter(problem, n_try=200000, rng=None):
    # ... sample many points
    X = lb + rng.random((n_try, problem.dim)) * (ub - lb)
    if problem.enforce_integer_teeth: X[:, 2] = np.rint(X[:, 2])
    
    # Check easy constraints without full evaluation
    mask = (
        (x2 * x3 <= 40.0) &
        (x1 / x2 >= 5.0) &
        (x1 / x2 <= 12.0) &
        (1.5 * x6 + 1.9 <= x4)
    )
    return X[mask]
```
**Purpose**: Efficiently filter out obviously infeasible points
- Checks linear constraints (g7, g8, g9, g10) without expensive evaluation
- **Reference**: Not explicitly in paper, but computational efficiency measure
- Used in `find_feasible_seed` to bootstrap with feasible initial point

### Feasible Seed Finding (Lines 65-81)

```python
def find_feasible_seed(problem, max_batches=5, per_batch=50000, verbose=True):
    for b in range(max_batches):
        cand = cheap_filter(problem, n_try=per_batch)
        if cand.size == 0:
            continue
        _, G = problem.evaluate(cand)
        feas = np.all(G <= 0.0, axis=1)
        if np.any(feas):
            return cand[feas][0]
    return None
```
**Purpose**: Ensure at least one feasible point in initial DoE
- **Why Important**: Enables immediate use of feasibility-preserving acquisition
- **Reference**: Section 4.1 states "a feasible solution is always preferred over an infeasible one"
- Paper notes: "efficiently identified, even if all points in the DOE at iteration k = 0 were initially infeasible"

### Feasibility-First Acquisition (Lines 84-96)

```python
@torch.no_grad()
def pick_candidate_feasibility_first(models_c, cand_t):
    """
    When no feasible points exist: maximize probability of feasibility (PoF)
    """
    normal = torch.distributions.Normal(0.0, 1.0)
    log_pof = torch.zeros(cand_t.shape[0], dtype=cand_t.dtype, device=cand_t.device)
    
    for mc in models_c:
        mu, sd = posterior_mean_std(mc, cand_t)
        z = -mu / sd  # Standardized distance to feasibility boundary
        log_pof = log_pof + normal.log_cdf(z)  # Product of probabilities (sum of logs)
    
    return torch.argmax(log_pof)
```

**Theoretical Foundation**:
- **Goal**: Find x that maximizes ∏_{j=1}^G P(c_j(x) ≤ 0)
- **Implementation**: 
  - For each constraint GP: compute P(c_j(x) ≤ 0) = Φ(-μ_j(x)/σ_j(x))
  - Φ is standard normal CDF
  - Product of probabilities = sum of log probabilities (numerical stability)

**Reference**: Section 2.3, Equation (12):
```
α_c(x|D) = α(x|D) ∏_{j=1}^G P(ĉ_j(x) ≤ 0)
```
- When no feasible points exist, α(x|D) is ignored
- Focus purely on constraint satisfaction
- **Why**: "actively search the design space to find promising regions" (Section 2.4)

---

## Cell 5: Model Building and Constrained Thompson Sampling

### Purpose
Implements three BO variants (SCBO, PCA-GP SCBO, kPCA-GP SCBO) and the Constrained Thompson Sampling acquisition function.

### Paper References
- **SCBO Baseline**: Section 2.3 - models each constraint independently  
- **PCA-GP SCBO**: Section 3 - projects constraints to g-dimensional latent space via PCA
- **kPCA-GP SCBO**: Section 3.2 - nonlinear projection via kernel PCA
- **Thompson Sampling**: Algorithm 1 - Constrained Thompson Sampling (from Eriksson & Poloczek 2021)

### 1. SCBO: Standard Constrained BO (Lines 7-13)

```python
def build_models_scbo(X, f, C):
    """
    Standard SCBO: model each of G constraints independently
    
    Reference: Eriksson & Poloczek (2021) - Section 2.3, Equation (11)
    """
    mf, _ = fit_gp(X, f)  # Objective GP
    models_c = []
    for i in range(C.shape[1]):  # G independent GPs
        mc, _ = fit_gp(X, C[:, i])
        models_c.append(mc)
    reducer = None  # No dimensionality reduction
    return mf, models_c, reducer
```

**Complexity Analysis**:
- **Training**: O((G+1)N³) - one GP per constraint + objective
- **Memory**: O((G+1)N²)
- **Problem**: Scales poorly with G (number of constraints)
- **Reference**: Section 3.4 discusses these limitations

### 2. PCA Reducer (Lines 16-32)

```python
class PCAReducer:
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)
    
    def fit_transform(self, C):
        """Project C ∈ R^(N×G) to Z ∈ R^(N×g) where g << G"""
        Z = self.pca.fit_transform(C)
        return Z
    
    def inverse_transform(self, Z):
        """Reconstruct C̃ ∈ R^(N×G) from Z ∈ R^(N×g)"""
        return self.pca.inverse_transform(Z)
```

**Mathematical Foundation**:
- **Input**: Constraint matrix C ∈ ℝ^(N×G)
- **Output**: Latent representation Z ∈ ℝ^(N×g), where g ≪ G

**PCA Steps** (from Section 3.1):
1. Center data: C̄ = C - 𝟙_N μ where μ = (1/N)Σc_i
2. Compute covariance: C = (1/(N-1))C̄ᵀC̄ ∈ ℝ^(G×G)
3. Eigendecomposition: C = ΨΛΨ⁻¹
   - Ψ = [ψ₁, ..., ψ_G] are eigenvectors
   - Λ = diag(λ₁, ..., λ_G) with λ₁ ≥ λ₂ ≥ ... ≥ λ_G
4. Truncate: Keep top g eigenvectors → Ψ_g ∈ ℝ^(G×g)
5. Project: Z = CΨ_g (Equation 20)

**Reference**: Section 3.1, Equations (16-20)

**Why PCA?**:
- Captures maximum variance with fewest components
- Linear, fast, interpretable
- **Assumption**: Constraints lie approximately on linear subspace
- **Paper Finding**: Figure 6 shows rapid eigenvalue decay for Speed Reducer → few components needed

### 3. Kernel PCA Reducer (Lines 35-51)

```python
class KPCAReducer:
    def __init__(self, n_components, gamma=0.1):
        self.kpca = KernelPCA(n_components=n_components, 
                              kernel='rbf',
                              gamma=gamma, 
                              fit_inverse_transform=True)
    
    def fit_transform(self, C):
        """Nonlinear projection via kernel trick"""
        return self.kpca.fit_transform(C)
    
    def inverse_transform(self, Z):
        """Approximate inverse (pre-image problem)"""
        return self.kpca.inverse_transform(Z)
```

**Mathematical Foundation** (Section 3.2):
1. **Kernel Function**: k(c_i, c_j) = exp(-γ||c_i - c_j||²)  (Equation 23, 28)
   - Maps to infinite-dimensional feature space F
   - φ: ℝ^G → F (implicit mapping)

2. **Kernel Matrix**: K_ij = ⟨φ(c_i), φ(c_j)⟩ ∈ ℝ^(N×N)  (Equation 24)

3. **Eigendecomposition**: Kα_i = λ_i α_i  (Equation 25)
   - Solve in N-dimensional space (avoid explicit F)

4. **Projection**: ṽ^(q)ᵀφ(c_+(x)) = Σᵢ αᵢ^(q) k(c_i, c_+(x))  (Equation 26)

**Why Kernel PCA?**:
- Handles nonlinear constraint relationships
- **Trade-off**: More expressive but requires pre-image approximation
- **Paper Finding**: Section 4.1 shows PCA slightly outperforms kPCA on Speed Reducer
  - Suggests near-linear constraint structure for this problem

**Gamma Parameter** (γ = 0.1 default):
- Controls kernel width
- Smaller γ → broader features
- **Reference**: Section 4.1 uses γ = 0.2 for Speed Reducer

### 4. PCA-GP SCBO Builder (Lines 54-65)

```python
def build_models_pca(X, f, C, n_components=4):
    """
    Main contribution: reduce G constraints to g latent GPs
    
    Reference: Algorithm 2 - SCBO with Latent Gaussian Processes
    """
    reducer = PCAReducer(n_components)
    Z = reducer.fit_transform(C)  # C ∈ R^(N×G) → Z ∈ R^(N×g)
    
    mf, _ = fit_gp(X, f)  # Objective GP (unchanged)
    
    models_z = []
    for i in range(Z.shape[1]):  # Only g GPs instead of G!
        mz, _ = fit_gp(X, Z[:, i])
        models_z.append(mz)
    
    return mf, models_z, reducer
```

**Complexity Comparison**:
- **SCBO**: O((G+1)N³) training, O((G+1)N²) memory
- **PCA-GP SCBO**: O((g+1)N³ + G³) training, O((g+1)N²) memory
  - **Savings**: Massive when G ≫ g (e.g., G=1786, g=35 in aeroelastic case)
  - **Cost**: O(G³) for PCA eigendecomposition (one-time per iteration)

**Reference**: Section 3.4 complexity analysis

### 5. kPCA-GP SCBO Builder (Lines 68-80)

```python
def build_models_kpca(X, f, C, n_components=4, gamma=0.1):
    """
    Nonlinear variant using kernel PCA
    
    Reference: Section 3.2 - Kernel Principal Component Analysis
    """
    reducer = KPCAReducer(n_components, gamma=gamma)
    Z = reducer.fit_transform(C)
    
    mf, _ = fit_gp(X, f)
    models_z = []
    for i in range(Z.shape[1]):
        mz, _ = fit_gp(X, Z[:, i])
        models_z.append(mz)
    
    return mf, models_z, reducer
```

**When to Use kPCA vs PCA**:
- **PCA**: Fast, interpretable, works well when constraints are approximately linear
- **kPCA**: More expressive, handles nonlinear constraint manifolds
- **Paper Recommendation**: Start with PCA; use kPCA if eigenvalue decay is slow

### 6. Constrained Thompson Sampling (Lines 83-110)

```python
@torch.no_grad()
def pick_candidate_cts(model_f, models_c_or_z, cand_t, reducer=None):
    """
    Constrained Thompson Sampling acquisition function
    
    Reference: Algorithm 1 - Constrained Thompson Sampling
    From: Hernández-Lobato et al. (2017), adapted in Eriksson & Poloczek (2021)
    """
    # 1. Sample objective function
    f_samp = model_f(cand_t).sample()
    
    # 2. Sample latent constraints (or constraints directly)
    c_cols = [mc(cand_t).sample().unsqueeze(-1) for mc in models_c_or_z]
    Z_samp = torch.cat(c_cols, dim=-1)  # (N_cand, g) or (N_cand, G)
    
    # 3. Project back to original constraint space if using latent models
    if reducer is not None:
        Z_np = Z_samp.cpu().numpy()
        C_np = reducer.inverse_transform(Z_np)  # Z → C̃
        C_samp = torch.as_tensor(C_np, dtype=cand_t.dtype, device=cand_t.device)
    else:
        C_samp = Z_samp
    
    # 4. Identify feasible samples
    feas = torch.all(C_samp <= 0.0, dim=-1)
    
    # 5. Select candidate
    if feas.any():
        # Among feasible: minimize objective
        idx = torch.argmin(f_samp[feas])
        best_idx = torch.nonzero(feas, as_tuple=False).squeeze(1)[idx]
    else:
        # No feasible samples: minimize constraint violation
        viol = torch.clamp(C_samp, min=0.0).sum(dim=-1)
        best_idx = torch.argmin(viol)
    
    return best_idx
```

**Thompson Sampling Theory**:
- **Idea**: Sample from posterior, optimize sample (random but informed)
- **Why TS**: 
  1. Naturally balances exploration vs exploitation
  2. Scales well to high dimensions (vs EI)
  3. Supports batch selection
  4. No hyperparameters (vs UCB's β)

**Algorithm Steps** (from Algorithm 1):
1. Sample θ from posterior p(θ|D_k)
2. Get realizations: f̂(x), ĉ₁(x), ..., ĉ_G(x)
3. Define feasible set: X_f = {x | ĉ_j(x) ≤ 0 ∀j}
4. If X_f ≠ ∅: return argmin_{x∈X_f} f̂(x)
5. Else: return argmin_x Σⱼ max(ĉⱼ(x), 0)

**Critical Detail - Latent Space Handling**:
```python
if reducer is not None:
    C_np = reducer.inverse_transform(Z_np)  # Z̃ → C̃
```
- **Why Necessary**: Feasibility check (c_j ≤ 0) must happen in original G-dimensional space
- **Reason**: Latent space Z doesn't have interpretable feasibility boundaries
- **Reference**: Section 3.3 emphasizes "validity of a feasible design is checked in the original space"

**Comparison to Feasibility-First**:
- **Feasibility-First** (`pick_candidate_feasibility_first`): Used when no feasible points exist
  - Ignores objective
  - Maximizes probability of feasibility
- **CTS** (`pick_candidate_cts`): Used once ≥1 feasible point exists
  - Considers both objective and constraints
  - Balances optimization and constraint satisfaction

---

## Cell 6: Unified BO Runner

### Purpose
Main optimization loop implementing Algorithm 2: SCBO with Latent Gaussian Processes.

### Paper References
- **Full Algorithm**: Algorithm 2 in Section 3.3
- **Trust Region Updates**: Lines 22-31 of Algorithm 2
- **Penalty Coefficient Updates**: Lines 9-10 of Algorithm 2

### Function Signature (Lines 7-9)

```python
def run_bo(method="scbo", seed=42, n_init=20, iters=100, n_cand=4096,
           n_components=4, kpca_gamma=0.1, verbose=True,
           enforce_integer_teeth=True, try_seed_feasible=True):
```

**Key Parameters**:
- `method`: "scbo" | "pca" | "kpca" - selects algorithm variant
- `n_init=20`: Initial DoE size (N)
- `iters=100`: BO iterations
- `n_cand=4096`: Candidate points per iteration (N_c in Algorithm 1)
- `n_components=4`: Latent dimensions (g) for PCA/kPCA
- `kpca_gamma=0.1`: RBF kernel width for kPCA

**Reference**: Section 4.1 experiments use these defaults

### Initialization Phase (Lines 10-30)

```python
# 1. Set random seed for reproducibility
set_global_seed(seed)
problem = SpeedReducerProblem(enforce_integer_teeth=enforce_integer_teeth)

# 2. Generate initial Design of Experiments (DoE)
X = problem.sample_lhs(n_init, rng=np.random.default_rng(seed))
f_list, C_list = [], []
for x in X:
    f_i, g_i = problem.evaluate(x)
    f_list.append(f_i); C_list.append(g_i)
f_vals = np.array(f_list, dtype=np.float64)
C_vals = np.array(C_list, dtype=np.float64)
```
**Reference**: Algorithm 2, line 2 - "Compute DoE D₀"

**3. Optional Feasible Seeding** (Lines 32-41):
```python
feas = np.all(C_vals <= 0.0, axis=1)
if (not np.any(feas)) and try_seed_feasible:
    x_seed = find_feasible_seed(problem, max_batches=5, per_batch=60000, verbose=verbose)
    if x_seed is not None:
        f_seed, g_seed = problem.evaluate(x_seed)
        X = np.vstack([X, x_seed])
        f_vals = np.append(f_vals, f_seed)
        C_vals = np.vstack([C_vals, g_seed])
```
**Purpose**: Bootstrap with ≥1 feasible point if DoE is entirely infeasible
- **Why**: Enables immediate use of CTS (rather than feasibility-first)
- **Reference**: Section 4.2 notes this helps "efficiently identify feasible design point even when D₀ only contains infeasible ones"

**4. Initialize Incumbent and Trust Region** (Lines 43-50):
```python
feas = np.all(C_vals <= 0.0, axis=1)
if np.any(feas):
    inc = X[feas][np.argmin(f_vals[feas])]  # Best feasible
else:
    inc = X[least_violation_index(C_vals)]  # Least infeasible

tr = TrustRegion(problem.lb, problem.ub, init_frac=0.8)
tr.set_center(inc.astype(np.float32))
```
**Reference**: Algorithm 2 doesn't show this explicitly, but SCBO paper (Eriksson & Poloczek 2021) initializes TR at best point

### Main Optimization Loop (Lines 56-103)

**Structure** (Algorithm 2, lines 4-14):
```python
for t in range(iters):
    # 1. Build surrogate models
    # 2. Generate candidates
    # 3. Select next evaluation point
    # 4. Evaluate and update
    # 5. Update trust region
```

**Step 1: Model Building** (Lines 60-66):
```python
if method == "scbo":
    mf, mcs, reducer = build_models_scbo(X, f_vals, C_vals)
elif method == "pca":
    mf, mcs, reducer = build_models_pca(X, f_vals, C_vals, n_components=n_components)
elif method == "kpca":
    mf, mcs, reducer = build_models_kpca(X, f_vals, C_vals, n_components=n_components, 
                                         gamma=kpca_gamma)
```
**Reference**: 
- Algorithm 2, line 5: "With c(x) ⊂ D_k compute projection P_k"
- Algorithm 2, line 6: "Project constraints... Fit GP for f(x) and c̃(x)"

**Dimensionality Reduction Happens Here**:
- **SCBO**: No reduction, mcs are G constraint GPs
- **PCA/kPCA**: mcs are g latent GPs, reducer stores projection P_k

**Step 2: Generate Candidates** (Lines 68-75):
```python
have_feas = np.any(np.all(C_vals <= 0.0, axis=1))
n_cand_this = n_cand if have_feas else max(n_cand, 8192)

cand_np = tr.sample(n_cand_this)
cand_t = torch.tensor(cand_np, dtype=torch.float64, device=device)
if problem.enforce_integer_teeth:
    cand_t[:, 2] = torch.round(cand_t[:, 2])
```
**Adaptive Candidate Count**:
- More candidates (≥8192) when no feasible point exists
- **Reason**: Feasibility-first acquisition needs more points to find feasible regions
- **Reference**: Common BO practice, not explicit in paper

**Step 3: Acquisition** (Lines 77-83):
```python
with torch.no_grad():
    if not have_feas:
        # No feasible points → feasibility-first
        best_idx = pick_candidate_feasibility_first(mcs, cand_t)
    else:
        # Have feasible points → constrained Thompson sampling
        best_idx = pick_candidate_cts(mf, mcs, cand_t, reducer=reducer)
    x_next = cand_t[best_idx].cpu().numpy()
```
**Two-Phase Strategy**:
1. **Phase 1** (no feasible): Maximize P(all constraints satisfied)
2. **Phase 2** (≥1 feasible): Optimize objective subject to constraints

**Reference**: Algorithm 2, line 7 - "x₊ ← CONSTRAINEDTHOMPSONSAMPLING"

**Step 4: Evaluation and Update** (Lines 85-88):
```python
f_next, g_next = problem.evaluate(x_next)
X = np.vstack([X, x_next])
f_vals = np.append(f_vals, f_next)
C_vals = np.vstack([C_vals, g_next])
```
**Reference**: Algorithm 2, lines 8-9 - "Evaluate x₊ and observe f(x₊), c(x₊)" and "D_{k+1} = D_k ∪ {x₊, f(x₊), c(x₊)}"

**Step 5: Trust Region Update** (Lines 90-101):
```python
# Determine if iteration was successful
prev_best = best_feasible_value(f_vals[:-1], C_vals[:-1])
curr_best = best_feasible_value(f_vals, C_vals)
success = (not np.isnan(curr_best)) and (np.isnan(prev_best) or curr_best < prev_best - 1e-12)
tr.step(success)

# Re-center trust region at current best
feas = np.all(C_vals <= 0.0, axis=1)
if np.any(feas):
    inc = X[feas][np.argmin(f_vals[feas])]
else:
    inc = X[least_violation_index(C_vals)]
tr.set_center(inc.astype(np.float32))

best_hist.append(curr_best)
```

**Success Definition**:
- Found new feasible point when none existed before, OR
- Found better feasible point than previous best (with tolerance 1e-12)

**Reference**: Algorithm 2, line 10 - "Update TuRBO state"

**Trust Region Dynamics**:
- **Success** → expand (after 3 consecutive)
- **Failure** → contract (after 3 consecutive)
- **Re-centering**: Always center at current incumbent
- **Why**: Keeps search focused on promising region

### Return Value (Lines 103-106)

```python
return {
    "X": X,               # All evaluated points
    "f": f_vals,          # All objective values
    "C": C_vals,          # All constraint values (G×N)
    "best_hist": np.array(best_hist, float),  # Best feasible per iteration
    "problem": problem
}
```

---

## Cell 7: Experiments and Visualization

### Purpose
Run comparative experiments (SCBO vs PCA-GP vs kPCA-GP) and visualize convergence.

### Experimental Setup (Lines 5-11)

```python
seed = 12345  # Reproducibility
n_init = 20   # Initial DoE size
iters = 100   # BO iterations  
n_cand = 4096 # Candidates per iteration
g_comp = 4    # PCA/kPCA latent dimensions
gamma = 0.2   # kPCA kernel width
```

**Reference**: Section 4.1 experimental settings
- Paper uses 20 experiments per method
- This notebook runs single seed for demonstration

### Running Experiments (Lines 13-24)

```python
print("=== SCBO ===")
res_scbo = run_bo("scbo", seed=seed, n_init=n_init, iters=iters, n_cand=n_cand)

print("\n=== PCA-GP SCBO ===")
res_pca = run_bo("pca", seed=seed, n_init=n_init, iters=iters, n_cand=n_cand, 
                 n_components=g_comp)

print("\n=== kPCA-GP SCBO ===")
res_kpca = run_bo("kpca", seed=seed, n_init=n_init, iters=iters, n_cand=n_cand,
                  n_components=g_comp, kpca_gamma=gamma)
```

**Same Seed**: Ensures fair comparison (same initial DoE)

### Visualization Function (Lines 27-40)

```python
def plot_convergence(*pairs, f_star=2996.3482):
    plt.figure(figsize=(7, 4))
    for name, res in pairs:
        y = res["best_hist"]
        x = np.arange(1, len(y) + 1)
        m = ~np.isnan(y)  # Mask infeasible iterations
        plt.plot(x[m], y[m], lw=2, label=name)
    plt.axhline(f_star, ls="--", c="k", lw=1.2, label="$f^*$≈{:.2f}".format(f_star))
    plt.xlabel("Evaluations"); plt.ylabel("Best feasible f")
    plt.title("Speed Reducer — Convergence (single seed)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
```

**Interpretation**:
- **X-axis**: Number of function evaluations (n_init + iteration count)
- **Y-axis**: Best feasible objective value found so far
- **Gaps**: When no feasible point exists yet (NaN values masked)
- **Horizontal line**: Known optimum f* = 2996.3482

### Results Comparison (Lines 42-51)

```python
plot_convergence(
    ("SCBO", res_scbo),
    ("PCA-GP SCBO (g=4)", res_pca),
    ("kPCA-GP SCBO (g=4)", res_kpca),
    f_star=res_scbo["problem"].f_star_ref
)

print("Final best (SCBO):     ", np.nanmin(res_scbo["best_hist"]))
print("Final best (PCA-GP):   ", np.nanmin(res_pca["best_hist"]))
print("Final best (kPCA-GP):  ", np.nanmin(res_kpca["best_hist"]))
```

**Expected Observations** (from Figure 3 in paper):
1. **All methods converge** to near-optimal solution (f* ≈ 2996.35)
2. **SCBO**: Fastest initial convergence (models full constraints)
3. **PCA-GP**: Slightly slower but competitive (g=4 captures 11 constraints well)
4. **kPCA-GP**: Similar to PCA (problem has approximately linear constraints)

**Reference**: Section 4.1, Table 1 shows comparative results:
- SCBO: 2996.3482 (best), 480s runtime
- PCA-GP SCBO: 2996.3482 (best), 192s runtime (60% faster!)
- kPCA-GP SCBO: 2996.3482 (best), 207s runtime (56% faster!)

---

## Summary of Key Contributions

### 1. Computational Efficiency (Section 3.4)

**Problem**: SCBO requires G+1 independent GPs
- Complexity: O((G+1)N³) training, O((G+1)N²) memory
- Infeasible for G > 1000 (e.g., aeroelastic tailoring with G=1786)

**Solution**: PCA/kPCA-GP SCBO reduces to g+1 GPs
- Complexity: O((g+1)N³ + G³) training, O((g+1)N²) memory  
- **Speed Reducer**: 60% faster with g=4, G=11
- **Aeroelastic**: Makes problem tractable (g=35 << G=1786)

### 2. When to Use Each Method

**SCBO**:
- ✅ Use when: G ≤ 20, sufficient computation available
- ✅ Advantages: Models constraints exactly, fastest convergence
- ❌ Limitations: Doesn't scale to large G

**PCA-GP SCBO**:
- ✅ Use when: G > 20, constraints have linear structure
- ✅ Advantages: Fast, interpretable, theoretically sound
- ⚠️ Check: Eigenvalue decay (Figure 6) - need rapid decay for few components

**kPCA-GP SCBO**:
- ✅ Use when: Constraints have nonlinear structure, slow eigenvalue decay
- ⚠️ Disadvantages: Requires tuning γ, pre-image approximation less accurate
- 📊 Paper finding: Minimal benefit over PCA for problems tested

### 3. Hyperparameter Guidance

**Number of Components (g)**:
- Start with eigenvalue threshold (λ_g > 0.01 in paper)
- Figure 4 in paper: Vary g ∈ {1, 2, 4, 6}
  - g=1: Too few, misses feasible region
  - g=2-4: Sweet spot for Speed Reducer
  - g=6: Marginal improvement, added cost

**kPCA Gamma (γ)**:
- Paper uses γ = 0.2 for Speed Reducer
- Smaller γ → smoother, broader features
- Larger γ → more local, complex features
- **Recommendation**: Start with γ = 1/G (rule of thumb)

### 4. Feasibility Handling Strategy

**Two-Phase Approach**:
```python
if no_feasible_points_yet:
    use_feasibility_first_acquisition()
else:
    use_constrained_thompson_sampling()
```

**Why Effective**:
- Phase 1: Rapidly explores to find *any* feasible point
- Phase 2: Efficiently optimizes once feasible region known
- **Reference**: Section 3.3 states this approach "finds feasible points efficiently"

### 5. Trust Region Best Practices

**Initialization**:
- Start at center of domain (or least infeasible point)
- Initial size: 80% of domain (init_frac=0.8)

**During Optimization**:
- Re-center at current incumbent every iteration
- Expand cautiously (after 3 successes)
- Contract aggressively (after 3 failures)

**Why It Works**:
- Focuses computation on promising regions
- Adapts to local landscape
- Prevents premature convergence

---

## Complete Algorithm Flow

```
1. INITIALIZATION
   ├─ Generate initial DoE (LHS)
   ├─ Evaluate all points
   └─ [Optional] Find feasible seed

2. FOR each iteration:
   │
   ├─ BUILD MODELS
   │  ├─ Fit GP for objective f(x)
   │  └─ Fit GPs for constraints:
   │     ├─ SCBO: G independent GPs for c₁, ..., c_G
   │     ├─ PCA-GP: g GPs for z₁, ..., z_g (via PCA)
   │     └─ kPCA-GP: g GPs for z₁, ..., z_g (via kPCA)
   │
   ├─ GENERATE CANDIDATES
   │  ├─ Sample N_cand points from trust region
   │  └─ [If integer vars] Round to nearest integer
   │
   ├─ SELECT NEXT POINT
   │  ├─ IF no feasible points yet:
   │  │  └─ Maximize P(all constraints satisfied)
   │  └─ ELSE:
   │     ├─ Sample from GP posteriors
   │     ├─ [If latent] Project z̃ → c̃
   │     ├─ Find feasible samples
   │     └─ Return argmin f̃(x) among feasible
   │
   ├─ EVALUATE
   │  ├─ Compute f(x_next), c(x_next)
   │  └─ Add to dataset D
   │
   └─ UPDATE TRUST REGION
      ├─ Determine if iteration successful
      ├─ Expand/contract accordingly
      └─ Re-center at current best

3. RETURN
   └─ Best feasible solution found
```

---

## References

### Primary Papers
1. **Maathuis, H.F., De Breuker, R., & Castro, S.G.P. (2025)**. High-Dimensional Bayesian Optimisation with Large-Scale Constraints via Latent Space Gaussian Processes. *arXiv preprint arXiv:2412.15679*.

2. **Eriksson, D. & Poloczek, M. (2021)**. Scalable Constrained Bayesian Optimization. *arXiv preprint arXiv:2002.08526*.

3. **Eriksson, D., Pearce, M., Gardner, J.R., Turner, R., & Poloczek, M. (2020)**. Scalable Global Optimization via Local Bayesian Optimization. *NeurIPS 2020*.

4. **Lemonge, A.C.C., Barbosa, H.J.C., Borges, C.C.H., & Silva, F.B.S. (2010)**. Constrained Optimization Problems in Mechanical Engineering Design Using a Real-Coded Steady-State Genetic Algorithm. *Mecánica Computacional, Vol XXIX*.

### Supporting Literature
5. **Rasmussen, C.E. & Williams, C.K.I. (2006)**. Gaussian Processes for Machine Learning. MIT Press.

6. **Hernández-Lobato, J.M., Requeima, J., Pyzer-Knapp, E.O., & Aspuru-Guzik, A. (2017)**. Parallel and Distributed Thompson Sampling for Large-scale Accelerated Exploration of Chemical Space. *ICML 2017*.

---

## Appendix: Quick Reference

### Key Equations from Paper

**GP Posterior** (Equations 7-8):
```
μ(x) = k(x,X)K⁻¹f
σ²(x) = k(x,x) - k(x,X)K⁻¹k(X,x)
```

**Constrained Acquisition** (Equation 12):
```
α_c(x|D) = α(x|D) ∏_{j=1}^G P(ĉ_j(x) ≤ 0)
```

**PCA Projection** (Equation 20):
```
Z = CΨ_g
```

**Trust Region Size** (Equation 14):
```
L_TR = (l_i L) / (∏_j l_j)^(1/D)
```

### Default Hyperparameters

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Population size | 200 | Section 4.1 |
| Initial DoE | 20 | Section 4.1 |
| BO iterations | 100 | Section 4.1 |
| Candidates/iter | 4096 | Section 4.1 |
| Latent dims (g) | 4 (Speed Reducer) | Section 4.1 |
| | 35 (Aeroelastic) | Section 4.2 |
| kPCA gamma | 0.2 | Section 4.1 |
| TR init size | 0.8 | TuRBO defaults |
| TR grow/shrink | 1.6 / 0.5 | TuRBO defaults |
| Success/fail tol | 3 / 3 | TuRBO defaults |

---

*End of Implementation Guide*

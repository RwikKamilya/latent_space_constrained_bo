"""
Full implementation of PCA-GP SCBO replication
Based on Maathuis et al. (2024): "High-Dimensional Bayesian Optimisation with 
Large-Scale Constraints via Latent Space Gaussian Processes"

This implementation follows:
- Algorithm 2 from the paper
- Uses BoTorch and GPyTorch as specified in Section 4
- Implements SCBO (Scalable Constrained BO) from Eriksson et al. (2020)
- Adds PCA/kPCA dimensionality reduction for constraints
"""

import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.decomposition import PCA, KernelPCA
from scipy.stats import qmc
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import problem definition (avoiding duplication)
from speed_reducer_problem import SpeedReducerProblem


# ============================================================================
# DESIGN OF EXPERIMENTS
# ============================================================================

def generate_lhs_samples(problem: SpeedReducerProblem, n_samples: int) -> np.ndarray:
    """
    Generate Latin Hypercube Samples (normalized to [0, 1])
    
    Args:
        problem: problem instance
        n_samples: number of samples
    
    Returns:
        X: samples (n_samples, dim) in [0, 1]
    """
    sampler = qmc.LatinHypercube(d=problem.dim)
    X_norm = sampler.random(n=n_samples)
    return X_norm


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

class ConstraintReducer:
    """Base class for constraint dimensionality reduction"""
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.model = None
    
    def fit_transform(self, C: np.ndarray) -> np.ndarray:
        """Fit reducer and transform constraints"""
        raise NotImplementedError
    
    def transform(self, C: np.ndarray) -> np.ndarray:
        """Transform constraints to latent space"""
        raise NotImplementedError
    
    def inverse_transform(self, C_reduced: np.ndarray) -> np.ndarray:
        """Transform from latent space back to original"""
        raise NotImplementedError


class PCAReducer(ConstraintReducer):
    """PCA-based constraint reduction (Equation 20 in paper)"""
    
    def __init__(self, n_components: int):
        super().__init__(n_components)
        self.model = None
    
    def fit_transform(self, C: np.ndarray) -> np.ndarray:
        n_components_adjusted = min(self.n_components, C.shape[0] - 1, C.shape[1])
        
        if n_components_adjusted != self.n_components:
            print(f"  Adjusting PCA components from {self.n_components} to {n_components_adjusted}")
        
        self.model = PCA(n_components=n_components_adjusted)
        C_reduced = self.model.fit_transform(C)
        
        # Diagnostic: Check variance explained
        var_ratio = self.model.explained_variance_ratio_
        total_var = var_ratio.sum()
        print(f"  PCA variance explained: {total_var:.2%} with {n_components_adjusted} components")
        if total_var < 0.90:
            print(f"  ⚠️  Warning: Only {total_var:.2%} variance captured! Consider more components.")
        
        return C_reduced
    
    def transform(self, C: np.ndarray) -> np.ndarray:
        return self.model.transform(C)
    
    def inverse_transform(self, C_reduced: np.ndarray) -> np.ndarray:
        return self.model.inverse_transform(C_reduced)


class KPCAReducer(ConstraintReducer):
    """Kernel PCA-based constraint reduction (Equation 26 in paper)"""
    
    def __init__(self, n_components: int, gamma: float = 0.1):
        super().__init__(n_components)
        self.gamma = gamma
        self.model = None
        self.fallback_to_pca = False
        self.pca_fallback = None
        
    def fit_transform(self, C: np.ndarray) -> np.ndarray:
        if not self.fallback_to_pca:
            try:
                n_components_adjusted = min(self.n_components, C.shape[0] - 1, C.shape[1])
                
                self.model = KernelPCA(
                    n_components=n_components_adjusted,
                    kernel='rbf',
                    gamma=self.gamma,
                    fit_inverse_transform=True
                )
                
                C_reduced = self.model.fit_transform(C)
                return C_reduced
                
            except (ValueError, np.linalg.LinAlgError) as e:
                print(f"  Warning: kPCA failed, falling back to PCA")
                self.fallback_to_pca = True
        
        if self.fallback_to_pca:
            n_components_adjusted = min(self.n_components, C.shape[0] - 1, C.shape[1])
            self.pca_fallback = PCA(n_components=n_components_adjusted)
            return self.pca_fallback.fit_transform(C)
    
    def transform(self, C: np.ndarray) -> np.ndarray:
        if self.fallback_to_pca:
            return self.pca_fallback.transform(C)
        return self.model.transform(C)
    
    def inverse_transform(self, C_reduced: np.ndarray) -> np.ndarray:
        if self.fallback_to_pca:
            return self.pca_fallback.inverse_transform(C_reduced)
        return self.model.inverse_transform(C_reduced)


# ============================================================================
# TRUST REGION MANAGEMENT (from SCBO/TuRBO)
# ============================================================================

class TrustRegion:
    """
    Trust Region management for SCBO
    Based on Eriksson et al. (2020)
    
    This implements the trust region logic mentioned in paper Section 2.4
    """
    
    def __init__(
        self,
        dim: int,
        L_init: float = 0.8,
        L_min: float = 0.5**7,
        L_max: float = 1.6,
        success_tolerance: int = 3,
        failure_tolerance: int = 4
    ):
        self.dim = dim
        self.L = L_init
        self.L_min = L_min
        self.L_max = L_max
        
        self.success_tolerance = success_tolerance
        self.failure_tolerance = failure_tolerance
        
        self.success_counter = 0
        self.failure_counter = 0
        
        self.center = None
        self.best_value = np.inf
        self.restart_triggered = False
    
    def update(self, x_new: np.ndarray, f_new: float, feasible: bool):
        """
        Update trust region based on new evaluation
        
        Args:
            x_new: new point evaluated
            f_new: objective value at x_new
            feasible: whether x_new is feasible
        """
        if feasible and f_new < self.best_value:
            # Success: better feasible point found
            self.success_counter += 1
            self.failure_counter = 0
            self.best_value = f_new
            self.center = x_new.copy()
            
            if self.success_counter == self.success_tolerance:
                # Expand trust region
                self.L = min(2.0 * self.L, self.L_max)
                self.success_counter = 0
        else:
            # Failure: no improvement
            self.success_counter = 0
            self.failure_counter += 1
            
            if self.failure_counter == self.failure_tolerance:
                # Shrink trust region
                self.L = max(0.5 * self.L, self.L_min)
                self.failure_counter = 0
                
                if self.L == self.L_min:
                    # Trigger restart
                    self.restart_triggered = True
    
    def get_bounds(self, problem_bounds: torch.Tensor) -> torch.Tensor:
        """
        Get current trust region bounds
        
        Args:
            problem_bounds: global bounds (2, dim)
        
        Returns:
            tr_bounds: trust region bounds (2, dim)
        """
        if self.center is None:
            return problem_bounds
        
        # Create bounds around center
        lb = torch.clamp(
            torch.tensor(self.center - self.L/2, dtype=torch.float64),
            problem_bounds[0],
            problem_bounds[1]
        )
        ub = torch.clamp(
            torch.tensor(self.center + self.L/2, dtype=torch.float64),
            problem_bounds[0],
            problem_bounds[1]
        )
        
        return torch.stack([lb, ub])
    
    def restart(self, x_new: np.ndarray):
        """Restart trust region at new location"""
        self.L = 0.8  # Reset to initial size
        self.center = x_new.copy()
        self.success_counter = 0
        self.failure_counter = 0
        self.restart_triggered = False


# ============================================================================
# CONSTRAINED THOMPSON SAMPLING (Algorithm 1 in Eriksson et al. 2020)
# ============================================================================

def constrained_thompson_sampling(
    model_f: SingleTaskGP,
    models_c: List[SingleTaskGP],
    reducer: Optional[ConstraintReducer],
    bounds: torch.Tensor,
    n_candidates: int = 1000,
    problem: Optional[SpeedReducerProblem] = None
) -> torch.Tensor:
    """
    Constrained Thompson Sampling acquisition (Algorithm 1 from SCBO paper)
    
    This implements the acquisition strategy from Eriksson et al. (2020),
    used in Algorithm 2 of Maathuis et al. (2024)
    
    Args:
        model_f: GP for objective
        models_c: list of GPs for (possibly reduced) constraints
        reducer: constraint reducer (PCA/kPCA) or None
        bounds: search bounds (2, dim)
        n_candidates: number of candidate points
        problem: problem instance (for inverse transform if needed)
    
    Returns:
        x_next: next point to evaluate (1, dim)
    """
    dim = bounds.shape[1]
    
    # Generate candidate points uniformly in trust region
    candidates = torch.rand(n_candidates, dim, dtype=torch.float64) * \
                (bounds[1] - bounds[0]) + bounds[0]
    
    # Sample from GP posteriors (Thompson Sampling)
    model_f.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_dist = model_f.posterior(candidates)
        f_samples = f_dist.rsample().squeeze()
    
    # Sample constraints (in latent or original space)
    c_samples_latent = []
    for model_c in models_c:
        model_c.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            c_dist = model_c.posterior(candidates)
            c_sample = c_dist.rsample().squeeze()
            c_samples_latent.append(c_sample.numpy())
    
    c_samples_latent = np.column_stack(c_samples_latent)  # (n_candidates, g or G)
    
    # Project back to original constraint space if using dimensionality reduction
    if reducer is not None:
        c_samples_original = reducer.inverse_transform(c_samples_latent)
    else:
        c_samples_original = c_samples_latent
    
    # Find feasible candidates
    feasible_mask = np.all(c_samples_original <= 0, axis=1)
    
    if np.any(feasible_mask):
        # Select feasible point with best (minimum) objective
        feasible_f = f_samples[feasible_mask]
        best_idx_in_feasible = torch.argmin(feasible_f)
        feasible_indices = np.where(feasible_mask)[0]
        best_idx = feasible_indices[best_idx_in_feasible]
    else:
        # Select point with minimum constraint violation
        violations = np.maximum(c_samples_original, 0).sum(axis=1)
        best_idx = np.argmin(violations)
    
    x_next = candidates[best_idx:best_idx+1]
    
    return x_next


# ============================================================================
# MAIN OPTIMIZATION ALGORITHM (Algorithm 2 from paper)
# ============================================================================

class PCAGPSCBO:
    """
    PCA-GP SCBO optimizer
    
    Implements Algorithm 2 from Maathuis et al. (2024):
    "SCBO with Latent Gaussian Processes"
    
    Uses BoTorch and GPyTorch as specified in Section 4 of the paper.
    """
    
    def __init__(
        self,
        problem: SpeedReducerProblem,
        n_components: int = 4,
        use_kpca: bool = False,
        use_pca: bool = True,
        n_initial: int = 20,
        max_evals: int = 100,
        n_candidates: int = 1000,
        batch_size: int = 1
    ):
        self.problem = problem
        self.n_components = n_components
        self.n_initial = n_initial
        self.max_evals = max_evals
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        self.use_pca = use_pca
        self.use_kpca = use_kpca
        
        # Choose reducer (only if using dimensionality reduction)
        if use_pca or use_kpca:
            if use_kpca:
                self.reducer = KPCAReducer(n_components=n_components)
            else:
                self.reducer = PCAReducer(n_components=n_components)
        else:
            self.reducer = None  # Standard SCBO without reduction
        
        # Trust Region for SCBO
        self.trust_region = TrustRegion(dim=problem.dim)
        
        # Data storage (normalized)
        self.X = None
        self.f = None
        self.C = None
        
        # Models
        self.model_f = None
        self.models_c = None
    
    def initialize(self):
        """Generate initial DoE using LHS"""
        print(f"Generating {self.n_initial} initial samples with LHS...")
        self.X = generate_lhs_samples(self.problem, self.n_initial)
        
        self.f = []
        self.C = []
        for x in self.X:
            f_val, c_val = self.problem.evaluate(x, normalized=True)
            self.f.append(f_val)
            self.C.append(c_val)
        
        self.f = np.array(self.f)
        self.C = np.array(self.C)
        
        # Initialize trust region at best feasible point (or best point if none feasible)
        feasible_mask = np.all(self.C <= 0, axis=1)
        if np.any(feasible_mask):
            best_idx = np.argmin(self.f[feasible_mask])
            best_x = self.X[feasible_mask][best_idx]
            best_f = self.f[feasible_mask][best_idx]
            self.trust_region.center = best_x
            self.trust_region.best_value = best_f
        else:
            best_idx = np.argmin(self.f)
            self.trust_region.center = self.X[best_idx]
            self.trust_region.best_value = self.f[best_idx]
        
        print(f"Initial best: {np.min(self.f):.4f}")
        print(f"Feasible in initial: {np.sum(feasible_mask)}/{self.n_initial}")
    
    def build_models(self):
        """Build GP models using BoTorch"""
        # Reduce constraints if using dimensionality reduction
        if self.reducer is not None:
            C_reduced = self.reducer.fit_transform(self.C)
            actual_n_components = C_reduced.shape[1]
            n_constraint_gps = min(actual_n_components, self.n_components)
            
            if actual_n_components != self.n_components:
                reduction_type = "kPCA" if self.use_kpca else "PCA"
                print(f"  {reduction_type} produced {actual_n_components} components")
        else:
            # Standard SCBO: model all constraints independently
            C_reduced = self.C
            n_constraint_gps = self.problem.n_constraints
        
        # Convert to tensors
        X_tensor = torch.tensor(self.X, dtype=torch.float64)
        f_tensor = torch.tensor(self.f, dtype=torch.float64).unsqueeze(-1)
        
        # Build GP for objective using BoTorch
        self.model_f = SingleTaskGP(X_tensor, f_tensor)
        mll_f = ExactMarginalLogLikelihood(self.model_f.likelihood, self.model_f)
        fit_gpytorch_mll(mll_f)
        
        # Build GPs for (possibly reduced) constraints using BoTorch
        self.models_c = []
        for i in range(n_constraint_gps):
            if i >= C_reduced.shape[1]:
                break
                
            c_tensor = torch.tensor(C_reduced[:, i], dtype=torch.float64).unsqueeze(-1)
            model_c = SingleTaskGP(X_tensor, c_tensor)
            mll_c = ExactMarginalLogLikelihood(model_c.likelihood, model_c)
            fit_gpytorch_mll(mll_c)
            self.models_c.append(model_c)
        
        print(f"  Built 1 objective + {len(self.models_c)} constraint GP models")
    
    def optimize(self):
        """Main optimization loop (Algorithm 2 from paper)"""
        self.initialize()
        
        for iteration in range(self.max_evals):
            print(f"\nIteration {iteration + 1}/{self.max_evals}, TR size: {self.trust_region.L:.4f}")
            
            # Build/update models
            self.build_models()
            
            # Get trust region bounds
            tr_bounds = self.trust_region.get_bounds(self.problem.bounds)
            
            # Acquisition: Constrained Thompson Sampling in trust region
            x_next = constrained_thompson_sampling(
                self.model_f,
                self.models_c,
                self.reducer,
                tr_bounds,
                self.n_candidates,
                self.problem
            )
            
            # Evaluate
            x_next_np = x_next.numpy()[0]
            f_next, c_next = self.problem.evaluate(x_next_np, normalized=True)
            feasible = np.all(c_next <= 0)
            
            # Update trust region
            self.trust_region.update(x_next_np, f_next, feasible)
            
            # Check for restart
            if self.trust_region.restart_triggered:
                print("  Trust region restart triggered")
                # Restart at a random feasible point or best point
                feasible_mask = np.all(self.C <= 0, axis=1)
                if np.any(feasible_mask):
                    restart_idx = np.random.choice(np.where(feasible_mask)[0])
                else:
                    restart_idx = np.argmin(self.f)
                self.trust_region.restart(self.X[restart_idx])
            
            # Update dataset
            self.X = np.vstack([self.X, x_next_np])
            self.f = np.append(self.f, f_next)
            self.C = np.vstack([self.C, c_next])
            
            # Report progress
            feasible_mask = np.all(self.C <= 0, axis=1)
            if np.any(feasible_mask):
                best_f = np.min(self.f[feasible_mask])
                print(f"  New point: f={f_next:.4f}, feasible={feasible}")
                print(f"  Best feasible: {best_f:.4f} (gap: {best_f - self.problem.optimal_value:.4f})")
            else:
                print(f"  New point: f={f_next:.4f}, feasible={feasible}")
                print(f"  No feasible solution yet")
    
    def get_result(self):
        """Get best result"""
        feasible_mask = np.all(self.C <= 0, axis=1)
        
        if np.any(feasible_mask):
            best_idx = np.argmin(self.f[feasible_mask])
            best_x = self.X[feasible_mask][best_idx]
            best_f = self.f[feasible_mask][best_idx]
            feasible = True
        else:
            best_idx = np.argmin(self.f)
            best_x = self.X[best_idx]
            best_f = self.f[best_idx]
            feasible = False
        
        return {
            'x': best_x,
            'f': best_f,
            'feasible': feasible,
            'X_history': self.X,
            'f_history': self.f,
            'C_history': self.C
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_single_trial(
    problem=None,
    n_initial_samples=20,
    budget=100,
    use_pca=False,
    use_kpca=False,
    n_components=4,
    batch_size=1,
    population_size=1000
):
    """Run a single optimization trial"""
    import time
    
    if problem is None:
        problem = SpeedReducerProblem()
    
    optimizer = PCAGPSCBO(
        problem=problem,
        n_components=n_components,
        use_pca=use_pca,
        use_kpca=use_kpca,
        n_initial=n_initial_samples,
        max_evals=budget,
        n_candidates=population_size,
        batch_size=batch_size
    )
    
    start_time = time.time()
    optimizer.optimize()
    computation_time = time.time() - start_time
    
    result = optimizer.get_result()
    
    # Format result
    feasible_mask = np.all(optimizer.C <= 0, axis=1)
    best_feasible_value = None
    if np.any(feasible_mask):
        best_feasible_value = np.min(optimizer.f[feasible_mask])
    
    return {
        'x': result['x'],
        'f': result['f'],
        'feasible': result['feasible'],
        'best_feasible_value': best_feasible_value,
        'X_history': result['X_history'],
        'f_history': result['f_history'],
        'C_history': result['C_history'],
        'history': optimizer.f.tolist(),
        'computation_time': computation_time
    }


def run_multiple_trials(
    problem=None,
    n_trials=50,
    n_initial_samples=20,
    budget=100,
    use_pca=False,
    use_kpca=False,
    n_components=4,
    batch_size=1,
    population_size=1000
):
    """Run multiple trials for statistics"""
    results = []
    
    for trial in range(n_trials):
        print(f"\n{'='*60}")
        print(f"TRIAL {trial + 1}/{n_trials}")
        print(f"{'='*60}")
        
        result = run_single_trial(
            problem=problem,
            n_initial_samples=n_initial_samples,
            budget=budget,
            use_pca=use_pca,
            use_kpca=use_kpca,
            n_components=n_components,
            batch_size=batch_size,
            population_size=population_size
        )
        
        results.append(result)
        
        best_val = result['best_feasible_value'] if result['best_feasible_value'] is not None else result['f']
        print(f"\nTrial {trial + 1} result: {best_val:.6f} (feasible: {result['feasible']})")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PCA-GP SCBO Implementation")
    print("Based on Maathuis et al. (2024)")
    print("Using BoTorch + GPyTorch + SCBO Trust Regions")
    print("="*80)
    
    # Run single trial to test
    print("\nRunning single trial with PCA-GP SCBO...")
    result = run_single_trial(
        n_initial_samples=20,
        budget=100,
        use_pca=True,
        use_kpca=False,
        n_components=4,
        population_size=1000
    )
    
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    if result['best_feasible_value']:
        print(f"Best objective: {result['best_feasible_value']:.6f}")
        print(f"Known optimal: 2996.3482")
        print(f"Gap: {result['best_feasible_value'] - 2996.3482:.6f}")
    else:
        print(f"Best (infeasible): {result['f']:.6f}")
    print(f"Feasible: {result['feasible']}")
    print(f"Computation time: {result['computation_time']:.2f}s")

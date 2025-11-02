"""
Skeleton implementation for PCA-GP SCBO replication
Based on Maathuis et al. (2024)
"""

import numpy as np
import torch
import gpytorch
import botorch
from sklearn.decomposition import PCA, KernelPCA
from scipy.stats import qmc
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PROBLEM DEFINITION: 7D Speed Reducer
# ============================================================================

class SpeedReducerProblem:
    """
    7D Speed Reducer optimization problem with 11 constraints
    
    Variables:
    x1: face width [2.6, 3.6]
    x2: module of teeth [0.7, 0.8]  
    x3: number of teeth [17, 28] (integer)
    x4: shaft 1 length [7.3, 8.3]
    x5: shaft 2 length [7.8, 8.3]
    x6: shaft 1 diameter [2.9, 3.9]
    x7: shaft 2 diameter [5.0, 5.5]
    """
    
    def __init__(self):
        self.dim = 7
        self.n_constraints = 11
        self.bounds_lower = np.array([2.6, 0.7, 17, 7.3, 7.8, 2.9, 5.0])
        self.bounds_upper = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])
        self.optimal_value = 2996.3482
    
    def evaluate(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Evaluate objective and constraints
        
        Args:
            x: design variables (7,)
        
        Returns:
            f: objective value (weight)
            c: constraint values (11,) where c[i] <= 0 is feasible
        """
        x1, x2, x3, x4, x5, x6, x7 = x
        
        # Objective: minimize weight
        f = (0.7854 * x1 * x2**2 * (3.3333*x3**2 + 14.9334*x3 - 43.0934)
             - 1.508 * x1 * (x6**2 + x7**2)
             + 7.4777 * (x6**3 + x7**3)
             + 0.7854 * (x4*x6**2 + x5*x7**2))
        
        # Constraints (reformulated as g(x) <= 0)
        c = np.zeros(11)
        c[0] = 27/(x1 * x2**2 * x3) - 1
        c[1] = 397.5/(x1 * x2**2 * x3**2) - 1
        c[2] = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1
        c[3] = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1
        
        term5 = np.sqrt((745*x4/(x2*x3))**2 + 16.9e6)
        c[4] = term5 / (0.1 * x6**3) - 1100
        
        term6 = np.sqrt((745*x5/(x2*x3))**2 + 157.5e6)
        c[5] = term6 / (0.1 * x7**3) - 850
        
        c[6] = x2*x3 - 40
        c[7] = 5 - x1/x2
        c[8] = x1/x2 - 12
        c[9] = (1.5*x6 + 1.9)/x4 - 1
        c[10] = (1.1*x7 + 1.9)/x5 - 1
        
        return f, c


# ============================================================================
# DESIGN OF EXPERIMENTS
# ============================================================================

def generate_lhs_samples(problem: SpeedReducerProblem, n_samples: int) -> np.ndarray:
    """
    Generate Latin Hypercube Samples
    
    Args:
        problem: problem instance
        n_samples: number of samples
    
    Returns:
        X: samples (n_samples, dim)
    """
    sampler = qmc.LatinHypercube(d=problem.dim)
    samples = sampler.random(n=n_samples)
    
    # Scale to problem bounds
    X = qmc.scale(samples, problem.bounds_lower, problem.bounds_upper)
    
    # Round x3 (index 2) to integer
    X[:, 2] = np.round(X[:, 2])
    
    return X


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
    """PCA-based constraint reduction"""
    
    def __init__(self, n_components: int):
        super().__init__(n_components)
        self.model = None
    
    def fit_transform(self, C: np.ndarray) -> np.ndarray:
        # Adjust n_components if necessary (can't exceed n_samples - 1 or n_features)
        n_components_adjusted = min(self.n_components, C.shape[0] - 1, C.shape[1])
        
        if n_components_adjusted != self.n_components:
            print(f"  Adjusting PCA components from {self.n_components} to {n_components_adjusted}")
        
        self.model = PCA(n_components=n_components_adjusted)
        return self.model.fit_transform(C)
    
    def transform(self, C: np.ndarray) -> np.ndarray:
        return self.model.transform(C)
    
    def inverse_transform(self, C_reduced: np.ndarray) -> np.ndarray:
        return self.model.inverse_transform(C_reduced)


class KPCAReducer(ConstraintReducer):
    """Kernel PCA-based constraint reduction with fallback to PCA"""
    
    def __init__(self, n_components: int, gamma: float = 0.1):
        super().__init__(n_components)
        self.gamma = gamma
        self.model = None
        self.fallback_to_pca = False
        self.pca_fallback = None
        
    def fit_transform(self, C: np.ndarray) -> np.ndarray:
        # Try kPCA first
        if not self.fallback_to_pca:
            try:
                # Adjust n_components if necessary (can't exceed n_samples - 1)
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
                print(f"  Warning: kPCA failed ({type(e).__name__}: {str(e)[:50]}...)")
                print(f"  Falling back to standard PCA")
                self.fallback_to_pca = True
            except Exception as e:
                # Catch any other unexpected errors
                print(f"  Warning: kPCA failed with unexpected error ({type(e).__name__})")
                print(f"  Falling back to standard PCA")
                self.fallback_to_pca = True
        
        # Fallback to PCA if kPCA failed
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
# GAUSSIAN PROCESS MODELS
# ============================================================================

class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model with RBF kernel"""
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(model, likelihood, train_x, train_y, training_iter=50):
    """Train GP hyperparameters"""
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    likelihood.eval()
    
    return model, likelihood


# ============================================================================
# CONSTRAINED THOMPSON SAMPLING
# ============================================================================

def constrained_thompson_sampling(
    model_f: ExactGPModel,
    models_c: List[ExactGPModel],
    reducer: ConstraintReducer,
    problem: SpeedReducerProblem,
    n_candidates: int = 1000
) -> np.ndarray:
    """
    Constrained Thompson Sampling acquisition
    
    Args:
        model_f: GP for objective
        models_c: list of GPs for (possibly reduced) constraints
        reducer: constraint reducer (PCA/kPCA) or None for standard SCBO
        problem: problem instance
        n_candidates: number of candidate points
    
    Returns:
        x_next: next point to evaluate
    """
    # Generate candidate points
    candidates = []
    for _ in range(n_candidates):
        x = np.random.uniform(problem.bounds_lower, problem.bounds_upper)
        x[2] = np.round(x[2])  # Integer constraint
        candidates.append(x)
    
    candidates = torch.tensor(candidates, dtype=torch.float32)
    
    # Sample from GP posteriors
    model_f.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_dist = model_f(candidates)
        f_samples = f_dist.sample()
    
    # Sample constraints (in latent or original space)
    c_samples_latent = []
    for model_c in models_c:
        model_c.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            c_dist = model_c(candidates)
            c_sample = c_dist.sample()
            c_samples_latent.append(c_sample.numpy())
    
    c_samples_latent = np.column_stack(c_samples_latent)  # (n_candidates, g or G)
    
    # Project back to original constraint space (if using dimensionality reduction)
    if reducer is not None:
        c_samples_original = reducer.inverse_transform(c_samples_latent)  # (n_candidates, G)
    else:
        # Standard SCBO: already in original space
        c_samples_original = c_samples_latent
    
    # Find feasible candidates
    feasible_mask = np.all(c_samples_original <= 0, axis=1)
    
    if np.any(feasible_mask):
        # Select feasible point with best objective
        feasible_f = f_samples[feasible_mask]
        best_idx_in_feasible = torch.argmin(feasible_f)
        feasible_indices = np.where(feasible_mask)[0]
        best_idx = feasible_indices[best_idx_in_feasible]
    else:
        # Select point with minimum constraint violation
        violations = np.maximum(c_samples_original, 0).sum(axis=1)
        best_idx = np.argmin(violations)
    
    x_next = candidates[best_idx].numpy()
    
    return x_next


# ============================================================================
# MAIN OPTIMIZATION ALGORITHM
# ============================================================================

class PCAGPSCBO:
    """PCA-GP SCBO optimizer"""
    
    def __init__(
        self,
        problem: SpeedReducerProblem,
        n_components: int = 4,
        use_kpca: bool = False,
        use_pca: bool = True,
        n_initial: int = 20,
        max_evals: int = 100,
        n_candidates: int = 1000
    ):
        self.problem = problem
        self.n_components = n_components
        self.n_initial = n_initial
        self.max_evals = max_evals
        self.n_candidates = n_candidates
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
        
        # Data storage
        self.X = None
        self.f = None
        self.C = None
        
        # Models
        self.model_f = None
        self.models_c = None
    
    def initialize(self):
        """Generate initial DoE"""
        print(f"Generating {self.n_initial} initial samples...")
        self.X = generate_lhs_samples(self.problem, self.n_initial)
        
        self.f = []
        self.C = []
        for x in self.X:
            f_val, c_val = self.problem.evaluate(x)
            self.f.append(f_val)
            self.C.append(c_val)
        
        self.f = np.array(self.f)
        self.C = np.array(self.C)
        
        print(f"Initial best: {np.min(self.f):.4f}")
        print(f"Feasible in initial: {np.sum(np.all(self.C <= 0, axis=1))}/{self.n_initial}")
    
    def build_models(self):
        """Build GP models"""
        # Reduce constraints (if using dimensionality reduction)
        if self.reducer is not None:
            C_reduced = self.reducer.fit_transform(self.C)
            
            # Check actual number of components produced
            # (PCA/kPCA can produce fewer than requested if insufficient variance or data)
            actual_n_components = C_reduced.shape[1]
            
            # Ensure we don't try to use more components than actually produced
            if actual_n_components != self.n_components:
                reduction_type = "kPCA" if self.use_kpca else "PCA"
                print(f"  Warning: {reduction_type} produced {actual_n_components} components "
                      f"(requested {self.n_components})")
            
            n_constraint_gps = min(actual_n_components, self.n_components, C_reduced.shape[1])
            print(f"  Building {n_constraint_gps} constraint GP models")
        else:
            # Standard SCBO: model all constraints independently
            C_reduced = self.C
            n_constraint_gps = self.problem.n_constraints
        
        # Safety check: ensure C_reduced has the right shape
        assert C_reduced.shape[0] == len(self.X), "Mismatch in number of samples"
        assert C_reduced.shape[1] >= n_constraint_gps, f"Not enough components: {C_reduced.shape[1]} < {n_constraint_gps}"
        
        # Convert to tensors
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        f_tensor = torch.tensor(self.f, dtype=torch.float32)
        
        # Build GP for objective
        likelihood_f = gpytorch.likelihoods.GaussianLikelihood()
        self.model_f = ExactGPModel(X_tensor, f_tensor, likelihood_f)
        self.model_f, likelihood_f = train_gp(self.model_f, likelihood_f, X_tensor, f_tensor)
        
        # Build GPs for (possibly reduced) constraints
        self.models_c = []
        for i in range(n_constraint_gps):
            # Safety check before indexing
            if i >= C_reduced.shape[1]:
                print(f"  ERROR: Trying to access component {i} but C_reduced only has {C_reduced.shape[1]} components")
                break
                
            c_tensor = torch.tensor(C_reduced[:, i], dtype=torch.float32)
            likelihood_c = gpytorch.likelihoods.GaussianLikelihood()
            model_c = ExactGPModel(X_tensor, c_tensor, likelihood_c)
            model_c, likelihood_c = train_gp(model_c, likelihood_c, X_tensor, c_tensor)
            self.models_c.append(model_c)
        
        print(f"  Built {len(self.models_c)} constraint GP models")
    
    def optimize(self):
        """Main optimization loop"""
        self.initialize()
        
        for iteration in range(self.max_evals):
            print(f"\nIteration {iteration + 1}/{self.max_evals}")
            
            # Build/update models
            self.build_models()
            
            # Acquisition: select next point
            x_next = constrained_thompson_sampling(
                self.model_f,
                self.models_c,
                self.reducer,
                self.problem,
                self.n_candidates
            )
            
            # Evaluate
            f_next, c_next = self.problem.evaluate(x_next)
            
            # Update dataset
            self.X = np.vstack([self.X, x_next])
            self.f = np.append(self.f, f_next)
            self.C = np.vstack([self.C, c_next])
            
            # Report progress
            feasible_mask = np.all(self.C <= 0, axis=1)
            if np.any(feasible_mask):
                best_f = np.min(self.f[feasible_mask])
                best_idx = np.argmin(self.f[feasible_mask])
                best_x = self.X[feasible_mask][best_idx]
                print(f"  New point: f={f_next:.4f}, feasible={np.all(c_next <= 0)}")
                print(f"  Best feasible so far: {best_f:.4f}")
            else:
                print(f"  New point: f={f_next:.4f}, feasible={np.all(c_next <= 0)}")
                print(f"  No feasible solution found yet")
    
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
# MAIN SCRIPT
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
    
    # Note: batch_size is not used in current implementation but kept for API compatibility
    
    optimizer = PCAGPSCBO(
        problem=problem,
        n_components=n_components,
        use_pca=use_pca,
        use_kpca=use_kpca,
        n_initial=n_initial_samples,
        max_evals=budget,
        n_candidates=population_size
    )
    
    start_time = time.time()
    optimizer.optimize()
    computation_time = time.time() - start_time
    
    result = optimizer.get_result()
    
    # Format result to match expected structure in run_experiments.py
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
    # Run single trial to test
    print("Running single trial with PCA-GP SCBO...")
    result = run_single_trial(
        n_initial_samples=20,
        budget=100,
        use_pca=True,
        use_kpca=False,
        n_components=4
    )
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best objective: {result['best_feasible_value']:.6f}" if result['best_feasible_value'] else f"Best (infeasible): {result['f']:.6f}")
    print(f"Feasible: {result['feasible']}")
    print(f"Known optimal: 2996.3482")
    if result['best_feasible_value']:
        print(f"Gap: {result['best_feasible_value'] - 2996.3482:.6f}")
    
    # Uncomment to run full statistical experiment
    # print("\n\nRunning 20 trials for statistical analysis...")
    # results = run_multiple_trials(n_trials=20, use_pca=True, use_kpca=False, n_components=4)

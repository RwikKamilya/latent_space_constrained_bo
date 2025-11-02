"""
Speed Reducer Design Problem

7D optimization problem with 11 black-box constraints from mechanical engineering.
This is a benchmark problem used in Maathuis et al. (2024) and many other papers.

Reference:
- Lemonge et al. (2010) - "Constrained optimization problems in mechanical 
  engineering design using a real-coded steady-state genetic algorithm"
- Maathuis et al. (2024) - Section 4.1

Known optimum: f* = 2996.3482
Optimal design variables:
  x* = [3.5, 0.7, 17, 7.3, 7.8, 3.350215, 5.286683]
"""

import numpy as np


class SpeedReducerProblem:
    """
    Speed Reducer Design Optimization Problem
    
    Minimize the weight of a speed reducer subject to constraints on:
    - Bending stress of gear teeth
    - Contact stress 
    - Transverse deflections of shafts
    - Stresses in the shafts
    
    Design Variables:
    - x1: face width (b) ∈ [2.6, 3.6]
    - x2: module of teeth (m) ∈ [0.7, 0.8]  
    - x3: number of teeth on pinion (n) ∈ [17, 28] (integer)
    - x4: length of shaft 1 between bearings (l1) ∈ [7.3, 8.3]
    - x5: length of shaft 2 between bearings (l2) ∈ [7.8, 8.3]
    - x6: diameter of shaft 1 (d1) ∈ [2.9, 3.9]
    - x7: diameter of shaft 2 (d2) ∈ [5.0, 5.5]
    """
    
    def __init__(self):
        self.name = "Speed Reducer"
        self.n_dim = 7
        self.n_constraints = 11
        
        # Variable bounds
        self.bounds = np.array([
            [2.6, 3.6],   # x1: face width
            [0.7, 0.8],   # x2: module of teeth
            [17, 28],     # x3: number of teeth (integer)
            [7.3, 8.3],   # x4: shaft 1 length
            [7.8, 8.3],   # x5: shaft 2 length
            [2.9, 3.9],   # x6: shaft 1 diameter
            [5.0, 5.5]    # x7: shaft 2 diameter
        ])
        
        # Known optimum
        self.optimal_value = 2996.3482
        self.optimal_solution = np.array([3.5, 0.7, 17, 7.3, 7.8, 3.350215, 5.286683])
        
        # Compatibility aliases for different naming conventions
        self.dim = self.n_dim
        self.bounds_lower = self.bounds[:, 0]
        self.bounds_upper = self.bounds[:, 1]
        
    def evaluate(self, x):
        """
        Evaluate objective function and constraints
        
        Parameters
        ----------
        x : array-like, shape (7,)
            Design variables
            
        Returns
        -------
        f : float
            Objective function value (weight)
        c : array, shape (11,)
            Constraint violations (c <= 0 is feasible)
        """
        x = np.atleast_1d(x)
        
        # Round x3 to nearest integer (number of teeth)
        x = x.copy()
        x[2] = np.round(x[2])
        
        # Unpack design variables
        x1, x2, x3, x4, x5, x6, x7 = x
        
        # Objective: minimize weight
        f = (0.7854 * x1 * x2**2 * 
             (3.3333 * x3**2 + 14.9334 * x3 - 43.0934) - 
             1.508 * x1 * (x6**2 + x7**2) + 
             7.4777 * (x6**3 + x7**3) + 
             0.7854 * (x4 * x6**2 + x5 * x7**2))
        
        # Constraints (all should be <= 0)
        c = np.zeros(11)
        
        # g1: Bending stress constraint 1
        c[0] = 27 / (x1 * x2**2 * x3) - 1
        
        # g2: Bending stress constraint 2  
        c[1] = 397.5 / (x1 * x2**2 * x3**2) - 1
        
        # g3: Transverse deflection constraint for shaft 1
        c[2] = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1
        
        # g4: Transverse deflection constraint for shaft 2
        c[3] = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1
        
        # g5: Stress constraint for shaft 1
        f_term = (745 * x4 / (x2 * x3))**2 + 16.9e6
        c[4] = np.sqrt(f_term) / (0.1 * x6**3) - 1100
        
        # g6: Stress constraint for shaft 2
        f_term = (745 * x5 / (x2 * x3))**2 + 157.5e6
        c[5] = np.sqrt(f_term) / (0.1 * x7**3) - 850
        
        # g7: Gear ratio constraint
        c[6] = x2 * x3 - 40
        
        # g8: Lower bound on face width to module ratio
        c[7] = 5 - x1 / x2
        
        # g9: Upper bound on face width to module ratio
        c[8] = x1 / x2 - 12
        
        # g10: Lower bound on shaft 1 geometry
        c[9] = 1.5 * x6 + 1.9 - x4
        
        # g11: Lower bound on shaft 2 geometry
        c[10] = 1.1 * x7 + 1.9 - x5
        
        return f, c
    
    def is_feasible(self, x):
        """Check if a design is feasible"""
        _, c = self.evaluate(x)
        return np.all(c <= 0)
    
    def get_random_sample(self, n_samples=1):
        """Generate random samples within bounds"""
        samples = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            size=(n_samples, self.n_dim)
        )
        # Round x3 (number of teeth) to integer
        samples[:, 2] = np.round(samples[:, 2])
        return samples
    
    def normalize(self, x):
        """Normalize design variables to [0, 1]"""
        return (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def denormalize(self, x_norm):
        """Denormalize from [0, 1] to original bounds"""
        x = x_norm * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        x[2] = np.round(x[2])  # Integer constraint on x3
        return x
    
    def get_initial_design(self, n_samples=20, method='lhs'):
        """
        Generate initial design of experiments
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        method : str
            'lhs' for Latin Hypercube Sampling or 'random'
            
        Returns
        -------
        X : array, shape (n_samples, n_dim)
            Initial design samples
        """
        if method == 'lhs':
            # Latin Hypercube Sampling
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=self.n_dim)
            X_norm = sampler.random(n=n_samples)
            X = self.denormalize(X_norm)
        else:
            # Random sampling
            X = self.get_random_sample(n_samples)
        
        return X
    
    def __str__(self):
        return (f"Speed Reducer Problem\n"
                f"  Dimensions: {self.n_dim}\n"
                f"  Constraints: {self.n_constraints}\n"
                f"  Known optimum: {self.optimal_value:.4f}")


def test_speed_reducer():
    """Test the speed reducer problem implementation"""
    print("Testing Speed Reducer Problem")
    print("=" * 60)
    
    problem = SpeedReducerProblem()
    print(problem)
    print()
    
    # Test with known optimal solution
    print("Testing known optimal solution:")
    x_opt = problem.optimal_solution
    f_opt, c_opt = problem.evaluate(x_opt)
    print(f"  x* = {x_opt}")
    print(f"  f* = {f_opt:.6f} (expected: {problem.optimal_value:.6f})")
    print(f"  Feasible: {problem.is_feasible(x_opt)}")
    print(f"  Max constraint violation: {np.max(c_opt):.6e}")
    print()
    
    # Test with random samples
    print("Testing random samples:")
    X = problem.get_initial_design(n_samples=10)
    n_feasible = 0
    
    for i, x in enumerate(X):
        f, c = problem.evaluate(x)
        is_feas = problem.is_feasible(x)
        n_feasible += is_feas
        print(f"  Sample {i+1}: f={f:.2f}, feasible={is_feas}, "
              f"violations={np.sum(c > 0)}/{len(c)}")
    
    print(f"\nFeasibility rate: {n_feasible}/{len(X)} ({n_feasible/len(X)*100:.1f}%)")
    print()
    
    # Test bounds
    print("Testing variable bounds:")
    for i, (lb, ub) in enumerate(problem.bounds):
        print(f"  x{i+1}: [{lb:.1f}, {ub:.1f}]")
    print()
    
    print("Test completed successfully!")


if __name__ == "__main__":
    test_speed_reducer()

"""
Visualization and analysis tools for PCA-GP SCBO results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_paper_figure_3_convergence(results_dict: Dict[str, List[Dict]], 
                                    save_path: str = None):
    """
    Plot convergence curves matching Figure 3 from the paper.
    Shows mean ± std over multiple runs for all methods.
    
    Args:
        results_dict: dict mapping method names to lists of result dicts
                     e.g., {'SCBO': [result1, result2, ...], 'PCA-GP SCBO': [...]}
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'SCBO': 'red', 'PCA-GP SCBO': 'cyan', 'kPCA-GP SCBO': 'blue'}
    
    for method_name, results in results_dict.items():
        # Collect best feasible trajectories from all runs
        all_trajectories = []
        
        for result in results:
            f_history = result['f_history']
            C_history = result['C_history']
            
            # Identify feasible points
            feasible_mask = np.all(C_history <= 0, axis=1)
            
            # Compute best feasible so far
            best_feasible = []
            current_best = np.inf
            
            for i in range(len(f_history)):
                if np.any(feasible_mask[:i+1]):
                    feasible_vals = f_history[:i+1][feasible_mask[:i+1]]
                    if len(feasible_vals) > 0:
                        current_best = min(current_best, np.min(feasible_vals))
                best_feasible.append(current_best if current_best != np.inf else np.nan)
            
            all_trajectories.append(best_feasible)
        
        # Convert to array and compute statistics
        all_trajectories = np.array(all_trajectories)
        
        # Handle NaNs in statistics
        mean_trajectory = np.nanmean(all_trajectories, axis=0)
        std_trajectory = np.nanstd(all_trajectories, axis=0)
        
        # Plot mean line
        x = np.arange(len(mean_trajectory))
        color = colors.get(method_name, 'gray')
        ax.plot(x, mean_trajectory, '-', linewidth=2, label=method_name, color=color)
        
        # Plot std shading
        ax.fill_between(x, 
                       mean_trajectory - std_trajectory, 
                       mean_trajectory + std_trajectory,
                       alpha=0.2, color=color)
    
    # Add known optimal line
    ax.axhline(y=2996.3482, color='k', linestyle='--', linewidth=2, label='f*')
    
    ax.set_xlabel('Number of evaluations')
    ax.set_ylabel('Function value')
    ax.set_ylim([3000, 6000])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 3 convergence plot to {save_path}")
    
    plt.show()


def plot_convergence(result: Dict, save_path: str = None):
    """
    Plot optimization convergence curve
    
    Args:
        result: result dict from optimizer.get_result()
        save_path: optional path to save figure
    """
    f_history = result['f_history']
    C_history = result['C_history']
    
    # Identify feasible points
    feasible_mask = np.all(C_history <= 0, axis=1)
    
    # Compute best feasible so far
    best_feasible = []
    for i in range(len(f_history)):
        if np.any(feasible_mask[:i+1]):
            best_feasible.append(np.min(f_history[:i+1][feasible_mask[:i+1]]))
        else:
            best_feasible.append(np.nan)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Convergence plot
    ax1.plot(f_history, 'o-', alpha=0.3, label='All evaluations', markersize=3)
    ax1.plot(np.where(feasible_mask)[0], f_history[feasible_mask], 
             'go', alpha=0.6, label='Feasible', markersize=5)
    ax1.plot(best_feasible, 'r-', linewidth=2, label='Best feasible')
    ax1.axhline(y=2996.3482, color='k', linestyle='--', label='Known optimal')
    ax1.set_xlabel('Evaluation')
    ax1.set_ylabel('Objective Value (Weight)')
    ax1.set_title('Optimization Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Feasibility plot
    n_violated = np.sum(C_history > 0, axis=1)
    ax2.plot(n_violated, 'o-', alpha=0.6, markersize=3)
    ax2.fill_between(range(len(n_violated)), 0, n_violated, alpha=0.3)
    ax2.set_xlabel('Evaluation')
    ax2.set_ylabel('Number of Constraints Violated')
    ax2.set_title('Constraint Satisfaction Progress')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
    
    plt.show()


def plot_paper_figure_3_complete(results_dict: Dict[str, List[Dict]], 
                                 C_matrix: np.ndarray,
                                 save_path: str = None):
    """
    Create complete Figure 3 from paper: convergence (left) + eigenvalues (right)
    
    Args:
        results_dict: dict mapping method names to lists of results
        C_matrix: constraint matrix for eigenvalue analysis
        save_path: optional path to save figure
    """
    from sklearn.decomposition import PCA
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT PANEL: Convergence curves
    colors = {'SCBO': 'red', 'PCA-GP SCBO': 'cyan', 'kPCA-GP SCBO': 'blue'}
    
    for method_name, results in results_dict.items():
        all_trajectories = []
        
        for result in results:
            f_history = result['f_history']
            C_history = result['C_history']
            feasible_mask = np.all(C_history <= 0, axis=1)
            
            best_feasible = []
            current_best = np.inf
            
            for i in range(len(f_history)):
                if np.any(feasible_mask[:i+1]):
                    feasible_vals = f_history[:i+1][feasible_mask[:i+1]]
                    if len(feasible_vals) > 0:
                        current_best = min(current_best, np.min(feasible_vals))
                best_feasible.append(current_best if current_best != np.inf else np.nan)
            
            all_trajectories.append(best_feasible)
        
        all_trajectories = np.array(all_trajectories)
        mean_trajectory = np.nanmean(all_trajectories, axis=0)
        std_trajectory = np.nanstd(all_trajectories, axis=0)
        
        x = np.arange(len(mean_trajectory))
        color = colors.get(method_name, 'gray')
        ax1.plot(x, mean_trajectory, '-', linewidth=2, label=method_name, color=color)
        ax1.fill_between(x, mean_trajectory - std_trajectory, 
                        mean_trajectory + std_trajectory, alpha=0.2, color=color)
    
    ax1.axhline(y=2996.3482, color='k', linestyle='--', linewidth=2, label='f*')
    ax1.set_xlabel('Number of evaluations')
    ax1.set_ylabel('Function value')
    ax1.set_ylim([3000, 6000])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RIGHT PANEL: Eigenvalue decay
    pca = PCA()
    pca.fit(C_matrix)
    eigenvalues = pca.explained_variance_[:10]
    
    ax2.semilogy(range(1, 11), eigenvalues, 'o-')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Eigenvalues λᵢ')
    ax2.set_ylim([1e-5, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 3 to {save_path}")
    
    plt.show()


def plot_eigenvalue_decay(C: np.ndarray, n_components: int = 11, save_path: str = None):
    """
    Plot eigenvalue decay for PCA analysis
    
    Args:
        C: constraint matrix (N, G)
        n_components: number of components to show
        save_path: optional path to save figure
    """
    from sklearn.decomposition import PCA
    
    # Fit PCA
    pca = PCA()
    pca.fit(C)
    
    eigenvalues = pca.explained_variance_
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Eigenvalue decay
    ax1.semilogy(range(1, n_components+1), eigenvalues[:n_components], 'o-')
    ax1.axvline(x=4, color='r', linestyle='--', label='g=4 (used in paper)')
    ax1.set_xlabel('Component Index')
    ax1.set_ylabel('Eigenvalue λᵢ')
    ax1.set_title('Eigenvalue Decay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance explained
    ax2.plot(range(1, n_components+1), cumulative_var[:n_components] * 100, 'o-')
    ax2.axvline(x=4, color='r', linestyle='--', label='g=4 (used in paper)')
    ax2.axhline(y=95, color='gray', linestyle=':', alpha=0.5, label='95% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Variance Explained by Principal Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved eigenvalue plot to {save_path}")
    
    # Print statistics
    print(f"\nPCA Statistics:")
    print(f"  Variance explained by g=4: {cumulative_var[3]*100:.2f}%")
    print(f"  First 4 eigenvalues: {eigenvalues[:4]}")
    
    plt.show()


def plot_multiple_trials(results_list: List[float], method_name: str = "PCA-GP SCBO", 
                        save_path: str = None):
    """
    Plot distribution of results from multiple trials
    
    Args:
        results_list: list of objective values from multiple trials
        method_name: name of the method
        save_path: optional path to save figure
    """
    results = np.array(results_list)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(results, bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(x=np.median(results), color='r', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(results):.2f}')
    ax1.axvline(x=np.mean(results), color='g', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(results):.2f}')
    ax1.axvline(x=2996.3482, color='k', linestyle=':', 
                linewidth=2, label='Known optimal: 2996.35')
    ax1.set_xlabel('Objective Value (Weight)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Results ({method_name})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    box_data = [results]
    ax2.boxplot(box_data, labels=[method_name])
    ax2.axhline(y=2996.3482, color='k', linestyle=':', 
                linewidth=2, label='Known optimal')
    ax2.set_ylabel('Objective Value (Weight)')
    ax2.set_title('Box Plot of Results')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trial results plot to {save_path}")
    
    # Print statistics
    print(f"\n{method_name} Statistics (n={len(results)}):")
    print(f"  Best:    {np.min(results):.6f}")
    print(f"  Median:  {np.median(results):.6f}")
    print(f"  Mean:    {np.mean(results):.6f}")
    print(f"  Std Dev: {np.std(results):.2e}")
    print(f"  Worst:   {np.max(results):.6f}")
    print(f"  Target:  2996.3482")
    
    plt.show()


def compare_methods(results_dict: Dict[str, List[float]], save_path: str = None):
    """
    Compare multiple methods side by side
    
    Args:
        results_dict: dict mapping method names to lists of results
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    data = [results_dict[method] for method in results_dict.keys()]
    labels = list(results_dict.keys())
    
    # Box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Reference line
    ax.axhline(y=2996.3482, color='k', linestyle='--', 
              linewidth=2, label='Known optimal: 2996.35')
    
    ax.set_ylabel('Objective Value (Weight)')
    ax.set_title('Comparison of Optimization Methods')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    # Print comparison table
    print("\nMethod Comparison:")
    print(f"{'Method':<20} {'Best':<12} {'Median':<12} {'Mean':<12} {'Worst':<12}")
    print("-" * 68)
    for method, results in results_dict.items():
        results = np.array(results)
        print(f"{method:<20} {np.min(results):<12.4f} {np.median(results):<12.4f} "
              f"{np.mean(results):<12.4f} {np.max(results):<12.4f}")
    
    plt.show()


def plot_constraint_violations(C_history: np.ndarray, save_path: str = None):
    """
    Analyze constraint violation patterns
    
    Args:
        C_history: constraint history (N_evals, G)
        save_path: optional path to save figure
    """
    # Compute violations over time
    violations = np.maximum(C_history, 0)
    n_violated = np.sum(violations > 0, axis=1)
    max_violation = np.max(violations, axis=1)
    total_violation = np.sum(violations, axis=1)
    
    # Which constraints are violated most often
    constraint_violation_freq = np.mean(C_history > 0, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Number of violated constraints
    axes[0, 0].plot(n_violated, alpha=0.7)
    axes[0, 0].set_xlabel('Evaluation')
    axes[0, 0].set_ylabel('Number of Constraints Violated')
    axes[0, 0].set_title('Constraint Violations Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Maximum violation
    axes[0, 1].plot(max_violation, alpha=0.7, color='red')
    axes[0, 1].set_xlabel('Evaluation')
    axes[0, 1].set_ylabel('Maximum Constraint Violation')
    axes[0, 1].set_title('Maximum Violation Magnitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Total violation
    axes[1, 0].plot(total_violation, alpha=0.7, color='purple')
    axes[1, 0].set_xlabel('Evaluation')
    axes[1, 0].set_ylabel('Total Constraint Violation')
    axes[1, 0].set_title('Total Violation Sum')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Constraint violation frequency
    axes[1, 1].bar(range(len(constraint_violation_freq)), 
                   constraint_violation_freq * 100)
    axes[1, 1].set_xlabel('Constraint Index')
    axes[1, 1].set_ylabel('Violation Frequency (%)')
    axes[1, 1].set_title('Which Constraints Are Most Often Violated')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved constraint analysis plot to {save_path}")
    
    plt.show()
    
    # Print statistics
    print("\nConstraint Violation Analysis:")
    print(f"  Total evaluations: {len(n_violated)}")
    print(f"  Feasible evaluations: {np.sum(n_violated == 0)} ({np.sum(n_violated == 0)/len(n_violated)*100:.1f}%)")
    print(f"  Average violations per eval: {np.mean(n_violated):.2f}")
    print(f"  Most violated constraint: {np.argmax(constraint_violation_freq)} "
          f"({constraint_violation_freq[np.argmax(constraint_violation_freq)]*100:.1f}%)")


# Example usage
if __name__ == "__main__":
    print("This module provides visualization tools.")
    print("\nExample usage:")
    print("""
from pca_gp_scbo import run_single_trial
from visualize_results import plot_convergence, plot_eigenvalue_decay

# Run optimization
result = run_single_trial(use_kpca=False)

# Plot convergence
plot_convergence(result, save_path='convergence.png')

# Analyze PCA
plot_eigenvalue_decay(result['C_history'], save_path='eigenvalues.png')

# Analyze constraint violations
plot_constraint_violations(result['C_history'], save_path='constraints.png')
    """)

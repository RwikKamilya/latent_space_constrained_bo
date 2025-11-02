#!/usr/bin/env python3
"""
Main script to reproduce experiments from:
"High-Dimensional Bayesian Optimisation with Large-Scale Constraints 
via Latent Space Gaussian Processes" (Maathuis et al., 2024)

This script runs the complete experimental pipeline and generates all figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm

from speed_reducer_problem import SpeedReducerProblem
from pca_gp_scbo import run_single_trial, run_multiple_trials
from visualize_results import (
    plot_convergence, 
    plot_eigenvalue_decay,
    plot_constraint_violations,
    compare_methods,
    plot_paper_figure_3_convergence,
    plot_paper_figure_3_complete
)


def create_output_directories():
    """Create directory structure for storing results."""
    base_dir = Path("results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_experiment_config(run_dir, config):
    """Save experimental configuration to JSON."""
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {config_path}")


def experiment_1_speed_reducer_scbo(run_dir, n_runs=50):
    """
    Experiment 1: Baseline SCBO on Speed Reducer Problem
    
    Paper reference: Table 1, Section 4.1
    Expected result: Mean ~3007.20, Time ~501.38s
    
    CORRECTED: budget = 36000 - 20 = 35980 (paper specifies 36,000 total evals)
    CORRECTED: n_runs = 50 (paper reports statistics over 50 trials)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Speed Reducer with Standard SCBO")
    print("="*80)
    print("⚠️  NOTE: This will take ~10-30 minutes per trial")
    print("         Total time for 50 trials: ~8-25 hours")
    
    problem = SpeedReducerProblem()
    
    results = run_multiple_trials(
        problem=problem,
        n_trials=n_runs,
        n_initial_samples=20,
        budget=35980,  # FIXED: 36000 total - 20 initial = 35980 optimization iters
        use_pca=False,  # Standard SCBO (models all constraints independently)
        use_kpca=False,
        n_components=11,  # All constraints (no reduction for baseline)
        batch_size=1,
        population_size=200
    )
    
    # Save results
    results_path = run_dir / "data" / "experiment1_scbo.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_values': [r['best_feasible_value'] for r in results],
            'computation_times': [r['computation_time'] for r in results],
            'n_feasible': sum(1 for r in results if r['best_feasible_value'] is not None)
        }, f, indent=2)
    
    # Print summary statistics
    best_vals = [r['best_feasible_value'] for r in results if r['best_feasible_value'] is not None]
    times = [r['computation_time'] for r in results]
    
    print(f"\nResults (n={len(best_vals)} feasible solutions):")
    print(f"  Best:     {np.min(best_vals):.2f}")
    print(f"  Median:   {np.median(best_vals):.2f}")
    print(f"  Mean:     {np.mean(best_vals):.2f} ± {np.std(best_vals):.2f}")
    print(f"  Worst:    {np.max(best_vals):.2f}")
    print(f"  Avg Time: {np.mean(times):.2f}s")
    print(f"  Paper comparison - Mean: 3007.20, Time: 501.38s")
    
    return results


def experiment_2_speed_reducer_pca(run_dir, n_runs=50, n_components=4):
    """
    Experiment 2: PCA-GP SCBO on Speed Reducer Problem
    
    Paper reference: Table 1, Section 4.1
    Expected result: Mean ~3053.30, Time ~201.38s (59.83% time saving)
    
    CORRECTED: budget = 35980, n_runs = 50
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT 2: Speed Reducer with PCA-GP SCBO (g={n_components})")
    print("="*80)
    print("⚠️  NOTE: This will take ~5-15 minutes per trial (faster than SCBO)")
    print("         Total time for 50 trials: ~4-12 hours")
    
    problem = SpeedReducerProblem()
    
    results = run_multiple_trials(
        problem=problem,
        n_trials=n_runs,
        n_initial_samples=20,
        budget=35980,  # FIXED: paper specifies 36,000 total function evaluations
        use_pca=True,
        use_kpca=False,
        n_components=n_components,
        batch_size=1,
        population_size=200
    )
    
    # Save results
    results_path = run_dir / "data" / "experiment2_pca.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_values': [r['best_feasible_value'] for r in results],
            'computation_times': [r['computation_time'] for r in results],
            'n_feasible': sum(1 for r in results if r['best_feasible_value'] is not None),
            'n_components': n_components
        }, f, indent=2)
    
    # Print summary statistics
    best_vals = [r['best_feasible_value'] for r in results if r['best_feasible_value'] is not None]
    times = [r['computation_time'] for r in results]
    
    print(f"\nResults (n={len(best_vals)} feasible solutions):")
    print(f"  Best:     {np.min(best_vals):.2f}")
    print(f"  Median:   {np.median(best_vals):.2f}")
    print(f"  Mean:     {np.mean(best_vals):.2f} ± {np.std(best_vals):.2f}")
    print(f"  Worst:    {np.max(best_vals):.2f}")
    print(f"  Avg Time: {np.mean(times):.2f}s")
    print(f"  Paper comparison - Mean: 3053.30, Time: 201.38s")
    
    return results


def experiment_3_speed_reducer_kpca(run_dir, n_runs=50, n_components=4):
    """
    Experiment 3: kPCA-GP SCBO on Speed Reducer Problem
    
    Paper reference: Table 1, Section 4.1
    Expected result: Mean ~3088.39, Time ~216.96s (56.73% time saving)
    
    CORRECTED: budget = 35980, n_runs = 50
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT 3: Speed Reducer with kPCA-GP SCBO (g={n_components})")
    print("="*80)
    print("⚠️  NOTE: This will take ~5-15 minutes per trial (faster than SCBO)")
    print("         Total time for 50 trials: ~4-12 hours")
    
    problem = SpeedReducerProblem()
    
    results = run_multiple_trials(
        problem=problem,
        n_trials=n_runs,
        n_initial_samples=20,
        budget=35980,  # FIXED: paper specifies 36,000 total function evaluations
        use_pca=False,
        use_kpca=True,
        n_components=n_components,
        batch_size=1,
        population_size=200
    )
    
    # Save results
    results_path = run_dir / "data" / "experiment3_kpca.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_values': [r['best_feasible_value'] for r in results],
            'computation_times': [r['computation_time'] for r in results],
            'n_feasible': sum(1 for r in results if r['best_feasible_value'] is not None),
            'n_components': n_components
        }, f, indent=2)
    
    # Print summary statistics
    best_vals = [r['best_feasible_value'] for r in results if r['best_feasible_value'] is not None]
    times = [r['computation_time'] for r in results]
    
    print(f"\nResults (n={len(best_vals)} feasible solutions):")
    print(f"  Best:     {np.min(best_vals):.2f}")
    print(f"  Median:   {np.median(best_vals):.2f}")
    print(f"  Mean:     {np.mean(best_vals):.2f} ± {np.std(best_vals):.2f}")
    print(f"  Worst:    {np.max(best_vals):.2f}")
    print(f"  Avg Time: {np.mean(times):.2f}s")
    print(f"  Paper comparison - Mean: 3088.39, Time: 216.96s")
    
    return results


def experiment_4_component_sensitivity(run_dir, n_runs=10):
    """
    Experiment 4: Sensitivity to Number of Principal Components
    
    Paper reference: Figure 4, Section 4.1
    Tests g = 1, 2, 4, 6 components
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: Component Number Sensitivity Analysis")
    print("="*80)
    
    problem = SpeedReducerProblem()
    component_values = [1, 2, 4, 6]
    all_results = {}
    
    for g in component_values:
        print(f"\nTesting g={g} components...")
        results = run_multiple_trials(
            problem=problem,
            n_trials=n_runs,
            n_initial_samples=20,
            budget=360,
            use_pca=True,
            use_kpca=False,
            n_components=g,
            batch_size=1,
            population_size=200
        )
        all_results[g] = results
    
    # Save results
    results_path = run_dir / "data" / "experiment4_sensitivity.json"
    with open(results_path, 'w') as f:
        json.dump({
            str(g): {
                'best_values': [r['best_feasible_value'] for r in results],
                'computation_times': [r['computation_time'] for r in results],
                'n_feasible': sum(1 for r in results if r['best_feasible_value'] is not None)
            }
            for g, results in all_results.items()
        }, f, indent=2)
    
    # Create comparison plot (Figure 4 from paper)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {1: 'purple', 2: 'orange', 4: 'red', 6: 'green'}
    
    for g in component_values:
        results = all_results[g]
        
        # Collect trajectories from all runs
        all_trajectories = []
        
        for result in results:
            f_history = result['f_history']
            C_history = result['C_history']
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
        mean_trajectory = np.nanmean(all_trajectories, axis=0)
        std_trajectory = np.nanstd(all_trajectories, axis=0)
        
        # Plot mean line with shaded std
        x = np.arange(len(mean_trajectory))
        color = colors.get(g, 'gray')
        ax.plot(x, mean_trajectory, '-', linewidth=2, label=f'g={g}', color=color)
        ax.fill_between(x, 
                       mean_trajectory - std_trajectory, 
                       mean_trajectory + std_trajectory,
                       alpha=0.2, color=color)
    
    ax.set_xlabel('Number of evaluations')
    ax.set_ylabel('Function value')
    ax.set_ylim([3000, 6000])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig_path = run_dir / "figures" / "experiment4_sensitivity.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Figure 4 (component sensitivity) to {fig_path}")
    
    # Print summary
    print("\n" + "-"*60)
    print("Component Sensitivity Results:")
    print("-"*60)
    for g in component_values:
        results = all_results[g]
        best_vals = [r['best_feasible_value'] for r in results if r['best_feasible_value'] is not None]
        if best_vals:
            print(f"g={g}: Mean={np.mean(best_vals):.2f}, "
                  f"Best={np.min(best_vals):.2f}, "
                  f"Feasible={len(best_vals)}/{n_runs}")
        else:
            print(f"g={g}: No feasible solutions found")
    
    return all_results


def experiment_5_comparison_plot(run_dir, exp1_results, exp2_results, exp3_results):
    """
    Experiment 5: Generate comparison plot (Figure 3 from paper)
    
    Compares SCBO, PCA-GP SCBO, and kPCA-GP SCBO side-by-side
    """
    print("\n" + "="*80)
    print("EXPERIMENT 5: Generating Paper Figure 3")
    print("="*80)
    
    # Generate Figure 3: Convergence comparison (mean ± std) + Eigenvalue decay
    fig_path = run_dir / "figures" / "paper_figure_3.png"
    
    # Use constraint matrix from one of the PCA runs for eigenvalue analysis
    C_matrix = exp2_results[0]['C_history']
    
    plot_paper_figure_3_complete(
        results_dict={
            'SCBO': exp1_results,
            'PCA-GP SCBO': exp2_results,
            'kPCA-GP SCBO': exp3_results
        },
        C_matrix=C_matrix,
        save_path=fig_path
    )
    
    # Also generate the convergence-only plot
    fig_path_conv = run_dir / "figures" / "paper_figure_3_convergence_only.png"
    plot_paper_figure_3_convergence(
        results_dict={
            'SCBO': exp1_results,
            'PCA-GP SCBO': exp2_results,
            'kPCA-GP SCBO': exp3_results
        },
        save_path=fig_path_conv
    )
    
    print(f"Saved Figure 3 to {fig_path}")


def generate_paper_table_1(run_dir, exp1_results, exp2_results, exp3_results):
    """
    Generate Table 1 from the paper with performance statistics.
    
    Table columns:
    - Method
    - ˜f* (mean best value)
    - (˜f* - f*)/f* [%] (percentage error)
    - Time [s] (mean time)
    - Time Saving [%] (compared to SCBO)
    - Successful runs (n feasible / n total)
    """
    print("\n" + "="*80)
    print("GENERATING PAPER TABLE 1")
    print("="*80)
    
    f_optimal = 2996.3482  # Known optimal from paper
    
    results_dict = {
        'SCBO': exp1_results,
        'PCA-GP SCBO': exp2_results,
        'kPCA-GP SCBO': exp3_results
    }
    
    # Compute statistics for each method
    table_data = []
    scbo_time = None
    
    for method_name, results in results_dict.items():
        # Extract feasible values and times
        feasible_vals = [r['best_feasible_value'] for r in results if r['best_feasible_value'] is not None]
        times = [r['computation_time'] for r in results]
        
        if len(feasible_vals) > 0:
            mean_val = np.mean(feasible_vals)
            pct_error = (mean_val - f_optimal) / f_optimal * 100
            mean_time = np.mean(times)
            n_feasible = len(feasible_vals)
            n_total = len(results)
            
            # Calculate time saving
            if method_name == 'SCBO':
                scbo_time = mean_time
                time_saving = 0.0
            else:
                time_saving = (1 - mean_time / scbo_time) * 100 if scbo_time else 0.0
            
            table_data.append({
                'Method': method_name,
                'f_tilde': mean_val,
                'error_pct': pct_error,
                'time': mean_time,
                'time_saving': time_saving,
                'success': f"{n_feasible}/{n_total}"
            })
    
    # Print table
    print("\nTable 1: Speed Reducer Problem Performance")
    print("="*100)
    print(f"{'Method':<15} {'˜f*':<12} {'(˜f*-f*)/f* [%]':<18} {'Time [s]':<12} {'Time Saving [%]':<18} {'Successful':<12}")
    print("-"*100)
    
    for row in table_data:
        print(f"{row['Method']:<15} "
              f"{row['f_tilde']:<12.2f} "
              f"{row['error_pct']:<18.2f} "
              f"{row['time']:<12.2f} "
              f"{row['time_saving']:<18.2f} "
              f"{row['success']:<12}")
    
    print("="*100)
    print(f"Known optimal: f* = {f_optimal}")
    
    # Save to file
    table_path = run_dir / "TABLE_1_results.txt"
    with open(table_path, 'w') as f:
        f.write("Table 1: Speed Reducer Problem Performance\n")
        f.write("="*100 + "\n")
        f.write(f"{'Method':<15} {'˜f*':<12} {'(˜f*-f*)/f* [%]':<18} {'Time [s]':<12} {'Time Saving [%]':<18} {'Successful':<12}\n")
        f.write("-"*100 + "\n")
        
        for row in table_data:
            f.write(f"{row['Method']:<15} "
                   f"{row['f_tilde']:<12.2f} "
                   f"{row['error_pct']:<18.2f} "
                   f"{row['time']:<12.2f} "
                   f"{row['time_saving']:<18.2f} "
                   f"{row['success']:<12}\n")
        
        f.write("="*100 + "\n")
        f.write(f"Known optimal: f* = {f_optimal}\n")
    
    print(f"\nTable saved to: {table_path}")
    
    return table_data


def generate_summary_report(run_dir, all_experiments):
    """Generate a comprehensive summary report of all experiments."""
    report_path = run_dir / "RESULTS_SUMMARY.md"
    
    with open(report_path, 'w') as f:
        f.write("# Experimental Results Summary\n\n")
        f.write(f"**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Paper Reference\n\n")
        f.write("Maathuis et al. (2024) - High-Dimensional Bayesian Optimisation with ")
        f.write("Large-Scale Constraints via Latent Space Gaussian Processes\n\n")
        
        f.write("## Speed Reducer Problem (7D, 11 constraints)\n\n")
        f.write("### Table 1 Comparison\n\n")
        f.write("| Method | Best | Median | Mean | Std | Time (s) | Paper Mean | Paper Time |\n")
        f.write("|--------|------|--------|------|-----|----------|------------|------------|\n")
        
        # Add rows for each experiment
        for exp_name, exp_data in all_experiments.items():
            if exp_data:
                results = exp_data
                best_vals = [r['best_feasible_value'] for r in results if r['best_feasible_value'] is not None]
                times = [r['computation_time'] for r in results]
                
                if best_vals:
                    f.write(f"| {exp_name} | ")
                    f.write(f"{np.min(best_vals):.2f} | ")
                    f.write(f"{np.median(best_vals):.2f} | ")
                    f.write(f"{np.mean(best_vals):.2f} | ")
                    f.write(f"{np.std(best_vals):.2f} | ")
                    f.write(f"{np.mean(times):.2f} | ")
                    
                    # Add paper comparison values
                    if "SCBO" in exp_name and "PCA" not in exp_name:
                        f.write("3007.20 | 501.38 |\n")
                    elif "PCA-GP" in exp_name and "kPCA" not in exp_name:
                        f.write("3053.30 | 201.38 |\n")
                    elif "kPCA" in exp_name:
                        f.write("3088.39 | 216.96 |\n")
                    else:
                        f.write("- | - |\n")
        
        f.write("## Generated Figures\n\n")
        f.write("- `experiment1_convergence.png`: SCBO convergence curve\n")
        f.write("- `experiment2_convergence.png`: PCA-GP SCBO convergence\n")
        f.write("- `experiment2_eigenvalues.png`: Eigenvalue decay analysis\n")
        f.write("- `experiment3_convergence.png`: kPCA-GP SCBO convergence\n")
        f.write("- `experiment4_sensitivity.png`: Component number sensitivity\n")
        f.write("- `experiment5_comparison.png`: Side-by-side method comparison\n\n")
    
    print(f"\nGenerated summary report: {report_path}")


def main():
    """Main experimental pipeline."""
    print("\n" + "="*80)
    print(" BAYESIAN OPTIMIZATION EXPERIMENTS")
    print(" Maathuis et al. (2024) - Replication Study")
    print("="*80)
    
    # Setup
    np.random.seed(42)  # For reproducibility
    run_dir = create_output_directories()
    
    # Save configuration
    config = {
        'seed': 42,
        'n_runs_per_experiment': 20,
        'speed_reducer': {
            'n_dim': 7,
            'n_constraints': 11,
            'initial_samples': 20,
            'budget': 100,
            'population_size': 200
        }
    }
    save_experiment_config(run_dir, config)
    
    # Run experiments
    start_time = time.time()
    
    all_experiments = {}
    
    # Experiment 1: Baseline SCBO
    exp1_results = experiment_1_speed_reducer_scbo(run_dir, n_runs=20)
    all_experiments['SCBO'] = exp1_results
    
    # Experiment 2: PCA-GP SCBO
    exp2_results = experiment_2_speed_reducer_pca(run_dir, n_runs=20, n_components=4)
    all_experiments['PCA-GP SCBO'] = exp2_results
    
    # Experiment 3: kPCA-GP SCBO
    exp3_results = experiment_3_speed_reducer_kpca(run_dir, n_runs=20, n_components=4)
    all_experiments['kPCA-GP SCBO'] = exp3_results
    
    # Experiment 4: Component sensitivity
    exp4_results = experiment_4_component_sensitivity(run_dir, n_runs=10)
    
    # Experiment 5: Comparison plots
    experiment_5_comparison_plot(run_dir, exp1_results, exp2_results, exp3_results)
    
    # Generate Paper Table 1
    generate_paper_table_1(run_dir, exp1_results, exp2_results, exp3_results)
    
    # Generate summary report
    generate_summary_report(run_dir, all_experiments)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(" EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {run_dir}")
    print(f"  - Table 1: {run_dir}/TABLE_1_results.txt")
    print(f"  - Figure 3: {run_dir}/figures/paper_figure_3.png")
    print(f"  - Figure 4: {run_dir}/figures/experiment4_sensitivity.png")
    print(f"  - Summary report: {run_dir}/RESULTS_SUMMARY.md")
    print(f"  - All figures: {run_dir}/figures/")
    print(f"  - Raw data: {run_dir}/data/")

if __name__ == "__main__":
    main()

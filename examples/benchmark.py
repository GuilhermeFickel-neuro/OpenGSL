import opengsl
import psutil
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class MemoryTracker:
    def __init__(self, log_file="memory_usage.csv"):
        """Initialize a memory tracker that logs memory usage to a CSV file.
        
        Args:
            log_file: Path to the CSV file where memory usage will be logged.
        """
        self.log_file = log_file
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.method = None
        self.dataset = None
        
        # Initialize log file with headers if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("timestamp,method,dataset,memory_mb,cpu_percent,elapsed_time_s,event\n")
    
    def start(self, method, dataset):
        """Start tracking for a new experiment.
        
        Args:
            method: The method name being tested.
            dataset: The dataset name being used.
        """
        self.start_time = time.time()
        self.method = method
        self.dataset = dataset
        self.log("experiment_start")
    
    def log(self, event):
        """Log the current memory usage.
        
        Args:
            event: A string describing the current event/stage.
        """
        memory_mb = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
        cpu_percent = self.process.cpu_percent()
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{self.method},{self.dataset},{memory_mb:.2f},{cpu_percent:.2f},{elapsed_time:.2f},{event}\n")
        
        return memory_mb, cpu_percent, elapsed_time
    
    def end(self):
        """End tracking for the current experiment."""
        self.log("experiment_end")
        
    def plot_memory_usage(self, methods=None, datasets=None):
        """Generate plots of memory usage across experiments.
        
        Args:
            methods: List of methods to include in the plot (None for all).
            datasets: List of datasets to include in the plot (None for all).
        """
        try:
            df = pd.read_csv(self.log_file)
            
            if methods:
                df = df[df['method'].isin(methods)]
            if datasets:
                df = df[df['dataset'].isin(datasets)]
            
            # Create unique experiment identifiers
            df['experiment'] = df['method'] + '_' + df['dataset']
            
            # Plot memory usage over time for each experiment
            plt.figure(figsize=(12, 6))
            for exp, group in df.groupby('experiment'):
                plt.plot(group['elapsed_time_s'], group['memory_mb'], label=exp)
            
            plt.xlabel('Elapsed Time (s)')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Over Time')
            plt.legend()
            plt.grid(True)
            
            plt.savefig('memory_usage_plot.png')
            print(f"Memory usage plot saved to memory_usage_plot.png")
            
            # Summary statistics
            summary = df.groupby(['method', 'dataset']).agg({
                'memory_mb': ['mean', 'max'],
                'cpu_percent': ['mean', 'max'],
                'elapsed_time_s': ['max']
            }).reset_index()
            
            summary.columns = ['method', 'dataset', 'avg_memory_mb', 'peak_memory_mb', 
                             'avg_cpu_percent', 'peak_cpu_percent', 'total_time_s']
            
            summary.to_csv('memory_usage_summary.csv', index=False)
            print(f"Summary statistics saved to memory_usage_summary.csv")
            
            return summary
        except Exception as e:
            print(f"Error generating plots: {e}")
            return None

def run_experiment(method, dataset, n_runs=10, feat_norm=True):
    """Run an experiment with a specific method and dataset while tracking memory.
    
    Args:
        method: The method name to use (e.g., "gcn", "gat").
        dataset: The dataset name to use (e.g., "cora", "citeseer").
        n_runs: Number of runs to perform.
        feat_norm: Whether to normalize features.
        
    Returns:
        The experiment results.
    """
    tracker = MemoryTracker()
    
    try:
        # Start tracking
        tracker.start(method, dataset)
        
        # Load configuration
        tracker.log("loading_config")
        conf = opengsl.config.load_conf(method=method, dataset=dataset)
        
        # Load dataset
        tracker.log("loading_dataset")
        dataset_obj = opengsl.data.Dataset(dataset, n_splits=1, feat_norm=conf.dataset['feat_norm'])
        
        # Initialize solver
        tracker.log("initializing_solver")
        if method == "gcn":
            solver = opengsl.method.GCNSolver(conf, dataset_obj)
        elif method == "gat":
            solver = opengsl.method.GATSolver(conf, dataset_obj)
        # Add other methods as needed
        else:
            # Try to dynamically get the solver class
            solver_class = getattr(opengsl.method, f"{method.upper()}Solver")
            solver = solver_class(conf, dataset_obj)
        
        # Run experiment
        tracker.log("running_experiment")
        exp = opengsl.ExpManager(solver)
        results = exp.run(n_runs=n_runs)
        
        # End tracking
        tracker.log("recording_results")
        tracker.end()
        
        # Process results based on what ExpManager.run() actually returns
        # Based on the error message, it appears to return a tuple, not a dict
        processed_results = {}
        
        # OpenGSL's ExpManager.run() typically returns results like:
        # (mean_accuracy, std_accuracy, mean_f1, std_f1, mean_runtime, std_runtime)
        # Let's extract these values properly
        if isinstance(results, tuple) and len(results) >= 6:
            processed_results = {
                'accuracy': results[0],  # mean accuracy
                'accuracy_std': results[1],  # std of accuracy
                'f1': results[2],  # mean f1
                'f1_std': results[3],  # std of f1
                'runtime': results[4],  # mean runtime
                'runtime_std': results[5]  # std of runtime
            }
        elif isinstance(results, dict):
            # If it's already a dict, use it directly
            processed_results = results
        else:
            # Otherwise, store the raw results
            processed_results = {'raw_results': results}
        
        return processed_results, tracker
    
    except Exception as e:
        logging.error(f"Error running experiment {method} on {dataset}: {e}", exc_info=True)
        tracker.log(f"error: {str(e)}")
        tracker.end()
        return None, tracker

def run_benchmark(methods, datasets, n_runs=10):
    """Run benchmarks across multiple methods and datasets.
    
    Args:
        methods: List of methods to test.
        datasets: List of datasets to test.
        n_runs: Number of runs per experiment.
        
    Returns:
        A DataFrame with benchmark results.
    """
    results = []
    tracker = MemoryTracker()
    
    for method in methods:
        for dataset in datasets:
            print(f"Running {method} on {dataset}...")
            exp_result, exp_tracker = run_experiment(method, dataset, n_runs)
            
            if exp_result is not None:
                # Check the structure of exp_result
                if isinstance(exp_result, dict):
                    result_dict = {
                        'method': method,
                        'dataset': dataset,
                        'accuracy': exp_result.get('accuracy', np.nan),
                        'f1': exp_result.get('f1', np.nan),
                        'runtime': exp_result.get('runtime', np.nan)
                    }
                else:
                    # If it's not a dict, extract metrics differently based on OpenGSL's ExpManager return structure
                    # This handles the case where ExpManager.run() returns a tuple or other format
                    result_dict = {
                        'method': method,
                        'dataset': dataset,
                        'accuracy': getattr(exp_result, 'accuracy', np.nan) if hasattr(exp_result, 'accuracy') else np.nan,
                        'f1': getattr(exp_result, 'f1', np.nan) if hasattr(exp_result, 'f1') else np.nan,
                        'runtime': getattr(exp_result, 'runtime', np.nan) if hasattr(exp_result, 'runtime') else np.nan
                    }
                results.append(result_dict)
    
    # Generate summary plots and statistics
    summary = tracker.plot_memory_usage()
    
    # Combine model performance with memory usage
    if summary is not None and results:
        performance_df = pd.DataFrame(results)
        combined_results = performance_df.merge(
            summary, on=['method', 'dataset'], how='left'
        )
        combined_results.to_csv('benchmark_results.csv', index=False)
        print(f"Complete benchmark results saved to benchmark_results.csv")
        return combined_results
    
    return pd.DataFrame(results) if results else None

# Example usage
if __name__ == "__main__":
    # Define methods and datasets to test
    methods_to_test = ["sublime"]
    datasets_to_test = ["ogbn-products"]#, "citeseer", "pubmed"]
    
    # Run benchmarks
    results = run_benchmark(methods_to_test, datasets_to_test, n_runs=1)
    
    # Alternatively, run a single experiment
    # result, tracker = run_experiment("gcn", "cora", n_runs=10)
    # tracker.plot_memory_usage()
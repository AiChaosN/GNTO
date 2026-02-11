import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib import ticker

# Define file paths and labels
# You can manually add/remove files here, OR use the --path argument to auto-populate
files = [
    # {
    #     "path": "/path/to/ablation_result/Full_Model/log.csv",
    #     "label": "Full Model"
    # },
    # {
    #     "path": "/path/to/ablation_result/No_Hist/log.csv",
    #     "label": "No Hist"
    # },
]

# Metrics to plot (We aim for 6 metrics to fit 2x3 grid)
# Note: Adjust these keys if your CSV headers differ (e.g. val_q_50 vs val_q50)
metrics = [
    'val_q50', 
    'val_q75',
    'val_q90', 
    'val_q95',
    'val_q99',
    'loss' # Using training loss as the 6th metric since train_q metrics might be missing
]

# Alternate names to check if the above are not found
metric_aliases = {
    'val_q50': ['val_q_50', 'q50'],
    'val_q75': ['val_q_75', 'q75'],
    'val_q90': ['val_q_90', 'q90'],
    'val_q95': ['val_q_95', 'q95'],
    'val_q99': ['val_q_99', 'q99'],
    'loss': ['train_loss', 'training_loss'],
    'train_q50': ['train_q_50'],
    'train_q75': ['train_q_75'],
    'train_q99': ['train_q_99']
}

def get_actual_metric_name(df, metric):
    if metric in df.columns:
        return metric
    if metric in metric_aliases:
        for alias in metric_aliases[metric]:
            if alias in df.columns:
                return alias
    return None

def main():
    parser = argparse.ArgumentParser(description='Plot GNTO Ablation Results (2x3 Grid)')
    parser.add_argument('--path', type=str, help='Path to results directory to auto-populate files')
    parser.add_argument('--output', type=str, default='ablation_comparison_2x3.pdf', help='Output filename')
    args = parser.parse_args()

    global files
    # Auto-populate if path is provided
    if args.path and os.path.exists(args.path):
        subdirs = sorted([d for d in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, d))])
        # Prioritize Full_Model to be first if present
        if "Full_Model" in subdirs:
            subdirs.remove("Full_Model")
            subdirs.insert(0, "Full_Model")
            
        files = []
        for d in subdirs:
            log_path = os.path.join(args.path, d, 'log.csv')
            if os.path.exists(log_path):
                files.append({"path": log_path, "label": d})
        print(f"Found {len(files)} logs in {args.path}")

    if not files:
        print("No files configured. Please set the 'files' list in the script or provide --path.")
        return

    # Load data
    dfs = []
    for f in files:
        if os.path.exists(f["path"]):
            try:
                df = pd.read_csv(f["path"])
                dfs.append({"df": df, "label": f["label"]})
            except Exception as e:
                print(f"Error reading {f['path']}: {e}")
        else:
            print(f"Warning: File not found: {f['path']}")

    if not dfs:
        print("No data loaded.")
        return

    # Setup plot style (Matching 0202_compare_logs_QFvsGNTO.py)
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 30,
        'axes.titlesize': 30,
        'legend.fontsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'grid.alpha': 0
    })

    # Create 2x3 subplot
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break
        ax = axes[i]
        
        has_data = False
        for item in dfs:
            df = item["df"]
            label = item["label"]
            
            actual_col = get_actual_metric_name(df, metric)
            
            if actual_col:
                has_data = True
                
                # Style logic
                marker = 'o'
                markersize = 6
                linewidth = 3
                alpha = 0.5 # Default: fainter for comparison
                zorder = 2
                
                # Optional: specific styles for Full_Model
                linestyle = '-'
                if "No_" in label: linestyle = '--'
                elif "Replace" in label: linestyle = '-.'
                
                if label == "Full Model" or label == "Full_Model":
                    linewidth = 4
                    linestyle = '-'
                    alpha = 1.0
                    zorder = 10

                ax.plot(df['epoch'], df[actual_col], label=label, 
                        marker=marker, markersize=markersize, linewidth=linewidth, 
                        linestyle=linestyle, alpha=alpha, zorder=zorder)
        
        if has_data:
            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_yscale('log')
            
            # Formatting (Exact match to 0202)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            
            # Dynamic tick locator
            if '99' in metric or 'loss' in metric:
                # Handle large range
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
            else:
                # Handle small range (1.0 - 2.0)
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=6))
            
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            
            ax.legend()
            # Adjust grid lines
            ax.grid(True, which="major", ls="-", alpha=0.4)
        else:
            ax.text(0.5, 0.5, f"Metric {metric} not found", ha='center', va='center')

    plt.tight_layout()
    
    # Determine output path
    output_dir = "."
    if args.path:
        output_dir = args.path
    elif files:
         output_dir = os.path.dirname(os.path.dirname(files[0]["path"]))
    
    save_path = os.path.join(output_dir, args.output)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # plt.show()

if __name__ == "__main__":
    main()

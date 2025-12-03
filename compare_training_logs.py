import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt errors
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths and labels
files = [
    {
        "path": "/home/AiChaosN/Project/Phd/project/QueryFormer_VLDB2022/results/full/cost/training_log.csv",
        "label": "QueryFormer (Cost)"
    },
    {
        "path": "/home/AiChaosN/Project/Phd/project/GNTO/results/GNTO_QF_1203_1912/training_log.csv",
        "label": "GNTO QF 1203"
    },
    {
        "path": "/home/AiChaosN/Project/Phd/project/GNTO/results/gnto_1101_training_log.csv",
        "label": "GNTO 1101"
    }
]

# Metrics to plot
metrics = [
    'loss', 
    'time',
    'grad_norm',
    'val_q_50', 
    'val_q_75',
    'val_q_90', 
    'val_q_95',
    'val_q_99',
    'train_q_50',
    'train_q_75',
    'train_q_90',
    'train_q_95',
    'train_q_99'
]

# Load data
dfs = []
for f in files:
    if os.path.exists(f["path"]):
        df = pd.read_csv(f["path"])
        dfs.append({"df": df, "label": f["label"]})
    else:
        print(f"Warning: File not found: {f['path']}")

if not dfs:
    print("No data files found.")
    exit()

# Setup plot style
try:
    plt.style.use('seaborn-v0_8-darkgrid') 
except:
    plt.style.use('ggplot') # Fallback if seaborn style is missing

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    if i >= len(axes):
        break
    ax = axes[i]
    
    for item in dfs:
        df = item["df"]
        label = item["label"]
        
        if metric in df.columns:
            # Use log scale for all metrics as requested due to large differences
            ax.plot(df['epoch'], df[metric], label=label, marker='.')
            ax.set_yscale('log')
        else:
            # print(f"Metric {metric} not found in {label}")
            pass

    ax.set_title(metric)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
output_file = "training_comparison.png"
plt.savefig(output_file)
print(f"Plot saved to {output_file}")
# plt.show() # Commented out for headless environments, but valid if running locally with display


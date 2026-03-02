import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt errors
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import ticker

# Define file paths and labels
files = [
    # {
    #     "path": "/home/AiChaosN/Project/Phd/project/QueryFormer_VLDB2022/results/full/cost/training_log.csv",
    #     "label": "QueryFormer"
    # },
    {
        "path": "/home/AiChaosN/Project/Phd/project/GNTO/results/GNTO_QF_1203_1912/training_log.csv",
        "label": "GNTO"
    },
    {
        "path": "/home/AiChaosN/Project/Phd/project/GNTO/results/GNTO_QF_1216_1708/training_log.csv",
        "label": "GATv2"
    },
    {
        "path": "/home/AiChaosN/Project/Phd/project/GNTO/results/GNTO_QF_1216_1737/training_log.csv",
        "label": "GATv2 + Plan Rows"
    },
    # {
    #     "path": "/home/AiChaosN/Project/Phd/project/GNTO/results/gnto_1101_training_log.csv",
    #     "label": "GNTO 1101"
    # }
]

# Metrics to plot
metrics = [
    # 'loss',
    # 'time',
    # 'grad_norm',
    'val_q_50', 
    'val_q_75',
    # 'val_q_90', 
    # 'val_q_95',
    # 'val_q_99',
    'train_q_50',
    'train_q_75',
    # 'train_q_90',
    # 'train_q_95',
    # 'train_q_99'
]

# Load data
dfs = []
for f in files:
    if os.path.exists(f["path"]):
        df = pd.read_csv(f["path"])
        # Filter data to start from epoch 10
        df = df[df['epoch'] >= 10]
        dfs.append({"df": df, "label": f["label"]})
    else:
        print(f"Warning: File not found: {f['path']}")

if not dfs:
    print("No data files found.")
    exit()

# Setup plot style
plt.style.use('seaborn-v0_8-whitegrid') 

# Update parameters for larger fonts and white background
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 30,
    'axes.titlesize': 30,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'grid.alpha': 0
})

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10)) # Adjusted for 4 plots
axes = axes.flatten()

for i, metric in enumerate(metrics):
    if i >= len(axes):
        break
    ax = axes[i]
    
    for item in dfs:
        df = item["df"]
        label = item["label"]
        
        if metric in df.columns:
            # Custom style based on label
            if "Plan Rows" in label:
                linewidth = 7  # Thicker and solid for the main model
                alpha = 1.0    # Opaque
                zorder = 10    # Draw on top
                linestyle = '-'
            else:
                linewidth = 4  # Also thick but transparent
                alpha = 0.7    # Transparent
                zorder = 5     # Draw below
                linestyle = '-'

            ax.plot(df['epoch'], df[metric], label=label, marker='o', markersize=8, 
                    linewidth=linewidth, alpha=alpha, zorder=zorder, linestyle=linestyle)

            ax.set_yscale('log')

            # 统一设置：数字显示格式
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            
            # 动态设置：刻度数量
            if metric == 'train_q_99':
                # 处理跨度大的情况（10 - 1000）
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
            else:
                # 处理跨度小的情况（1.0 - 2.0）
                # subs=(1.0, 1.2, 1.4, 1.6, 1.8) 这种写法可以手动控制显示的刻度点
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=6))
                    
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.grid(True, which="major", ls="-", alpha=0.4)
        else:
            pass

    # ax.set_title(metric) # Remove title
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    # ax.legend(loc='upper right')
    # 调整网格线
    # ax.grid(True, which="major", ls="-", alpha=0.4) # Moved inside loop

# 整个图共用一个图例，放在顶部中央
handles, labels = axes[0].get_legend_handles_labels()

# 2. 修改 fig.legend 参数
fig.legend(
    handles, 
    labels, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.0), # 将锚点稍微调低到 1.0，配合 tight_layout 的 rect 使用
    ncol=3,                    # 修改为 3 列一排
    fontsize=20, 
    frameon=True,
    columnspacing=1.0,         # 可选：调整列与列之间的间距
    handletextpad=0.5          # 可选：调整图标与文字之间的间距
)

# 3. 调整子图布局，为顶部图例留出更多空间
# rect=[左, 下, 右, 上] -> 将 0.96 调低（例如 0.90 或 0.88），顶部留白会变大
plt.tight_layout(rect=[0, 0, 1, 0.94])

# output_file = "training_comparison_1218.png"
output_file = "0226_comp_gntoVsGAT1VsGAT2.pdf" # Vector format
plt.savefig(output_file)
print(f"Plot saved to {output_file}")
# plt.show() # Commented out for headless environments, but valid if running locally with display


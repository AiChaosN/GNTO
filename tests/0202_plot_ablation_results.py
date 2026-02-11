import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np

# 设置风格
plt.style.use('bmh') # 使用 matplotlib 自带的样式替代 seaborn
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def plot_final_comparison(summary_path, output_dir):
    """绘制最终结果对比柱状图"""
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return

    df = pd.read_csv(summary_path)
    
    # 确保 config 列存在
    if 'config' not in df.columns:
        print("Error: 'config' column not found in summary.csv")
        return

    # 按 best_q90 排序
    df = df.sort_values('best_q90', ascending=True)

    # 绘制 Q90 对比图
    plt.figure(figsize=(12, 6))
    
    # 手动生成颜色
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(df)))
    
    bars = plt.bar(df['config'], df['best_q90'], color=colors)
    
    plt.title('Ablation Study: Best Q-Error (90th Percentile)', fontsize=16, pad=20)
    plt.ylabel('Q-Error (Lower is Better)', fontsize=14)
    plt.xlabel('Configuration', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 在柱状图上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ablation_summary_q90.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved summary plot to {save_path}")
    plt.close()

    # 如果有 final_q95 或 final_q99，也可以画一个多指标对比图
    if 'final_q50' in df.columns and 'final_q95' in df.columns:
        metrics = ['final_q50', 'best_q90', 'final_q95', 'final_q99']
        valid_metrics = [m for m in metrics if m in df.columns]
        
        plt.figure(figsize=(14, 7))
        
        x = np.arange(len(df['config']))  # 标签位置
        width = 0.8 / len(valid_metrics)  # 柱子宽度

        for i, metric in enumerate(valid_metrics):
            plt.bar(x + i*width, df[metric], width, label=metric)

        plt.title('Ablation Study: Multi-Metric Comparison', fontsize=16)
        plt.xlabel('Configuration')
        plt.ylabel('Q-Error (Log Scale)')
        plt.yscale('log')
        plt.xticks(x + width * (len(valid_metrics) - 1) / 2, df['config'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        save_path_multi = os.path.join(output_dir, 'ablation_summary_multi_metric.png')
        plt.savefig(save_path_multi, dpi=300)
        print(f"Saved multi-metric plot to {save_path_multi}")
        plt.close()

def plot_training_curves(results_dir, output_dir):
    """读取所有子目录的 log.csv 并绘制 2x2 收敛曲线 (Q50, Q75, Q90, Q99)"""
    subdirs = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])
    
    if not subdirs:
        print("No subdirectories found.")
        return

    # 读取所有数据
    data = {}
    for config_name in subdirs:
        log_path = os.path.join(results_dir, config_name, 'log.csv')
        if os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                data[config_name] = df
            except Exception as e:
                print(f"Error reading {log_path}: {e}")
    
    if not data:
        print("No valid log.csv files found.")
        return

    # 定义要绘制的指标
    metrics = [
        ('val_q50', 'Validation Q50'),
        ('val_q75', 'Validation Q75'),
        ('val_q90', 'Validation Q90'),
        ('val_q99', 'Validation Q99')
    ]

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8)) # 稍微增大画布尺寸
    axes = axes.flatten()
    
    # 设置字体大小
    plt.rcParams.update({'font.size': 20}) # 增大全局字体
    
    # 使用 matplotlib 默认的颜色循环
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        
        # 遍历所有配置进行绘制
        for idx, config_name in enumerate(subdirs):
            if config_name not in data:
                continue
                
            df = data[config_name]
            if metric in df.columns:
                color = colors[idx % len(colors)]
                
                # Full_Model 用实线，其他用虚线
                if "Full_Model" in config_name:
                    linestyle = '-'
                    linewidth = 4.0
                    alpha = 1.0
                else:
                    linestyle = '--'
                    linewidth = 3.5
                    alpha = 0.6
                
                ax.plot(df['epoch'], df[metric], label=config_name, linewidth=linewidth, linestyle=linestyle, marker=None, color=color, alpha=alpha)
        
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_ylabel('Q-Error', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_yscale('log')
        
        # 只在右上角的图 (索引 1) 里面添加图例
        if i == 1:
            ax.legend(title="Configuration", loc='upper right', fontsize=12, title_fontsize=14, framealpha=0.9)
    
    # 移除之前的全局图例代码
    
    plt.tight_layout()
    # 不需要 subplots_adjust 了，因为图例在子图内部
    
    save_path = os.path.join(output_dir, 'ablation_convergence_2x2.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved 2x2 convergence plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot Ablation Study Results')
    # 默认路径设为
    default_path = "/home/AiChaosN/Project/Phd/project/GNTO/results/Ablation_QF_0129_1208"
    parser.add_argument('--path', type=str, default=default_path, help='Path to the ablation results directory')
    
    args = parser.parse_args()
    
    results_dir = args.path
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        return

    print(f"Processing results from: {results_dir}")
    
    # 绘制 Summary Bar Chart
    plot_final_comparison(os.path.join(results_dir, "summary.csv"), results_dir)
    
    # 绘制 Convergence Line Chart
    plot_training_curves(results_dir, results_dir)
    
    print("\nDone! Check the result directory for the generated images.")

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 定义文件路径
files_4090 = [
    "sputnik_performance_results_all.csv", 
    "cusparse_performance_results.csv", 
    "spmm_performance_results_main_v2.csv",
    "sparta_performance_results_main_v1.csv"
]

# 定义方法的名称及对应的颜色
methods = ['cuSPARSE', 'Sputnik', 'SparTA', 'Flash-LLM', 'SpInfer-SpMMV3']
colors = ['#000', '#C00000', '#800080','#0000FF','#4d8076']

def process_data(files):
    data = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file)  # 改为read_csv
        data = pd.concat([data, df])
    
    data = data[['M', 'K', 'N', 'SplitK', 'Sparsity', 'Kernel', 'Duration(ns)', 'TFLOPS']]
    
    def calculate_speedup(group):
        if 'cuBLAS_TC' in group['Kernel'].values:
            cublas_duration = group.loc[group['Kernel'] == 'cuBLAS_TC', 'Duration(ns)'].values[0]
            group['Speedup'] = cublas_duration / group['Duration(ns)']
        else:
            group['Speedup'] = np.nan
        return group

    data_grouped = data.groupby(['M', 'K', 'N', 'Sparsity'], group_keys=False).apply(calculate_speedup)
    return data_grouped.reset_index(drop=True)

# 处理数据
data_4090 = process_data(files_4090)

# 选择感兴趣的N值和Sparsity
Ns = [8, 16, 32]
sparsities = [40, 50, 60, 70]

# 设置全局字体大小
plt.rcParams.update({'font.size': 24})

# 创建1x3的画布
fig, axs = plt.subplots(1, 3, figsize=(30, 6), sharex=True)

def plot_data(data):
    for i, N in enumerate(Ns):
        subset = data[data['N'] == N]
        subset = subset[subset['Kernel'] != 'cuBLAS_TC']
        
        if not subset.index.is_unique:
            subset = subset.reset_index(drop=True)

        subset['Sparsity_Kernel'] = subset['Sparsity'].astype(str) + '_' + subset['Kernel']

        sns.boxplot(
            x='Sparsity', y='Speedup', hue='Kernel', data=subset,
            ax=axs[i], palette=colors, 
            hue_order=methods, order=sparsities,
            showfliers=False, whis=np.inf, meanline=True, showmeans=True,
            medianprops=dict(color="red", linewidth=2)
        )
        
        # 添加抖动的数据点
        sns.stripplot(
            x='Sparsity', y='Speedup', hue='Kernel', data=subset,
            ax=axs[i], palette=colors, 
            hue_order=methods, order=sparsities,
            dodge=True, alpha=0.5, size=5
        )
        
        axs[i].axhline(1, color='red', linewidth=2, linestyle='--')
        axs[i].set_xlabel('Sparsity (%)', fontsize=24)
        axs[i].set_ylabel('Speedup vs cuBLAS_TC' if i == 0 else '', fontsize=24)
        axs[i].get_legend().remove()
        axs[i].set_ylim(bottom=0)
        axs[i].tick_params(axis='both', which='major', labelsize=20)
        
        # 将 N=8, N=16, N=32 放在对应 figure 的内部顶上居中
        axs[i].text(0.5, 0.9, f'N={N}', transform=axs[i].transAxes, 
                    ha='center', va='bottom', fontsize=28, fontweight='bold')
        
        # 在箱子上方添加中位数文本
        for j, artist in enumerate(axs[i].artists):
            median = subset[subset['Kernel'] == methods[j % len(methods)]]['Speedup'].median()
            axs[i].text(artist.get_x() + artist.get_width() / 2., 
                        artist.get_y() + artist.get_height(),
                        f'{median:.2f}', ha='center', va='bottom', fontsize=16)

    # 设置统一的y轴范围
    y_max = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(0, y_max)

# 绘制RTX4090数据
plot_data(data_4090)

# 添加GPU型号标签
axs[0].text(-0.2, 1.1, 'RTX 4090', transform=axs[0].transAxes, 
            fontsize=28, fontweight='bold', ha='left', va='top')

# 在整个图的顶部添加一个统一的图例
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, methods, loc='upper center', ncol=len(methods), 
           bbox_to_anchor=(0.5, 1.01), fontsize=24, title_fontsize=28)

# 调整布局
plt.tight_layout()

# 保存图片到文件
plt.savefig('output_boxplot_1x3_updated.png', dpi=300, bbox_inches='tight')

# 关闭当前的图形对象
plt.close()
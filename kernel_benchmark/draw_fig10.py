import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 定义文件路径
files_4090 = [
    "sputnik_performance_results_all_done.csv",
    "cusparse_performance_results_done.csv",
    "spmm_performance_results_main_v2_done.csv",
    "sparta_performance_results_main_v1_done.csv"
]

# 定义方法的名称及对应的颜色
methods = ['cuSPARSE', 'Sputnik', 'SparTA', 'Flash-LLM', 'SpInfer']
colors = ['#000', '#C00000', '#800080','#0000FF','#4d8076']

def process_data(files):
    data = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file)
        if 'SpInfer' in df['Kernel'].values:
            # 只保留主要的SpInfer结果，过滤掉变体
            df = df[~df['Kernel'].isin(['SpInfer-SpMMV1', 'SpInfer-SpMMV2'])]
        data = pd.concat([data, df])

    # 只保留需要的列和方法
    data = data[data['Kernel'].isin(methods + ['cuBLAS_TC'])]
    data = data[['M', 'K', 'N', 'Sparsity', 'Kernel', 'TFLOPS']]

    def calculate_speedup(group):
        if 'cuBLAS_TC' in group['Kernel'].values:
            cublas_tflops = group.loc[group['Kernel'] == 'cuBLAS_TC', 'TFLOPS'].values[0]
            # 使用TFLOPS计算加速比
            group['Speedup'] = group['TFLOPS'] / cublas_tflops
        else:
            group['Speedup'] = np.nan
        return group

    data_grouped = data.groupby(['M', 'K', 'N', 'Sparsity']).apply(calculate_speedup)
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

        # 打印每种方法的加速比范围进行验证
        print(f"\nN={N} Speedup summary over cuBLAS_TC:")
        for method in methods:
            method_data = subset[subset['Kernel'] == method]
            if not method_data.empty:
                print(f"{method}: {method_data['Speedup'].min():.2f} - {method_data['Speedup'].max():.2f}")

        sns.boxplot(
            x='Sparsity', y='Speedup', hue='Kernel', data=subset,
            ax=axs[i], palette=colors,
            hue_order=methods, order=sparsities,
            showfliers=False, whis=np.inf, meanline=True, showmeans=True,
            medianprops=dict(color="red", linewidth=2)
        )

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

        # 添加N值标签
        axs[i].text(0.5, 0.9, f'N={N}', transform=axs[i].transAxes,
                    ha='center', va='bottom', fontsize=28, fontweight='bold')

        # 在箱子上方添加中位数标签
        for j, artist in enumerate(axs[i].artists):
            method = methods[j % len(methods)]
            method_data = subset[subset['Kernel'] == method]['Speedup']
            if not method_data.empty:
                median = method_data.median()
                axs[i].text(artist.get_x() + artist.get_width() / 2.,
                           artist.get_y() + artist.get_height(),
                           f'{median:.2f}x', ha='center', va='bottom', fontsize=16)

    # 统一y轴范围
    y_max = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(0, y_max)

# 绘制图表
plot_data(data_4090)

# 添加GPU型号标签
axs[0].text(-0.2, 1.1, 'RTX 4090', transform=axs[0].transAxes,
            fontsize=28, fontweight='bold', ha='left', va='top')

# 添加统一图例
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles[:len(methods)], methods, 
          loc='upper center', ncol=len(methods),
          bbox_to_anchor=(0.5, 1.01), fontsize=24)

plt.tight_layout()
plt.savefig('Figure10.png', dpi=300, bbox_inches='tight')
plt.close()
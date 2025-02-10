import pandas as pd

# 读取CSV文件
df = pd.read_csv('sputnik_performance_results_all.csv')

# 重命名列
df = df.rename(columns={
    'Sputnik_Duration(ns)': 'Duration(ns)',
    'Sputnik_TFLOPS': 'TFLOPS'
})

# 添加新列Kernel
df['Kernel'] = 'Sputnik'

# 保存修改后的CSV文件
df.to_csv('sputnik_performance_results_all_done.csv', index=False)




# 读取CSV文件
df = pd.read_csv('sparta_performance_results_main_v1.csv')

# 重命名列
df = df.rename(columns={
    'SparTA_Duration(ns)': 'Duration(ns)',
    'SparTA_TFLOPS': 'TFLOPS'
})

# 添加新列Kernel
df['Kernel'] = 'SparTA'

# 保存修改后的CSV文件
df.to_csv('sparta_performance_results_main_v1_done.csv', index=False)





# 读取CSV文件
df = pd.read_csv('cusparse_performance_results.csv')

# 删除cuSPARSE_C相关列
df = df.drop(['cuSPARSE_C_Duration(ns)', 'cuSPARSE_C_TFLOPS'], axis=1)

# 重命名列
df = df.rename(columns={
    'cuSPARSE_R_Duration(ns)': 'Duration(ns)',
    'cuSPARSE_R_TFLOPS': 'TFLOPS'
})

# 添加新列Kernel
df['Kernel'] = 'cuSPARSE'

# 保存修改后的CSV文件
df.to_csv('cusparse_performance_results_done.csv', index=False)

print("CSV文件已成功修改！")



# 读取CSV文件
df = pd.read_csv('spmm_performance_results_main_v2.csv')

# 将'SpInfer-SpMMV3'替换为'SpInfer'
df['Kernel'] = df['Kernel'].replace('SpInfer-SpMMV3', 'SpInfer')

# 保存修改后的CSV文件
df.to_csv('spmm_performance_results_main_v2_done.csv', index=False)



print("csv handled.")
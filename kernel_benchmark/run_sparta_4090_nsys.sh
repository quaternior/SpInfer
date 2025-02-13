#!/bin/bash

# 定义测试参数
M=(8192 4096 32000 3584 32000 1024 28672 5120 5120 3584 3584 4096 13824 8192 18944 14336 4096 8192 11008 32000 20480 1024 3584 2560 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
K=(29568 4096 5120 2560 8192 8192 8192 5120 13824 20480 3584 11008 5120 8192 3584 4096 14336 28672 4096 4096 3584 4096 18944 3584 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)
N=(8 16 32)
SPLIT_K=(1)
SPARSITY=(40 50 60 70)

# 设置输出文件
output_csv="sparta_performance_results_main_v1.csv"
debug_log="sparta_debug_main_v1.log"

# 创建或清空输出文件
echo "M,K,N,SplitK,Sparsity,SparTA_Duration(ns),SparTA_TFLOPS" > "$output_csv"
> "$debug_log"

# 计算 TFLOPS
calculate_tflops() {
    local m=$1
    local k=$2
    local n=$3
    local duration_ns=$4
    awk -v m="$m" -v k="$k" -v n="$n" -v d="$duration_ns" 'BEGIN {print (2 * m * k * n) / (d / 1e9) / 1e12}'
}

# 处理 nsys 输出
process_test_case() {
    local m=$1
    local k=$2
    local n=$3
    local s=$4
    local sk=$5

    echo "Debug: Running test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"

    # 运行 nsys 并捕获输出
    nsys_output=$(nsys nvprof ./spmm_test_sparta $m $k $n $s $sk 2>&1)

    echo "Debug: nsys output:" >> "$debug_log"
    echo "$nsys_output" >> "$debug_log"

    # 提取 SparTA 运行时间（ms）
    sparTA_time_ms=$(echo "$nsys_output" | grep -oP 'sparTA -> Time/ms: \K[0-9]+(\.[0-9]+)?')
    if [[ -z "$sparTA_time_ms" ]]; then
        echo "Error: Unable to extract SparTA execution time" >> "$debug_log"
        return
    fi

    # 转换为纳秒
    sparTA_time_ns=$(awk -v time_ms="$sparTA_time_ms" 'BEGIN {print time_ms * 1e6}')

    # 提取 Sparse GEMM 内核的最小时间（ns）
    sparse_gemm_min_time=""
    while read -r line; do
        kernel_time_ns=$(echo "$line" | awk '{print $3}' | tr -d ',')
        if [[ -n "$kernel_time_ns" ]]; then
            if [[ -z "$sparse_gemm_min_time" || "$(awk "BEGIN {print ($kernel_time_ns < $sparse_gemm_min_time)}")" -eq 1 ]]; then
                sparse_gemm_min_time=$kernel_time_ns
            fi
        fi
    done < <(echo "$nsys_output" | grep 'sm86_xmma_sparse_gemm')

    # 提取 Sputnik 内核时间（ns）
    sputnik_kernel_time=""
    while read -r line; do
        kernel_time_ns=$(echo "$line" | awk '{print $3}' | tr -d ',')
        if [[ -n "$kernel_time_ns" ]]; then
            sputnik_kernel_time=$kernel_time_ns
        fi
    done < <(echo "$nsys_output" | grep 'void sputnik::<unnamed>::Kernel')

    # 确保变量有默认值
    if [[ -z "$sparse_gemm_min_time" ]]; then sparse_gemm_min_time=0; fi
    if [[ -z "$sputnik_kernel_time" ]]; then sputnik_kernel_time=0; fi

    # 计算 SparTA 总时间（ns）
    sparTA_total_time=$(awk "BEGIN {print $sparse_gemm_min_time + $sputnik_kernel_time}")

    echo "Debug: SparTA total time: $sparTA_total_time ns" >> "$debug_log"

    # 计算 TFLOPS
    tflops=$(calculate_tflops $m $k $n $sparTA_total_time)

    # 输出结果到 CSV
    echo "$m,$k,$n,$sk,$s,${sparTA_total_time},${tflops}" >> "$output_csv"
    echo "Debug: Output to CSV - SparTA_Total_Time: $sparTA_total_time ns, TFLOPS: $tflops" >> "$debug_log"
}

# 确保 M 和 K 数组长度相同
if [ ${#M[@]} -ne ${#K[@]} ]; then
    echo "Error: M and K arrays must have the same length."
    exit 1
fi

# 遍历参数组合
for ((i=0; i<${#M[@]}; i++)); do
    m=${M[i]}
    k=${K[i]}
    for n in "${N[@]}"; do
        for s in "${SPARSITY[@]}"; do
            for sk in "${SPLIT_K[@]}"; do
                process_test_case $m $k $n $s $sk
            done
        done
    done
done

echo "SparTA performance testing completed. Results saved in $output_csv"
echo "Debug log saved in $debug_log"

# 检查 CSV 文件是否为空
if [ -s "$output_csv" ]; then
    echo "CSV file is not empty."
else
    echo "CSV file is empty. Please check the debug log for more information."
fi

# 显示 CSV 文件的前几行
echo "First few lines of the CSV file:"
head -n 5 "$output_csv"
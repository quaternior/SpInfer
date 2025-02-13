#!/bin/bash
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

# 定义函数来计算 TFLOPS
calculate_tflops() {
    local m=$1
    local k=$2
    local n=$3
    local duration_ns=$4  # 输入是纳秒
    awk -v m="$m" -v k="$k" -v n="$n" -v d="$duration_ns" 'BEGIN {print (2 * m * k * n) / (d / 1e9) / 1e12}'
}

process_test_case() {
    local m=$1
    local k=$2
    local n=$3
    local s=$4
    local sk=$5

    echo "Debug: Starting test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"

    # 初始化变量
    local sparse_gemm_min_time=""  # 保持为空字符串，与原始逻辑一致
    local sputnik_kernel_time=0
    local in_gpukernsum=false

    # 运行 nsys 并捕获输出
    echo "Debug: Running nsys command..." >> "$debug_log"
    nsys_output=$(nsys nvprof spmm_test_sparta $m $k $n $s $sk 2>&1)
    echo "Debug: nsys command completed" >> "$debug_log"

    # 将完整输出写入日志文件
    echo "Debug: nsys output:" >> "$debug_log"
    echo "$nsys_output" >> "$debug_log"

    # 处理输出
    while IFS= read -r line; do
        # 检测gpukernsum部分的开始
        if [[ $line =~ "Executing 'gpukernsum' stats report" ]]; then
            in_gpukernsum=true
            continue
        fi

        # 如果不在gpukernsum部分，继续下一行
        if [[ "$in_gpukernsum" != true ]]; then
            continue
        fi

        # 处理sparse_gemm kernel行
        if [[ $line =~ sm86_xmma_sparse_gemm ]]; then
            # 提取时间（纳秒）
            kernel_time=$(echo "$line" | awk '{print $2}')
            if [[ -n "$kernel_time" ]]; then
                if [[ -z "$sparse_gemm_min_time" || $(awk "BEGIN {print ($kernel_time < $sparse_gemm_min_time)}") -eq 1 ]]; then
                    sparse_gemm_min_time=$kernel_time
                    echo "Debug: Updated sparse_gemm_min_time: $sparse_gemm_min_time ns" >> "$debug_log"
                fi
            fi
        # 处理sputnik kernel行
        elif [[ $line =~ "void sputnik::<unnamed>::Kernel" ]]; then
            # 提取时间（纳秒）
            sputnik_kernel_time=$(echo "$line" | awk '{print $2}')
            echo "Debug: Found sputnik kernel time: $sputnik_kernel_time ns" >> "$debug_log"
        fi
    done <<< "$nsys_output"

    # 计算总时间，与原始逻辑保持一致
    if [[ -n "$sparse_gemm_min_time" && $(awk "BEGIN {print ($sputnik_kernel_time > 0)}") -eq 1 ]]; then
        sparTA_total_time=$(awk "BEGIN {print $sparse_gemm_min_time + $sputnik_kernel_time}")
        echo "Debug: SparTA total time: $sparTA_total_time ns" >> "$debug_log"

        # 计算 TFLOPS（使用纳秒作为输入）
        tflops=$(calculate_tflops $m $k $n $sparTA_total_time)

        # 输出结果到 CSV
        echo "$m,$k,$n,$sk,$s,${sparTA_total_time},${tflops}" >> "$output_csv"
        echo "Debug: Output to CSV - SparTA_Total_Time: $sparTA_total_time ns, TFLOPS: $tflops" >> "$debug_log"
    else
        echo "Debug: Missing sparse_gemm or sputnik kernel data for M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"
    fi

    echo "Debug: Finished test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"
    echo "" >> "$debug_log"
}

# 确保 M 和 K 数组长度相同
if [ ${#M[@]} -ne ${#K[@]} ]; then
    echo "Error: M and K arrays must have the same length."
    exit 1
fi

# 主循环
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
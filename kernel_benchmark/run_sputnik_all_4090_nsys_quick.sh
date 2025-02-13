#!/bin/bash

# M=(8192 4096 32000 3584 32000 1024 28672 5120 5120  3584  3584 4096 13824 8192  18944 14336 4096 8192  11008 32000 20480 1024 3584 2560 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
# K=(29568 4096 5120 2560 8192  8192 8192  5120 13824 20480  3584 11008 5120 8192 3584  4096  14336 28672 4096 4096 3584 4096  18944 3584 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)
M=(8192 4096 32000)
K=(29568 4096 5120)
N=(8 16 32)
SPLIT_K=(1)
SPARSITY=(40 50 60 70)

# 设置输出文件
output_csv="sputnik_performance_results_all.csv"
debug_log="sputnik_debug_all.log"

# 创建或清空输出文件
echo "M,K,N,SplitK,Sparsity,Sputnik_Duration(ns),Sputnik_TFLOPS" > "$output_csv"
> "$debug_log"

# 计算 TFLOPS
calculate_tflops() {
    local m=$1
    local k=$2
    local n=$3
    local duration_ns=$4  # ns 级别
    awk -v m="$m" -v k="$k" -v n="$n" -v d="$duration_ns" 'BEGIN {print (2 * m * k * n) / (d / 1e9) / 1e12}'
}

process_test_case() {
    local m=$1
    local k=$2
    local n=$3
    local s=$4
    local sk=$5

    echo "Debug: Starting test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"

    # 运行 nsys 并捕获输出
    echo "Debug: Running nsys command..." >> "$debug_log"
    nsys_output=$(nsys nvprof ./spmm_test_sputnik $m $k $n $s $sk 2>&1)
    echo "Debug: nsys command completed" >> "$debug_log"

    # 写入日志文件
    echo "Debug: nsys output:" >> "$debug_log"
    echo "$nsys_output" >> "$debug_log"

    # 初始化变量
    local sputnik_total_time=0

    echo "Debug: Extracting Sputnik kernel time (Avg)..." >> "$debug_log"

    # 提取 `sputnik::<unnamed>::Kernel` 内核的 `Avg (ns)`
    sputnik_kernel_lines=$(echo "$nsys_output" | grep 'void sputnik::<unnamed>::Kernel')

    echo "Debug: Sputnik Kernel Lines Found:" >> "$debug_log"
    echo "$sputnik_kernel_lines" >> "$debug_log"

    while read -r line; do
        echo "Debug: Sputnik Kernel Line: $line" >> "$debug_log"

        kernel_avg_ns=$(echo "$line" | awk '{print $4}' | tr -d ',')  # 取 `Avg (ns)`（第 4 列）

        echo "Debug: Extracted Sputnik Avg (ns) = $kernel_avg_ns" >> "$debug_log"

        if [[ -n "$kernel_avg_ns" && "$kernel_avg_ns" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            sputnik_total_time=$kernel_avg_ns
            echo "Debug: Updated Sputnik Kernel Avg Time = $sputnik_total_time" >> "$debug_log"
        fi
    done <<< "$sputnik_kernel_lines"

    echo "Debug: Final Sputnik Kernel Avg Time = $sputnik_total_time" >> "$debug_log"

    # 确保变量有默认值
    if [[ -z "$sputnik_total_time" ]]; then sputnik_total_time=0; fi

    # 计算 TFLOPS
    tflops=$(calculate_tflops $m $k $n $sputnik_total_time)

    # 输出结果到 CSV
    echo "$m,$k,$n,$sk,$s,${sputnik_total_time},${tflops}" >> "$output_csv"
    echo "Debug: Output to CSV - Sputnik_Total_Time: $sputnik_total_time ns, TFLOPS: $tflops" >> "$debug_log"

    # 清理 nsys 生成的报告文件
    echo "Debug: Removing report files..." >> "$debug_log"
    rm -f report*
    echo "Debug: Report files removed." >> "$debug_log"

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
                echo "Running sputnik test case: M=$m, K=$k, N=$n, S=$s, SK=$sk"
                process_test_case $m $k $n $s $sk
            done
        done
    done
done

echo "Sputnik performance testing completed. Results saved in $output_csv"
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
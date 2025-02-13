#!/bin/bash

# 数组定义部分保持不变
M=(8192 4096 32000 3584 32000 1024 28672 5120 5120  3584  3584 4096 13824 8192  18944 14336 4096 8192  11008 32000 20480 1024 3584 2560 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
K=(29568 4096 5120 2560 8192  8192 8192  5120 13824 20480  3584 11008 5120 8192 3584  4096  14336 28672 4096 4096 3584 4096  18944 3584 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)
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
    local duration_us=$4  # 假设输入是微秒
    awk -v m="$m" -v k="$k" -v n="$n" -v d="$duration_us" 'BEGIN {print (2 * m * k * n) / (d / 1e6) / 1e12}'
}

process_test_case() {
    local m=$1
    local k=$2
    local n=$3
    local s=$4
    local sk=$5

    echo "Debug: Starting test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"

    # 初始化变量
    local sparse_gemm_time=0
    local sputnik_kernel_time=0

    # 运行 nsys 并捕获输出
    echo "Debug: Running nsys command..." >> "$debug_log"
    nsys_output=$(nsys profile --stats=true -o sparta_profile ./spmm_test_sparta $m $k $n $s $sk 2>&1)
    echo "Debug: nsys command completed" >> "$debug_log"

    # 将完整 nsys 输出写入日志文件
    echo "Debug: nsys output:" >> "$debug_log"
    echo "$nsys_output" >> "$debug_log"

    # 查找 sparse_gemm kernel 的时间（从 gpukernsum 部分）
    sparse_gemm_line=$(echo "$nsys_output" | grep -E "sparse_gemm.*$")
    if [[ -n "$sparse_gemm_line" ]]; then
        sparse_gemm_time=$(echo "$sparse_gemm_line" | awk '{gsub(/,/, "", $2); print $2/1000}')  # 转换为微秒
        echo "Detected sparse_gemm time: $sparse_gemm_time us" >> "$debug_log"
    fi

    # 查找 sputnik kernel 的时间
    sputnik_line=$(echo "$nsys_output" | grep -E "void sputnik::\<unnamed\>::Kernel.*$")
    if [[ -n "$sputnik_line" ]]; then
        sputnik_kernel_time=$(echo "$sputnik_line" | awk '{gsub(/,/, "", $2); print $2/1000}')  # 转换为微秒
        echo "Detected sputnik kernel time: $sputnik_kernel_time us" >> "$debug_log"
    fi

    # 计算总时间并输出结果
    if [[ $sparse_gemm_time != "0" && $sputnik_kernel_time != "0" ]]; then
        local sparTA_total_time=$(awk "BEGIN {print $sparse_gemm_time + $sputnik_kernel_time}")
        echo "Debug: SparTA total time: $sparTA_total_time us" >> "$debug_log"

        # 计算 TFLOPS
        local tflops=$(calculate_tflops $m $k $n $sparTA_total_time)

        # 输出结果到 CSV
        echo "$m,$k,$n,$sk,$s,${sparTA_total_time},${tflops}" >> "$output_csv"
        echo "Debug: Output to CSV - SparTA_Total_Time: $sparTA_total_time us, TFLOPS: $tflops" >> "$debug_log"
    else
        echo "Debug: Missing kernel timing data for M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"
    fi

    echo "Debug: Finished test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"
    echo "" >> "$debug_log"

    # 清理临时文件
    rm -f sparta_profile.*
}

# 主程序部分保持不变
if [ ${#M[@]} -ne ${#K[@]} ]; then
    echo "Error: M and K arrays must have the same length."
    exit 1
fi

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

if [ -s "$output_csv" ]; then
    echo "CSV file is not empty."
else
    echo "CSV file is empty. Please check the debug log for more information."
fi

echo "First few lines of the CSV file:"
head -n 5 "$output_csv"
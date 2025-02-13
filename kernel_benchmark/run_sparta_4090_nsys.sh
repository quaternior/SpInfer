#!/bin/bash

M=(8192 4096 32000 3584 32000 1024 28672 5120 5120  3584  3584 4096 13824 8192  18944 14336 4096 8192  11008 32000 20480 1024 3584 2560 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
K=(29568 4096 5120 2560 8192  8192 8192  5120 13824 20480  3584 11008 5120 8192 3584  4096  14336 28672 4096 4096 3584 4096  18944 3584 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)
N=(8 16 32)
SPLIT_K=(1)
SPARSITY=(40 50 60 70)

output_csv="cusparse_performance_results.csv"
debug_log="cusparse_debug.log"

echo "M,K,N,SplitK,Sparsity,cuSPARSE_C_Duration(ns),cuSPARSE_R_Duration(ns),cuSPARSE_C_TFLOPS,cuSPARSE_R_TFLOPS" > "$output_csv"
> "$debug_log"

calculate_tflops() {
    local m=$1
    local k=$2
    local n=$3
    local duration_us=$4

    if [[ -z "$duration_us" || "$duration_us" == "0" ]]; then
        echo "N/A"
    else
        awk -v m="$m" -v k="$k" -v n="$n" -v d="$duration_us" 'BEGIN {print (2 * m * k * n) / (d / 1e6) / 1e12}'
    fi
}

process_test_case() {
    local m=$1
    local k=$2
    local n=$3
    local s=$4
    local sk=$5

    echo "Debug: Starting test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"

    local cusparse_c_time=0
    local cusparse_r_time=0

    echo "Debug: Running nsys command..." >> "$debug_log"
    nsys_output=$(nsys profile --stats=true -o /dev/null ./spmm_test_cusparse $m $k $n $s $sk 2>&1)
    echo "Debug: nsys command completed" >> "$debug_log"

    echo "Debug: nsys output:" >> "$debug_log"
    echo "$nsys_output" >> "$debug_log"

    # 提取 cuSPARSE_C 内核时间（ns）
    cusparse_c_line=$(echo "$nsys_output" | grep -E "cusparse::load_balancing_kernel")
    if [[ -n "$cusparse_c_line" ]]; then
        cusparse_c_time_ns=$(echo "$cusparse_c_line" | awk '{gsub(/,/, "", $2); print $2}')
        cusparse_c_time=$((cusparse_c_time_ns / 1000))  # 转换为微秒
    fi

    # 提取 cuSPARSE_R 内核时间（ns）
    cusparse_r_line=$(echo "$nsys_output" | grep -E "cusparse::csrmm_alg2_kernel")
    if [[ -n "$cusparse_r_line" ]]; then
        cusparse_r_time_ns=$(echo "$cusparse_r_line" | awk '{gsub(/,/, "", $2); print $2}')
        cusparse_r_time=$((cusparse_r_time_ns / 1000))  # 转换为微秒
    fi

    # 计算 TFLOPS
    cusparse_c_tflops=$(calculate_tflops $m $k $n $cusparse_c_time)
    cusparse_r_tflops=$(calculate_tflops $m $k $n $cusparse_r_time)

    # 处理无效时间
    [[ $cusparse_c_time -eq 0 ]] && cusparse_c_time="N/A"
    [[ $cusparse_r_time -eq 0 ]] && cusparse_r_time="N/A"

    echo "$m,$k,$n,$sk,$s,${cusparse_c_time},${cusparse_r_time},${cusparse_c_tflops},${cusparse_r_tflops}" >> "$output_csv"
    echo "Debug: Output to CSV - C:${cusparse_c_time}us R:${cusparse_r_time}us" >> "$debug_log"
    echo "" >> "$debug_log"
}

# 数组长度检查
[[ ${#M[@]} -ne ${#K[@]} ]] && echo "Error: M/K array mismatch" && exit 1

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

echo "Results saved to $output_csv"
echo "Debug log: $debug_log"
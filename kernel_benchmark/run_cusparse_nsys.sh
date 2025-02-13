#!/bin/bash

# M=(20480)
# K=(4096)
# N=(8 16 32)
# SPLIT_K=(1)
# SPARSITY=(40 50 60 70)

M=(8192 4096 32000 3584 32000 1024 28672 5120 5120  3584  3584 4096 13824 8192  18944 14336 4096 8192  11008 32000 20480 1024 3584 2560 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
K=(29568 4096 5120 2560 8192  8192 8192  5120 13824 20480  3584 11008 5120 8192 3584  4096  14336 28672 4096 4096 3584 4096  18944 3584 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)
N=(8 16 32)
SPLIT_K=(1)
SPARSITY=(40 50 60 70)


# M=(3584 3584 1024 2560)
# K=(2560 3584 4096 3584)
# N=(8 16 32)
# SPLIT_K=(1)
# SPARSITY=(40 50 60 70)
# 设置输出文件
output_csv="cusparse_performance_results.csv"
debug_log="cusparse_debug.log"

# 创建或清空输出文件
echo "M,K,N,SplitK,Sparsity,cuSPARSE_C_Duration(ns),cuSPARSE_R_Duration(ns),cuSPARSE_C_TFLOPS,cuSPARSE_R_TFLOPS" > "$output_csv"
> "$debug_log"

calculate_tflops() {
    local m=$1
    local k=$2
    local n=$3
    local duration_us=$4  # 假设输入是微秒
    
    if [[ -z "$duration_us" || "$duration_us" == "0" ]]; then
        # 如果时间为空或为0，返回 N/A
        echo "N/A"
    else
        # 计算 TFLOPS = (2 * M * K * N) / (time in seconds) / 1e12
        awk -v m="$m" -v k="$k" -v n="$n" -v d="$duration_us" 'BEGIN {print (2 * m * k * n) / (d / 1e6) / 1e12}'
    fi
}
# 定义函数来处理 ncu 输出中的时间单位
convert_duration_to_useconds() {
    local duration=$1
    local unit=$2

    if [[ $unit == "msecond" ]]; then
        # 将毫秒转换为微秒
        echo $(awk -v d="$duration" 'BEGIN {print d * 1000}')
    elif [[ $unit == "usecond" ]]; then
        # 如果已经是微秒，直接返回
        echo "$duration"
    else
        # 如果单位未知，返回 0 并记录错误
        echo "Error: Unknown time unit $unit" >> "$debug_log"
        echo "0"
    fi
}
process_test_case() {
    local m=$1
    local k=$2
    local n=$3
    local s=$4
    local sk=$5

    echo "Debug: Starting test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"

    # 初始化变量
    local cusparse_c_time=0
    local cusparse_r_time=0

    # 运行 nsys 并捕获输出
    echo "Debug: Running nsys command..." >> "$debug_log"
    nsys_output=$(nsys nvprof ./spmm_test_cusparse $m $k $n $s $sk 2>&1)
    echo "Debug: nsys command completed" >> "$debug_log"

    # 将完整 nsys 输出写入日志文件
    echo "Debug: nsys output:" >> "$debug_log"
    echo "$nsys_output" >> "$debug_log"

    # 处理 nsys 输出
    while IFS= read -r line; do
        # 匹配 cuSPARSE_C 和 cuSPARSE_R 的输出行
        if [[ $line =~ CuSparse_C[[:space:]]*-\>[[:space:]]*Time/ms:[[:space:]]*([0-9.]+) ]]; then
            # 将毫秒转换为微秒
            cusparse_c_time=$(echo "${BASH_REMATCH[1]} * 1000" | bc)
            echo "Detected cuSPARSE_C time: $cusparse_c_time us" >> "$debug_log"
        elif [[ $line =~ CuSparse_R[[:space:]]*-\>[[:space:]]*Time/ms:[[:space:]]*([0-9.]+) ]]; then
            # 将毫秒转换为微秒
            cusparse_r_time=$(echo "${BASH_REMATCH[1]} * 1000" | bc)
            echo "Detected cuSPARSE_R time: $cusparse_r_time us" >> "$debug_log"
        fi
    done <<< "$nsys_output"

    # 检查 cuSPARSE_C 和 cuSPARSE_R 的时间
    if (( $(echo "$cusparse_c_time > 0" | bc -l) )) || (( $(echo "$cusparse_r_time > 0" | bc -l) )); then
        echo "Debug: cuSPARSE_C time: $cusparse_c_time us, cuSPARSE_R time: $cusparse_r_time us" >> "$debug_log"

        # 计算 TFLOPS
        if (( $(echo "$cusparse_c_time > 0" | bc -l) )); then
            cusparse_c_tflops=$(calculate_tflops $m $k $n $cusparse_c_time)
        else
            cusparse_c_tflops="N/A"
        fi

        if (( $(echo "$cusparse_r_time > 0" | bc -l) )); then
            cusparse_r_tflops=$(calculate_tflops $m $k $n $cusparse_r_time)
        else
            cusparse_r_tflops="N/A"
        fi

        # 输出结果到 CSV
        echo "$m,$k,$n,$sk,$s,${cusparse_c_time},${cusparse_r_time},${cusparse_c_tflops},${cusparse_r_tflops}" >> "$output_csv"
        echo "Debug: Output to CSV - cuSPARSE_C_Time: $cusparse_c_time us, cuSPARSE_R_Time: $cusparse_r_time us, cuSPARSE_C_TFLOPS: $cusparse_c_tflops, cuSPARSE_R_TFLOPS: $cusparse_r_tflops" >> "$debug_log"
    else
        echo "Debug: Missing cuSPARSE_C or cuSPARSE_R kernel data for M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"
    fi
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
                echo "Running cusparse test case: M=$m, K=$k, N=$n, S=$s, SK=$sk"
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
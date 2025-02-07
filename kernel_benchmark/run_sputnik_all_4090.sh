#!/bin/bash

# M=(28672)
# K=(8192)
# N=(16)
# SPLIT_K=(1)
# SPARSITY=(60)



M=(8192 4096 32000 3584 32000 1024 28672 5120 5120  3584  3584 4096 13824 8192  18944 14336 4096 8192  11008 32000 20480 1024 3584 2560 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
K=(29568 4096 5120 2560 8192  8192 8192  5120 13824 20480  3584 11008 5120 8192 3584  4096  14336 28672 4096 4096 3584 4096  18944 3584 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)
N=(8 16 32)
SPLIT_K=(1)
SPARSITY=(40 50 60 70)
# 设置输出文件
output_csv="sputnik_performance_results_all.csv"
debug_log="sputnik_debug_all.log"

# 创建或清空输出文件
echo "M,K,N,SplitK,Sparsity,Sputnik_Duration(ns),Sputnik_TFLOPS" > "$output_csv"
> "$debug_log"

# 定义函数来计算 TFLOPS
calculate_tflops() {
    local m=$1
    local k=$2
    local n=$3
    local duration_us=$4  # 假设输入是微秒
    awk -v m="$m" -v k="$k" -v n="$n" -v d="$duration_us" 'BEGIN {print (2 * m * k * n) / (d / 1e6) / 1e12}'
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
    local sputnik_total_time=0
    local in_sputnik_kernel=false
    local kernel_name=""

    # 运行 ncu 并捕获输出
    echo "Debug: Running ncu command..." >> "$debug_log"
    ncu_output=$(ncu --metrics gpu__time_duration.sum --kernel-name 'regex:.*' ./spmm_test_sputnik $m $k $n $s $sk 2>&1)
    echo "Debug: ncu command completed" >> "$debug_log"

    # 将完整 ncu 输出写入日志文件
    echo "Debug: ncu output:" >> "$debug_log"
    echo "$ncu_output" >> "$debug_log"

    # 处理 ncu 输出
    while IFS= read -r line; do
        if [[ $line =~ ^[[:space:]]*void[[:space:]]*sputnik::\<unnamed\>::Kernel ]]; then
            kernel_name=$(echo "$line" | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
            echo "Detected sputnik kernel: $kernel_name" >> "$debug_log"
            in_sputnik_kernel=true
        elif [[ $line =~ gpu__time_duration.sum ]] && [[ "$in_sputnik_kernel" = true ]]; then
            # 使用正则表达式提取时间和单位
            duration=$(echo "$line" | grep -oP '[0-9]+(\.[0-9]+)?$')
            unit=$(echo "$line" | grep -oP '\b(usecond|msecond)\b')

            if [[ -n "$duration" && -n "$unit" ]]; then
                # 转换时间为微秒
                duration_us=$(convert_duration_to_useconds "$duration" "$unit")

                echo "Processing sputnik kernel, Duration: $duration_us us" >> "$debug_log"
                sputnik_total_time=$(awk "BEGIN {print $sputnik_total_time + $duration_us}")
                in_sputnik_kernel=false
            else
                echo "Error: Invalid duration or unit in line: $line" >> "$debug_log"
            fi
        fi
    done <<< "$ncu_output"

    if [[ $(awk "BEGIN {print ($sputnik_total_time > 0)}") -eq 1 ]]; then
        echo "Debug: Sputnik total time: $sputnik_total_time microseconds" >> "$debug_log"

        # 计算 TFLOPS（注意：sputnik_total_time 现在是微秒）
        tflops=$(calculate_tflops $m $k $n $sputnik_total_time)

        # 输出结果到 CSV
        echo "$m,$k,$n,$sk,$s,${sputnik_total_time},${tflops}" >> "$output_csv"
        echo "Debug: Output to CSV - Sputnik_Total_Time: $sputnik_total_time microseconds, TFLOPS: $tflops" >> "$debug_log"
    else
        echo "Debug: Missing sputnik kernel data for M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"
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

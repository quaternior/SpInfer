#!/bin/bash


# M=(12288)
# K=(49152)
# N=(32)
# SPLIT_K=(6)
# SPARSITY=(70)

M=(20480)
K=(4096)
N=(8 16 32)
SPLIT_K=(1)
SPARSITY=(40 50 60 70)


# M=(8192 4096 32000 3584 32000 1024 28672 5120 5120  3584  3584 4096 13824 8192  18944 14336 4096 8192  11008 32000 20480 1024 3584 2560 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
# K=(29568 4096 5120 2560 8192  8192 8192  5120 13824 20480  3584 11008 5120 8192 3584  4096  14336 28672 4096 4096 3584 4096  18944 3584 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)
# N=(8 16 32)
# SPLIT_K=(7 7 3 7 3 7 4 5 5 7 7 7 3 7 7 7 7 7 7 3 6 7 7 7 1 3 4 3 7 5 2 5 2 6 3 6)
# SPARSITY=(40 50 60 70)

process_kernel_name() {
    local name="$1"
    if [[ "$name" == *cutlass*tensorop* || "$name" == *s1688gemm* || "$name" == *s161616gemm* || "$name" == *s16816gemm* ]]; then
        echo "cuBLAS_TC"
    elif [[ "$name" == *sgemm* ]]; then
        echo "cublas-cuda-core"
    elif [[ "$name" == *SpMM_Kernel_bitmap_v1* ]]; then
        echo "SpInfer-SpMMV1"
    elif [[ "$name" == *SpMM_Kernel_bitmap_v2* ]]; then
        echo "SpInfer-SpMMV2"
    elif [[ "$name" == *SpMM_Kernel_bitmap_v3* ]]; then
        echo "SpInfer-SpMMV3"
    elif [[ "$name" == *SpMM_Kernel\<* ]]; then
        echo "Flash-LLM"
    else
        echo "$name"
    fi
}


# 设置输出文件
output_csv="spmm_performance_results_main_v2.csv"
debug_log="debug_main_v2.log"

# 创建或清空输出文件
echo "M,K,N,SplitK,Sparsity,Kernel,Duration(ns),TFLOPS" > "$output_csv"
> "$debug_log"

# process_kernel_name 函数保持不变

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
    declare -A accumulated_times
    declare -A kernel_counts
    local splitk_reduction_time=0
    local splitk_reduction_count=0
    local splitkreduce_kernel_time=0
    local splitkreduce_kernel_count=0
    local kernel_name=""
    local last_processed_name=""

    # 运行 ncu 并捕获输出
    echo "Debug: Running ncu command..." >> "$debug_log"
    ncu_output=$(ncu --metrics gpu__time_duration.sum --kernel-name 'regex:.*' ./spmm_test $m $k $n $s $sk 2>&1)
    echo "Debug: ncu command completed" >> "$debug_log"
    echo "$ncu_output" >> "$debug_log"

    # 处理 ncu 输出
    while IFS= read -r line; do
        if [[ $line =~ ^[[:space:]]*(void|ampere_fp16|cutlass|cublasLt|cublas|SplitK_Reduction|ampere_sgemm|ampere_) ]]; then
            kernel_name=$(echo "$line" | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
            echo "Detected kernel: $kernel_name" >> "$debug_log"
        elif [[ $line =~ gpu__time_duration.sum ]]; then
            # 使用正则表达式提取时间和单位
            duration=$(echo "$line" | grep -oP '[0-9]+(\.[0-9]+)?$')
            unit=$(echo "$line" | grep -oP '\b(usecond|msecond)\b')

            if [[ -n "$duration" && -n "$unit" ]]; then
                # 转换时间为微秒
                duration_us=$(convert_duration_to_useconds "$duration" "$unit")
                
                if [[ "$kernel_name" == *splitKreduce_kernel* ]]; then
                    echo "Processing splitKreduce_kernel, Duration: $duration_us us" >> "$debug_log"
                    splitkreduce_kernel_time=$(awk "BEGIN {print $splitkreduce_kernel_time + $duration_us}")
                    ((splitkreduce_kernel_count++))
                elif [[ "$kernel_name" == SplitK_Reduction* ]]; then
                    echo "Processing SplitK_Reduction, Duration: $duration_us us" >> "$debug_log"
                    splitk_reduction_time=$(awk "BEGIN {print $splitk_reduction_time + $duration_us}")
                    ((splitk_reduction_count++))
                else
                    processed_name=$(process_kernel_name "$kernel_name")
                    echo "Processed kernel name: $processed_name, Duration: $duration_us us" >> "$debug_log"
                    
                    current_time=${accumulated_times[$processed_name]:-0}
                    accumulated_times[$processed_name]=$(awk "BEGIN {print $current_time + $duration_us}")
                    ((kernel_counts[$processed_name]++))
                fi
            else
                echo "Error: Invalid duration or unit in line: $line" >> "$debug_log"
            fi
        fi
    done <<< "$ncu_output"

    # 计算平均值并加到对应的 kernel 上
    if [[ $splitkreduce_kernel_count -gt 0 ]]; then
        avg_splitkreduce_kernel_time=$(awk "BEGIN {print $splitkreduce_kernel_time / $splitkreduce_kernel_count}")
        echo "Debug: Average splitKreduce_kernel time: $avg_splitkreduce_kernel_time us" >> "$debug_log"
        if [[ -v "accumulated_times[cuBLAS_TC]" ]]; then
            accumulated_times[cuBLAS_TC]=$(awk "BEGIN {print ${accumulated_times[cuBLAS_TC]} + $avg_splitkreduce_kernel_time}")
            echo "Debug: Added average splitKreduce_kernel time to cuBLAS_TC: ${accumulated_times[cuBLAS_TC]} us" >> "$debug_log"
        fi
    fi

    if [[ $splitk_reduction_count -gt 0 ]]; then
        avg_splitk_reduction_time=$(awk "BEGIN {print $splitk_reduction_time / $splitk_reduction_count}")
        echo "Debug: Average SplitK_Reduction time: $avg_splitk_reduction_time us" >> "$debug_log"
        for key in "SpInfer-SpMMV1" "SpInfer-SpMMV2" "SpInfer-SpMMV3" "Flash-LLM"; do
            if [[ -v "accumulated_times[$key]" ]]; then
                accumulated_times[$key]=$(awk "BEGIN {print ${accumulated_times[$key]} + $avg_splitk_reduction_time}")
                echo "Debug: Added average SplitK_Reduction time to $key: ${accumulated_times[$key]} us" >> "$debug_log"
            fi
        done
    fi

    # 输出结果到 CSV
    echo "Debug: Preparing to write results to CSV" >> "$debug_log"
    for kernel in "${!accumulated_times[@]}"; do
        duration=${accumulated_times[$kernel]}
        count=${kernel_counts[$kernel]}
        if [[ "$duration" =~ ^[0-9]+(\.[0-9]+)?$ && $count -gt 0 ]]; then
            avg_duration=$(awk "BEGIN {print $duration / $count}")
            tflops=$(calculate_tflops $m $k $n $avg_duration)
            echo "$m,$k,$n,$sk,$s,\"$kernel\",${avg_duration},${tflops}" >> "$output_csv"
            echo "Debug: Output to CSV - Kernel: $kernel, Avg Duration: $avg_duration us, TFLOPS: $tflops" >> "$debug_log"
        else
            echo "Debug: Invalid duration value: $duration or count: $count for kernel: $kernel" >> "$debug_log"
        fi
    done

    echo "Debug: Finished test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"
    echo "" >> "$debug_log"
}
# 确保 M 和 K 数组长度相同
if [ ${#M[@]} -ne ${#K[@]} ]; then
    echo "Error: M and K arrays must have the same length."
    exit 1
fi
# 确保 M 和 sK 数组长度相同
if [ ${#M[@]} -ne ${#SPLIT_K[@]} ]; then
    echo "Error: M and SK arrays must have the same length."
    exit 1
fi

# 主循环
for ((i=0; i<${#M[@]}; i++)); do
    m=${M[i]}
    k=${K[i]}
    sk=${SPLIT_K[i]}
    for n in "${N[@]}"; do
        for s in "${SPARSITY[@]}"; do
            process_test_case $m $k $n $s $sk
        done
    done
done

echo "Performance testing completed. Results saved in $output_csv"
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

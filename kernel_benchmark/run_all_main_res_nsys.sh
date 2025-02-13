#!/bin/bash

M=(8192 4096 32000 3584 32000 1024 28672 5120 5120  3584  3584 4096 13824 8192  18944 14336 4096 8192  11008 32000 20480 1024 3584 2560 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
K=(29568 4096 5120 2560 8192  8192 8192  5120 13824 20480  3584 11008 5120 8192 3584  4096  14336 28672 4096 4096 3584 4096  18944 3584 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)
N=(8 16 32)
SPLIT_K=(7 7 3 7 3 7 4 5 5 7 7 7 3 7 7 7 7 7 7 3 6 7 7 7 1 3 4 3 7 5 2 5 2 6 3 6)
SPARSITY=(40 50 60 70)
# M=(4096)
# K=(4096)
# N=(32)
# SPLIT_K=(6)
# SPARSITY=(50)
# 处理内核名称
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
output_csv="spmm_performance_results_main_nsys.csv"
debug_log="debug_main_nsys.log"

# 创建或清空输出文件
echo "M,K,N,SplitK,Sparsity,Kernel,Duration(ns),TFLOPS" > "$output_csv"
> "$debug_log"

# 计算 TFLOPS
calculate_tflops() {
    local m=$1
    local k=$2
    local n=$3
    local duration_ns=$4
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
    declare -A accumulated_times
    declare -A kernel_counts
    local splitkreduce_kernel_time=0
    local splitkreduce_kernel_count=0
    local splitkreduction_time=0
    local splitkreduction_count=0
    local kernel_name=""

    # 运行 nsys 并捕获输出
    echo "Debug: Running nsys command..." >> "$debug_log"
    nsys_output=$(nsys nvprof ./spmm_test $m $k $n $s $sk 2>&1)
    echo "Debug: nsys command completed" >> "$debug_log"

    # 解析 `gpukernsum` 结果
    echo "Debug: Extracting kernel times from nsys_output..." >> "$debug_log"
    kernel_lines=$(echo "$nsys_output" | grep -E 'cutlass.*tensorop|s1688gemm|s161616gemm|s16816gemm|sgemm|SpMM_Kernel_bitmap_v[1-3]|SpMM_Kernel<|SplitK_Reduction|splitKreduce_kernel')
    if [[ -z "$kernel_lines" ]]; then
        echo "Error: No kernel execution times found in nsys output" >> "$debug_log"
        return
    fi

    echo "$kernel_lines" >> "$debug_log"

    while read -r line; do
        # 提取 `Avg (ns)` 和 `Name`
        duration_ns=$(echo "$line" | awk '{print $4}' | tr -d ',' | sed 's/[[:space:]]//g')
        kernel_name=$(echo "$line" | awk '{for (i=15; i<=NF; i++) printf "%s ", $i; print ""}' | sed 's/[ \t]*$//')
        echo "$kernel_name"
        # 过滤掉无效 `kernel_name`
        if [[ -z "$kernel_name" || "$kernel_name" =~ ^[0-9]+$ || "$kernel_name" =~ ^[[:punct:]]+$ ]]; then
            echo "Error: Skipping invalid kernel name: $kernel_name" >> "$debug_log"
            continue
        fi
        
        # 确保 `duration_ns` 是有效数字
        if [[ -z "$duration_ns" || ! "$duration_ns" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "Error: Invalid duration for kernel $kernel_name: $duration_ns, skipping CSV output" >> "$debug_log"
            continue
        fi

        # 处理 SplitK 内核
        if [[ "$kernel_name" == *splitKreduce_kernel* ]]; then
            echo "Processing splitKreduce_kernel, Duration: $duration_ns ns" >> "$debug_log"
            splitkreduce_kernel_time=$(awk "BEGIN {print $splitkreduce_kernel_time + $duration_ns}")
            ((splitkreduce_kernel_count++))
        elif [[ "$kernel_name" ==  *"SplitK_Reduction"* ]]; then
            echo "Processing SplitK_Reduction, Duration: $duration_ns ns" >> "$debug_log"
            splitkreduction_time=$(awk "BEGIN {print $splitkreduction_time + $duration_ns}")
            ((splitkreduction_count++))
        else
            processed_name=$(process_kernel_name "$kernel_name")
            accumulated_times[$processed_name]=$(awk "BEGIN {print ${accumulated_times[$processed_name]:-0} + $duration_ns}")
            ((kernel_counts[$processed_name]++))
        fi
    done <<< "$kernel_lines"

    echo "Debug: Accumulated kernel times:" >> "$debug_log"
    declare -p accumulated_times >> "$debug_log"

    if [[ ${#accumulated_times[@]} -eq 0 ]]; then
        echo "Error: No valid kernel times found, skipping CSV output" >> "$debug_log"
        return
    fi

    # 计算 `splitKreduce_kernel` 平均时间，并加到 `cuBLAS_TC`
    if [[ $splitkreduce_kernel_count -gt 0 ]]; then
        avg_splitkreduce_kernel_time=$(awk "BEGIN {print $splitkreduce_kernel_time / $splitkreduce_kernel_count}")
        echo "Debug: Average splitKreduce_kernel time: $avg_splitkreduce_kernel_time ns" >> "$debug_log"
        accumulated_times[cuBLAS_TC]=$(awk "BEGIN {print ${accumulated_times[cuBLAS_TC]:-0} + $avg_splitkreduce_kernel_time}")
        echo "Debug: Added average splitKreduce_kernel time to cuBLAS_TC: ${accumulated_times[cuBLAS_TC]} ns" >> "$debug_log"
    fi

     if [[ $splitkreduction_count -gt 0 ]]; then
        avg_splitkreduction_time=$(awk "BEGIN {print $splitkreduction_time / $splitkreduction_count}")
        echo "Debug: Average SplitK_Reduction time: $avg_splitkreduction_time ns" >> "$debug_log"
        for key in "${!accumulated_times[@]}"; do
            # 匹配处理后的SpMM内核名称，例如以"SpInfer-SpMM"开头
            if [[ "$key" == SpInfer-SpMM* ]]; then
                accumulated_times[$key]=$(awk "BEGIN {print ${accumulated_times[$key]:-0} + $avg_splitkreduction_time}")
                echo "Debug: Added average SplitK_Reduction time to $key: ${accumulated_times[$key]} ns" >> "$debug_log"
            fi
        done
        # 加到 `Flash-LLM`
        accumulated_times[Flash-LLM]=$(awk "BEGIN {print ${accumulated_times[Flash-LLM]:-0} + $avg_splitkreduction_time}")
        echo "Debug: Added average SplitK_Reduction time to Flash-LLM: ${accumulated_times[Flash-LLM]} ns" >> "$debug_log"
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
            echo "Debug: Output to CSV - Kernel: $kernel, Avg Duration: $avg_duration ns, TFLOPS: $tflops" >> "$debug_log"
        else
            echo "Debug: Invalid duration value: $duration or count: $count for kernel: $kernel" >> "$debug_log"
        fi
    done
    rm report*
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
            echo "Running test case: M=$m, K=$k, N=$n, S=$s, SK=$sk"
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

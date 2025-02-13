#!/bin/bash

M=(8192 4096 32000 3584 32000 1024 28672 5120 5120  3584  3584 4096 13824 8192  18944 14336 4096 8192  11008 32000 20480 1024 3584 2560 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
K=(29568 4096 5120 2560 8192  8192 8192  5120 13824 20480  3584 11008 5120 8192 3584  4096  14336 28672 4096 4096 3584 4096  18944 3584 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)
N=(8 16 32)
SPLIT_K=(7 7 3 7 3 7 4 5 5 7 7 7 3 7 7 7 7 7 7 3 6 7 7 7 1 3 4 3 7 5 2 5 2 6 3 6)
SPARSITY=(40 50 60 70)

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
output_csv="spmm_performance_results_main_v2.csv"
debug_log="debug_main_v2.log"

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
    local splitk_reduction_time=0
    local splitk_reduction_count=0
    local splitkreduce_kernel_time=0
    local splitkreduce_kernel_count=0
    local kernel_name=""

    # 运行 nsys 并捕获输出
    echo "Debug: Running nsys command..." >> "$debug_log"
    nsys_output=$(nsys nvprof ./spmm_test $m $k $n $s $sk 2>&1)
    echo "Debug: nsys command completed" >> "$debug_log"
    echo "$nsys_output" >> "$debug_log"

    # 处理 nsys 输出
    while IFS= read -r line; do
        if [[ $line =~ ^[[:space:]]*(void|ampere_fp16|cutlass|cublasLt|cublas|SplitK_Reduction|ampere_sgemm|ampere_) ]]; then
            kernel_name=$(echo "$line" | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
            echo "Detected kernel: $kernel_name" >> "$debug_log"
        elif [[ $line =~ gpu__time_duration.sum ]]; then
            duration_ns=$(echo "$line" | awk '{print $4}' | tr -d ',')
            if [[ -n "$duration_ns" && "$duration_ns" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                if [[ "$kernel_name" == *splitKreduce_kernel* ]]; then
                    echo "Processing splitKreduce_kernel, Duration: $duration_ns ns" >> "$debug_log"
                    splitkreduce_kernel_time=$(awk "BEGIN {print $splitkreduce_kernel_time + $duration_ns}")
                    ((splitkreduce_kernel_count++))
                elif [[ "$kernel_name" == SplitK_Reduction* ]]; then
                    echo "Processing SplitK_Reduction, Duration: $duration_ns ns" >> "$debug_log"
                    splitk_reduction_time=$(awk "BEGIN {print $splitk_reduction_time + $duration_ns}")
                    ((splitk_reduction_count++))
                else
                    processed_name=$(process_kernel_name "$kernel_name")
                    echo "Processed kernel name: $processed_name, Duration: $duration_ns ns" >> "$debug_log"
                    accumulated_times[$processed_name]=$(awk "BEGIN {print ${accumulated_times[$processed_name]:-0} + $duration_ns}")
                    ((kernel_counts[$processed_name]++))
                fi
            else
                echo "Error: Invalid duration in line: $line" >> "$debug_log"
            fi
        fi
    done <<< "$nsys_output"

    # 计算 SplitK 平均时间并加到对应的 kernel 上
    if [[ $splitkreduce_kernel_count -gt 0 ]]; then
        avg_splitkreduce_kernel_time=$(awk "BEGIN {print $splitkreduce_kernel_time / $splitkreduce_kernel_count}")
        accumulated_times[cuBLAS_TC]=$(awk "BEGIN {print ${accumulated_times[cuBLAS_TC]:-0} + $avg_splitkreduce_kernel_time}")
    fi

    if [[ $splitk_reduction_count -gt 0 ]]; then
        avg_splitk_reduction_time=$(awk "BEGIN {print $splitk_reduction_time / $splitk_reduction_count}")
        for key in "SpInfer-SpMMV1" "SpInfer-SpMMV2" "SpInfer-SpMMV3" "Flash-LLM"; do
            accumulated_times[$key]=$(awk "BEGIN {print ${accumulated_times[$key]:-0} + $avg_splitk_reduction_time}")
        done
    fi

    # 输出结果到 CSV
    for kernel in "${!accumulated_times[@]}"; do
        duration=${accumulated_times[$kernel]}
        avg_duration=$(awk "BEGIN {print $duration / ${kernel_counts[$kernel]}}")
        tflops=$(calculate_tflops $m $k $n $avg_duration)
        echo "$m,$k,$n,$sk,$s,\"$kernel\",${avg_duration},${tflops}" >> "$output_csv"
    done

    # 清理 nsys 生成的报告文件
    rm -f report*

    echo "Debug: Finished test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"
}

# 运行全部测试
for ((i=0; i<${#M[@]}; i++)); do
    for n in "${N[@]}"; do
        for s in "${SPARSITY[@]}"; do
            process_test_case ${M[i]} ${K[i]} $n $s ${SPLIT_K[i]}
        done
    done
done
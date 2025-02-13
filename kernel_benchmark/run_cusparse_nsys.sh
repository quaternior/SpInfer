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

    echo "Debug: Finished test case M=$m K=$k N=$n S=$s SK=$sk" >> "$debug_log"
    echo "" >> "$debug_log"
}
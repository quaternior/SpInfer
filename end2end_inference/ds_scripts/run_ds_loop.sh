#!/bin/bash

EXE="${SpInfer_HOME}/end2end_inference/ds_scripts/inference-test.py"

# Define the output directory path
OUTPUT_DIR="${SpInfer_HOME}/end2end_inference/ds_scripts/ds_result/2-gpu"

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# Define the batch size and output length ranges
BATCH_SIZES=(8 16 32 64)
OUTPUT_LENS=(64 128 256 512 1024)
# BATCH_SIZES=(8)
# OUTPUT_LENS=(128)

# gpus=2
for batch_size in "${BATCH_SIZES[@]}"; do
  for output_len in "${OUTPUT_LENS[@]}"; do
    # Define the output file name
    OUTPUT_FILE="$OUTPUT_DIR/output_batch_${batch_size}_output_len_${output_len}.log"

    # Run the command and save the output to the file
    CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 inference-test.py --ds_inference --greedy --use_kernel --name /mnt/models/opt-13b \
            --batch_size ${batch_size} --max_new_tokens ${output_len} --max_tokens 1500 > "$OUTPUT_FILE" 2>&1

    # Print the status
    echo "Completed run with batch_size=$batch_size, output_len=$output_len. Output saved to $OUTPUT_FILE"
  done
done

# gpus=4
# Define the output directory path
OUTPUT_DIR="${SpInfer_HOME}/end2end_inference/ds_scripts/ds_result/4-gpu"

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"
for batch_size in "${BATCH_SIZES[@]}"; do
  for output_len in "${OUTPUT_LENS[@]}"; do
    # Define the output file name
    OUTPUT_FILE="$OUTPUT_DIR/output_batch_${batch_size}_output_len_${output_len}.log"

    # Run the command and save the output to the file
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus 4 inference-test.py --ds_inference --greedy --use_kernel --name /mnt/models/opt-30b \
            --batch_size ${batch_size} --max_new_tokens ${output_len} --max_tokens 1500 > "$OUTPUT_FILE" 2>&1

    # Print the status
    echo "Completed run with batch_size=$batch_size, output_len=$output_len. Output saved to $OUTPUT_FILE"
  done
done

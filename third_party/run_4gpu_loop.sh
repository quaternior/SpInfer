#!/bin/bash

# Define the configuration file path
CONFIG_FILE="${SpInfer_HOME}/third_party/FasterTransformer/examples/cpp/multi_gpu_gpt/gpt_config.ini"

sed -i -E "s/(tensor_para_size=)[0-9]+/\14/" "$CONFIG_FILE"
sed -i -E "s#(model_dir\s*=\s*).*#\1/data2/fanruibo/models/opt-30b/spinfer-model/4-gpu#" "$CONFIG_FILE"

# Define the output directory path
OUTPUT_DIR="${SpInfer_HOME}/third_party/FasterTransformer/Result_13B/4-gpu"

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

cd ${SpInfer_HOME}/third_party/FasterTransformer/build
# Define the batch size and output length ranges
BATCH_SIZES=(8 16 32 64)
OUTPUT_LENS=(64 128 256 512 1024)

# Loop through all combinations
for batch_size in "${BATCH_SIZES[@]}"; do
  for output_len in "${OUTPUT_LENS[@]}"; do
    # Modify the configuration file
    sed -i "s/^request_batch_size=.*/request_batch_size=$batch_size/" $CONFIG_FILE
    sed -i "s/^request_output_len=.*/request_output_len=$output_len/" $CONFIG_FILE

    # Define the output file name
    OUTPUT_FILE="$OUTPUT_DIR/output_batch_${batch_size}_output_len_${output_len}.log"

    # Run the command and save the output to the file
    CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -n 4 --allow-run-as-root ./bin/multi_gpu_gpt_example > "$OUTPUT_FILE" 2>&1

    # Print the status
    echo "Completed run with batch_size=$batch_size, output_len=$output_len. Output saved to $OUTPUT_FILE"
  done
done

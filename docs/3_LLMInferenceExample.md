# LLM Inference Example

#### 1. Building Faster-Transformer with SpInfer integration
Note: cudnn should be installed/linked.
```sh
cd $SpInfer_HOME/third_party/FasterTransformer/
mkdir -p build && cd build

cmake -DSM=89 -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON -DSpInfer=ON ..
make -j
```
Note: if you want run standard **Faster-Transformer** (cuBLAS will be used for all MatMuls), use the following cmake command instead:
```sh
cmake -DSM=89 -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON -DFLASH_LLM=OFF ..
make -j
```
Note: if you want run Faster-Transformer with **Flash-llm**, use the following cmake command instead:
```sh
cmake -DSM=89 -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON -DFLASH_LLM=ON ..
make -j
```

#### 2. Downloading & Converting OPT models

Downloading the OPT model checkpoint (taking OPT-30B as an example) :
```sh
cd $SpInfer_HOME/end2end_inference/models
git clone https://huggingface.co/facebook/opt-30b && opt-30b
apt install git-lfs
git lfs pull --include="pytorch_model*"
```
Converting from PyTorch format to Faster-Transformer format, here **-i_g** means the number of GPUs you will use for inference while -p means the number of CPU threads you use for this model format transformation. This script will load pytorch model from -i PATH and output the generated model to -o PATH.
```sh
cd $SpInfer_HOME/end2end_inference/ft_tools
python huggingface_opt_convert_Phase1.py \
      -i $SpInfer_HOME/end2end_inference/models/opt-30b \
      -o $SpInfer_HOME/end2end_inference/models/opt-30b/c-model \
      -i_g 1 \
      -weight_data_type fp16 \
      -p 64
```

Converting from Faster-Transformer format to SpInfer format. Note that here **-i_g** means the number of GPUs you will use for inference. We fake 80% sparsity in this script for demonstration purpose, which can be disabled if you already have your OPT models pruned. Besides, hand-tuned SplitK is used for different input shapes.
```sh
python huggingface_opt_convert_Phase2.py \
      -m $SpInfer_HOME/end2end_inference/models/opt-30b/c-model \
      -l $SpInfer_HOME/build/libSpMM_API.so \
      -i_g 1 \
      -p 64
```

#### 3. Configuration
First, you need to modify the config file for the inference task in path:
*$SpInfer_HOME/third_party/FasterTransformer/examples/cpp/multi_gpu_gpt/gpt_config.ini*
The most important thing is to change the model_dir according to your absolute file path.
The model_name should also be changed if you are running other OPT models.
Besides, the tensor_para_size should be the same as the **-i_g** you used in step 2.

#### 4. Running Inference
Note that **-n** here means the number of GPUs you use for inference, which should be the same as **-ig** in step 2.
```sh
cd $SpInfer_HOME/third_party/FasterTransformer/build
mpirun -n 1 --allow-run-as-root ./bin/multi_gpu_gpt_example
```

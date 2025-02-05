# SpInfer Artifact for EuroSys'25.

## 1. Clone this project.
```
git clone https://github.com/xxyux/SpInfer.git
cd SpInfer
git submodule update --init --recursive
source Init_SpInfer.sh
cd $SpInfer_HOME/third_party/FasterTransformer && git apply ../ft_spinfer.patch
```

+ **Requirements**: 
> + `Ubuntu 16.04+`
> + `gcc >= 7.3`
> + `cmake >= 3.30.3`
> + `CUDA >= 12.2` and `nvcc >= 12.0`
> + NVIDIA GPU with `sm >= 80` (i.e., Ampere, like A6000. Ada, like RTX4090).

## 2. Environment Setup.(Install via Conda)
+ 2.1 Install **`conda`** on system **[Toturial](https://docs.anaconda.com/miniconda/)**.
+ 2.2 Create a **`conda`** environment: 
```
cd $SpInfer_HOME
conda env create -f spinfer.yml
conda activate spinfer
```

## 3. Install **`SpInfer`**.
The libSpMM_API.so and SpMM_API.cuh will be available for easy integration after:
```
cd $SpInfer_HOME/build && make -j
``` 

## 4. Runing **SpInfer** in kernel benchmark.
TODO:

## 5. Running End-to-end model.
#### 5.1 Building
Follow the steps in **[SpInfer/docs/3_LLMInferenceExample](https://github.com/xxyux/SpInfer/blob/main/docs/3_LLMInferenceExample.md#llm-inference-example)**
+ Building Faster-Transformer with (SpInfer, Flash-llm or Standard) integration
+ Downloading & Converting OPT models
+ Configuration
Note: Model_dir is different for SpInfer, Flash-llm and Faster-Transformer.
#### 5.2 Running Inference (SpInfer, Flash-llm && Faster-Transformer)
> + `cd $SpInfer_HOME/SpInfer/third_party/`
> + `bash run_1gpu_loop.sh`
> + Check the results in `$SpInfer_HOME/third_party/FasterTransformer/OutputFile_1gpu_our_60_inlen64/`
> + Test tensor_para_size=2 using `bash run_2gpu_loop.sh`

#### 5.3 Runing DeepSpeed baseline
```sh
cd $SpInfer_HOME/end2end_inference/ds_scripts
pip install -r requirements.txt
deepspeed --num_gpus 1 inference-test.py --ds_inference --greedy --use_meta_tensor --use_kernel --name facebook/opt-30b --batch_size 8 --max_new_tokens 512 --max_tokens 576
```
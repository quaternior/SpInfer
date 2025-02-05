# Preparations

#### 1. Download source code
```sh
git clone https://github.com/xxyux/SpInfer.git
cd SpInfer
git submodule update --init --recursive
source Init_FlashLLM.sh
cd $SpInfer_HOME/third_party/FasterTransformer && git apply ../ft_spinfer.patch
```

#### 2. Build conda env

```sh
cd $SpInfer_HOME
conda env create -f spinfer.yml
conda activate spinfer
```

#### 3. Building
The libSpMM_API.so and SpMM_API.cuh will be available for easy integration after:
```sh
cd $SpInfer_HOME/build && make -j
```
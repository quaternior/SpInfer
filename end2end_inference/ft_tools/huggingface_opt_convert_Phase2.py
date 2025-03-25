# Copyright 2025 The SpInfer Authors. All rights reserved.
# Copyright 2023 The FLash-LLM Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###################################################################
weightNameList = [  
    "model.layers.{}.attention.query_key_value.weight.{}", 
    "model.layers.{}.attention.dense.weight.{}",
    "model.layers.{}.mlp.dense_h_to_4h.weight.{}",
    "model.layers.{}.mlp.dense_4h_to_h.weight.{}"
]
# Dictionary for OPT-30B, 66B and 175B from 1-GPU to 4 GPU
# SplitKDict = {16384:2, 4096:2, 12288:2, 21504:5, 7168:7, 28672:7, 27648:2, 9216:6, 36864:3, 12288:9, 49152:9, 10752:5, 14336:7, 13824:4, 18432:3, 24576:9, 5376:10, 6912:8}
# Dictionary for OPT-6.7B, 13B, 30B
SplitKDict = {4096:6, 12288:2, 16384:2, 5120:6, 15360:2, 20480:3, 7680:4, 10240:2, 7168:4, 21504:3, 28672:1}
THRESHOLD_SKIPPING_REPLACE = 0.45
FAKE_SPARSITY = True
REMOVE_ORGINAL_DENSE_WEIGHTS = False
###################################################################
# python huggingface_opt_convert_Phase2.py \
#       -m /data/opt-13b/ft-model-60 \
#       -l $SpInfer_HOME/build/libSpMM_API.so \
#       -i_g 1 \
#       -p 64
import ctypes
import configparser
import numpy as np
import argparse
import os
import multiprocessing
from datetime import datetime
import time

# Python wrapper for C++ function in libSpMM.so
# def GenBinaryFile(DenseMatrixFileName_without_suffix, M, K):
#     #
#     DenseMatrixFileName = DenseMatrixFileName_without_suffix + ".bin"
#     NZWeightsFileName   = DenseMatrixFileName_without_suffix + ".NZWeights.bin"
#     TileOffsetsFileName = DenseMatrixFileName_without_suffix + ".TileOffsets.bin"
#     OutputSizesFileName = DenseMatrixFileName_without_suffix + ".OutputSizes.bin"
#     '''
#     extern "C"
#     void GenSparseMatrixBinFile(char* DenseMatrixFileName,
#                               int M,
#                               int K,
#                               char* NZWeightsFileName,
#                               char* TileOffsetsFileName,
#                               char* OutputSizesFileName);       // NNZ -> NumOffsets                      
#     '''
#     SpMM_Lib = ctypes.cdll.LoadLibrary(SpMM_Lib_PATH)
#     SpMM_Lib.GenSparseMatrixBinFile.argtypes = [ctypes.c_char_p, 
#                                                 ctypes.c_int, ctypes.c_int, 
#                                                 ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p] 
#     DenseMatrixFileName_ = ctypes.create_string_buffer(DenseMatrixFileName.encode('utf-8'))
#     NZWeightsFileName_   = ctypes.create_string_buffer(NZWeightsFileName.encode('utf-8'))
#     TileOffsetsFileName_ = ctypes.create_string_buffer(TileOffsetsFileName.encode('utf-8'))
#     OutputSizesFileName_ = ctypes.create_string_buffer(OutputSizesFileName.encode('utf-8'))
#     SpMM_Lib.GenSparseMatrixBinFile(DenseMatrixFileName_, M, K, NZWeightsFileName_, TileOffsetsFileName_, OutputSizesFileName_)
def GenBinaryFile(DenseMatrixFileName_without_suffix, M, K):
    #
    DenseMatrixFileName = DenseMatrixFileName_without_suffix + ".bin"
    CompressedvalFileName   = DenseMatrixFileName_without_suffix + ".NZWeights.bin"
    BitmapglobalFileName = DenseMatrixFileName_without_suffix + ".gtile.bin"
    BitmapmedianFileName = DenseMatrixFileName_without_suffix + ".mtile.bin"
    BitmapFileName = DenseMatrixFileName_without_suffix + ".ltile.bin"
    NnzintileFileName = DenseMatrixFileName_without_suffix + ".intile.bin"
    OutputSizesFileName = DenseMatrixFileName_without_suffix + ".OutputSizes.bin"
    '''
    extern "C"
    void Our_GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                    int   M,
                                    int   K,
                                    char* Compressed_ValFileName,
                                    char* bitmap_TileOffsets_globalFileName,
                                    char* bitmap_TileOffsets_medianFileName,
                                    char* bitmapFileName,
                                    char* max_nnz_intileFileName,
                                    char* OutputSizesFileName);                   
    '''
    SpMM_Lib = ctypes.cdll.LoadLibrary(SpMM_Lib_PATH)
    SpMM_Lib.Our_GenSparseMatrixBinFile.argtypes = [ctypes.c_char_p, 
                                                ctypes.c_int, ctypes.c_int, 
                                                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                                                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p] 
    DenseMatrixFileName_ = ctypes.create_string_buffer(DenseMatrixFileName.encode('utf-8'))
    CompressedvalFileName_   = ctypes.create_string_buffer(CompressedvalFileName.encode('utf-8'))
    BitmapglobalFileName_ = ctypes.create_string_buffer(BitmapglobalFileName.encode('utf-8'))
    BitmapmedianFileName_   = ctypes.create_string_buffer(BitmapmedianFileName.encode('utf-8'))
    BitmapFileName_ = ctypes.create_string_buffer(BitmapFileName.encode('utf-8'))
    NnzintileFileName_   = ctypes.create_string_buffer(NnzintileFileName.encode('utf-8'))
    OutputSizesFileName_ = ctypes.create_string_buffer(OutputSizesFileName.encode('utf-8'))
    SpMM_Lib.Our_GenSparseMatrixBinFile(DenseMatrixFileName_, M, K, 
                                        CompressedvalFileName_, BitmapglobalFileName_, BitmapmedianFileName_, 
                                        BitmapFileName_, NnzintileFileName_, OutputSizesFileName_)


def matrix_replace_process(FT_ModelPath, weightNameList, i, j, k, M, K):
    try:
        DenseFileName = FT_ModelPath + "/" + weightNameList[k].format(j,i)
        denseMatrix = np.fromfile(DenseFileName+".bin", dtype=np.float16)
        # Firstly, we should transpose the weight matrix
        denseMatrix = denseMatrix.reshape((M[k], K[k])).transpose()
        if FAKE_SPARSITY:
            # Then, we should sparsify the dense matrix optionally if the model is not yet sparsified.
            # denseMatrix *= np.random.binomial(size=denseMatrix.shape, n=1, p=0.2) #80% Sparsity #denseMatrix *= np.random.randint(0, 2, denseMatrix.shape)  # 50% Sparsity
            denseMatrix *= np.random.binomial(size=denseMatrix.shape, n=1, p=0.4) #50% Sparsity
            denseMatrix.tofile(DenseFileName+".bin")
        #
        NNZ = np.count_nonzero(denseMatrix)
        Sparsity = 1.0 - float(NNZ)/(M[k]*K[k])
        print("Name: "+DenseFileName+" NNZ: "+str(NNZ)+" TensorSize: "+str(M[k]*K[k])+" Sparsity: "+ str(Sparsity))
        if(Sparsity < THRESHOLD_SKIPPING_REPLACE):
            # used for faking sparse weight
            #denseMatrix *= np.random.binomial(size=denseMatrix.shape, n=1, p=0.2) #80% Sparsity #denseMatrix *= np.random.randint(0, 2, denseMatrix.shape)  # 50% Sparsity
            #denseMatrix.tofile(DenseFileName+".bin")
            #GenBinaryFile(DenseFileName, M[k], K[k])
            print("Skipping replacing "+DenseFileName+" as it is very dense!")
        else:
            # At last, we should replace the dense matrix with the sparse version
            GenBinaryFile(DenseFileName, M[k], K[k])
            if REMOVE_ORGINAL_DENSE_WEIGHTS:
                if os.path.exists(DenseFileName+".bin"):
                    os.remove(DenseFileName+".bin")
                else:
                    print("Phase 2 Error: Can not remove file "+DenseFileName+".bin"+" since it does not exist.")
                    exit()
            #time.sleep(12-j)
            print("Succeeded in processing: " + DenseFileName + ".bin")
    except Exception as e:
        print("Failed in processing: "+DenseFileName + ".bin")
        print(repr(e))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
#     parser.add_argument('-model_dir', '-m', type=str, help='file path for reading and saving the model files', required=True)
#     parser.add_argument('-spmm_lib_path', '-l', type=str, help='file path for spmm lib file', required=True)
#     parser.add_argument('-infer_gpu_num', '-i_g', type=int, help='How many gpus for inference', required=True)
#     parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 1)", default=1)
#     args = parser.parse_args()
#     print("\n=============== Argument ===============")
#     for key in vars(args):
#         print(f"{key}: {vars(args)[key]}")
#     print("========================================")    
#     #
#     nGPU = args.infer_gpu_num
#     FT_ModelPath = args.model_dir + "/%d-gpu/" % nGPU
#     SpMM_Lib_PATH = args.spmm_lib_path
#     nProc = args.processes
#     #
#     print("Reading FT_Model's config.ini...")
#     config = configparser.ConfigParser()
#     config.read(FT_ModelPath+"/config.ini")
#     assert( len(config.sections())==1 )
#     ModelName = config.sections()[0]
#     print( "Model Name:" + ModelName )
#     assert("weight_data_type" in config[ModelName])
#     assert(config[ModelName]["weight_data_type"] == "fp16")
#     assert("num_layer" in config[ModelName])
#     assert("head_num" in config[ModelName])
#     assert("size_per_head" in config['gpt'])
#     assert("inter_size" in config[ModelName])
#     num_layer       = int( config[ModelName]["num_layer"]     )
#     head_num        = int( config[ModelName]["head_num"]      )
#     size_per_head   = int( config[ModelName]["size_per_head"] )
#     hidden_size     = head_num * size_per_head
#     inter_size      = int( config[ModelName]['inter_size']    )
#     assert(inter_size == hidden_size*4)

#     assert(hidden_size%nGPU == 0)
#     M = [ 3*hidden_size//nGPU, hidden_size     , hidden_size*4//nGPU, hidden_size        ]
#     K = [ hidden_size       , hidden_size//nGPU, hidden_size       , hidden_size*4//nGPU ]

#     #self_attention_weights.query_weight -> self_attention_weights.attention_output_weight 
#     #-> ffn_weights.intermediate_weight  -> ffn_weights.output_weight     
#     NNZ_List_AllGPU = []
#     NumOffsets_List_AllGPU = []
#     SplitK_List_AllGPU = []
#     for i in range(nGPU):
#         NNZ_List_AllGPU.append([])
#         NumOffsets_List_AllGPU.append([])
#         SplitK_List_AllGPU.append([])
#     ##################################################################################################
#     start_time = datetime.now()
#     print(str(nProc)+" threads are used!")
#     pool = multiprocessing.Pool(nProc)
#     for i in range(nGPU):
#         for j in range(num_layer):
#             for k in range(len(weightNameList)):
#                 pool.apply_async(matrix_replace_process,
#                             args=(FT_ModelPath, weightNameList, i, j, k, M, K, ) )
#     pool.close()
#     pool.join()
#     ##################################################################################################
#     SkippedMatrixCount = [0, 0, 0, 0]
#     ReplacedMatrixCount = [0, 0, 0, 0]
#     for i in range(nGPU):
#         for j in range(num_layer):
#             for k in range(len(weightNameList)):
#                 DenseFileName = FT_ModelPath + "/" + weightNameList[k].format(j,i)
#                 if os.path.exists(DenseFileName+".OutputSizes.bin"):
#                     Sizes = np.fromfile(DenseFileName+".OutputSizes.bin", dtype=np.int32)
#                     assert(Sizes.shape[0]==2)
#                     NNZ = Sizes[0]
#                     NumOffsets = Sizes[1]
#                     NNZ_List_AllGPU[i].append(NNZ)
#                     NumOffsets_List_AllGPU[i].append(NumOffsets)
#                     assert M[k] in SplitKDict, "M = {} is not in the SplitKDict!".format(M[k])
#                     SplitK_List_AllGPU[i].append( SplitKDict[M[k]] )
#                     #
#                     ReplacedMatrixCount[k] = ReplacedMatrixCount[k]+1
#                 else:
#                     assert(os.path.exists(DenseFileName+".bin"))
#                     NNZ_List_AllGPU[i].append(0)
#                     NumOffsets_List_AllGPU[i].append(0)
#                     SplitK_List_AllGPU[i].append(0)
#                     #
#                     SkippedMatrixCount[k] = SkippedMatrixCount[k]+1

#     '''
#     dir_path + "/NNZ_List."          + std::to_string(tensor_para_rank_) + ".bin"
#     dir_path + "/NumOffsets_List."   + std::to_string(tensor_para_rank_) + ".bin"
#     dir_path + "/SplitK_List."       + std::to_string(tensor_para_rank_) + ".bin"
#     '''
#     print("Saving NNZ_List.x.bin NumOffsets_List.x.bin SplitK_List.x.bin...")
#     for i in range(nGPU):
#         NNZ_List_FileName        = FT_ModelPath + "/NNZ_List.{}.bin".format(i)
#         NumOffsets_List_FileName = FT_ModelPath + "/NumOffsets_List.{}.bin".format(i)
#         SplitK_List_FileName     = FT_ModelPath + "/SplitK_List.{}.bin".format(i)
#         np.array(NNZ_List_AllGPU[i]).astype(np.int32).tofile(NNZ_List_FileName)
#         np.array(NumOffsets_List_AllGPU[i]).astype(np.int32).tofile(NumOffsets_List_FileName)
#         np.array(SplitK_List_AllGPU[i]).astype(np.int32).tofile(SplitK_List_FileName)
    
#     stop_time = datetime.now()
#     run_time = (stop_time - start_time)
#     print(f"[INFO] Spend {run_time} (h:m:s) to convert the model: Phase-2")
#     for i in range(4):
#         print("MatMul"+str(i)+": "+str(ReplacedMatrixCount[i])+" Matrices replaced, "+str(SkippedMatrixCount[i])+" skipped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-model_dir', '-m', type=str, help='file path for reading and saving the model files', required=True)
    parser.add_argument('-spmm_lib_path', '-l', type=str, help='file path for spmm lib file', required=True)
    parser.add_argument('-infer_gpu_num', '-i_g', type=int, help='How many gpus for inference', required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 1)", default=1)
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")    
    #
    nGPU = args.infer_gpu_num
    FT_ModelPath = args.model_dir + "/%d-gpu/" % nGPU
    SpMM_Lib_PATH = args.spmm_lib_path
    nProc = args.processes
    #
    print("Reading FT_Model's config.ini...")
    config = configparser.ConfigParser()
    config.read(FT_ModelPath+"/config.ini")
    assert( len(config.sections())==1 )
    ModelName = config.sections()[0]
    print( "Model Name:" + ModelName )
    assert("weight_data_type" in config[ModelName])
    assert(config[ModelName]["weight_data_type"] == "fp16")
    assert("num_layer" in config[ModelName])
    assert("head_num" in config[ModelName])
    assert("size_per_head" in config['gpt'])
    assert("inter_size" in config[ModelName])
    num_layer       = int( config[ModelName]["num_layer"]     )
    head_num        = int( config[ModelName]["head_num"]      )
    size_per_head   = int( config[ModelName]["size_per_head"] )
    hidden_size     = head_num * size_per_head
    inter_size      = int( config[ModelName]['inter_size']    )
    assert(inter_size == hidden_size*4)

    assert(hidden_size%nGPU == 0)
    M = [ 3*hidden_size//nGPU, hidden_size     , hidden_size*4//nGPU, hidden_size        ]
    K = [ hidden_size       , hidden_size//nGPU, hidden_size       , hidden_size*4//nGPU ]

    #self_attention_weights.query_weight -> self_attention_weights.attention_output_weight 
    #-> ffn_weights.intermediate_weight  -> ffn_weights.output_weight     
    val_count_List_AllGPU = []
    num_gtiles_List_AllGPU = []
    num_mtiles_List_AllGPU = []
    num_ltiles_List_AllGPU = []
    SplitK_List_AllGPU = []
    for i in range(nGPU):
        val_count_List_AllGPU.append([])
        num_gtiles_List_AllGPU.append([])
        num_mtiles_List_AllGPU.append([])
        num_ltiles_List_AllGPU.append([])
        SplitK_List_AllGPU.append([])
    ##################################################################################################
    start_time = datetime.now()
    print(str(nProc)+" threads are used!")
    pool = multiprocessing.Pool(nProc)
    for i in range(nGPU):
        for j in range(num_layer):
            for k in range(len(weightNameList)):
                pool.apply_async(matrix_replace_process,
                            args=(FT_ModelPath, weightNameList, i, j, k, M, K, ) )
    pool.close()
    pool.join()
    ##################################################################################################
    SkippedMatrixCount = [0, 0, 0, 0]
    ReplacedMatrixCount = [0, 0, 0, 0]
    for i in range(nGPU):
        for j in range(num_layer):
            for k in range(len(weightNameList)):
                DenseFileName = FT_ModelPath + "/" + weightNameList[k].format(j,i)
                if os.path.exists(DenseFileName+".OutputSizes.bin"):
                    Sizes = np.fromfile(DenseFileName+".OutputSizes.bin", dtype=np.int32)
                    assert(Sizes.shape[0]==4)
                    val_count = Sizes[0]
                    num_gtiles = Sizes[1]
                    num_mtiles = Sizes[2]
                    num_ltiles = Sizes[3]
                    val_count_List_AllGPU[i].append(val_count)
                    num_gtiles_List_AllGPU[i].append(num_gtiles)
                    num_mtiles_List_AllGPU[i].append(num_mtiles)
                    num_ltiles_List_AllGPU[i].append(num_ltiles)
                    assert M[k] in SplitKDict, "M = {} is not in the SplitKDict!".format(M[k])
                    SplitK_List_AllGPU[i].append( SplitKDict[M[k]] )
                    #
                    ReplacedMatrixCount[k] = ReplacedMatrixCount[k]+1
                else:
                    assert(os.path.exists(DenseFileName+".bin"))
                    val_count_List_AllGPU[i].append(0)
                    num_gtiles_List_AllGPU[i].append(0)
                    num_mtiles_List_AllGPU[i].append(0)
                    num_ltiles_List_AllGPU[i].append(0)
                    #
                    SkippedMatrixCount[k] = SkippedMatrixCount[k]+1

    '''
    dir_path + "/val_count_List."    + std::to_string(tensor_para_rank_) + ".bin"
    dir_path + "/num_gtiles_List."   + std::to_string(tensor_para_rank_) + ".bin"
    dir_path + "/num_mtiles_List."   + std::to_string(tensor_para_rank_) + ".bin"
    dir_path + "/num_ltiles_List."   + std::to_string(tensor_para_rank_) + ".bin"
    dir_path + "/SplitK_List."       + std::to_string(tensor_para_rank_) + ".bin"
    '''
    print("Saving NNZ_List.x.bin NumOffsets_List.x.bin SplitK_List.x.bin...")
    for i in range(nGPU):
        Val_List_FileName        = FT_ModelPath + "/val_count_List.{}.bin".format(i)
        Gtiles_List_FileName     = FT_ModelPath + "/num_gtiles_List.{}.bin".format(i)
        Mtiles_List_FileName     = FT_ModelPath + "/num_mtiles_List.{}.bin".format(i)
        Ltiles_List_FileName     = FT_ModelPath + "/num_ltiles_List.{}.bin".format(i)
        SplitK_List_FileName     = FT_ModelPath + "/SplitK_List.{}.bin".format(i)
        np.array(val_count_List_AllGPU[i]).astype(np.int32).tofile(Val_List_FileName)
        np.array(num_gtiles_List_AllGPU[i]).astype(np.int32).tofile(Gtiles_List_FileName)
        np.array(num_mtiles_List_AllGPU[i]).astype(np.int32).tofile(Mtiles_List_FileName)
        np.array(num_ltiles_List_AllGPU[i]).astype(np.int32).tofile(Ltiles_List_FileName)
        np.array(SplitK_List_AllGPU[i]).astype(np.int32).tofile(SplitK_List_FileName)
    
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    print(f"[INFO] Spend {run_time} (h:m:s) to convert the model: Phase-2")
    for i in range(4):
        print("MatMul"+str(i)+": "+str(ReplacedMatrixCount[i])+" Matrices replaced, "+str(SkippedMatrixCount[i])+" skipped.")
        
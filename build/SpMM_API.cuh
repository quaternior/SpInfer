/***************************************************************************
 * Copyright 2025 The SpInfer Authors. All rights reserved.
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
// Adapted from https://github.com/AlibabaResearch/flash-llm/blob/main/build/SpMM_API.cuh
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// template<typename TilingConfig, typename SparseKernelConfig>
// static void SpMM_SplitK_Kernel_Ex(cudaStream_t stream,
//                                   const half*  A,
//                                   const uint4* Compressed_A,
//                                   const int*   TileOffsets,
//                                   const half*  B,
//                                   half*        Reduction_Workspace,
//                                   const int    M_Global,
//                                   const int    N_Global,
//                                   const int    K_Global,
//                                   int          Split_K);
// /*
// half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output
// tensors
//                             2. Reduction_Workspace_Size = max( Split_K * M_Global * N_Global ) * sizeof(fp16)
// int Split_K:                Split K dimension into Split_K Parts
// */
// cudaError_t SpMM_SplitK_API(cudaStream_t stream,
//                             const half*  A,
//                             const uint4* Compressed_A,
//                             const int*   TileOffsets,
//                             const half*  B,
//                             half*        C,
//                             const int    M_Global,
//                             const int    N_Global,
//                             const int    K_Global,
//                             half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
//                             int          Split_K);

cudaError_t SpMM_SplitK_API_bitmap_v3(cudaStream_t stream,
                                    const half*  A,
                                    const half*  Compressed_A,
                                    const int*   TileOffsets,
                                    const int* TileOffsets_Median,
                                    const uint64_t* bitmap,
                                    const int* max_nnz_intile,
                                    const half*  B,
                                    half*        C,
                                    const int    M_Global,
                                    const int    N_Global,
                                    const int    K_Global,
                                    half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launchesSpMM_SplitK_Kernel_Ex_bitmap
                                    int          Split_K);
// // Generating Tiled-CSL format from dense format
// __host__ int InitSparseMatrixA_API(half* A_h, int M, int N, int K, uint32_t** Compressed_A, int** TileOffsets);
// // Generating Tiled-CSL format from dense format, without the optimization named "Ahead of Time Sparse Data Reordering"
// __host__ int InitSparseMatrixA_API_NoReorder(half*      A_h,
//                                              int        M,
//                                              int        N,
//                                              int        K,
//                                              uint32_t** Compressed_A,  // CPU PTR
//                                              int**      TileOffsets);       // CPU_PTR
// __host__ int InitSparseMatrixA_API_NoReorder_unstructured(half*      A_h,
//                                              int        M,
//                                              int        N,
//                                              int        K,
//                                              uint32_t** Compressed_A,  // CPU PTR
//                                              int**      TileOffsets);       // CPU_PTR
// Our sparsity_llm
__host__ int InitSparseMatrixA_bitmap(
                                        half* A_h,
                                        int M,  // 行数
                                        int K,  // 列数
                                        int tile_M,  // 8
                                        int tile_M_median,  // 16
                                        int tile_M_global,  // 64
                                        int tile_K,  // 8
                                        int tile_K_median,  // 64
                                        int tile_K_global,  // 64
                                        half** Compressed_Val,
                                        int** TileOffsets,
                                        int** TileOffsets_median,
                                        int** TileOffsets_global,
                                        uint64_t** bitmap,
                                        int& max_nnz_count);

// // Used by ft-tools
// extern "C" void GenSparseMatrixBinFile(char* DenseMatrixFileName,
//                                        int   M,
//                                        int   N,
//                                        int   K,
//                                        char* NZWeightsFileName,
//                                        char* TileOffsetsFileName,
//                                        char* OutputSizesFileName);
// Our sparsity_llm
extern "C" void Our_GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                            int   M,
                                            int   K,
                                            char* Compressed_ValFileName,
                                            char* bitmap_TileOffsets_globalFileName,
                                            char* bitmap_TileOffsets_medianFileName,
                                            char* bitmapFileName,
                                            char* max_nnz_intileFileName,
                                            char* OutputSizesFileName);
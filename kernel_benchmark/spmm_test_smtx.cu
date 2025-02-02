/***************************************************************************
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

#define USE_CUBLAS
#define USE_FLASH_LLM
//#define USE_SPUTNIK
// #define USE_CUSPARSE
// #define USE_SPARTA

#include "./spmm_test_utils.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>

#ifdef USE_FLASH_LLM
#include "SpMM_API.cuh"
#endif

#ifdef USE_SPUTNIK
#include "./sputnik_utils.h"
#include "sputnik/sputnik.h"
#endif

#ifdef USE_SPARTA
#include "sparTA.h"
#endif
// 定义读取 .smtx 文件的函数
void read_smtx_file(const char* file_path, int* num_rows, int* num_cols, int* nnz, int** indptr, int** indices) {
    // 打开文件
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    // 读取矩阵的形状和非零元素数量
    if (fscanf(file, "%d,%d,%d", num_rows, num_cols, nnz) != 3) {
        printf("Error reading matrix dimensions\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // 分配内存给 indptr 和 indices
    *indptr = (int*)malloc((*num_rows + 1) * sizeof(int));
    *indices = (int*)malloc(*nnz * sizeof(int));
    if (*indptr == NULL || *indices == NULL) {
        printf("Error allocating memory\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // 读取 indptr
    for (int i = 0; i <= *num_rows; i++) {
        if (fscanf(file, "%d", &(*indptr)[i]) != 1) {
            printf("Error reading indptr\n");
            fclose(file);
            free(*indptr);
            free(*indices);
            exit(EXIT_FAILURE);
        }
    }

    // 读取 indices
    for (int i = 0; i < *nnz; i++) {
        if (fscanf(file, "%d", &(*indices)[i]) != 1) {
            printf("Error reading indices\n");
            fclose(file);
            free(*indptr);
            free(*indices);
            exit(EXIT_FAILURE);
        }
    }

    // 关闭文件
    fclose(file);
}

__host__ void
init_host_matrices_B(half* b, half* b_trans,int K_GLOBAL, int N_GLOBAL)
{
    for (int i = 0; i < N_GLOBAL * K_GLOBAL; i++)
        //b[i] = __float2half_rn(1.0);
        b[i] = __float2half_rn(static_cast<float>((rand() % 5)) / 5 - 0.5f);
    for (int i = 0; i < K_GLOBAL; i++)
        for (int j = 0; j < N_GLOBAL; j++)
            b_trans[i * N_GLOBAL + j] =  b[i + j * K_GLOBAL];
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Wrong Inputs! Correct input format: ./spmm_test M K N Sparsity SplitK input_path\n");
        return -1;
    }

    int N_GLOBAL                    = atoi(argv[1]);
    int SPLIT_K                     = atoi(argv[2]);
    const char* input_path          = argv[3];  // 输入文件路径

    // Host memory
    half* A_h            = NULL;  // row major
    half* B_h            = NULL;  // col major
    half* B_Transposed_h = NULL;  // row major

    // Device memory
    half* A            = NULL;
    half* B            = NULL;
    half* B_Transposed = NULL;
    cublasStatus_t cublas_status;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int M_GLOBAL, K_GLOBAL, nnz;
    int* indptr = NULL;
    int* indices = NULL;
    read_smtx_file(input_path, &M_GLOBAL, &K_GLOBAL, &nnz, &indptr, &indices);

    // 为矩阵 A_h 和 B_h 分配内存
    A_h = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    B_Transposed_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    if (A_h == NULL || B_h == NULL || B_Transposed_h == NULL) {
        printf("Error in CPU Malloc!\n");
        exit(-1);
    }

    // 读取 smtx 文件
    int num_rows = M_GLOBAL;
    int num_cols = K_GLOBAL;
    // 使用读取的数据初始化 A_h
    for (int i = 0; i < num_rows; i++) {
        for (int j = indptr[i]; j < indptr[i + 1]; j++) {
            A_h[i * K_GLOBAL + indices[j]] = __float2half(1.0f);  // 假设非零元素为 1.0
        }
    }

    // 初始化 B 矩阵
    init_host_matrices_B(B_h, B_Transposed_h, K_GLOBAL, N_GLOBAL);

    // CUDA 内存分配
    cudaMalloc(reinterpret_cast<void**>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B_Transposed), sizeof(half) * N_GLOBAL * K_GLOBAL);
    checkLastCudaError(__LINE__);

    // 将数据从主机传输到设备
    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_Transposed, B_Transposed_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);

    //#ifdef USE_CUBLAS
    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching CuBlas...\n");
    half* D_cublas = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_cublas), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_cublas, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);

    // Tensor core not enabled
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    cudaDeviceSynchronize();
    int              m = M_GLOBAL, n = N_GLOBAL, k = K_GLOBAL;
    const float      alpha     = 1.0;
    const float      beta      = 0.0;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     A,
                                     CUDA_R_16F,
                                     k,
                                     B,
                                     CUDA_R_16F,
                                     k,
                                     &beta,
                                     D_cublas,
                                     CUDA_R_16F,
                                     m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m,
                     n,
                     k,
                     &alpha,
                     A,
                     CUDA_R_16F,
                     k,
                     B,
                     CUDA_R_16F,
                     k,
                     &beta,
                     D_cublas,
                     CUDA_R_16F,
                     m,
                     CUDA_R_32F,
                     CuBlasALG);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //
    float milliseconds_cublas = 0;
    cudaEventElapsedTime(&milliseconds_cublas, start, stop);
    milliseconds_cublas = milliseconds_cublas / BENCHMARK_ITERATION;
    float tflops_cublas =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_cublas / 1000.))
        / 1e12;
    // Tensor core enabled
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaDeviceSynchronize();
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     A,
                                     CUDA_R_16F,
                                     k,
                                     B,
                                     CUDA_R_16F,
                                     k,
                                     &beta,
                                     D_cublas,
                                     CUDA_R_16F,
                                     m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m,
                     n,
                     k,
                     &alpha,
                     A,
                     CUDA_R_16F,
                     k,
                     B,
                     CUDA_R_16F,
                     k,
                     &beta,
                     D_cublas,
                     CUDA_R_16F,
                     m,
                     CUDA_R_32F,
                     CuBlasALG);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //
    float milliseconds_cublas_tc = 0;
    cudaEventElapsedTime(&milliseconds_cublas_tc, start, stop);
    milliseconds_cublas_tc = milliseconds_cublas_tc / BENCHMARK_ITERATION;
    float tflops_cublas_tc = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)
                                                 / (milliseconds_cublas_tc / 1000.))
                             / 1e12;
    half* D_cublas_h = NULL;  // col major
    D_cublas_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas_h == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(D_cublas_h, D_cublas, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_cublas);
    /////////////////////////////////////////////////////////////////////////////////////////////////
//#endif
#ifdef USE_FLASH_LLM
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // split A_h to A_h_structured and A_h_unstructured
    auto A_h_structured = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    auto A_h_unstructured = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    std::vector<std::vector<std::tuple<int, int>>> metadata;
    std::vector<std::vector<std::tuple<int, int>>> metadata_reverse;
    std::vector<std::vector<std::tuple<int, int>>> metadata_unstructured;
    splitMatrix(A_h, M_GLOBAL, K_GLOBAL, A_h_structured, A_h_unstructured, metadata, metadata_unstructured);
    auto compressedMat = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL / 2);
    auto compressedMat_unstructured = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL / 2);
    compressSparseMat(A_h_structured, M_GLOBAL, K_GLOBAL, compressedMat, metadata);
    compressSparseMat(A_h_unstructured, M_GLOBAL, K_GLOBAL, compressedMat_unstructured, metadata_unstructured);
    // for (const auto& row : metadata) {
    //     for (const auto& entry : row) {
    //         std::cout << "(" << std::get<0>(entry) << ", " << std::get<1>(entry) << "), ";
    //     }
    //     std::cout << std::endl;
    // }
    auto bin_meta = metadataToBinary(metadata);
    int totalMetaElements = (M_GLOBAL/16) * (K_GLOBAL/2);
    // uint32_t* bin_meta_array = new uint32_t[totalMetaElements];
    auto bin_meta_array = (uint32_t*)malloc(sizeof(uint32_t) * totalMetaElements);
    binMetaToArray(bin_meta, bin_meta_array);
    // printMetaE(bin_meta_array, M_GLOBAL/16, K_GLOBAL/2, "MetaE"); 
    printMatrix(A_h, M_GLOBAL, K_GLOBAL, M_GLOBAL - 16, M_GLOBAL, K_GLOBAL - 16, K_GLOBAL, "A_h");
    printMatrix(A_h_structured,  M_GLOBAL, K_GLOBAL, M_GLOBAL - 16, M_GLOBAL, K_GLOBAL - 16, K_GLOBAL, "A_h_structured");
    printMatrix(A_h_unstructured,  M_GLOBAL, K_GLOBAL, M_GLOBAL - 16, M_GLOBAL, K_GLOBAL - 16, K_GLOBAL, "A_h_unstructured"); 
    printMatrix(compressedMat,  M_GLOBAL, K_GLOBAL/2, M_GLOBAL - 16, M_GLOBAL, K_GLOBAL/2 - 16, K_GLOBAL/2, "compressedMat"); 
    printMatrix(compressedMat_unstructured, M_GLOBAL, K_GLOBAL/2 , M_GLOBAL - 16, M_GLOBAL, K_GLOBAL/2 - 16, K_GLOBAL/2, "compressedMat_unstructured"); 
    uint32_t* NZWeights_CPU_unstructured  = NULL;
    int*      TileOffsets_CPU_unstructured = NULL;
    auto NumOffsets_unstructured = InitSparseMatrixA_API_NoReorder_unstructured(compressedMat_unstructured, M_GLOBAL, N_GLOBAL, K_GLOBAL/2, &NZWeights_CPU_unstructured, &TileOffsets_CPU_unstructured);
    auto NNZ_unstructured       = TileOffsets_CPU_unstructured[NumOffsets_unstructured - 1] * 4;  // VectorSize = 4
    printf("NumOffsets_unstructured: %d, NNZ_unstructured: %d\n", NumOffsets_unstructured, NNZ_unstructured);
    
    uint32_t* NZWeights_GPU_unstructured   = NULL;
    int*  TileOffsets_GPU_unstructured = NULL;
    cudaMalloc(&TileOffsets_GPU_unstructured, sizeof(int) * NumOffsets_unstructured);
    if (NNZ_unstructured == 0)
      NNZ_unstructured = 1;  // For 100% sparsity, NNZ = 0, malloc will return NULL
    cudaMalloc(&NZWeights_GPU_unstructured, sizeof(uint32_t) * NNZ_unstructured);
    if (TileOffsets_GPU_unstructured == NULL || NZWeights_GPU_unstructured == NULL) {
        printf("Error in malloc memory from device memory!\n");
        exit(-1);
    }
    cudaMemcpy(NZWeights_GPU_unstructured, NZWeights_CPU_unstructured, sizeof(uint32_t) * NNZ_unstructured, cudaMemcpyHostToDevice);
    cudaMemcpy(TileOffsets_GPU_unstructured, TileOffsets_CPU_unstructured, sizeof(int) * NumOffsets_unstructured, cudaMemcpyHostToDevice);
    free(TileOffsets_CPU_unstructured);
    free(NZWeights_CPU_unstructured);
    
    //
    half*  A_GPU_structured = NULL;
    cudaMalloc(&A_GPU_structured, sizeof(half) *  M_GLOBAL * K_GLOBAL / 2);
    cudaMemcpy(A_GPU_structured, compressedMat, sizeof(half) *  M_GLOBAL * K_GLOBAL / 2, cudaMemcpyHostToDevice);

    uint32_t*  MetaE = NULL;
    cudaMalloc(&MetaE, sizeof(uint32_t) *  totalMetaElements);
    cudaMemcpy(MetaE, bin_meta_array, sizeof(uint32_t) *  totalMetaElements, cudaMemcpyHostToDevice);

    //
    printf("Launching Flash-LLM without Ahead of Time Sparse Data Reordering...\n");
    int Split_K = SPLIT_K;
    // printf("Split_K = %d\n", Split_K);
    half* Reduction_Workspace1 = NULL;
    cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace1), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
    if (Reduction_Workspace1 == NULL) {
        printf("Error in cudaMalloc\n");
        exit(-1);
    }
    //
    half* D_SpMM1 = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_SpMM1), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_SpMM1 == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_SpMM1, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    // for (int i = 0; i < WARM_UP_ITERATION; i++)
    //     SpMM_SplitK_APIv1(0,
    //                     A_GPU_structured,
    //                     MetaE,
    //                     reinterpret_cast<uint4*>(NZWeights_GPU_unstructured),
    //                     TileOffsets_GPU_unstructured,
    //                     B,
    //                     D_SpMM1,
    //                     M_GLOBAL,
    //                     N_GLOBAL,
    //                     K_GLOBAL,
    //                     Reduction_Workspace1,
    //                     Split_K);
    // cudaEventRecord(start);
    // for (int i = 0; i < BENCHMARK_ITERATION; i++)
    //     SpMM_SplitK_APIv1(0,
    //                     A_GPU_structured,
    //                     MetaE,
    //                     reinterpret_cast<uint4*>(NZWeights_GPU_unstructured),
    //                     TileOffsets_GPU_unstructured,
    //                     B,
    //                     D_SpMM1,
    //                     M_GLOBAL,
    //                     N_GLOBAL,
    //                     K_GLOBAL,
    //                     Reduction_Workspace1,
    //                     Split_K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);
    //
    float milliseconds_SpMM1 = 0.0f;
    cudaEventElapsedTime(&milliseconds_SpMM1, start, stop);
    milliseconds_SpMM1 = milliseconds_SpMM1 / BENCHMARK_ITERATION;
    float tflops_SpMM1 =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM1 / 1000.))
        / 1e12;
    half* D_SpMM_h1 = NULL;  // col major
    D_SpMM_h1       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(D_SpMM_h1, D_SpMM1, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    printMatrixColumnMajor(D_SpMM_h1, M_GLOBAL, N_GLOBAL, 0, 32, 0, N_GLOBAL, "D_SpMM_h1"); 
    printMatrixColumnMajor(D_cublas_h, M_GLOBAL, N_GLOBAL, 0, 32, 0, N_GLOBAL, "D_cublas_h"); 
    
    compareMatrices(D_SpMM_h1, D_cublas_h, M_GLOBAL*N_GLOBAL);
    cudaFree(D_SpMM1);
    cudaFree(NZWeights_GPU_unstructured);
    cudaFree(TileOffsets_GPU_unstructured);
    cudaFree(Reduction_Workspace1);
    cudaFree(MetaE);
    cudaFree(A_GPU_structured);
    
    
    
    
    ////////////////////////////////////////////////////////////////////////////////////////////////
    uint32_t* NZWeights_CPU   = NULL;
    int*      TileOffsets_CPU = NULL;
    // int Split_K = SPLIT_K;
    half* D_SpMM2 = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_SpMM2), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_SpMM2 == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_SpMM2, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    auto NumOffsets = InitSparseMatrixA_API_NoReorder(A_h, M_GLOBAL, N_GLOBAL, K_GLOBAL, &NZWeights_CPU, &TileOffsets_CPU);
    auto NNZ        = TileOffsets_CPU[NumOffsets - 1] * 4;  // VectorSize = 4
    printf("NumOffsets: %d, NNZ: %d\n", NumOffsets, NNZ);
    //
    uint32_t* NZWeights_GPU   = NULL;
    int*  TileOffsets_GPU = NULL;
    cudaMalloc(&TileOffsets_GPU, sizeof(int) * NumOffsets);
    if (NNZ == 0)
        NNZ = 1;  // For 100% sparsity, NNZ = 0, malloc will return NULL
    cudaMalloc(&NZWeights_GPU, sizeof(uint32_t) * NNZ);
    if (TileOffsets_GPU == NULL || NZWeights_GPU == NULL) {
        printf("Error in malloc memory from device memory!\n");
        exit(-1);
    }
    cudaMemcpy(NZWeights_GPU, NZWeights_CPU, sizeof(uint32_t) * NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(TileOffsets_GPU, TileOffsets_CPU, sizeof(int) * NumOffsets, cudaMemcpyHostToDevice);
    free(TileOffsets_CPU);
    free(NZWeights_CPU);
    // printf("Done! Compressed A matrix for GPU kernel: MM_Sparse_TC.\n");
    //
    printf("Launching Flash-LLM without Ahead of Time Sparse Data Reordering...\n");
    Split_K = SPLIT_K;
    // printf("Split_K = %d\n", Split_K);
    half* Reduction_Workspace = NULL;
    cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
    if (Reduction_Workspace == NULL) {
        printf("Error in cudaMalloc\n");
        exit(-1);
    }
    //
    for (int i = 0; i < WARM_UP_ITERATION; i++)
        SpMM_SplitK_API(0,
                        A,
                        reinterpret_cast<uint4*>(NZWeights_GPU),
                        TileOffsets_GPU,
                        B,
                        D_SpMM2,
                        M_GLOBAL,
                        N_GLOBAL,
                        K_GLOBAL,
                        Reduction_Workspace,
                        Split_K);
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        SpMM_SplitK_API(0,
                        A,
                        reinterpret_cast<uint4*>(NZWeights_GPU),
                        TileOffsets_GPU,
                        B,
                        D_SpMM2,
                        M_GLOBAL,
                        N_GLOBAL,
                        K_GLOBAL,
                        Reduction_Workspace,
                        Split_K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);
    //
    float milliseconds_SpMM2 = 0.0f;
    cudaEventElapsedTime(&milliseconds_SpMM2, start, stop);
    milliseconds_SpMM2 = milliseconds_SpMM2 / BENCHMARK_ITERATION;
    float tflops_SpMM2 =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM2 / 1000.))
        / 1e12;
    half* D_SpMM_h2 = NULL;  // col major
    D_SpMM_h2       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(D_SpMM_h2, D_SpMM2, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_SpMM2);
    cudaFree(NZWeights_GPU);
    cudaFree(TileOffsets_GPU);
    cudaFree(Reduction_Workspace);
    /////////////////////////////////////////////////////////////////////////////////////////////////
#endif

// #ifdef USE_FLASH_LLM
    double totalError_SpMM2 = ComputeTotalError(D_cublas_h, D_SpMM_h2, M_GLOBAL, N_GLOBAL);
    double totalError_SpMM1 = ComputeTotalError(D_cublas_h, D_SpMM_h1, M_GLOBAL, N_GLOBAL);
// PrintMismatch("MySpMM", 10, 0.5, D_cublas_h, D_SpMM_h, M_GLOBAL, N_GLOBAL);
    free(D_SpMM_h2);
    free(D_SpMM_h1);
    PrintPerformance("FlashLLM_v1", milliseconds_SpMM2, tflops_SpMM2, totalError_SpMM2);
    PrintPerformance("FlashLLM_vuns", milliseconds_SpMM1, tflops_SpMM1, totalError_SpMM1);
    PrintPerformance("CuBlas_TC", milliseconds_cublas_tc, tflops_cublas_tc, 0.0);
    // PrintPerformance("FlashLLM_v2", milliseconds_SpMM, tflops_SpMM, totalError_SpMM);
// #endif

    free(D_cublas_h);
    free(A_h);
    free(A_h_structured);
    free(A_h_unstructured);
    free(compressedMat);
    free(compressedMat_unstructured);
    free(bin_meta_array);
    free(B_h);
    free(B_Transposed_h);
    cudaFree(A);
    cudaFree(B);
    cudaFree(B_Transposed);

    return 0;
}

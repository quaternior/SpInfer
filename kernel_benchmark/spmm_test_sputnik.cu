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



#include "./spmm_test_utils.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include "./sputnik_utils.h"
#include "sputnik/sputnik.h"

int main(int argc, char** argv)
{
    if (argc != 6) {
        printf("Wrong Inputs! Correct input format: ./spmm_test M K N Sparsity SplitK\n");
        return;
    }
    int M_GLOBAL                    = atoi(argv[1]);
    int K_GLOBAL                    = atoi(argv[2]);
    int N_GLOBAL                    = atoi(argv[3]);
    int MATRIX_A_PRUNING_PERCENTAGE = atoi(argv[4]);
    int SPLIT_K                     = atoi(argv[5]);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Host memory
    half* A_h            = NULL;  // row major
    half* B_h            = NULL;  // col major
    half* B_Transposed_h = NULL;  // row major
    // Device memory
    half* A            = NULL;
    half* B            = NULL;
    half* B_Transposed = NULL;
    //
    A_h            = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h            = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    B_Transposed_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    if (A_h == NULL || B_h == NULL || B_Transposed_h == NULL) {
        printf("Error in CPU Malloc!\n");
        exit(-1);
    }
    cudaMalloc(reinterpret_cast<void**>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B_Transposed), sizeof(half) * N_GLOBAL * K_GLOBAL);
    checkLastCudaError(__LINE__);
    if (A == NULL || B == NULL || B_Transposed == NULL) {
        printf("Error in cudaMalloc!\n");
        exit(-1);
    }
    init_host_matrices(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE);
    for (int i = 0; i < K_GLOBAL; i++)
        for (int j = 0; j < N_GLOBAL; j++)
            B_Transposed_h[i * N_GLOBAL + j] = B_h[i + j * K_GLOBAL];
    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_Transposed, B_Transposed_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
  

    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching Sputnik...\n");
    half* D_Sputnik = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_Sputnik), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_Sputnik == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_Sputnik, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    //
    float* A_float_h = NULL;
    A_float_h        = (float*)malloc(sizeof(float) * M_GLOBAL * K_GLOBAL);
    for (int i = 0; i < M_GLOBAL * K_GLOBAL; i++)
        A_float_h[i] = __half2float(A_h[i]);
    sputnik_utils::SparseMatrix            sparse_matrix(M_GLOBAL, K_GLOBAL, A_float_h, sputnik_utils::IDENTITY, 4);
    sputnik_utils::CudaSparseMatrix<half2> sparse_matrix_gpu(sparse_matrix);

    for (int i = 0; i < WARM_UP_ITERATION; i++)
        CUDA_CALL(sputnik::CudaSpmm(M_GLOBAL,
                                    K_GLOBAL,
                                    N_GLOBAL,
                                    sparse_matrix_gpu.NumElementsWithPadding(),
                                    sparse_matrix_gpu.RowIndices(),
                                    sparse_matrix_gpu.Values(),
                                    sparse_matrix_gpu.RowOffsets(),
                                    sparse_matrix_gpu.ColumnIndices(),
                                    reinterpret_cast<half2*>(B_Transposed),
                                    reinterpret_cast<half2*>(D_Sputnik),
                                    0));
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        CUDA_CALL(sputnik::CudaSpmm(M_GLOBAL,
                                    K_GLOBAL,
                                    N_GLOBAL,
                                    sparse_matrix_gpu.NumElementsWithPadding(),
                                    sparse_matrix_gpu.RowIndices(),
                                    sparse_matrix_gpu.Values(),
                                    sparse_matrix_gpu.RowOffsets(),
                                    sparse_matrix_gpu.ColumnIndices(),
                                    reinterpret_cast<half2*>(B_Transposed),
                                    reinterpret_cast<half2*>(D_Sputnik),
                                    0));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_Sputnik = 0;
    cudaEventElapsedTime(&milliseconds_Sputnik, start, stop);
    milliseconds_Sputnik = milliseconds_Sputnik / BENCHMARK_ITERATION;
    float tflops_Sputnik =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_Sputnik / 1000.))
        / 1e12;
    cudaFree(D_Sputnik);

    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("******************************************Problem Size******************************************\n");
    printf("M: %d N: %d K: %d Pruning Rate: %d SplitK: %d\n",
           M_GLOBAL,
           N_GLOBAL,
           K_GLOBAL,
           MATRIX_A_PRUNING_PERCENTAGE,
           SPLIT_K);
// printf("******************************************Performance*******************************************\n");
    PrintPerformance("Sputnik", milliseconds_Sputnik, tflops_Sputnik, 0.0);

    SaveSputnikPerformanceData("sputnik_performance_results.csv",
        M_GLOBAL, K_GLOBAL, N_GLOBAL, 
        SPLIT_K, MATRIX_A_PRUNING_PERCENTAGE,
        milliseconds_Sputnik, tflops_Sputnik);
    free(A_h);
    free(B_h);
    free(B_Transposed_h);
    cudaFree(A);
    cudaFree(B);
    cudaFree(B_Transposed);
    return 0;
}

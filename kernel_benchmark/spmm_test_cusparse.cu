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


#define USE_CUSPARSE

#include "./spmm_test_utils.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>
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
    //
    // printf("M: %d N: %d K: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    //
    cublasStatus_t cublas_status;
    // cusparseStatus_t  cusparse_status;
    // cudaError_t       cuda_error;
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
    //
    init_host_matrices(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE);
    for (int i = 0; i < K_GLOBAL; i++)
        for (int j = 0; j < N_GLOBAL; j++)
            B_Transposed_h[i * N_GLOBAL + j] = B_h[i + j * K_GLOBAL];
    //
    // printf("Preparing dense data for GPU...\n");
    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_Transposed, B_Transposed_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
  
#ifdef USE_CUSPARSE
    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching CuSparse_ColMajor...\n");
    half* D_CuSparse = NULL;
    cudaMalloc(&D_CuSparse, N_GLOBAL * M_GLOBAL * sizeof(half));
    if (D_CuSparse == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_CuSparse, 0.0f, N_GLOBAL * M_GLOBAL * sizeof(half));
    //
    cusparseHandle_t sp_handle = 0;
    cusparseCreate(&sp_handle);
    cusparseSetStream(sp_handle, 0);
    cusparseSpMatDescr_t SpMatA;
    cusparseDnMatDescr_t DnMatA, DnMatB, DnMatC;
    // Create Dense Matrix
    CHECK_CUSPARSE(cusparseCreateDnMat(&DnMatA,
                                       M_GLOBAL,
                                       K_GLOBAL,
                                       K_GLOBAL,
                                       A,
                                       CUDA_R_16F,
                                       CUSPARSE_ORDER_ROW))  // Very critical!!! Weight Matrix must be row major,
                                                             // otherwise causing significant performance problems
    CHECK_CUSPARSE(cusparseCreateDnMat(&DnMatB, K_GLOBAL, N_GLOBAL, K_GLOBAL, B, CUDA_R_16F, CUSPARSE_ORDER_COL))
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&DnMatC, M_GLOBAL, N_GLOBAL, M_GLOBAL, D_CuSparse, CUDA_R_16F, CUSPARSE_ORDER_COL))
    // Create Sparse Matrix in CSR format
    int* csrRowPtr;
    cudaMalloc(&csrRowPtr, sizeof(int) * (M_GLOBAL + 1));
    CHECK_CUSPARSE(cusparseCreateCsr(&SpMatA,
                                     M_GLOBAL,
                                     K_GLOBAL,
                                     0,
                                     csrRowPtr,
                                     NULL,
                                     NULL,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_16F))
    // execute Sparse to Dense conversion
    void*  Buffer     = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(
        cusparseDenseToSparse_bufferSize(sp_handle, DnMatA, SpMatA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize))
    cudaMalloc(&Buffer, bufferSize);
    CHECK_CUSPARSE(
        cusparseDenseToSparse_analysis(sp_handle, DnMatA, SpMatA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, Buffer))
    //
    int64_t numRowTMP, numColTMP, NNZ_1;
    CHECK_CUSPARSE(cusparseSpMatGetSize(SpMatA, &numRowTMP, &numColTMP, &NNZ_1))
    //
    int*  csrColInd;
    half* csrVal;
    cudaMalloc(&csrColInd, NNZ_1 * sizeof(int));
    cudaMalloc(&csrVal, NNZ_1 * sizeof(half));
    //
    CHECK_CUSPARSE(cusparseCsrSetPointers(SpMatA, csrRowPtr, csrColInd, csrVal))
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(sp_handle, DnMatA, SpMatA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, Buffer))
    //
    cusparseSpMMAlg_t CuSparse_Algorithm;
    CuSparse_Algorithm = CUSPARSE_SPMM_ALG_DEFAULT;
    CuSparse_Algorithm =
        CUSPARSE_SPMM_CSR_ALG1;  // csrmm_kernel faster: Provide the best performance with column-major layout
    // CuSparse_Algorithm = CUSPARSE_SPMM_CSR_ALG2;    // csrmm_v2_kernel: Provide the best performance with row-major
    // layout!!! How about try row major of B&C? CuSparse_Algorithm = CUSPARSE_SPMM_CSR_ALG3;
    //
    // printf("CuSparse Algorithm: %d \n", CuSparse_Algorithm);
    //
    const float alpha_float = 1.0;
    const float beta_float  = 0.0;
    //
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_float,
                                           SpMatA,
                                           DnMatB,
                                           &beta_float,
                                           DnMatC,
                                           CUDA_R_32F,
                                           CuSparse_Algorithm,
                                           &bufferSize))
    cudaFree(Buffer);
    cudaMalloc(&Buffer, bufferSize);
    for (int i = 0; i < CUSPARSE_ITERATION; i++)
        CHECK_CUSPARSE(cusparseSpMM(sp_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_float,
                                    SpMatA,
                                    DnMatB,
                                    &beta_float,
                                    DnMatC,
                                    CUDA_R_32F,
                                    CuSparse_Algorithm,
                                    Buffer))
    cudaEventRecord(start);
    for (int i = 0; i < CUSPARSE_ITERATION; i++)
        CHECK_CUSPARSE(cusparseSpMM(sp_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_float,
                                    SpMatA,
                                    DnMatB,
                                    &beta_float,
                                    DnMatC,
                                    CUDA_R_32F,
                                    CuSparse_Algorithm,
                                    Buffer))
    cudaEventRecord(stop);
    //
    float milliseconds_CuSparse_ColMajor = 0.0f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_CuSparse_ColMajor, start, stop);
    milliseconds_CuSparse_ColMajor = milliseconds_CuSparse_ColMajor / CUSPARSE_ITERATION;
    float tflops_CuSparse_ColMajor = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)
                                                         / (milliseconds_CuSparse_ColMajor / 1000.))
                                     / 1e12;
    //
    half* D_CuSparse_h;
    D_CuSparse_h = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_CuSparse_h == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(D_CuSparse_h, D_CuSparse, N_GLOBAL * M_GLOBAL * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(D_CuSparse);
    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);
    cudaFree(Buffer);
    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching CuSparse_RowMajor...\n");
    half* D_CuSparse_2;
    cudaMalloc(&D_CuSparse_2, N_GLOBAL * M_GLOBAL * sizeof(half));
    if (D_CuSparse_2 == NULL) {
        printf("Error in Test_SpMM_v2.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_CuSparse_2, 0.0f, N_GLOBAL * M_GLOBAL * sizeof(half));
    //
    cusparseHandle_t sp_handle_2 = 0;
    cusparseCreate(&sp_handle_2);
    cusparseSpMatDescr_t SpMatA_2;
    cusparseDnMatDescr_t DnMatA_2, DnMatB_2, DnMatC_2;
    // Create Dense Matrix
    CHECK_CUSPARSE(cusparseCreateDnMat(&DnMatA_2,
                                       M_GLOBAL,
                                       K_GLOBAL,
                                       K_GLOBAL,
                                       A,
                                       CUDA_R_16F,
                                       CUSPARSE_ORDER_ROW))  // Very critical!!! Weight Matrix must be row major,
                                                             // otherwise causing significant performance problems

    CHECK_CUSPARSE(
        cusparseCreateDnMat(&DnMatB_2, K_GLOBAL, N_GLOBAL, N_GLOBAL, B_Transposed, CUDA_R_16F, CUSPARSE_ORDER_ROW))
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&DnMatC_2, M_GLOBAL, N_GLOBAL, N_GLOBAL, D_CuSparse_2, CUDA_R_16F, CUSPARSE_ORDER_ROW))
    // Create Sparse Matrix in CSR format
    int* csrRowPtr_2;
    cudaMalloc(&csrRowPtr_2, sizeof(int) * (M_GLOBAL + 1));
    CHECK_CUSPARSE(cusparseCreateCsr(&SpMatA_2,
                                     M_GLOBAL,
                                     K_GLOBAL,
                                     0,
                                     csrRowPtr_2,
                                     NULL,
                                     NULL,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_16F))
    // execute Sparse to Dense conversion
    void*  Buffer_2     = NULL;
    size_t bufferSize_2 = 0;
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
        sp_handle_2, DnMatA_2, SpMatA_2, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize_2))
    cudaMalloc(&Buffer_2, bufferSize_2);
    CHECK_CUSPARSE(
        cusparseDenseToSparse_analysis(sp_handle_2, DnMatA_2, SpMatA_2, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, Buffer_2))
    //
    int64_t numRowTMP_2, numColTMP_2, NNZ_2;
    CHECK_CUSPARSE(cusparseSpMatGetSize(SpMatA_2, &numRowTMP_2, &numColTMP_2, &NNZ_2))
    //
    int*  csrColInd_2;
    half* csrVal_2;
    cudaMalloc(&csrColInd_2, NNZ_2 * sizeof(int));
    cudaMalloc(&csrVal_2, NNZ_2 * sizeof(half));
    //
    CHECK_CUSPARSE(cusparseCsrSetPointers(SpMatA_2, csrRowPtr_2, csrColInd_2, csrVal_2))
    CHECK_CUSPARSE(
        cusparseDenseToSparse_convert(sp_handle_2, DnMatA_2, SpMatA_2, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, Buffer_2))
    //
    cusparseSpMMAlg_t CuSparse_Algorithm_2;
    CuSparse_Algorithm_2 = CUSPARSE_SPMM_ALG_DEFAULT;
    CuSparse_Algorithm_2 =
        CUSPARSE_SPMM_CSR_ALG1;  // csrmm_kernel faster: Provide the best performance with column-major layout
    CuSparse_Algorithm_2 = CUSPARSE_SPMM_CSR_ALG2;  // csrmm_v2_kernel: Provide the best performance with row-major
                                                    // layout!!! How about try row major of B&C?
    // CuSparse_Algorithm_2 = CUSPARSE_SPMM_CSR_ALG3;
    // printf("CuSparse Algorithm: %d \n", CuSparse_Algorithm_2);
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle_2,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_float,
                                           SpMatA_2,
                                           DnMatB_2,
                                           &beta_float,
                                           DnMatC_2,
                                           CUDA_R_32F,
                                           CuSparse_Algorithm_2,
                                           &bufferSize_2))
    cudaFree(Buffer_2);
    cudaMalloc(&Buffer_2, bufferSize_2);
    //
    for (int i = 0; i < CUSPARSE_ITERATION; i++)
        CHECK_CUSPARSE(cusparseSpMM(sp_handle_2,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_float,
                                    SpMatA_2,
                                    DnMatB_2,
                                    &beta_float,
                                    DnMatC_2,
                                    CUDA_R_32F,
                                    CuSparse_Algorithm_2,
                                    Buffer_2))
    cudaEventRecord(start);
    for (int i = 0; i < CUSPARSE_ITERATION; i++)
        CHECK_CUSPARSE(cusparseSpMM(sp_handle_2,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_float,
                                    SpMatA_2,
                                    DnMatB_2,
                                    &beta_float,
                                    DnMatC_2,
                                    CUDA_R_32F,
                                    CuSparse_Algorithm_2,
                                    Buffer_2))
    cudaEventRecord(stop);
    //
    float milliseconds_CuSparse_RowMajor;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_CuSparse_RowMajor, start, stop);
    milliseconds_CuSparse_RowMajor = milliseconds_CuSparse_RowMajor / CUSPARSE_ITERATION;
    float tflops_CuSparse_RowMajor = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)
                                                         / (milliseconds_CuSparse_RowMajor / 1000.))
                                     / 1e12;
    // transpose result to col-major
    half* D_CuSparse_h_2_row_major;
    half* D_CuSparse_h_2;
    D_CuSparse_h_2           = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    D_CuSparse_h_2_row_major = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_CuSparse_h_2 == NULL || D_CuSparse_h_2_row_major == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(D_CuSparse_h_2_row_major,
               D_CuSparse_2,
               N_GLOBAL * M_GLOBAL * sizeof(half),
               cudaMemcpyDeviceToHost);  // row major
    for (int i = 0; i < N_GLOBAL; i++)
        for (int j = 0; j < M_GLOBAL; j++)
            D_CuSparse_h_2[i * M_GLOBAL + j] = D_CuSparse_h_2_row_major[i + j * N_GLOBAL];
    free(D_CuSparse_h_2_row_major);
    cudaFree(D_CuSparse_2);
    cudaFree(csrRowPtr_2);
    cudaFree(csrColInd_2);
    cudaFree(csrVal_2);
    cudaFree(Buffer_2);
    /////////////////////////////////////////////////////////////////////////////////////////////////
#endif



    printf("******************************************Problem Size******************************************\n");
    printf("M: %d N: %d K: %d Pruning Rate: %d SplitK: %d\n",
           M_GLOBAL,
           N_GLOBAL,
           K_GLOBAL,
           MATRIX_A_PRUNING_PERCENTAGE,
           SPLIT_K);
// printf("******************************************Performance*******************************************\n");

#ifdef USE_CUSPARSE
    PrintPerformance("CuSparse_C", milliseconds_CuSparse_ColMajor, tflops_CuSparse_ColMajor, 0.0);
    PrintPerformance("CuSparse_R", milliseconds_CuSparse_RowMajor, tflops_CuSparse_RowMajor, 0.0);
#endif

    free(A_h);
    free(B_h);
    free(B_Transposed_h);
    cudaFree(A);
    cudaFree(B);
    cudaFree(B_Transposed);
    return 0;
}

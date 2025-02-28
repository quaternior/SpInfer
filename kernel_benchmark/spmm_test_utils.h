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
#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <iomanip>
#include <algorithm> 
#include <fstream>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

// Performance Benchmark
#define WARM_UP_ITERATION 0
#define BENCHMARK_ITERATION 1000
#ifdef USE_CUSPARSE
#define CUSPARSE_ITERATION 10
#endif


void checkCublasError(cublasStatus_t status, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error at line %d, Error Code: %d\n", line, status);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUSPARSE(func)                                                                                           \
    {                                                                                                                  \
        cusparseStatus_t status = (func);                                                                              \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                                       \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",                                             \
                   __LINE__,                                                                                           \
                   cusparseGetErrorString(status),                                                                     \
                   status);                                                                                            \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }
#define CHECK_CUDA(func)                                                                                               \
    {                                                                                                                  \
        cudaError_t status = (func);                                                                                   \
        if (status != cudaSuccess) {                                                                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status);  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }


#define CUDA_CALL(code)                                                                                                \
    do {                                                                                                               \
        cudaError_t status = code;                                                                                     \
        std::string err    = cudaGetErrorString(status);                                                               \
        CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err;                                                        \
    } while (0)

void checkCudaError(cudaError_t error, int line)
{
    if (error != cudaSuccess) {
        printf("Cuda Error at line %d, Error Code: %d\n", line, error);
        exit(EXIT_FAILURE);
    }
}

void checkLastCudaError(int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Last Cuda Error Detected at line: %d, Error: %s.\n", line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

__host__ void
init_host_matrices(half* a, half* b, int M_GLOBAL, int K_GLOBAL, int N_GLOBAL, int MATRIX_A_PRUNING_PERCENTAGE)
{
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            int r = rand() % 100;
            if (r >= MATRIX_A_PRUNING_PERCENTAGE)
                a[j + i * K_GLOBAL] = __float2half_rn(static_cast<float>((rand() % 5)) / 5 - 0.5f);
            else
                a[j + i * K_GLOBAL] = __float2half_rn(0.0f);
        }
    }
    for (int i = 0; i < N_GLOBAL * K_GLOBAL; i++)
        b[i] = __float2half_rn(static_cast<float>((rand() % 5)) / 5 - 0.5f);
}

double ComputeTotalError(half* CuBlas, half* Other, int m, int n)
{
    double totalError = 0.0;
    for (int i = 0; i < m * n; i++)
        totalError += fabs(__half2float(CuBlas[i]) - __half2float(Other[i]));
    return totalError;
}

void PrintMismatch(const char* KernelName,
                   int         MaxNumMismatch,
                   float       ErrorThreshold,
                   half*       CuBlas,
                   half*       Other,
                   int         M_GLOBAL,
                   int         N_GLOBAL)
{
    printf("First 10 Mismatches between Cublas and %s:\n", KernelName);
    int count = 0;
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < N_GLOBAL; j++) {
            if (fabs(__half2float(CuBlas[i + j * M_GLOBAL]) - __half2float(Other[i + j * M_GLOBAL])) > ErrorThreshold) {
                count++;
                printf("(%d,%d) CuBlas=%f %s=%f\n",
                       i,
                       j,
                       __half2float(CuBlas[i + j * M_GLOBAL]),
                       KernelName,
                       __half2float(Other[i + j * M_GLOBAL]));
            }
            if (count == MaxNumMismatch)
                break;
        }
        if (count == MaxNumMismatch)
            break;
    }
}
void PrintMismatchV1(const char* KernelName,
                   int         MaxNumMismatch,
                   float       ErrorThreshold,
                   half*       CuBlas,
                   half*       Other,
                   int         M_GLOBAL,
                   int         N_GLOBAL)
{
    printf("First 10 Mismatches between Cublas and %s:\n", KernelName);
    int count = 0;
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < N_GLOBAL; j++) {
            if (fabs(__half2float(CuBlas[i*N_GLOBAL + j]) - __half2float(Other[i*N_GLOBAL + j])) > ErrorThreshold) {
                count++;
                printf("(%d,%d) CuBlas=%f %s=%f\n",
                       i,
                       j,
                       __half2float(CuBlas[i*N_GLOBAL + j]),
                       KernelName,
                       __half2float(Other[i*N_GLOBAL + j]));
            }
            if (count == MaxNumMismatch)
                break;
        }
        if (count == MaxNumMismatch)
            break;
    }
}

void PrintMatrix(const char* MatrixName,
                 half*       Matrix,
                 int         M_GLOBAL,
                 int         N_GLOBAL,
                 int         MaxRows,
                 int         MaxCols,
                 int         start_row,
                 int         start_col)
{
    printf("Printing Matrix %s (showing up to %d rows and %d columns, starting from row %d and column %d):\n", 
           MatrixName, MaxRows, MaxCols, start_row, start_col);

    // Determine the number of rows and columns to print
    int rows_to_print = std::min(MaxRows, M_GLOBAL - start_row);
    int cols_to_print = std::min(MaxCols, N_GLOBAL - start_col);
    
    // Print row indices
    printf("     ");
    for (int j = start_col; j < start_col + cols_to_print; j++) {
        printf("%8d ", j);
    }
    printf("\n");

    for (int i = start_row; i < start_row + rows_to_print; i++) {
        // Print row index
        printf("%3d: ", i);
        for (int j = start_col; j < start_col + cols_to_print; j++) {
            printf("%8.4f ", __half2float(Matrix[i*N_GLOBAL + j]));
        }
        if (start_col + cols_to_print < N_GLOBAL) {
            printf("...");  // Indicate there are more columns
        }
        printf("\n");
    }
    
    if (start_row + rows_to_print < M_GLOBAL) {
        printf("...\n");  // Indicate there are more rows
    }
    
    printf("\n");
}
void PrintMatrixVec(const char* MatrixName,
                 half*       Matrix,
                 int         M_GLOBAL,
                 int         N_GLOBAL,
                 int         MaxRows,
                 int         MaxCols)
{
    printf("Printing Matrix %s (showing up to %d rows and %d columns):\n", MatrixName, MaxRows, MaxCols);
    
    // Determine the number of rows and columns to print
    int rows_to_print = (MaxRows < M_GLOBAL) ? MaxRows : M_GLOBAL;
    int cols_to_print = (MaxCols < N_GLOBAL) ? MaxCols : N_GLOBAL;
    
    for (int i = 0; i < rows_to_print; i++) {
        for (int j = 0; j < cols_to_print; j++) {
            printf("%8.4f ", __half2float(Matrix[i + j*M_GLOBAL]));
        }
        if (cols_to_print < N_GLOBAL) {
            printf("...");  // Indicate there are more columns
        }
        printf("\n");
    }
    
    if (rows_to_print < M_GLOBAL) {
        printf("...\n");  // Indicate there are more rows
    }
    
    printf("\n");
}

void PrintPerformance(const char* KernelName, float milliseconds, float tflops, double error)
{
    printf("%-10s \t -> \t\t Time/ms: %5.3f \t Performance/TFLOPs: %4.2f \t TotalError: %.2lf\n",
           KernelName,
           milliseconds,
           tflops,
           error);
}

void SavePerformanceData(const char* filename, int M, int K, int N, int SplitK, int Sparsity, 
                        float duration_cublas_tc, float tflops_cublas_tc,
                        float duration_SpMM2, float tflops_SpMM2,
                        float duration_SpMM_bitmapv3, float tflops_SpMM_bitmapv3) {
    FILE* fp;
    // Try to open file to check if it exists
    fp = fopen(filename, "r");
    bool fileExists = (fp != NULL);
    if (fp) fclose(fp);
    
    // Open file in append mode
    fp = fopen(filename, "a");
    if (!fp) {
        printf("Error opening file for writing!\n");
        return;
    }

    // Write header if file is new
    if (!fileExists) {
        fprintf(fp, "M,K,N,SplitK,Sparsity,Kernel,Duration(ns),TFLOPS\n");
    }

    // Convert milliseconds to nanoseconds
    float duration_cublas_tc_ns = duration_cublas_tc * 1000000;
    float duration_SpMM2_ns = duration_SpMM2 * 1000000;
    float duration_SpMM_bitmapv3_ns = duration_SpMM_bitmapv3 * 1000000;

    // Write data for each kernel
    fprintf(fp, "%d,%d,%d,%d,%d,%s,%.1f,%.5f\n", 
            M, K, N, SplitK, Sparsity, "SpInfer", 
            duration_SpMM_bitmapv3_ns, tflops_SpMM_bitmapv3);
    
    fprintf(fp, "%d,%d,%d,%d,%d,%s,%.1f,%.5f\n", 
            M, K, N, SplitK, Sparsity, "cuBLAS_TC", 
            duration_cublas_tc_ns, tflops_cublas_tc);
    
    fprintf(fp, "%d,%d,%d,%d,%d,%s,%.1f,%.5f\n", 
            M, K, N, SplitK, Sparsity, "Flash-LLM", 
            duration_SpMM2_ns, tflops_SpMM2);

    fclose(fp);
}

void SaveCuSparsePerformanceData(const char* filename, int M, int K, int N, int SplitK, int Sparsity, 
                                float duration_CuSparse_ColMajor, float tflops_CuSparse_ColMajor) {
    FILE* fp;
    // Try to open file to check if it exists
    fp = fopen(filename, "r");
    bool fileExists = (fp != NULL);
    if (fp) fclose(fp);
    
    // Open file in append mode
    fp = fopen(filename, "a");
    if (!fp) {
        printf("Error opening file for writing!\n");
        return;
    }

    // Write header if file is new
    if (!fileExists) {
        fprintf(fp, "M,K,N,SplitK,Sparsity,Kernel,Duration(ns),TFLOPS\n");
    }

    // Select the better performance between CuSparse Row and Col Major
    float cusparse_duration_ns, cusparse_tflops;
    cusparse_duration_ns = duration_CuSparse_ColMajor * 1000000; // convert to nanoseconds
    cusparse_tflops = tflops_CuSparse_ColMajor;

    // Write data for cuSPARSE
    fprintf(fp, "%d,%d,%d,%d,%d,%s,%.1f,%.5f\n", 
            M, K, N, SplitK, Sparsity, "cuSPARSE", 
            cusparse_duration_ns, cusparse_tflops);

    fclose(fp);
}
void SaveSputnikPerformanceData(const char* filename, int M, int K, int N, int SplitK, int Sparsity, 
                               float duration_Sputnik, float tflops_Sputnik) {
    FILE* fp;
    // Try to open file to check if it exists
    fp = fopen(filename, "r");
    bool fileExists = (fp != NULL);
    if (fp) fclose(fp);
    
    // Open file in append mode
    fp = fopen(filename, "a");
    if (!fp) {
        printf("Error opening file for writing!\n");
        return;
    }

    // Write header if file is new
    if (!fileExists) {
        fprintf(fp, "M,K,N,SplitK,Sparsity,Kernel,Duration(ns),TFLOPS\n");
    }

    // Convert milliseconds to nanoseconds
    float sputnik_duration_ns = duration_Sputnik * 1000000;

    // Write data for Sputnik
    fprintf(fp, "%d,%d,%d,%d,%d,%s,%.1f,%.5f\n", 
            M, K, N, SplitK, Sparsity, "Sputnik", 
            sputnik_duration_ns, tflops_Sputnik);

    fclose(fp);
}
void SaveSparTAPerformanceData(const char* filename, int M, int K, int N, int SplitK, int Sparsity, 
                              float duration_sparTA, float tflops_sparTA) {
    FILE* fp;
    // Try to open file to check if it exists
    fp = fopen(filename, "r");
    bool fileExists = (fp != NULL);
    if (fp) fclose(fp);
    
    // Open file in append mode
    fp = fopen(filename, "a");
    if (!fp) {
        printf("Error opening file for writing!\n");
        return;
    }

    // Write header if file is new
    if (!fileExists) {
        fprintf(fp, "M,K,N,SplitK,Sparsity,Kernel,Duration(ns),TFLOPS\n");
    }

    // Convert milliseconds to nanoseconds
    float sparta_duration_ns = duration_sparTA * 1000000;

    // Write data for sparTA
    fprintf(fp, "%d,%d,%d,%d,%d,%s,%.1f,%.5f\n", 
            M, K, N, SplitK, Sparsity, "SparTA", 
            sparta_duration_ns, tflops_sparTA);

    fclose(fp);
}
std::vector<int> findRemainingValues(int first, int second) {
    std::vector<int> allValues = {3, 2, 1, 0};
    std::vector<int> remainingValues;

    for (int val : allValues) {
        if (val != first && val != second) {
            remainingValues.push_back(val);
        }
    }

    return remainingValues;
}
void splitMatrix(const half* A_h, int M_GLOBAL, int K_GLOBAL, 
                 half* A_h_structured, half* A_h_unstructured,
                 std::vector<std::vector<std::tuple<int, int>>>& metadata,
                 std::vector<std::vector<std::tuple<int, int>>>& metadata_unstructured) {
    const int WINDOW_SIZE = 4;
    metadata.resize(M_GLOBAL);
    metadata_unstructured.resize(M_GLOBAL);

    std::fill(A_h_structured, A_h_structured + M_GLOBAL * K_GLOBAL, 0);
    std::fill(A_h_unstructured, A_h_unstructured + M_GLOBAL * K_GLOBAL, 0);

    for (int i = 0; i < M_GLOBAL; ++i) {
        for (int j = 0; j < K_GLOBAL; j += WINDOW_SIZE) {
            std::vector<int> nonZeroPositions;
            for (int k = j; k < j + WINDOW_SIZE && k < K_GLOBAL; ++k) {
                if (__half2float(A_h[i * K_GLOBAL + k]) != 0) {
                    nonZeroPositions.push_back(k % WINDOW_SIZE);
                }
            }

            // 特殊情况：三个非零元素位于前三个位置
            if (nonZeroPositions.size() == 3 && nonZeroPositions[0] == 1 && nonZeroPositions[1] == 2 && nonZeroPositions[2] == 3) {
                metadata[i].emplace_back(1, 3); // 保存位置 1 和 3
                metadata_unstructured[i].emplace_back(2, 0); // 保存位置 2 和 0

                A_h_structured[i * K_GLOBAL + j + 1] = A_h[i * K_GLOBAL + j + 1];
                A_h_structured[i * K_GLOBAL + j + 3] = A_h[i * K_GLOBAL + j + 3];
                A_h_unstructured[i * K_GLOBAL + j + 2] = A_h[i * K_GLOBAL + j + 2];
            } else {
                // 一般情况处理
                int count = 0;
                for (int pos : nonZeroPositions) {
                    if (count < 2) {
                        A_h_structured[i * K_GLOBAL + j + pos] = A_h[i * K_GLOBAL + j + pos];
                    } else {
                        A_h_unstructured[i * K_GLOBAL + j + pos] = A_h[i * K_GLOBAL + j + pos];
                    }
                    ++count;
                }

                // 填充 metadata
                int first = nonZeroPositions.empty() ? 0 : nonZeroPositions[0];
                int second = nonZeroPositions.size() > 1 ? nonZeroPositions[1] : (first + 1) % WINDOW_SIZE;
                metadata[i].emplace_back(first, second);

                // 使用辅助函数填充 metadata_unstructured
                std::vector<int> remainingValues = findRemainingValues(first, second);
                if (remainingValues.size() >= 2) {
                    metadata_unstructured[i].emplace_back(remainingValues[0], remainingValues[1]);
                }
            }
        }
    }
}



void printMatrix(const half* mat, int M, int K, int StartRow, int EndRow, int StartCol, int EndCol, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = StartRow; i < EndRow; ++i) {
        for (int j = StartCol; j < EndCol; ++j) {
            std::cout << std::setw(4) << __half2float(mat[i * K + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
void printMatrixColumnMajor(const half* mat, int M, int K, int StartRow, int EndRow, int StartCol, int EndCol, const std::string& name) {
    std::cout << name << " (Column Major):\n";
    for (int i = StartRow; i < EndRow; ++i) {
        for (int j = StartCol; j < EndCol; ++j) {
            // 对于列优先存储，首先遍历列，然后是行
            std::cout << std::setw(4) << __half2float(mat[j * M + i]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
void printMetaE(const uint32_t* mat, int M, int K, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << std::setw(5) << mat[i * K + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
void compressSparseMat(const half* A_h_structured, int M_GLOBAL, int K_GLOBAL, 
                       half* compressedMat,
                       const std::vector<std::vector<std::tuple<int, int>>>& metadata) {
    for (int row_index = 0; row_index < M_GLOBAL; ++row_index) {
        const auto& metadata_row = metadata[row_index];
        for (int i = 0; i < metadata_row.size(); ++i) {
            int offset = i * 4;
            int compressedIdx = i * 2; // 索引在压缩矩阵中的位置
            for (const auto& idx : {std::get<0>(metadata_row[i]), std::get<1>(metadata_row[i])}) {
                compressedMat[row_index * (K_GLOBAL / 2) + compressedIdx] = A_h_structured[row_index * K_GLOBAL + offset + idx];
                ++compressedIdx;
            }
        }
    }
}
std::vector<std::vector<uint32_t>> metadataToBinary(const std::vector<std::vector<std::tuple<int, int>>>& metadata) {
    int row = metadata.size();
    int col = metadata[0].size() * 2;
    int meta_row = row / 16;
    int meta_col = col;
    std::vector<std::vector<uint32_t>> bin_meta(meta_row, std::vector<uint32_t>(meta_col, 0));

    int half_row_num = 8;
    int half_col_num = 4;

    for (int i = 0; i < row; i += 16) {
        for (int j = 0; j < col / 2; j += 8) {
            for (int row_id = 0; row_id < half_row_num; ++row_id) {
                for (int sub_col = 0; sub_col < 2; ++sub_col) {
                    uint32_t bit_offset = 0;
                    std::vector<int> whole_row;
                    // 构建完整的行
                    for (int k = 0; k < half_col_num; ++k) {
                        if (i + row_id < metadata.size() && j + sub_col * half_col_num + k < metadata[i + row_id].size()) {
                            whole_row.push_back(std::get<0>(metadata[i + row_id][j + sub_col * half_col_num + k]));
                            whole_row.push_back(std::get<1>(metadata[i + row_id][j + sub_col * half_col_num + k]));
                        }
                    }
                    for (int k = 0; k < half_col_num; ++k) {
                        if (i + row_id + half_row_num < metadata.size() && j + sub_col * half_col_num + k < metadata[i + row_id + half_row_num].size()) {
                            whole_row.push_back(std::get<0>(metadata[i + row_id + half_row_num][j + sub_col * half_col_num + k]));
                            whole_row.push_back(std::get<1>(metadata[i + row_id + half_row_num][j + sub_col * half_col_num + k]));
                        }
                    }
                    // 将整行编码为二进制
                    for (const auto& idx : whole_row) {
                        // std::cout << " idx: " <<  idx;
                        bin_meta[i / 16][j * 2 + row_id * 2 + sub_col] |= (idx << bit_offset);
                        bit_offset += 2;
                    }
                    // std::cout << std::endl;
                }
            }
        }
    }

    return bin_meta;
}
std::vector<std::vector<std::tuple<int, int>>> loadMetadata(const std::string& filename) {
    std::vector<std::vector<std::tuple<int, int>>> metadata;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::tuple<int, int>> row;
        std::istringstream iss(line);
        int idx1, idx2;
        while (iss >> idx1 >> idx2) {
            row.emplace_back(idx1, idx2);
        }
        metadata.push_back(row);
    }
    return metadata;
}
void calculateZeroNonZeroRatio(const half* compressedMat, int M_GLOBAL, int K_GLOBAL) {
    int totalElements = M_GLOBAL * (K_GLOBAL / 2);
    int zeroCount = 0;
    int nonZeroCount = 0;

    for (int i = 0; i < totalElements; ++i) {
        if (__half2float(compressedMat[i]) == 0) {
            ++zeroCount;
        } else {
            ++nonZeroCount;
        }
    }

    double zeroRatio = static_cast<double>(zeroCount) / totalElements;
    double nonZeroRatio = static_cast<double>(nonZeroCount) / totalElements;

    std::cout << "Total Elements: " << totalElements << std::endl;
    std::cout << "Zero Count: " << zeroCount << std::endl;
    std::cout << "Non-Zero Count: " << nonZeroCount << std::endl;
    std::cout << "Zero Ratio: " << zeroRatio << std::endl;
    std::cout << "Non-Zero Ratio: " << nonZeroRatio << std::endl;
}
void binMetaToArray(const std::vector<std::vector<uint32_t>>& bin_meta, uint32_t* array) {
    int rowIndex = 0;
    for (const auto& row : bin_meta) {
        for (uint32_t val : row) {
            array[rowIndex++] = val;
        }
    }
}
bool checkBitwiseInversion(const std::vector<std::vector<uint32_t>>& bin_meta, 
                           const std::vector<std::vector<uint32_t>>& bin_meta_unstructured) {
    if (bin_meta.size() != bin_meta_unstructured.size()) {
        return false;
    }

    for (size_t i = 0; i < bin_meta.size(); ++i) {
        if (bin_meta[i].size() != bin_meta_unstructured[i].size()) {
            return false;
        }

        for (size_t j = 0; j < bin_meta[i].size(); ++j) {
            if (~bin_meta[i][j] != bin_meta_unstructured[i][j]) {
                std::cout << "Mismatch at [" << i << "][" << j << "]: " 
                          << "bin_meta = " << bin_meta[i][j] 
                          << ", bin_meta_unstructured = " << bin_meta_unstructured[i][j] << std::endl;
                return false;
            }
        }
    }

    return true;
}
void invertMetadata(const std::vector<std::vector<std::tuple<int, int>>>& metadata,
                    std::vector<std::vector<std::tuple<int, int>>>& metadata_reverse) {
    metadata_reverse.resize(metadata.size());

    for (size_t i = 0; i < metadata.size(); ++i) {
        metadata_reverse[i].resize(metadata[i].size());
        for (size_t j = 0; j < metadata[i].size(); ++j) {
            int first = std::get<0>(metadata[i][j]);
            int second = std::get<1>(metadata[i][j]);

            // 按位取反操作
            uint32_t inverted_first = ~static_cast<uint32_t>(first);
            uint32_t inverted_second = ~static_cast<uint32_t>(second);

            // 仅保留原始数据位数
            inverted_first &= 0x3; // 假设 first 是 2 位数
            inverted_second &= 0x3; // 假设 second 是 2 位数

            metadata_reverse[i][j] = std::make_tuple(inverted_first, inverted_second);
        }
    }
}

bool compareMatrices(const half* mat1, const half* mat2, int size) {
    for (int i = 0; i < size; ++i) {
        if (__half2float(mat1[i]) != __half2float(mat2[i])) {
            std::cout << "Mismatch at index " << i << ": " << __half2float(mat1[i]) << " != " << __half2float(mat2[i]) << std::endl;
            return false;
        }
    }
    return true;
}
__host__ int InitSparseMatrixA_bitmap_v2(
    half* A_h,
    int M,  // 行数
    int K,  // 列数
    int tile_M,  // 8
    int tile_M_global,  // 16
    int tile_K,  // 8
    int tile_K_global,  // 64
    half** Compressed_Val,  // 压缩后 tile 重排的 Val 数组
    int** TileOffsets,      // 每个 tile 的局部非零值 offset
    int** TileOffsets_global, // 每个 global tile 的非零值 offset
    uint64_t** bitmap)      // 每个 tile 的 bitmap
{
    // 计算有多少个 tile 和 global tile
    int num_tiles_M = M / tile_M;
    int num_tiles_K = K / tile_K;
    int num_tiles = num_tiles_M * num_tiles_K;
    
    int num_global_tiles_M = M / tile_M_global;
    int num_global_tiles_K = K / tile_K_global;
    int num_global_tiles = num_global_tiles_M * num_global_tiles_K;

    // 为 Compressed_Val, TileOffsets, TileOffsets_global 和 bitmap 分配内存
    *Compressed_Val = (half*)malloc(M * K * sizeof(half));
    *TileOffsets = (int*)malloc(num_tiles * sizeof(int)); // 从第一个小tile的位置开始
    *TileOffsets_global = (int*)malloc((num_global_tiles + 1) * sizeof(int));
    *bitmap = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));

    if (*Compressed_Val == nullptr || *TileOffsets == nullptr || *TileOffsets_global == nullptr || *bitmap == nullptr) {
        return -1;
    }

    int val_count = 0; // 总共压缩后值的数量
    int tile_idx = 0;  // 当前处理的 tile 索引
    std::vector<int> global_val_counts(num_global_tiles + 1, 0);

    // 遍历所有 global tiles
    for (int global_tile_m = 0; global_tile_m < num_global_tiles_M; ++global_tile_m) {
        for (int global_tile_k = 0; global_tile_k < num_global_tiles_K; ++global_tile_k) {
            int global_row_start = global_tile_m * tile_M_global;
            int global_col_start = global_tile_k * tile_K_global;

            int local_val_count = 0; // 当前 global tile 内部的局部计数

            // 遍历 global tile 内的所有 tiles
            for (int local_tile_m = 0; local_tile_m < tile_M_global / tile_M; ++local_tile_m) {
                for (int local_tile_k = 0; local_tile_k < tile_K_global / tile_K; ++local_tile_k) {
                    int row_start = global_row_start + local_tile_m * tile_M;
                    int col_start = global_col_start + local_tile_k * tile_K;

                    uint64_t tile_bitmap = 0;

                    // 处理偶数列 0, 2, 4, 6
                    for (int row_offset = 0; row_offset < tile_M; ++row_offset) {
                        for (int col_offset = 0; col_offset < tile_K; col_offset += 2) {
                            int row = row_start + row_offset;
                            int col = col_start + col_offset;

                            if (row < M && col < K) {
                                half val = A_h[row * K + col];
                                if (__half2float(val) != 0.0f) {
                                    tile_bitmap |= (1ULL << (row_offset * 4 + (col_offset / 2)));
                                    (*Compressed_Val)[val_count++] = val;
                                    local_val_count++;
                                    global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1]++;
                                }
                            }
                        }
                    }

                    // 处理奇数列 1, 3, 5, 7
                    for (int row_offset = 0; row_offset < tile_M; ++row_offset) {
                        for (int col_offset = 1; col_offset < tile_K; col_offset += 2) {
                            int row = row_start + row_offset;
                            int col = col_start + col_offset;

                            if (row < M && col < K) {
                                half val = A_h[row * K + col];
                                if (__half2float(val) != 0.0f) {
                                    tile_bitmap |= (1ULL << (32 + row_offset * 4 + (col_offset / 2)));
                                    (*Compressed_Val)[val_count++] = val;
                                    local_val_count++;
                                    global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1]++;
                                }
                            }
                        }
                    }

                    (*bitmap)[tile_idx] = tile_bitmap;
                    (*TileOffsets)[tile_idx] = local_val_count; // 记录局部偏移
                    ++tile_idx;
                }
            }

            // Padding with zeros to make non-zero count a multiple of 4
            int current_count = global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1];
            int padding = (4 - (current_count % 4)) % 4;
            for (int p = 0; p < padding; ++p) {
                (*Compressed_Val)[val_count++] = __float2half(0.0f);
            }
            global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] += padding;
        }
    }

    (*TileOffsets_global)[0] = 0;

    // 计算每个 global tile 的非零值起始地址
    for (int i = 1; i <= num_global_tiles; ++i) {
        global_val_counts[i] += global_val_counts[i - 1];
    }

    for (int i = 0; i <= num_global_tiles; ++i) {
        (*TileOffsets_global)[i] = global_val_counts[i];
    }

    // 减少 Compressed_Val 的大小到实际需要的大小
    *Compressed_Val = (half*)realloc(*Compressed_Val, val_count * sizeof(half));

    return num_global_tiles;
}

__host__ int InitSparseMatrixA_bitmap_v3(
    half* A_h,
    int M,  // 行数
    int K,  // 列数
    int tile_M,  // 8
    int tile_M_global,  // 16
    int tile_K,  // 8
    int tile_K_global,  // 64
    half** Compressed_Val,  // 压缩后 tile 重排的 Val 数组
    int** TileOffsets,      // 每个 tile 的局部非零值 offset
    int** TileOffsets_global, // 每个 global tile 的非零值 offset
    uint64_t** bitmap,
    int& max_nnz_count)      // 每个 tile 的 bitmap
{
    // 计算有多少个 tile 和 global tile
    int num_tiles_M = M / tile_M;
    int num_tiles_K = K / tile_K;
    int num_tiles = num_tiles_M * num_tiles_K;
    
    int num_global_tiles_M = M / tile_M_global;
    int num_global_tiles_K = K / tile_K_global;
    int num_global_tiles = num_global_tiles_M * num_global_tiles_K;

    // 为 Compressed_Val, TileOffsets, TileOffsets_global 和 bitmap 分配内存
    *Compressed_Val = (half*)malloc(M * K * sizeof(half));
    *TileOffsets = (int*)malloc(num_tiles * sizeof(int)); // 从第一个小tile的位置开始
    *TileOffsets_global = (int*)malloc((num_global_tiles + 1) * sizeof(int));
    *bitmap = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));

    if (*Compressed_Val == nullptr || *TileOffsets == nullptr || *TileOffsets_global == nullptr || *bitmap == nullptr) {
        return -1;
    }

    int val_count = 0; // 总共压缩后值的数量
    int tile_idx = 0;  // 当前处理的 tile 索引
    std::vector<int> global_val_counts(num_global_tiles + 1, 0);
    max_nnz_count = 0; // 初始化最大非零元素计数
    // 遍历所有 global tiles
    for (int global_tile_m = 0; global_tile_m < num_global_tiles_M; ++global_tile_m) {
        for (int global_tile_k = 0; global_tile_k < num_global_tiles_K; ++global_tile_k) {
            int global_row_start = global_tile_m * tile_M_global;
            int global_col_start = global_tile_k * tile_K_global;

            int local_val_count = 0; // 当前 global tile 内部的局部计数

            // 遍历 global tile 内的所有 tiles
            for (int local_tile_m = 0; local_tile_m < tile_M_global / tile_M; ++local_tile_m) {
                for (int local_tile_k = 0; local_tile_k < tile_K_global / tile_K; ++local_tile_k) {
                    int row_start = global_row_start + local_tile_m * tile_M;
                    int col_start = global_col_start + local_tile_k * tile_K;

                    uint64_t tile_bitmap = 0;

                    // 处理所有列
                    for (int row_offset = 0; row_offset < tile_M; ++row_offset) {
                        for (int col_offset = 0; col_offset < tile_K; ++col_offset) {
                            int row = row_start + row_offset;
                            int col = col_start + col_offset;

                            if (row < M && col < K) {
                                half val = A_h[row * K + col];
                                if (__half2float(val) != 0.0f) {
                                    tile_bitmap |= (1ULL << (row_offset * tile_K + col_offset));
                                    (*Compressed_Val)[val_count++] = val;
                                    local_val_count++;
                                    global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1]++;
                                }
                            }
                        }
                    }

                    (*bitmap)[tile_idx] = tile_bitmap;
                    (*TileOffsets)[tile_idx] = local_val_count; // 记录局部偏移
                    ++tile_idx;
                }
            }

            // Padding with zeros to make non-zero count a multiple of 8
            int current_count = global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1];
            int padding = (8 - (current_count % 8)) % 8;
            for (int p = 0; p < padding; ++p) {
                (*Compressed_Val)[val_count++] = __float2half(0.0f);
            }
            global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] += padding;
                       // 更新最大非零元素计数
            current_count = global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1];
            if (current_count > max_nnz_count) {
                max_nnz_count = current_count;
            }
        }
    }

    (*TileOffsets_global)[0] = 0;

    // 计算每个 global tile 的非零值起始地址
    for (int i = 1; i <= num_global_tiles; ++i) {
        global_val_counts[i] += global_val_counts[i - 1];
    }

    for (int i = 0; i <= num_global_tiles; ++i) {
        (*TileOffsets_global)[i] = global_val_counts[i];
    }

    // 减少 Compressed_Val 的大小到实际需要的大小
    *Compressed_Val = (half*)realloc(*Compressed_Val, val_count * sizeof(half));

    return num_global_tiles;
}
int InitSparseMatrixA_bitmap_v4(
    half* A_h,
    int M,  // 行数
    int K,  // 列数
    int tile_M,  // 8
    int tile_M_global,  // 16
    int tile_K,  // 8
    int tile_K_global,  // 64
    half** Compressed_Val,
    int** TileOffsets,
    int** TileOffsets_global,
    uint64_t** bitmap,
    int& max_nnz_count)
{
    // 计算有多少个 tile 和 global tile
    int num_tiles_M = M / tile_M;
    int num_tiles_K = K / tile_K;
    int num_tiles = num_tiles_M * num_tiles_K;
    
    int num_global_tiles_M = M / tile_M_global;
    int num_global_tiles_K = K / tile_K_global;
    int num_global_tiles = num_global_tiles_M * num_global_tiles_K;

    // 为 Compressed_Val, TileOffsets, TileOffsets_global 和 bitmap 分配内存
    *Compressed_Val = (half*)malloc(M * K * sizeof(half));
    *TileOffsets = (int*)malloc(num_tiles * sizeof(int));
    *TileOffsets_global = (int*)malloc((num_global_tiles + 1) * sizeof(int));
    *bitmap = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));

    if (*Compressed_Val == nullptr || *TileOffsets == nullptr || *TileOffsets_global == nullptr || *bitmap == nullptr) {
        return -1;
    }

    int val_count = 0;
    int tile_idx = 0;
    std::vector<int> global_val_counts(num_global_tiles + 1, 0);
    max_nnz_count = 0;

    // 遍历所有 global tiles
    for (int global_tile_m = 0; global_tile_m < num_global_tiles_M; ++global_tile_m) {
        for (int global_tile_k = 0; global_tile_k < num_global_tiles_K; ++global_tile_k) {
            int global_row_start = global_tile_m * tile_M_global;
            int global_col_start = global_tile_k * tile_K_global;

            int local_val_count = 0;

            // 改进：按 2x2 的顺序遍历 global tile 内的小 tiles
            for (int local_tile_m_group = 0; local_tile_m_group < tile_M_global / tile_M; local_tile_m_group += 2) {
                for (int local_tile_k_group = 0; local_tile_k_group < tile_K_global / tile_K; local_tile_k_group += 2) {
                    // 处理 2x2 的小 tile 组，改为按列访问
                    for (int j = 0; j < 2; ++j) {  // 外循环遍历列
                        for (int i = 0; i < 2; ++i) {  // 内循环遍历行
                            int local_tile_m = local_tile_m_group + i;
                            int local_tile_k = local_tile_k_group + j;

                            if (local_tile_m < tile_M_global / tile_M && local_tile_k < tile_K_global / tile_K) {
                                int row_start = global_row_start + local_tile_m * tile_M;
                                int col_start = global_col_start + local_tile_k * tile_K;

                                uint64_t tile_bitmap = 0;

                                // 处理小 tile 中的所有元素
                                for (int row_offset = 0; row_offset < tile_M; ++row_offset) {
                                    for (int col_offset = 0; col_offset < tile_K; ++col_offset) {
                                        int row = row_start + row_offset;
                                        int col = col_start + col_offset;

                                        if (row < M && col < K) {
                                            half val = A_h[row * K + col];
                                            if (__half2float(val) != 0.0f) {
                                                tile_bitmap |= (1ULL << (row_offset * tile_K + col_offset));
                                                (*Compressed_Val)[val_count++] = val;
                                                local_val_count++;
                                                global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1]++;
                                            }
                                        }
                                    }
                                }

                                (*bitmap)[tile_idx] = tile_bitmap;
                                (*TileOffsets)[tile_idx] = local_val_count;
                                ++tile_idx;
                            }
                        }
                    }
                }
            }


            // Padding with zeros to make non-zero count a multiple of 8
            int current_count = global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1];
            int padding = (8 - (current_count % 8)) % 8;
            for (int p = 0; p < padding; ++p) {
                (*Compressed_Val)[val_count++] = __float2half(0.0f);
            }
            global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] += padding;

            // 更新最大非零元素计数
            current_count = global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1];
            if (current_count > max_nnz_count) {
                max_nnz_count = current_count;
            }
        }
    }

    (*TileOffsets_global)[0] = 0;

    // 计算每个 global tile 的非零值起始地址
    for (int i = 1; i <= num_global_tiles; ++i) {
        global_val_counts[i] += global_val_counts[i - 1];
    }

    for (int i = 0; i <= num_global_tiles; ++i) {
        (*TileOffsets_global)[i] = global_val_counts[i];
    }

    // 减少 Compressed_Val 的大小到实际需要的大小
    *Compressed_Val = (half*)realloc(*Compressed_Val, val_count * sizeof(half));

    return num_global_tiles;
}

int InitSparseMatrixA_bitmap_v5(
    half* A_h,
    int M,  // 行数
    int K,  // 列数
    int tile_M,  // 8
    int tile_M_global,  // 64
    int tile_K,  // 8
    int tile_K_global,  // 64
    half** Compressed_Val,
    int** TileOffsets,
    int** TileOffsets_global,
    uint64_t** bitmap,
    int& max_nnz_count)
{
    // 计算有多少个 tile 和 global tile
    int num_tiles_M = M / tile_M;
    int num_tiles_K = K / tile_K;
    int num_tiles = num_tiles_M * num_tiles_K;
    
    int num_global_tiles_M = M / tile_M_global;
    int num_global_tiles_K = K / tile_K_global;
    int num_global_tiles = num_global_tiles_M * num_global_tiles_K;

    // 为 Compressed_Val, TileOffsets, TileOffsets_global 和 bitmap 分配内存
    *Compressed_Val = (half*)malloc(M * K * sizeof(half));
    *TileOffsets = (int*)malloc(num_tiles * sizeof(int));
    *TileOffsets_global = (int*)malloc((num_global_tiles + 1) * sizeof(int));
    *bitmap = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));

    if (*Compressed_Val == nullptr || *TileOffsets == nullptr || *TileOffsets_global == nullptr || *bitmap == nullptr) {
        return -1;
    }

    int val_count = 0;
    int tile_idx = 0;
    std::vector<int> global_val_counts(num_global_tiles + 1, 0);
    max_nnz_count = 0;

    // 遍历所有 global tiles
    for (int global_tile_m = 0; global_tile_m < num_global_tiles_M; ++global_tile_m) {
        for (int global_tile_k = 0; global_tile_k < num_global_tiles_K; ++global_tile_k) {
            int global_row_start = global_tile_m * tile_M_global;
            int global_col_start = global_tile_k * tile_K_global;

            int local_val_count = 0;

            // 改进：按列优先的顺序遍历 global tile 内的小 tiles
            for (int local_tile_k_group = 0; local_tile_k_group < tile_K_global / tile_K; local_tile_k_group += 2) {
                for (int local_tile_m_group = 0; local_tile_m_group < tile_M_global / tile_M; local_tile_m_group += 2) {
                    // 处理 2x2 的小 tile 组，按列访问
                    for (int j = 0; j < 2; ++j) {  // 外循环遍历列
                        for (int i = 0; i < 2; ++i) {  // 内循环遍历行
                            int local_tile_k = local_tile_k_group + j;
                            int local_tile_m = local_tile_m_group + i;

                            if (local_tile_k < tile_K_global / tile_K && local_tile_m < tile_M_global / tile_M) {
                                int col_start = global_col_start + local_tile_k * tile_K;
                                int row_start = global_row_start + local_tile_m * tile_M;

                                uint64_t tile_bitmap = 0;

                                // 处理小 tile 中的所有元素
                                for (int row_offset = 0; row_offset < tile_M; ++row_offset) {
                                    for (int col_offset = 0; col_offset < tile_K; ++col_offset) {
                                        int row = row_start + row_offset;
                                        int col = col_start + col_offset;

                                        if (row < M && col < K) {
                                            half val = A_h[row * K + col];
                                            if (__half2float(val) != 0.0f) {
                                                tile_bitmap |= (1ULL << (row_offset * tile_K + col_offset));
                                                (*Compressed_Val)[val_count++] = val;
                                                local_val_count++;
                                                global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1]++;
                                            }
                                        }
                                    }
                                }


                                (*bitmap)[tile_idx] = tile_bitmap;
                                (*TileOffsets)[tile_idx] = local_val_count;
                                ++tile_idx;
                            }
                        }
                    }
                }
            }

            // Padding with zeros to make non-zero count a multiple of 8
            int current_count = global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1];
            int padding = (8 - (current_count % 8)) % 8;
            for (int p = 0; p < padding; ++p) {
                (*Compressed_Val)[val_count++] = __float2half(0.0f);
            }
            global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] += padding;

            // 更新最大非零元素计数
            current_count = global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1];
            if (current_count > max_nnz_count) {
                max_nnz_count = current_count;
            }
        }
    }

    (*TileOffsets_global)[0] = 0;

    // 计算每个 global tile 的非零值起始地址
    for (int i = 1; i <= num_global_tiles; ++i) {
        global_val_counts[i] += global_val_counts[i - 1];
    }

    for (int i = 0; i <= num_global_tiles; ++i) {
        (*TileOffsets_global)[i] = global_val_counts[i];
    }

    // 减少 Compressed_Val 的大小到实际需要的大小
    *Compressed_Val = (half*)realloc(*Compressed_Val, val_count * sizeof(half));

    return num_global_tiles;
}

int InitSparseMatrixA_bitmap_v6(
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
    int& max_nnz_count)
{
    // 计算各层的tile数量
    int num_tiles_M = M / tile_M;
    int num_tiles_K = K / tile_K;
    int num_tiles = num_tiles_M * num_tiles_K;
    
    int num_median_tiles_M = M / tile_M_median;
    int num_median_tiles_K = K / tile_K_median;
    int num_median_tiles = num_median_tiles_M * num_median_tiles_K;

    int num_global_tiles_M = M / tile_M_global;
    int num_global_tiles_K = K / tile_K_global;
    int num_global_tiles = num_global_tiles_M * num_global_tiles_K;

    // 为各数据结构分配内存
    *Compressed_Val = (half*)malloc(M * K * sizeof(half));
    *TileOffsets = (int*)malloc(num_tiles * sizeof(int));
    *TileOffsets_median = (int*)malloc(num_median_tiles * (tile_M_median / tile_M * tile_K_median / tile_K) * sizeof(int));
    *TileOffsets_global = (int*)malloc((num_global_tiles + 1) * sizeof(int));
    *bitmap = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));

    if (*Compressed_Val == nullptr || *TileOffsets == nullptr || 
        *TileOffsets_median == nullptr || *TileOffsets_global == nullptr || *bitmap == nullptr) {
        return -1;
    }

    int val_count = 0;
    int tile_idx = 0;
    int median_offset_idx = 0;
    std::vector<int> global_val_counts(num_global_tiles + 1, 0);
    max_nnz_count = 0;

    // 遍历所有 global tiles
    for (int global_tile_m = 0; global_tile_m < num_global_tiles_M; ++global_tile_m) {
        for (int global_tile_k = 0; global_tile_k < num_global_tiles_K; ++global_tile_k) {
            int global_row_start = global_tile_m * tile_M_global;
            int global_col_start = global_tile_k * tile_K_global;
            int global_val_count = 0;
            
            int median_val_count = 0;
            (*TileOffsets_median)[median_offset_idx++] = 0;  // 每个median tile的起始偏移量为0
            // 遍历 global tile 内的 median tiles (按行顺序)
            for (int median_tile_m = 0; median_tile_m < tile_M_global / tile_M_median; ++median_tile_m) {
                for (int median_tile_k = 0; median_tile_k < tile_K_global / tile_K_median; ++median_tile_k) {
                    int median_row_start = global_row_start + median_tile_m * tile_M_median;
                    int median_col_start = global_col_start + median_tile_k * tile_K_median;
                    // 处理 median tile 内的 2x2 小 tile 组
                    for (int local_tile_m_group = 0; local_tile_m_group < tile_M_median / tile_M; local_tile_m_group += 2) {
                        for (int local_tile_k_group = 0; local_tile_k_group < tile_K_median / tile_K; local_tile_k_group += 2) {
                            // 按列优先处理 2x2 的小 tile 组
                            for (int j = 0; j < 2; ++j) {
                                for (int i = 0; i < 2; ++i) {
                                    int local_tile_k = local_tile_k_group + j;
                                    int local_tile_m = local_tile_m_group + i;

                                    int col_start = median_col_start + local_tile_k * tile_K;
                                    int row_start = median_row_start + local_tile_m * tile_M;

                                    uint64_t tile_bitmap = 0;
                                    int local_val_count = 0;

                                    // 处理小 tile 中的所有元素
                                    for (int row_offset = 0; row_offset < tile_M; ++row_offset) {
                                        for (int col_offset = 0; col_offset < tile_K; ++col_offset) {
                                            int row = row_start + row_offset;
                                            int col = col_start + col_offset;

                                            if (row < M && col < K) {
                                                half val = A_h[row * K + col];
                                                if (__half2float(val) != 0.0f) {
                                                    tile_bitmap |= (1ULL << (row_offset * tile_K + col_offset));
                                                    (*Compressed_Val)[val_count++] = val;
                                                    local_val_count++;
                                                    median_val_count++;
                                                    global_val_count++;
                                                }
                                            }
                                        }
                                    }

                                    (*bitmap)[tile_idx] = tile_bitmap;
                                    (*TileOffsets)[tile_idx] = local_val_count;
                                    ++tile_idx;
                                }
                            }
                        }
                    }
                    if(median_tile_m < (tile_M_global / tile_M_median - 1) or median_tile_k < (tile_K_global / tile_K_median - 1)){
                        // 更新 TileOffsets_median
                        (*TileOffsets_median)[median_offset_idx] = median_val_count;
                        median_offset_idx++;
                    } 

                }
            }

            // Additional padding for global tiles (if necessary)
            int global_padding = (8 - (global_val_count % 8)) % 8;
            for (int p = 0; p < global_padding; ++p) {
                (*Compressed_Val)[val_count++] = __float2half(0.0f);
            }
            global_val_count += global_padding;

            // Update global_val_counts and max_nnz_count
            global_val_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] = global_val_count;
            if (global_val_count > max_nnz_count) {
                max_nnz_count = global_val_count;
            }
        }
    }

    // Calculate offsets for global tiles
    (*TileOffsets_global)[0] = 0;
    for (int i = 1; i <= num_global_tiles; ++i) {
        global_val_counts[i] += global_val_counts[i - 1];
        (*TileOffsets_global)[i] = global_val_counts[i];
    }

    // 减少 Compressed_Val 的大小到实际需要的大小
    *Compressed_Val = (half*)realloc(*Compressed_Val, val_count * sizeof(half));

    return num_global_tiles;
}
void printBinary(uint64_t number) {
    for (int bit = 63; bit >= 0; --bit) {
        std::cout << ((number >> bit) & 1);
    }
}
void print_bitmap_results(half* Compressed_Val, int* TileOffsets, uint64_t* bitmap, int num_tiles, int val_count) {
    // 打印压缩后的非零值数组
    std::cout << "Compressed_Val: ";
    for (int i = 0; i < val_count; ++i) {
        std::cout << __half2float(Compressed_Val[i]) << " ";
    }
    std::cout << std::endl;

    // 打印 TileOffsets
    std::cout << "TileOffsets: ";
    for (int i = 0; i < num_tiles+1; ++i) {
        std::cout << "-i: " << i << " " << TileOffsets[i] << " ";
    }
    std::cout << std::endl;

    // 打印 bitmap
    std::cout << "Bitmaps: ";
    for (int i = 0; i < num_tiles; ++i) {
        std::cout << i << " " << std::hex << bitmap[i] << " " << std::dec << bitmap[i] << " ";
        printBinary(bitmap[i]);
        std::cout << std::endl;
    }
    std::cout << std::dec << std::endl; // 恢复到十进制格式
}
void print_bitmap_v3_results(half* Compressed_Val, int* TileOffsets, int* TileOffsets_global, uint64_t* bitmap, int num_tiles, int num_global_tiles, int val_count) {

    std::cout << "Compressed_Val: ";
    for (int i = 0; i < val_count; ++i) {
        std::cout << __half2float(Compressed_Val[i]) << " ";
    }
    std::cout << std::endl;


    std::cout << "TileOffsets: ";
    for (int i = 0; i < num_tiles; ++i) {
        std::cout << "-i: " << i << " " << TileOffsets[i] << " ";
    }
    std::cout << std::endl;


    std::cout << "TileOffsets_global: ";
    for (int i = 0; i <= num_global_tiles; ++i) {
        std::cout << "-i: " << i << " " << TileOffsets_global[i] << " ";
    }
    std::cout << std::endl;


    std::cout << "Bitmaps: ";
    for (int i = 0; i < num_tiles; ++i) {
        std::cout << i << " " << std::hex << bitmap[i] << " " << std::dec << bitmap[i] << " ";
        printBinary(bitmap[i]);
        std::cout << std::endl;
    }
    std::cout << std::dec << std::endl;
}
void print_bitmap_v6_results(half* Compressed_Val, int* TileOffsets, int* TileOffsets_median, int* TileOffsets_global, uint64_t* bitmap, int num_tiles, int num_median_tiles, int num_global_tiles, int val_count) {
    std::cout << "TileOffsets: ";
    for (int i = 0; i < num_tiles; ++i) {
        std::cout << "-i: " << i << " " << TileOffsets[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "TileOffsets_median: ";
    for (int i = 0; i < num_median_tiles; ++i) {
        std::cout << "-i: " << i << " " << TileOffsets_median[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "TileOffsets_global: ";
    for (int i = 0; i <= num_global_tiles; ++i) {
        std::cout << "-i: " << i << " " << TileOffsets_global[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Bitmaps: ";
    for (int i = 0; i < num_tiles; ++i) {
        std::cout << i << " " << std::hex << bitmap[i] << " " << std::dec << bitmap[i] << " ";
        printBinary(bitmap[i]);
        std::cout << std::endl;
    }
    std::cout << std::dec << std::endl; // 恢复到十进制格式
}
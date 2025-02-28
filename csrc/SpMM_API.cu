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
 // Extended from https://github.com/AlibabaResearch/flash-llm/blob/main/csrc/SpMM_API.cuh
#include "./MatMulUtilities.cuh"
#include "./Reduction_Kernel.cuh"
#include "./SpMM_Kernel.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template<typename TilingConfig, typename SparseKernelConfig>
static void SpMM_SplitK_Kernel_Ex(cudaStream_t stream,
                                  const half*  A,
                                  const uint4* Compressed_A,
                                  const int*   TileOffsets,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K)
{
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    cudaFuncSetAttribute(
        SpMM_Kernel<TilingConfig, SparseKernelConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    //
    SpMM_Kernel<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, Compressed_A, TileOffsets, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
}





template<typename TilingConfig>
static void SpMM_SplitK_Kernel_Ex_bitmap_v3(cudaStream_t stream,
                                  const half*  A,
                                  const half* Compressed_A,
                                  const int*   TileOffsets,
                                  const int* TileOffsets_Median,
                                  const uint64_t*   bitmap,
                                  const int* max_nnz_intile,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K)
{
    // 13b: 2304
    static int SHMEM_SZ = max((TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2 + 2304 * sizeof(half) + (TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3) * sizeof(uint64_t),
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    cudaFuncSetAttribute(
        SpMM_Kernel_bitmap_v3<TilingConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    SpMM_Kernel_bitmap_v3<TilingConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, Compressed_A, TileOffsets, TileOffsets_Median, bitmap, max_nnz_intile, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
}

/*
half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output
tensors
                            2. Reduction_Workspace_Size = max( Split_K * M_Global * N_Global ) * sizeof(fp16)
int Split_K:                Split K dimension into Split_K Parts
*/
cudaError_t SpMM_SplitK_API(cudaStream_t stream,
                            const half*  A,
                            const uint4* Compressed_A,
                            const int*   TileOffsets,
                            const half*  B,
                            half*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
                            int          Split_K)
{
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1)
        SpMM_SplitK_OutputPTR = C;
    else
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    // Batched SpMM
    switch (N_Global) {
        case 8:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<96>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 16:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1>, SparseKernelConfig<96>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 32:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 2>, SparseKernelConfig<96>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 64:
            // return SpMM_SplitK_Kernel_Ex< TilingConfig<4, 1, 4>, SparseKernelConfig<64> >
            SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 2>, SparseKernelConfig<64>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 128:
            SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 4>, SparseKernelConfig<32>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        default:
            if (N_Global % 128 == 0)
                SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 4>, SparseKernelConfig<32>>(stream,
                                                                                     A,
                                                                                     Compressed_A,
                                                                                     TileOffsets,
                                                                                     B,
                                                                                     SpMM_SplitK_OutputPTR,
                                                                                     M_Global,
                                                                                     N_Global,
                                                                                     K_Global,
                                                                                     Split_K);
            else {
                printf("MM_Sparse_API Error: Unsupported N dimension %d!\n", N_Global);
                return cudaErrorUnknown;
            }
            break;
    }
    //
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;

    if (Split_K == 1)
        return Error;
    dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    return cudaGetLastError();
}
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
                            int          Split_K)
{
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1)
        SpMM_SplitK_OutputPTR = C;
    else
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    // Batched SpMM
    switch (N_Global) {
        case 8:
            SpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 1, 1>>(
                stream, A, Compressed_A, TileOffsets, TileOffsets_Median, bitmap, max_nnz_intile, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 16:
            SpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 1>>(
                stream, A, Compressed_A, TileOffsets, TileOffsets_Median, bitmap, max_nnz_intile, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 32:
            SpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 2>>(
                stream, A, Compressed_A, TileOffsets, TileOffsets_Median, bitmap, max_nnz_intile, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 64:
            SpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 4>>(
                stream, A, Compressed_A, TileOffsets, TileOffsets_Median, bitmap, max_nnz_intile, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 128:
            SpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 4>>(
                stream, A, Compressed_A, TileOffsets, TileOffsets_Median, bitmap,  max_nnz_intile, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        default:
            if (N_Global % 128 == 0)
            SpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 4>>(stream,
                                                                                     A,
                                                                                     Compressed_A,
                                                                                     TileOffsets,
                                                                                     TileOffsets_Median,
                                                                                     bitmap,
                                                                                     max_nnz_intile,
                                                                                     B,
                                                                                     SpMM_SplitK_OutputPTR,
                                                                                     M_Global,
                                                                                     N_Global,
                                                                                     K_Global,
                                                                                     Split_K);
            else {
                printf("MM_Sparse_API Error: Unsupported N dimension %d!\n", N_Global);
                return cudaErrorUnknown;
            }
            break;
    }
    
    //
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;

    if (Split_K == 1)
        return Error;
    
    dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    return cudaGetLastError();
}

static int BankID_Minimum(std::vector<unsigned int> ItemsInBank[])
{
    int ID           = 0;
    int MinItemCount = ItemsInBank[0].size();
    for (int i = 1; i < 32; i++) {
        if (ItemsInBank[i].size() < MinItemCount) {
            ID           = i;
            MinItemCount = ItemsInBank[i].size();
        }
    }
    return ID;
}

static int BankID_Maximum(std::vector<unsigned int> ItemsInBank[])
{
    int ID           = 0;
    int MaxItemCount = ItemsInBank[0].size();
    for (int i = 1; i < 32; i++) {
        if (ItemsInBank[i].size() > MaxItemCount) {
            ID           = i;
            MaxItemCount = ItemsInBank[i].size();
        }
    }
    return ID;
}

/*
return: Number of Element in array TileOffsets
Note: TileOffsets[return-1] = NNZ / SparseKernelConfig::VECTOR_SIZE    (SparseKernelConfig::VECTOR_SIZE = 4)
*/
// template<typename TilingConfig, typename SparseKernelConfig>
__host__ int InitSparseMatrixA_API(half*      A_h,
                                   int        M,
                                   int        N,
                                   int        K,
                                   uint32_t** Compressed_A,  // CPU PTR
                                   int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % TILE_M == 0 && K % TILE_K == 0);
    int       TotalNZCount = 0;
    uint32_t* Ptr_SubArray = *Compressed_A;
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half*        CurrentTilePTR    = A_h + (i * TILE_M) * K + (j * TILE_K);
            int          TileNZCount       = 0;
            int          remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];
            unsigned int Item              = 0;
            // Processing each tile
            std::vector<unsigned int> ItemsInBank[32];
            int                       ZeroPositionForBank[32];
            for (int k = 0; k < 32; k++)
                ZeroPositionForBank[k] = -1;
            //
            // printf("Starting Processing Tile i:%d j:%d...\n", i, j);
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    // Row permutation for bank-conflict-free shared memory layout
                    int      row            = m;
                    int      col            = n;
                    uint32_t mask           = (row % 8) << 3;
                    int      col_permutated = col ^ mask;
                    int      bank_smem      = (col_permutated / 2) % 32;
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(&Item);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        *short_ptr       = static_cast<short>(row * TILE_K + col_permutated);
                        ItemsInBank[bank_smem].push_back(Item);
                        //
                        TileNZCount++;
                    }
                    else {
                        if (ZeroPositionForBank[bank_smem] == -1)
                            ZeroPositionForBank[bank_smem] = row * TILE_K + col_permutated;
                    }
                }
            }
            //
            // printf("Starting Weight Padding...\n");
            for (int k = 0; k < remainingPaddings; k++) {
                int BankID = BankID_Minimum(ItemsInBank);
                assert(BankID >= 0 && BankID < 32);
                int ZeroPosition = ZeroPositionForBank[BankID];
                assert(ZeroPosition != -1);
                //
                half* half_ptr   = reinterpret_cast<half*>(&Item);
                *half_ptr        = __float2half_rn(0.0f);
                short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                *short_ptr       = static_cast<short>(ZeroPosition);
                ItemsInBank[BankID].push_back(Item);
                //
                TileNZCount++;
            }
            /*
            if(i==0 && j==0)
            {
              printf("For tile i:%d j:%d\n",i,j);
              for(int h=0; h<32; h++)
                printf("%ld ", ItemsInBank[h].size());
              printf("\n");
            }
            */
            //
            // printf("Starting Weight Shuffle...\n");
            std::vector<unsigned int> MainPart[32];
            std::vector<unsigned int> TailPart[32];
            int                       TileVectorCount = TileNZCount / VECTOR_SIZE;
            assert(TileNZCount % VECTOR_SIZE == 0);
            int Repeat_Vector   = TileVectorCount / WARP_SIZE;
            int Remained_Vector = TileVectorCount % WARP_SIZE;
            // Filing the TailPart
            for (int v = 0; v < VECTOR_SIZE; v++) {
                for (int b = 0; b < Remained_Vector; b++) {
                    int BankID = BankID_Maximum(ItemsInBank);
                    Item       = ItemsInBank[BankID].back();
                    ItemsInBank[BankID].pop_back();
                    TailPart[b].push_back(Item);
                }
            }
            // Filing the MainPart
            // printf("Starting Filing the MainPart...\n");
            for (int r = 0; r < Repeat_Vector; r++) {
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    for (int b = 0; b < WARP_SIZE; b++) {
                        int BankID = BankID_Maximum(ItemsInBank);
                        Item       = ItemsInBank[BankID].back();
                        ItemsInBank[BankID].pop_back();
                        MainPart[b].push_back(Item);
                    }
                }
            }
            // Writing to the Sub-Array
            // printf("Starting Writing to the Sub-Array...\n");
            for (int r = 0; r < Repeat_Vector; r++) {
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    for (int b = 0; b < 32; b++) {
                        Item = MainPart[b].back();
                        MainPart[b].pop_back();
                        int V_Size                                     = VECTOR_SIZE;
                        Ptr_SubArray[r * V_Size * 32 + b * V_Size + v] = Item;
                    }
                }
            }
            Ptr_SubArray += Repeat_Vector * VECTOR_SIZE * WARP_SIZE;
            for (int v = 0; v < VECTOR_SIZE; v++) {
                for (int b = 0; b < Remained_Vector; b++) {
                    Item = TailPart[b].back();
                    TailPart[b].pop_back();
                    Ptr_SubArray[b * VECTOR_SIZE + v] = Item;
                }
            }
            Ptr_SubArray += VECTOR_SIZE * Remained_Vector;
            //
            TotalNZCount += TileNZCount;
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    //
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //
    return (M / TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}

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


// A_h is host memory pointer, Compressed_A and TileOffsets are device memory pointers
__host__ int InitSparseMatrixA_API_NoReorder(half*      A_h,
                                             int        M,
                                             int        N,
                                             int        K,
                                             uint32_t** Compressed_A,  // CPU PTR
                                             int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Format for different N, in our kernel, TILE_M=128 or 256
    const int TILE_M                       = 128;
    // const int TILE_M                       = 16;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is NOT Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % TILE_M == 0 && K % TILE_K == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR    = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount       = 0;
            int   remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        // Row permutation for bank-conflict-free shared memory layout
                        int      row            = m;
                        int      col            = n;
                        uint32_t mask           = (row % 8) << 3;
                        int      col_permutated = col ^ mask;
                        *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                        //
                        TileNZCount++;
                        TotalNZCount++;
                    }
                    else {
                        if (remainingPaddings > 0) {
                            remainingPaddings--;
                            half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                            *half_ptr        = value;  // zero
                            short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                            // Row permutation for bank-conflict-free shared memory layout
                            int      row            = m;
                            int      col            = n;
                            uint32_t mask           = (row % 8) << 3;
                            int      col_permutated = col ^ mask;
                            *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                            //
                            TileNZCount++;
                            TotalNZCount++;
                        }
                    }
                }
            }
            //
            assert(TileNZCount % VECTOR_SIZE == 0);
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //

    //
    return (M / TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}

// A_h is host memory pointer, Compressed_A and TileOffsets are device memory pointers
__host__ int InitSparseMatrixA_API_NoReorder_unstructured(half*      A_h,
                                             int        M,
                                             int        N,
                                             int        K,
                                             uint32_t** Compressed_A,  // CPU PTR
                                             int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    // const int TILE_M                       = 128;
    // const int TILE_M                       = 64;
    const int TILE_M                       = 16;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is NOT Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / TILE_M;
    int   NumCol_offsets = K / TILE_K_HALF;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K_HALF; j++) {
            half* CurrentTilePTR = A_h + (i * TILE_M) * K + (j * TILE_K_HALF);
            int   TileNZCount    = 0;
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K_HALF; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K_HALF) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % TILE_M == 0 && K % TILE_K_HALF == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K_HALF; j++) {
            half* CurrentTilePTR    = A_h + (i * TILE_M) * K + (j * TILE_K_HALF);
            int   TileNZCount       = 0;
            int   remainingPaddings = PaddingForEachTile[i * (K / TILE_K_HALF) + j];
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K_HALF; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        // Row permutation for bank-conflict-free shared memory layout
                        int      row            = m;
                        int      col            = n;
                        uint32_t mask           = ((row % 8)/2) << 3;
                        int      col_permutated = col ^ mask;
                        *short_ptr              = static_cast<short>(row * TILE_K_HALF + col_permutated);
                        //
                        TileNZCount++;
                        TotalNZCount++;
                    }
                    else {
                        if (remainingPaddings > 0) {
                            remainingPaddings--;
                            half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                            *half_ptr        = value;  // zero
                            short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                            // Row permutation for bank-conflict-free shared memory layout
                            int      row            = m;
                            int      col            = n;
                            uint32_t mask           = ((row % 8)/2) << 3;
                            int      col_permutated = col ^ mask;
                            *short_ptr              = static_cast<short>(row * TILE_K_HALF + col_permutated);
                            //
                            TileNZCount++;
                            TotalNZCount++;
                        }
                    }
                }
            }
            //
            assert(TileNZCount % VECTOR_SIZE == 0);
            (*TileOffsets)[i * K / TILE_K_HALF + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / TILE_M) * (K / TILE_K_HALF) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //

    //
    return (M / TILE_M) * (K / TILE_K_HALF) + 2;  // number of Elements in array TileOffsets
}


/*
input:    char* DenseMatrixFileName
          int   M
          int   N                   // N is used by void InitSparseMatrixA_API()
          int   K
          char* NZWeightsFileName
          char* TileOffsetsFileName
          char* OutputSizesFileName // NNZ -> NumOffsets
*/
extern "C" void GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                       int   M,
                                       int   K,
                                       char* NZWeightsFileName,
                                       char* TileOffsetsFileName,
                                       char* OutputSizesFileName)
{
    std::vector<half> host_array(M * K);
    std::ifstream     in(DenseMatrixFileName, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        printf("file %s cannot be opened, loadDataArrayFromBin fails. \n", DenseMatrixFileName);
        exit(-1);
    }
    size_t loaded_data_size = sizeof(half) * M * K;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
#ifdef DEBUG_MODE
    printf("Read %ld bytes from %s.\n", loaded_data_size, DenseMatrixFileName);
#endif
    in.read((char*)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        printf("file %s only has %ld, but request %ld, loading DenseMatrix fails! \n",
               DenseMatrixFileName,
               in_get_size,
               loaded_data_size);
        exit(-1);
    }
    in.close();
    // Step 2: Dense to Sparse Transformation
    unsigned int* NZWeights_CPU   = nullptr;
    int*          TileOffsets_CPU = nullptr;
    int           NumOffsets      = InitSparseMatrixA_API(host_array.data(), M, 0, K, &NZWeights_CPU, &TileOffsets_CPU);
    int           NNZ             = TileOffsets_CPU[NumOffsets - 1] * 4;  // VectorSize = 4
    // Step 3: Write to FILE(OutputSizesFileName)
    //         Write to FILE(NZWeightsFileName), FILE(TileOffsetsFileName)
    std::ofstream out_SizesFile(OutputSizesFileName, std::ios::out | std::ios::binary);
    std::ofstream out_NZWeightsFile(NZWeightsFileName, std::ios::out | std::ios::binary);
    std::ofstream out_TileOffsetsFile(TileOffsetsFileName, std::ios::out | std::ios::binary);
    if (!out_SizesFile.is_open() || !out_NZWeightsFile.is_open() || !out_TileOffsetsFile.is_open()) {
        printf("GenSparseMatrixBinFile() ERROR: file %s, %s, or %s cannot be opened or creaetd. \n",
               OutputSizesFileName,
               NZWeightsFileName,
               TileOffsetsFileName);
        exit(-1);
    }
    //
    // out_SizesFile << NNZ << NumOffsets;
    out_SizesFile.write((char*)&NNZ, sizeof(int));
    out_SizesFile.write((char*)&NumOffsets, sizeof(int));
    out_SizesFile.close();
    out_NZWeightsFile.write((char*)NZWeights_CPU, sizeof(uint32_t) * NNZ);
    out_NZWeightsFile.close();
    out_TileOffsetsFile.write((char*)TileOffsets_CPU, sizeof(int) * NumOffsets);
    out_TileOffsetsFile.close();
}

extern "C" void Our_GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                       int   M,
                                       int   K,
                                       char* Compressed_ValFileName,
                                       char* bitmap_TileOffsets_globalFileName,
                                       char* bitmap_TileOffsets_medianFileName,
                                       char* bitmapFileName,
                                       char* max_nnz_intileFileName,
                                       char* OutputSizesFileName)
{
    std::vector<half> host_array(M * K);
    std::ifstream     in(DenseMatrixFileName, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        printf("file %s cannot be opened, loadDataArrayFromBin fails. \n", DenseMatrixFileName);
        exit(-1);
    }
    size_t loaded_data_size = sizeof(half) * M * K;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
#ifdef DEBUG_MODE
    printf("Read %ld bytes from %s.\n", loaded_data_size, DenseMatrixFileName);
#endif
    in.read((char*)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        printf("file %s only has %ld, but request %ld, loading DenseMatrix fails! \n",
               DenseMatrixFileName,
               in_get_size,
               loaded_data_size);
        exit(-1);
    }
    in.close();
    // Step 2: Dense to Sparse Transformation
    // Define output pointer
    half* Compressed_Val_cpu_v3 = nullptr;
    int* bitmap_TileOffsets_cpu_v3 = nullptr;
    int* bitmap_TileOffsets_median_cpu_v3 = nullptr;
    int* bitmap_TileOffsets_global_cpu_v3 = nullptr;
    uint64_t* bitmap_cpu_v3 = nullptr;
    int max_nnz_intilev3 = 0;
    // Call InitSparseMatrixA_bitmap
    auto num_gtilesv3 = InitSparseMatrixA_bitmap(host_array.data(), M, K, 8, 16, 64, 8, 64, 64, &Compressed_Val_cpu_v3, &bitmap_TileOffsets_cpu_v3, &bitmap_TileOffsets_median_cpu_v3, &bitmap_TileOffsets_global_cpu_v3, &bitmap_cpu_v3, max_nnz_intilev3);
    auto local_tile_numv3 = 8*8;
    auto median_tile_numv3 = 4*1;
    auto num_ltilesv3 = num_gtilesv3*local_tile_numv3;
    auto num_mtilesv3 = num_gtilesv3*median_tile_numv3;
    int val_count_v3 = bitmap_TileOffsets_global_cpu_v3[num_gtilesv3]; // 最后一个 tile 的偏移即为压缩后的非零值总数
    // 将 max_nnz_intilev3 调整为 64 的倍数
    if (max_nnz_intilev3 % 64 != 0) {
        max_nnz_intilev3 = ((max_nnz_intilev3 / 64) + 1) * 64;
    }
    printf("num_global_tiles: %d, bitmap v3 NNZ: %d, max_nnz_intilev3: %d \n", num_gtilesv3, val_count_v3, max_nnz_intilev3);
    // Step 3: Write to FILE(OutputSizesFileName), size[4]
    //         Write to FILE(Compressed_ValFileName), size[val_count_v3]
    //         Write to FILE(bitmap_TileOffsets_globalFileName), size[num_gtilesv3 + 1], FILE(bitmap_TileOffsets_medianFileName), size[num_mtilesv3], FILE(BitmapFileName), size[num_ltilesv3]
    //         Write to FILE(max_nnz_intileFileName), size[1]
    std::ofstream out_SizesFile(OutputSizesFileName, std::ios::out | std::ios::binary);   // 4
    std::ofstream out_CompressedvalFile(Compressed_ValFileName, std::ios::out | std::ios::binary); // val_count_v3
    std::ofstream out_BitmapglobalFile(bitmap_TileOffsets_globalFileName, std::ios::out | std::ios::binary);  // num_gtilesv3 + 1
    std::ofstream out_BitmapmedianFile(bitmap_TileOffsets_medianFileName, std::ios::out | std::ios::binary);  // num_mtilesv3
    std::ofstream out_BitmapFile(bitmapFileName, std::ios::out | std::ios::binary);       // num_ltilesv3
    std::ofstream out_NnzintileFile(max_nnz_intileFileName, std::ios::out | std::ios::binary);     // 1
    if (!out_SizesFile.is_open() || !out_CompressedvalFile.is_open() || !out_BitmapglobalFile.is_open() || !out_BitmapmedianFile.is_open() || !out_BitmapFile.is_open() | !out_NnzintileFile.is_open()) {
        printf("Our_GenSparseMatrixBinFile() ERROR: file %s, %s, %s, %s, %s or %s cannot be opened or creaetd. \n",
               OutputSizesFileName, Compressed_ValFileName,
               bitmap_TileOffsets_globalFileName, bitmap_TileOffsets_medianFileName,
               bitmapFileName, max_nnz_intileFileName);
        exit(-1);
    }
    out_SizesFile.write((char*)&val_count_v3, sizeof(int));
    num_gtilesv3++;
    out_SizesFile.write((char*)&num_gtilesv3, sizeof(int));
    out_SizesFile.write((char*)&num_mtilesv3, sizeof(int));
    out_SizesFile.write((char*)&num_ltilesv3, sizeof(int));
    out_SizesFile.close();
    out_CompressedvalFile.write((char*)Compressed_Val_cpu_v3, sizeof(half) * val_count_v3);
    out_CompressedvalFile.close();
    out_BitmapglobalFile.write((char*)bitmap_TileOffsets_global_cpu_v3, sizeof(int) * num_gtilesv3);
    out_BitmapglobalFile.close();
    out_BitmapmedianFile.write((char*)bitmap_TileOffsets_median_cpu_v3, sizeof(int) * num_mtilesv3);
    out_BitmapmedianFile.close();
    out_BitmapFile.write((char*)bitmap_cpu_v3, sizeof(uint64_t) * num_ltilesv3);
    out_BitmapFile.close();
    out_NnzintileFile.write((char*)&max_nnz_intilev3, sizeof(int));
    out_NnzintileFile.close();
}

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

#include "MatMulUtilities.cuh"
#include <vector>
#include <inttypes.h>
#define __STDC_FORMAT_MACROS
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_CopyFromGlobalToReg(uint32_t*    Registers_GlobalToShared1,
                                                         uint32_t*    NNZ_VECTOR_ThreadLocal1,
                                                         const uint4* GlobalPTR1,
                                                         int          NNZ_VECTOR_ThisTile1,
                                                         uint32_t*    Registers_GlobalToShared2,
                                                         uint32_t*    NNZ_VECTOR_ThreadLocal2,
                                                         const uint4* GlobalPTR2,
                                                         int          NNZ_VECTOR_ThisTile2)
{
    // Load Global to registers
    int Num_NNZ_Vector1 = NNZ_VECTOR_ThisTile1 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
    if (threadIdx.x < (NNZ_VECTOR_ThisTile1 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
        Num_NNZ_Vector1++;
    *NNZ_VECTOR_ThreadLocal1 = Num_NNZ_Vector1;
    if (TilingConfig::TILE_M == 256) {
        int Num_NNZ_Vector2 = NNZ_VECTOR_ThisTile2 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
        if (threadIdx.x < (NNZ_VECTOR_ThisTile2 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
            Num_NNZ_Vector2++;
        *NNZ_VECTOR_ThreadLocal2 = Num_NNZ_Vector2;
    }
    //
    int Max_NNZ_VECTOR_ThisTile =
        (TilingConfig::TILE_M == 256) ? max(NNZ_VECTOR_ThisTile1, NNZ_VECTOR_ThisTile2) : NNZ_VECTOR_ThisTile1;
#pragma unroll
    for (int i = 0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / SparseKernelConfig::VECTOR_SIZE; i++) {
        int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));
        if (index >= Max_NNZ_VECTOR_ThisTile)
            break;
        if (index < NNZ_VECTOR_ThisTile1
            || TilingConfig::TILE_M != 256)  // if TILE_M!=256, not need to compare since we have break();
        {
            Registers_GlobalToShared1[i * 4 + 0] = GlobalPTR1[index].x;
            Registers_GlobalToShared1[i * 4 + 1] = GlobalPTR1[index].y;
            Registers_GlobalToShared1[i * 4 + 2] = GlobalPTR1[index].z;
            Registers_GlobalToShared1[i * 4 + 3] = GlobalPTR1[index].w;
        }
        if (TilingConfig::TILE_M == 256)
            if (index < NNZ_VECTOR_ThisTile2) {
                Registers_GlobalToShared2[i * 4 + 0] = GlobalPTR2[index].x;
                Registers_GlobalToShared2[i * 4 + 1] = GlobalPTR2[index].y;
                Registers_GlobalToShared2[i * 4 + 2] = GlobalPTR2[index].z;
                Registers_GlobalToShared2[i * 4 + 3] = GlobalPTR2[index].w;
            }
    }
}

// Only used for kernel pipeline analysis, to make sure the global load for sparse encoding is not optimied by NVCC, we
// have to store the data loaded from GMem stored in SMem
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_CopyFromGlobalToShared(int          tid,
                                                            half*        smem,
                                                            uint32_t*    Registers_GlobalToShared1,
                                                            uint32_t*    NNZ_VECTOR_ThreadLocal1,
                                                            const uint4* GlobalPTR1,
                                                            int          NNZ_VECTOR_ThisTile1,
                                                            uint32_t*    Registers_GlobalToShared2,
                                                            uint32_t*    NNZ_VECTOR_ThreadLocal2,
                                                            const uint4* GlobalPTR2,
                                                            int          NNZ_VECTOR_ThisTile2)
{
    uint32_t*    smem_int_ptr = reinterpret_cast<uint32_t*>(smem);
    unsigned int tmp1         = 0;
    unsigned int tmp2         = 0;
    // Load Global to registers
    int Num_NNZ_Vector1 = NNZ_VECTOR_ThisTile1 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
    if (threadIdx.x < (NNZ_VECTOR_ThisTile1 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
        Num_NNZ_Vector1++;
    *NNZ_VECTOR_ThreadLocal1 = Num_NNZ_Vector1;
    if (TilingConfig::TILE_M == 256) {
        int Num_NNZ_Vector2 = NNZ_VECTOR_ThisTile2 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
        if (threadIdx.x < (NNZ_VECTOR_ThisTile2 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
            Num_NNZ_Vector2++;
        *NNZ_VECTOR_ThreadLocal2 = Num_NNZ_Vector2;
    }
    //
    int Max_NNZ_VECTOR_ThisTile =
        (TilingConfig::TILE_M == 256) ? max(NNZ_VECTOR_ThisTile1, NNZ_VECTOR_ThisTile2) : NNZ_VECTOR_ThisTile1;
#pragma unroll
    for (int i = 0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / SparseKernelConfig::VECTOR_SIZE; i++) {
        int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));
        if (index >= Max_NNZ_VECTOR_ThisTile)
            break;
        if (index < NNZ_VECTOR_ThisTile1
            || TilingConfig::TILE_M != 256)  // if TILE_M!=256, not need to compare since we have break();
        {
            tmp1 = GlobalPTR1[index].x + GlobalPTR1[index].y + GlobalPTR1[index].z + GlobalPTR1[index].w;
        }
        if (TilingConfig::TILE_M == 256)
            if (index < NNZ_VECTOR_ThisTile2) {
                tmp2 = GlobalPTR2[index].x + GlobalPTR2[index].y + GlobalPTR2[index].z + GlobalPTR2[index].w;
            }
    }
    smem_int_ptr[tid] = tmp1 + tmp2;
}

// Init Shared Memory to 0
template<typename TilingConfig>
__device__ __forceinline__ void SpMM_InitSharedMemory(half* __restrict__ SharedPTR)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    //
    static_assert(TilingConfig::TILE_M % TilingConfig::BLOCK_WARPS == 0,
                  "TILE_M must be an integer multiple to BLOCK_WARPS");
    constexpr int RowsPerWarp = TilingConfig::TILE_M / TilingConfig::BLOCK_WARPS;
    //
    static_assert(TILE_K == 64, "For now, TILE_K is assumed to be 64.\n");
    const int StartRowNum         = warp_id * RowsPerWarp;
    half*     SharedPTR_PerThread = SharedPTR + StartRowNum * TILE_K + HALF_PER_128B * lane_id;
    //
    static_assert(RowsPerWarp % (WARP_SIZE * HALF_PER_128B / TILE_K) == 0,
                  "RowsPerWarp%(WARP_SIZE*HALF_PER_128B/TILE_K) should be 0\n");
    constexpr int ITERATIONS_PER_THREAD = RowsPerWarp / (WARP_SIZE * HALF_PER_128B / TILE_K);
#pragma unroll
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        cp_async_ignore_src<16>(SharedPTR_PerThread, (half*)NULL);
        SharedPTR_PerThread += WARP_SIZE * HALF_PER_128B;
    }
}

template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_DecompressFromRegisterToShared(half* __restrict__ SharedPTR1,
                                                                    uint32_t* Registers_For_SparseTiles1,
                                                                    uint32_t  NNZ_ThreadLocal1,
                                                                    half* __restrict__ SharedPTR2,
                                                                    uint32_t* Registers_For_SparseTiles2,
                                                                    uint32_t  NNZ_ThreadLocal2)
{
    int Max_NNZ_ThreadLocal =
        (TilingConfig::TILE_M == 256) ? max(NNZ_ThreadLocal1, NNZ_ThreadLocal2) : NNZ_ThreadLocal1;
#pragma unroll
    for (int i = 0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / SparseKernelConfig::VECTOR_SIZE; i++) {
        if (i >= Max_NNZ_ThreadLocal)
            break;

        if (i < NNZ_ThreadLocal1
            || (TilingConfig::TILE_M != 256))  // if TILE_M!=256, not need to compare since we have break();
#pragma unroll
            for (int j = 0; j < SparseKernelConfig::VECTOR_SIZE; j++) {
                half* half_ptr =
                    reinterpret_cast<half*>(&(Registers_For_SparseTiles1[i * SparseKernelConfig::VECTOR_SIZE + j]));
                short* short_ptr  = reinterpret_cast<short*>(half_ptr + 1);
                half   value      = *half_ptr;
                short  index      = *short_ptr;
                SharedPTR1[index] = value;
            }

        if (TilingConfig::TILE_M == 256)
            if (i < NNZ_ThreadLocal2)
#pragma unroll
                for (int j = 0; j < SparseKernelConfig::VECTOR_SIZE; j++) {
                    half* half_ptr =
                        reinterpret_cast<half*>(&(Registers_For_SparseTiles2[i * SparseKernelConfig::VECTOR_SIZE + j]));
                    short* short_ptr  = reinterpret_cast<short*>(half_ptr + 1);
                    half   value      = *half_ptr;
                    short  index      = *short_ptr;
                    SharedPTR2[index] = value;
                }
    }
}

__device__ __forceinline__ half2 maskloadingv1(uint64_t bitmap, const half* __restrict__ startpos, int lane_id) {
    int lid_offset = lane_id << 1;
    uint64_t bit1 = 1ULL << lid_offset;
    uint64_t bit2 = 2ULL << lid_offset;

    // Calculate the number of ones before lane_id * 2
    int num_ones_before = __popcll(bitmap & ((1ULL << lid_offset) - 1));

    // Load A_val1 and adjust the offset for A_val2
    half A_val1 = (bitmap & bit1) ? startpos[num_ones_before++] : __float2half(0.0f);
    half A_val2 = (bitmap & bit2) ? startpos[num_ones_before] : __float2half(0.0f);

    // Combine two half values into a half2
    return __halves2half2(A_val1, A_val2);
}

// __device__ __forceinline__ void SpMM_LoadFragAwithBitmapFromShem(uint32_t __restrict__ a[][4],
//                                                          const half* __restrict__ ShemVal,
//                                                          const uint64_t* __restrict__ SharedBitmap,
//                                                          bool        Pred = true)
// {
//     int lane_id = threadIdx.x % 32;
//     int start_pos = 0;
//     if (Pred == true) {
//         #pragma unroll
//         for (int i = 0; i < 2; i++) {
//             #pragma unroll
//             for (int j = 0; j < 8; j++) {
//                 uint64_t bitmap = SharedBitmap[i * 8 + j];
//                 // if(threadIdx.x == 0){
//                 //     printf("i: %d, j: %d, startpos: %d \n", i,j,start_pos);
//                 // }
//                 // 根据mask1加载半精度值
//                 half2 val = maskloadingv1(bitmap, ShemVal+start_pos, lane_id);
//                 // 将加载的两个half值组合成half2，并存储到 a[][] 中
//                 a[(j/2)][(j%2)*2 + (i%2)] = *reinterpret_cast<const uint32_t*>(&val);
//                 // 计算mask1中位为1的数量，来确定start_pos2的位置
//                 start_pos += __popcll(bitmap);  // 计算mask1中位为1的数量
//             }
//         }
//     }
// }
__device__ __forceinline__ void SpMM_LoadFragAwithBitmapFromShem(uint32_t __restrict__ a[][4],
                                                         const half* __restrict__ ShemVal,
                                                         const uint64_t* __restrict__ SharedBitmap,
                                                         bool        Pred = true)
{
    int lane_id = threadIdx.x % 32;
    int start_pos = 0;
    if (Pred == true) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t bitmap = SharedBitmap[i * 4 + j];
                // if(threadIdx.x == 0){
                //     printf("i: %d, j: %d, startpos: %d \n", i,j,start_pos);
                // }
                // 根据mask1加载半精度值
                half2 val = maskloadingv1(bitmap, ShemVal+start_pos, lane_id);
                // 将加载的两个half值组合成half2，并存储到 a[][] 中
                a[i][j] = *reinterpret_cast<const uint32_t*>(&val);
                // 计算mask1中位为1的数量，来确定start_pos2的位置
                start_pos += __popcll(bitmap);  // 计算mask1中位为1的数量
            }
        }
    }
}

// template<typename TilingConfig>
// __global__ void SpMM_Kernel_bitmap_v1(const half*  A,
//                             const half* Compressed_A,
//                             const int*   TileOffsets,
//                             const uint64_t*   bitmap,
//                             int max_nnz_intile,
//                             const half*  B,
//                             half*        Reduction_Workspace,
//                             const int    M_Global,
//                             const int    N_Global,
//                             const int    K_Global,
//                             int          Split_K)
// {
//     const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);
//     const int IsLastBatch = (BatchID == (Split_K - 1));
//     const int x           = blockIdx.x;
//     const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);
//     //
//     const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
//     const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
//     const int RoundedKBlock    = AverageNumKBlock * Split_K;
//     const int PaddingKBlock    = RoundedKBlock - NumKBlock;
//     int       NumIter          = 0;
//     if (IsLastBatch)
//         NumIter = AverageNumKBlock - PaddingKBlock;
//     else
//         NumIter = AverageNumKBlock;
//     const int* TileOffsets_ThisBlock = nullptr;
//     TileOffsets_ThisBlock = TileOffsets + K_Global / TILE_K * y + BatchID * AverageNumKBlock;
//     int NNZ_ThisTile = TileOffsets_ThisBlock[1] - TileOffsets_ThisBlock[0];
//     // if (threadIdx.x % 32 == 0){
//     //     printf("blockIdx.y: %d, TileOffsets_ThisBlock: %d, nnz_this_block: %d \n", blockIdx.y, TileOffsets_ThisBlock[0], TileOffsets_ThisBlock[1]-TileOffsets_ThisBlock[0]);
//     // }
// ////////
//     extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
//     uint64_t* smem_Bitmap = reinterpret_cast<uint64_t*>(&smem[max_nnz_intile+(TILE_K * TilingConfig::TILE_N)*2]);
//     half* smem_B = &smem[max_nnz_intile];
// // Warp and lane identification.
//     const unsigned int warpId       = threadIdx.x / WARP_SIZE;
//     const int          Tile_Start_M = y * TilingConfig::TILE_M;
//     const int          Tile_Start_Bitmap = y * TilingConfig::TILE_BITMAP_M_V1;
//     const int          Tile_Start_N = x * TilingConfig::TILE_N;
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // Compute a grid of C matrix tiles in each warp.
//     int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
//     int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
//     int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
//     int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
//     uint32_t __restrict__ a[WARP_ROW_TENSORS * BLOCK_K_TENSORS][4]; //4*4 or 1*4个mma指令的寄存器全部加载（64 or 16个寄存器）
//     // uint32_t __restrict__ a[WARP_ROW_TENSORS][4]; //4*4个mma指令的寄存器全部加载（64个寄存器）
//     uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
      
// // gld addr of copying B tile from GlobalMemory to SharedMemory
//     const half* BTileGlobalPTR =
//         B + Tile_Start_N * K_Global
//         + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
// // gld addr of  copying Bitmap tile from GlobalMemory to SharedMemory
//     const uint64_t* BitmapTileGlobalPTR =
//         bitmap + Tile_Start_Bitmap * (K_Global/4)
//         + BatchID * AverageNumKBlock * TILE_BITMAP_K_V1;  // Address for matrix bitmap, taking SplitK into consideration

// // 将1*16的bitmap加载在double buffer B shared tile之后
//     CopyTileFromGlobalToShared_X_16_1<TilingConfig::TILE_BITMAP_M_V1, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR, K_Global/4);  // 将2*8的bitmap加载在double buffer B shared tile之后
//     CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + TileOffsets_ThisBlock[0], NNZ_ThisTile);
//     cp_async_group_commit();
// // 加载B到shared mem   
//     CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(smem_B, BTileGlobalPTR, K_Global); //将B加载到shared mem
//     cp_async_group_commit();
    
// // Initilazing C Matrix to Zeros.
//     float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
//     for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
//         for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
//             c[i][j] = 0.0f;
// // Waiting for A to complete.
//     cp_async_wait_group<1>(); //bitmap加载完成
//     __syncthreads();
// // Logging 
//     // if (threadIdx.x == 0){
//     //     for (int j = 0; j < 16; j++){
//     //         printf("blockidy: %d, i: %d, bitmap val: %"PRIu64"\n", blockIdx.y, j, smem_Bitmap[j]);
//     //     }
//     // }
//     // if (threadIdx.x == 0){
//     //     for (int j = 0; j < max_nnz_intile; j++){
//     //        printf("i: %d, A val: %f ",  j, __half2float(smem[j]));
//     //     }
//     //     printf("\n");
//     // }
// // loading A to reg from shared mem with bitmap.
//     SpMM_LoadFragAwithBitmapFromShem(a, smem, smem_Bitmap);
// // waiting for B to complete
//     cp_async_wait_group<0>(); // B加载完成
//     __syncthreads();
//     // if (threadIdx.x == 0){
//     //     for (int j = 0; j < 64*8; j++){
//     //         printf("blockidy: %d, i: %d, B val: %f \n", blockIdx.y, j, __half2float(smem_B[j]));
//     //     }
//     // }
// //Prefetch
//     int StartIndex_SparseTiles_Prefetch = TileOffsets_ThisBlock[0 + 1];
//     int NNZ_ThisTile_Prefetch           = TileOffsets_ThisBlock[0 + 2] - TileOffsets_ThisBlock[0 + 1];

// // Go through the global K dimension by a fixed step at a time.
// // write buffer[1] first, read buffer[0] first
// #pragma unroll(1)
//     for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
//         //
//         int StartIndex_SparseTiles = StartIndex_SparseTiles_Prefetch;
//         int NNZ_ThisTile          = NNZ_ThisTile_Prefetch;
//         StartIndex_SparseTiles_Prefetch = TileOffsets_ThisBlock[tile_id_k + 1 + 1];
//         NNZ_ThisTile_Prefetch = TileOffsets_ThisBlock[tile_id_k + 1 + 2] - TileOffsets_ThisBlock[tile_id_k + 1 + 1];
//         // if (threadIdx.x == 0){
//         //     printf("blockidy: %d, NNZ_ThisTile: %d \n", blockIdx.y, NNZ_ThisTile);
//         // }
//         // copying A&B tile from GlobalMemory to SharedMemory (next tile)
//         BTileGlobalPTR = BTileGlobalPTR + TILE_K;
//         BitmapTileGlobalPTR = BitmapTileGlobalPTR + TILE_BITMAP_K_V1; 
        
//         // double buffer
//         half* __restrict__ smem_write_B_PTR = smem_B;
//         half* __restrict__ smem_read_B_PTR  = smem_B;
//         smem_write_B_PTR = smem_B + ((tile_id_k + 1) % 2) * (TILE_K * TilingConfig::TILE_N); //当前B写入的地址
//         smem_read_B_PTR  = smem_B + ((tile_id_k) % 2) * (TILE_K * TilingConfig::TILE_N); //当前读取B的地址

//         // COPY indicator
//         bool GlobalCopy = (tile_id_k + 1) < NumIter;
        
//         // Copying next Bitmap Tile to write shem
//         CopyTileFromGlobalToShared_X_16_1<TilingConfig::TILE_BITMAP_M_V1, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR, K_Global/4, GlobalCopy);  // 将2*8的bitmap加载在double buffer B shared tile之后
//         // Copying next Sparse A Tile to write shem
//         CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + StartIndex_SparseTiles, NNZ_ThisTile, GlobalCopy);
//         cp_async_group_commit();

//         // Copying next B Tile to write shem
//         CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
//             smem_write_B_PTR, BTileGlobalPTR, K_Global, GlobalCopy);
//         cp_async_group_commit();

//         PipelinedCoreComputationsBitmap<TilingConfig>(c, a, b, smem_read_B_PTR, warp_start_row, warp_start_col);
       
//         cp_async_wait_group<1>();
//         __syncthreads();
//         // loading next A from shared to reg a. This can be concurrent with loading next B to shem
//         SpMM_LoadFragAwithBitmapFromShem(a, smem, smem_Bitmap, GlobalCopy);

//         cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
//         __syncthreads();
//     }
//     /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     // Store the C fragments to shared memory.
//     float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
//         reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
//     StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
//     __syncthreads();
//     // Now that shared memory contains all the D tiles, stream them to global memory.
//     half* BlockGlobalPTR =
//         Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
// #pragma unroll
//     for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
// #pragma unroll
//         for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
//             BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
// }



template<typename TilingConfig>
__global__ void SpMM_Kernel_bitmap_v1(const half*  A,
                            const half* Compressed_A,
                            const int*   TileOffsets,
                            const uint64_t*   bitmap,
                            int max_nnz_intile,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K)
{
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    const int* TileOffsets_ThisBlock = nullptr;
    TileOffsets_ThisBlock = TileOffsets + K_Global / TILE_K * y + BatchID * AverageNumKBlock;
    int NNZ_ThisTile = TileOffsets_ThisBlock[1] - TileOffsets_ThisBlock[0];
    // if (threadIdx.x % 32 == 0){
    //     printf("blockIdx.y: %d, TileOffsets_ThisBlock: %d, nnz_this_block: %d \n", blockIdx.y, TileOffsets_ThisBlock[0], TileOffsets_ThisBlock[1]-TileOffsets_ThisBlock[0]);
    // }
////////
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
    uint64_t* smem_Bitmap = reinterpret_cast<uint64_t*>(&smem[max_nnz_intile+(TILE_K * TilingConfig::TILE_N)*2]);
    half* smem_B = &smem[max_nnz_intile];
// Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_Bitmap = y * TilingConfig::TILE_BITMAP_M_V1;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS_BITMAP_V1 * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    uint32_t __restrict__ a[WARP_ROW_TENSORS_BITMAP_V1 * BLOCK_K_TENSORS][4];
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
      
// gld addr of copying B tile from GlobalMemory to SharedMemory
    const half* BTileGlobalPTR =
        B + Tile_Start_N * K_Global
        + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
// gld addr of  copying Bitmap tile from GlobalMemory to SharedMemory
    const uint64_t* BitmapTileGlobalPTR =
        bitmap + Tile_Start_Bitmap * (K_Global/4)
        + BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V1;  // Address for matrix bitmap, taking SplitK into consideration

// 将1*16的bitmap加载在double buffer B shared tile之后
    CopyTileFromGlobalToShared_X_16_1<TilingConfig::TILE_BITMAP_M_V1, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR, K_Global/4);  // 将2*8的bitmap加载在double buffer B shared tile之后
    CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + TileOffsets_ThisBlock[0], NNZ_ThisTile);
    cp_async_group_commit();
// 加载B到shared mem   
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(smem_B, BTileGlobalPTR, K_Global); //将B加载到shared mem
    cp_async_group_commit();
    
// Initilazing C Matrix to Zeros.
    float c[WARP_ROW_TENSORS_BITMAP_V1 * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V1 * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
// Waiting for A to complete.
    cp_async_wait_group<1>(); //bitmap加载完成
    __syncthreads();
// Logging 
    // if (threadIdx.x == 0){
    //     for (int j = 0; j < 16; j++){
    //         printf("blockidy: %d, i: %d, bitmap val: %"PRIu64"\n", blockIdx.y, j, smem_Bitmap[j]);
    //     }
    // }
    // if (threadIdx.x == 0){
    //     for (int j = 0; j < max_nnz_intile; j++){
    //        printf("i: %d, A val: %f ",  j, __half2float(smem[j]));
    //     }
    //     printf("\n");
    // }
// loading A to reg from shared mem with bitmap.
    SpMM_LoadFragAwithBitmapFromShem(a, smem, smem_Bitmap);
// waiting for B to complete
    cp_async_wait_group<0>(); // B加载完成
    __syncthreads();
    // if (threadIdx.x == 0){
    //     for (int j = 0; j < 64*8; j++){
    //         printf("blockidy: %d, i: %d, B val: %f \n", blockIdx.y, j, __half2float(smem_B[j]));
    //     }
    // }
//Prefetch
    int StartIndex_SparseTiles_Prefetch = TileOffsets_ThisBlock[0 + 1];
    int NNZ_ThisTile_Prefetch           = TileOffsets_ThisBlock[0 + 2] - TileOffsets_ThisBlock[0 + 1];

// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first
#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        //
        int StartIndex_SparseTiles = StartIndex_SparseTiles_Prefetch;
        int NNZ_ThisTile          = NNZ_ThisTile_Prefetch;
        StartIndex_SparseTiles_Prefetch = TileOffsets_ThisBlock[tile_id_k + 1 + 1];
        NNZ_ThisTile_Prefetch = TileOffsets_ThisBlock[tile_id_k + 1 + 2] - TileOffsets_ThisBlock[tile_id_k + 1 + 1];
        // if (threadIdx.x == 0){
        //     printf("blockidy: %d, NNZ_ThisTile: %d \n", blockIdx.y, NNZ_ThisTile);
        // }
        // copying A&B tile from GlobalMemory to SharedMemory (next tile)
        BTileGlobalPTR = BTileGlobalPTR + TILE_K;
        BitmapTileGlobalPTR = BitmapTileGlobalPTR + TilingConfig::TILE_BITMAP_K_V1; 
        
        // double buffer
        half* __restrict__ smem_write_B_PTR = smem_B;
        half* __restrict__ smem_read_B_PTR  = smem_B;
        smem_write_B_PTR = smem_B + ((tile_id_k + 1) % 2) * (TILE_K * TilingConfig::TILE_N); //当前B写入的地址
        smem_read_B_PTR  = smem_B + ((tile_id_k) % 2) * (TILE_K * TilingConfig::TILE_N); //当前读取B的地址

        // COPY indicator
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        
        // Copying next Bitmap Tile to write shem
        CopyTileFromGlobalToShared_X_16_1<TilingConfig::TILE_BITMAP_M_V1, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR, K_Global/4, GlobalCopy);  // 将2*8的bitmap加载在double buffer B shared tile之后
        // Copying next Sparse A Tile to write shem
        CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + StartIndex_SparseTiles, NNZ_ThisTile, GlobalCopy);
        cp_async_group_commit();

        // Copying next B Tile to write shem
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_B_PTR, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();

        PipelinedCoreComputationsBitmap<TilingConfig>(c, a, b, smem_read_B_PTR, warp_start_row, warp_start_col);
       
        cp_async_wait_group<1>();
        __syncthreads();
        // loading next A from shared to reg a. This can be concurrent with loading next B to shem
        SpMM_LoadFragAwithBitmapFromShem(a, smem, smem_Bitmap, GlobalCopy);

        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegisterBitmapV1<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}



template<typename TilingConfig>
__global__ void SpMM_Kernel_bitmap_v2(const half*  A,
                            const half* Compressed_A,
                            const int*   TileOffsets,
                            const uint64_t*   bitmap,
                            int max_nnz_intile,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K)
{
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    //
    // if(threadIdx.x == 0) {
    //     printf("TILE_M: %d, TILE_BITMAP_M_V2:%d \n", TilingConfig::TILE_M, TilingConfig::TILE_BITMAP_M_V2);
    // }
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    const int* TileOffsets_ThisBlock = nullptr;
    TileOffsets_ThisBlock = TileOffsets + K_Global / TILE_K * y + BatchID * AverageNumKBlock;
    int NNZ_ThisTile = TileOffsets_ThisBlock[1] - TileOffsets_ThisBlock[0];
    // if (threadIdx.x % 32 == 0){
    //     printf("blockIdx.y: %d, TileOffsets_ThisBlock: %d, nnz_this_block: %d \n", blockIdx.y, TileOffsets_ThisBlock[0], TileOffsets_ThisBlock[1]-TileOffsets_ThisBlock[0]);
    // }
////////
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
    uint64_t* smem_Bitmap = reinterpret_cast<uint64_t*>(&smem[max_nnz_intile*2+(TILE_K * TilingConfig::TILE_N)*2]);
    half* smem_B = &smem[max_nnz_intile*2];
// Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_Bitmap = y * TilingConfig::TILE_BITMAP_M_V2;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS_BITMAP_V2 * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    uint32_t __restrict__ a[WARP_ROW_TENSORS_BITMAP_V2 * 2][4];
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
      
// gld addr of copying B tile from GlobalMemory to SharedMemory
    const half* BTileGlobalPTR =
        B + Tile_Start_N * K_Global
        + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
// gld addr of  copying Bitmap tile from GlobalMemory to SharedMemory
    const uint64_t* BitmapTileGlobalPTR =
        bitmap + Tile_Start_Bitmap * K_Global
        + BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V2;  // Address for matrix bitmap, taking SplitK into consideration

// 将1*16的bitmap加载在double buffer B shared tile之后
    CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V2, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR);  // 将2*8的bitmap加载在double buffer B shared tile之后
    CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + TileOffsets_ThisBlock[0], NNZ_ThisTile);
    // cp_async_group_commit();
// 加载B到shared mem   
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(smem_B, BTileGlobalPTR, K_Global); //将B加载到shared mem
    cp_async_group_commit();
    
// Initilazing C Matrix to Zeros.
    float c[WARP_ROW_TENSORS_BITMAP_V2 * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V2 * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
// Waiting for A to complete.
    // cp_async_wait_group<1>(); //bitmap加载完成
    // __syncthreads();

// loading A to reg from shared mem with bitmap.
    // SpMM_LoadFragAwithBitmapFromShem(a, smem, smem_Bitmap);
// waiting for B to complete
    cp_async_wait_group<0>(); // B加载完成
    __syncthreads();
    // if (threadIdx.x == 0){
    //     for (int j = 0; j < 64*8; j++){
    //         printf("blockidy: %d, i: %d, B val: %f \n", blockIdx.y, j, __half2float(smem_B[j]));
    //     }
    // }
    // Logging 
    // if (threadIdx.x == 0){
    //     for (int j = 0; j < 64; j++){
    //         printf("blockidy: %d, i: %d, bitmap val: %"PRIu64"\n", blockIdx.y, j, smem_Bitmap[j]);
    //     }
    // }
    // if (threadIdx.x == 0){
    //     for (int j = 0; j < max_nnz_intile; j++){
    //        printf("i: %d, A val: %f ",  j, __half2float(smem[j]));
    //     }
    //     printf("\n");
    // }
//Prefetch
    int StartIndex_SparseTiles_Prefetch = TileOffsets_ThisBlock[0 + 1];
    int NNZ_ThisTile_Prefetch           = TileOffsets_ThisBlock[0 + 2] - TileOffsets_ThisBlock[0 + 1];

// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first
#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        //
        int StartIndex_SparseTiles = StartIndex_SparseTiles_Prefetch;
        int NNZ_ThisTile          = NNZ_ThisTile_Prefetch;
        StartIndex_SparseTiles_Prefetch = TileOffsets_ThisBlock[tile_id_k + 1 + 1];
        NNZ_ThisTile_Prefetch = TileOffsets_ThisBlock[tile_id_k + 1 + 2] - TileOffsets_ThisBlock[tile_id_k + 1 + 1];
        // if (threadIdx.x == 0){
        //     printf("blockidy: %d, NNZ_ThisTile: %d \n", blockIdx.y, NNZ_ThisTile);
        // }
        // copying A&B tile from GlobalMemory to SharedMemory (next tile)
        BTileGlobalPTR = BTileGlobalPTR + TILE_K;
        BitmapTileGlobalPTR = BitmapTileGlobalPTR + TilingConfig::TILE_BITMAP_K_V2; 
        
        // double buffer
        half* __restrict__ smem_write_B_PTR = smem_B;
        half* __restrict__ smem_read_B_PTR  = smem_B;
        smem_write_B_PTR = smem_B + ((tile_id_k + 1) % 2) * (TILE_K * TilingConfig::TILE_N); //当前B写入的地址
        smem_read_B_PTR  = smem_B + ((tile_id_k) % 2) * (TILE_K * TilingConfig::TILE_N); //当前读取B的地址

        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (max_nnz_intile); //当前B写入的地址
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (max_nnz_intile); //当前读取B的地址


        uint64_t* __restrict__ smem_write_Bitmap_PTR = smem_Bitmap;
        uint64_t* __restrict__ smem_read_Bitmap_PTR  = smem_Bitmap;
        smem_write_Bitmap_PTR = smem_Bitmap + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_BITMAP_K_V2 * TilingConfig::TILE_BITMAP_M_V2); //当前B写入的地址
        smem_read_Bitmap_PTR  = smem_Bitmap + ((tile_id_k) % 2) * (TilingConfig::TILE_BITMAP_K_V2 * TilingConfig::TILE_BITMAP_M_V2); //当前读取B的地址


        // COPY indicator
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        
        // Copying next Bitmap Tile to write shem
        CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V2, TilingConfig>(smem_write_Bitmap_PTR, BitmapTileGlobalPTR, GlobalCopy);  // 将2*8的bitmap加载在double buffer B shared tile之后
        // Copying next Sparse A Tile to write shem
        CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem_write_PTR, Compressed_A + StartIndex_SparseTiles, NNZ_ThisTile, GlobalCopy);
        // cp_async_group_commit();

        // Copying next B Tile to write shem
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_B_PTR, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();


            // Logging 
            // if (threadIdx.x == 0){
            //     for (int j = 0; j < 64*8; j++){
            //         printf("blockidy: %d, i: %d, B val 111: %f \n", blockIdx.y, j, __half2float(smem_read_B_PTR[j]));
            //     }
            // }
        PipelinedCoreComputationsBitmapV2<TilingConfig>(c, a, b, smem_read_PTR, smem_read_Bitmap_PTR, smem_read_B_PTR, warp_start_row, warp_start_col);
       
        // cp_async_wait_group<1>();
        // __syncthreads();
        // loading next A from shared to reg a. This can be concurrent with loading next B to shem
        // SpMM_LoadFragAwithBitmapFromShem(a, smem_write_PTR, smem_write_Bitmap_PTR, GlobalCopy);

        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegisterBitmapV2<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}


template<typename TilingConfig>
__global__ void SpMM_Kernel_bitmap_v3(const half*  A,
                            const half* Compressed_A,
                            const int*   TileOffsets,
                            const int*   TileOffsets_Median,
                            const uint64_t*   bitmap,
                            const int* ptr_max_nnz_intile,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K)
{
    // if (blockIdx.x==0&&blockIdx.y==0&& threadIdx.x%32==0){
    //     printf("bid=%d, tid=%d\n", blockIdx.x, threadIdx.x);
    // }
    int max_nnz_intile=*ptr_max_nnz_intile;
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    const int* TileOffsets_ThisBlock = nullptr;
    const int BlockOffset = K_Global / TILE_K * y + BatchID * AverageNumKBlock;
    TileOffsets_ThisBlock = TileOffsets + BlockOffset;
    int NNZ_ThisTile = TileOffsets_ThisBlock[1] - TileOffsets_ThisBlock[0];

    // *************************
    // if (threadIdx.x % 32 == 0){
    //     printf("blockIdx.y: %d, TileOffsets_ThisBlock: %d, nnz_this_block: %d \n", blockIdx.y, TileOffsets_ThisBlock[0], TileOffsets_ThisBlock[1]-TileOffsets_ThisBlock[0]);
    // }
    // *************************
////////
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
    uint64_t* smem_Bitmap = reinterpret_cast<uint64_t*>(&smem[max_nnz_intile+(TILE_K * TilingConfig::TILE_N)*2]);
    half* smem_B = &smem[max_nnz_intile];
// Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_Bitmap = y * TilingConfig::TILE_BITMAP_M_V3;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS_BITMAP_V3 * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    uint32_t __restrict__ a[WARP_ROW_TENSORS_BITMAP_V3 * BLOCK_K_TENSORS][4];
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
    uint64_t* smem_BitmapWarp = smem_Bitmap + Warp_i*16;
    const int* TileOffsets_ThisWarp = nullptr;
    const int WarpOffset = BlockOffset*4 + Warp_i;
    TileOffsets_ThisWarp = TileOffsets_Median + WarpOffset;
// gld addr of copying B tile from GlobalMemory to SharedMemory
    const half* BTileGlobalPTR =
        B + Tile_Start_N * K_Global
        + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
// gld addr of  copying Bitmap tile from GlobalMemory to SharedMemory
    const uint64_t* BitmapTileGlobalPTR =
        bitmap + Tile_Start_Bitmap * K_Global
        + BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V3;  // Address for matrix bitmap, taking SplitK into consideration

// 将1*16的bitmap加载在double buffer B shared tile之后
    CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR);  // 将1*64的bitmap加载在double buffer B shared tile之后
    CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + TileOffsets_ThisBlock[0], NNZ_ThisTile);
    cp_async_group_commit();
// 加载B到shared mem   
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(smem_B, BTileGlobalPTR, K_Global); //将B加载到shared mem
    cp_async_group_commit();
    
// Initilazing C Matrix to Zeros.
    float c[WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
// Waiting for A to complete.
    cp_async_wait_group<1>(); //bitmap加载完成
    __syncthreads();
// Logging 
    // *************************
    // if (threadIdx.x == 0){
    //     for (int j = 0; j < 16; j++){
    //         printf("blockidy: %d, i: %d, bitmap val: %"PRIu64"\n", blockIdx.y, j, smem_Bitmap[j]);
    //     }
    // }
    // *************************
    // if (threadIdx.x == 0){
    //     for (int j = 0; j < max_nnz_intile; j++){
    //        printf("i: %d, A val: %f ",  j, __half2float(smem[j]));
    //     }
    //     printf("\n");
    // }
// loading A to reg from shared mem with bitmap.
    SpMM_LoadFragAwithBitmapFromShem(a, smem + TileOffsets_ThisWarp[0], smem_BitmapWarp);
// waiting for B to complete
    cp_async_wait_group<0>(); // B加载完成
    __syncthreads();
    // if (threadIdx.x == 0){
    //     for (int j = 0; j < 64*8; j++){
    //         printf("blockidy: %d, i: %d, B val: %f \n", blockIdx.y, j, __half2float(smem_B[j]));
    //     }
    // }
//Prefetch
    int StartIndex_SparseTiles_Prefetch = TileOffsets_ThisBlock[0 + 1];
    int NNZ_ThisTile_Prefetch           = TileOffsets_ThisBlock[0 + 2] - TileOffsets_ThisBlock[0 + 1];

// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first
#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        //
        int StartIndex_SparseTiles = StartIndex_SparseTiles_Prefetch;
        int NNZ_ThisTile          = NNZ_ThisTile_Prefetch;
        StartIndex_SparseTiles_Prefetch = TileOffsets_ThisBlock[tile_id_k + 1 + 1];
        NNZ_ThisTile_Prefetch = TileOffsets_ThisBlock[tile_id_k + 1 + 2] - TileOffsets_ThisBlock[tile_id_k + 1 + 1];
        // if (threadIdx.x == 0){
        //     printf("blockidy: %d, NNZ_ThisTile: %d \n", blockIdx.y, NNZ_ThisTile);
        // }
        // copying A&B tile from GlobalMemory to SharedMemory (next tile)
        BTileGlobalPTR = BTileGlobalPTR + TILE_K;
        BitmapTileGlobalPTR = BitmapTileGlobalPTR + TilingConfig::TILE_BITMAP_K_V3; 
        
        // double buffer
        half* __restrict__ smem_write_B_PTR = smem_B;
        half* __restrict__ smem_read_B_PTR  = smem_B;
        smem_write_B_PTR = smem_B + ((tile_id_k + 1) % 2) * (TILE_K * TilingConfig::TILE_N); //当前B写入的地址
        smem_read_B_PTR  = smem_B + ((tile_id_k) % 2) * (TILE_K * TilingConfig::TILE_N); //当前读取B的地址

        // COPY indicator
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        
        // Copying next Bitmap Tile to write shem
        CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR, GlobalCopy);  // 将2*8的bitmap加载在double buffer B shared tile之后
        // Copying next Sparse A Tile to write shem
        CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + StartIndex_SparseTiles, NNZ_ThisTile, GlobalCopy);
        cp_async_group_commit();

        // Copying next B Tile to write shem
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_B_PTR, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();

        PipelinedCoreComputationsBitmap<TilingConfig>(c, a, b, smem_read_B_PTR, warp_start_row, warp_start_col);
       
        cp_async_wait_group<1>();
        __syncthreads();
        // loading next A from shared to reg a. This can be concurrent with loading next B to shem
        SpMM_LoadFragAwithBitmapFromShem(a, smem + TileOffsets_ThisWarp[(tile_id_k+1)*4], smem_BitmapWarp, GlobalCopy);

        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegisterBitmapV3<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
    
}



template<typename TilingConfig, typename SparseKernelConfig>
__global__ void SpMM_Kernel(const half*  A,
                            const uint4* Compressed_A,
                            const int*   TileOffsets,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K)
{
    //
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    //
    const int* TileOffsets_ThisBlock1 = nullptr;
    const int* TileOffsets_ThisBlock2 = nullptr;
    if (TilingConfig::TILE_M == 256) {
        TileOffsets_ThisBlock1 =
            TileOffsets + K_Global / TILE_K * y * 2
            + BatchID * AverageNumKBlock;  // Address for matrix A, taking SplitK into consideration
        TileOffsets_ThisBlock2 =
            TileOffsets + K_Global / TILE_K * (y * 2 + 1)
            + BatchID * AverageNumKBlock;  // Address for matrix A, taking SplitK into consideration
    }
    else {
        TileOffsets_ThisBlock1 = TileOffsets + K_Global / TILE_K * y + BatchID * AverageNumKBlock;
        TileOffsets_ThisBlock2 = TileOffsets_ThisBlock1;  // otherwise will cause problem when passing
                                                          // TileOffsets_ThisBlock2[0] to SpMM_CopyFromGlobalToReg()
    }
    //
    uint32_t Registers_GlobalToShared[SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL];
    uint32_t NNZ_ThreadLocal1 = 0;
    uint32_t NNZ_ThreadLocal2 = 0;
    //
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
    // Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4];
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
    // copying B tile from GlobalMemory to SharedMemory
    const half* BTileGlobalPTR =
        B + Tile_Start_N * K_Global
        + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
    //
    int NNZ_ThisTile1 = TileOffsets_ThisBlock1[1] - TileOffsets_ThisBlock1[0];
    int NNZ_ThisTile2 = 0;
    if (TilingConfig::TILE_M == 256)
        NNZ_ThisTile2 = TileOffsets_ThisBlock2[1] - TileOffsets_ThisBlock2[0];
    // printf("NNZ_ThisTile: %d ", NNZ_ThisTile);
    //
    SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_GlobalToShared,
                                                               &NNZ_ThreadLocal1,
                                                               Compressed_A + TileOffsets_ThisBlock1[0],
                                                               NNZ_ThisTile1,
                                                               Registers_GlobalToShared
                                                                   + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
                                                               &NNZ_ThreadLocal2,
                                                               Compressed_A + TileOffsets_ThisBlock2[0],
                                                               NNZ_ThisTile2);
    SpMM_InitSharedMemory<TilingConfig>(smem);
    cp_async_group_commit();
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global);
    cp_async_group_commit();
    // Initilazing C Matrix to Zeros
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    //
    cp_async_wait_group<1>();
    __syncthreads();
    SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
        smem,
        Registers_GlobalToShared,
        NNZ_ThreadLocal1,
        smem + TilingConfig::TILE_M * TILE_K / 2,
        Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
        NNZ_ThreadLocal2);
    //
    cp_async_wait_group<0>();
    __syncthreads();
    // Prefetch to reduce stall_long_sb
    int StartIndex_SparseTiles_Prefetch1 = TileOffsets_ThisBlock1[0 + 1];
    int NNZ_ThisTile_Prefetch1           = TileOffsets_ThisBlock1[0 + 2] - TileOffsets_ThisBlock1[0 + 1];
    int StartIndex_SparseTiles_Prefetch2 = 0;
    int NNZ_ThisTile_Prefetch2           = 0;
    if (TilingConfig::TILE_M == 256) {
        StartIndex_SparseTiles_Prefetch2 = TileOffsets_ThisBlock2[0 + 1];
        NNZ_ThisTile_Prefetch2           = TileOffsets_ThisBlock2[0 + 2] - TileOffsets_ThisBlock2[0 + 1];
    }
// Debug
// printf("NNZ_ThisTile_Prefetch: %d ", NNZ_ThisTile_Prefetch);
//
// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first
#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        // Using the previous prefetched value
        int StartIndex_SparseTiles1 = StartIndex_SparseTiles_Prefetch1;
        int NNZ_ThisTile1           = NNZ_ThisTile_Prefetch1;
        int StartIndex_SparseTiles2 = 0;
        int NNZ_ThisTile2           = 0;
        if (TilingConfig::TILE_M == 256) {
            StartIndex_SparseTiles2 = StartIndex_SparseTiles_Prefetch2;
            NNZ_ThisTile2           = NNZ_ThisTile_Prefetch2;
        }
        //
        StartIndex_SparseTiles_Prefetch1 = TileOffsets_ThisBlock1[tile_id_k + 1 + 1];
        NNZ_ThisTile_Prefetch1 = TileOffsets_ThisBlock1[tile_id_k + 1 + 2] - TileOffsets_ThisBlock1[tile_id_k + 1 + 1];
        if (TilingConfig::TILE_M == 256) {
            StartIndex_SparseTiles_Prefetch2 = TileOffsets_ThisBlock2[tile_id_k + 1 + 1];
            NNZ_ThisTile_Prefetch2 =
                TileOffsets_ThisBlock2[tile_id_k + 1 + 2] - TileOffsets_ThisBlock2[tile_id_k + 1 + 1];
        }
        // copying B tile from GlobalMemory to SharedMemory
        BTileGlobalPTR = B + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K);
        // double buffer
        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter;

        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR);
        cp_async_group_commit();
        SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(
            Registers_GlobalToShared,
            &NNZ_ThreadLocal1,
            Compressed_A + StartIndex_SparseTiles1,
            NNZ_ThisTile1,
            Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
            &NNZ_ThreadLocal2,
            Compressed_A + StartIndex_SparseTiles2,
            NNZ_ThisTile2);

        // Copying B Tile
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();

        // only used for kernel pipeline analysis
        // SpMM_CopyFromGlobalToShared<TilingConfig, SparseKernelConfig>
        //               ( threadIdx.x,
        //                 smem_write_PTR,
        //                 Registers_GlobalToShared,
        //                 &NNZ_ThreadLocal1,
        //                 Compressed_A+StartIndex_SparseTiles1,
        //                 NNZ_ThisTile1,
        //                 Registers_GlobalToShared+SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL/2,
        //                 &NNZ_ThreadLocal2,
        //                 Compressed_A+StartIndex_SparseTiles2,
        //                 NNZ_ThisTile2);

        PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col);
        //

        cp_async_wait_group<1>();
        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet
        SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
            smem_write_PTR,
            Registers_GlobalToShared,
            NNZ_ThreadLocal1,
            smem_write_PTR + TilingConfig::TILE_M * TILE_K / 2,
            Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
            NNZ_ThreadLocal2);
        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}







__device__ __forceinline__ half maskloading(uint32_t mask, const half* __restrict__ startpos, int lane_id)
{
    // 计算当前线程的 lane_id 之前的位数中有多少位为1
    uint32_t mask_shifted = mask & ((1U << lane_id) - 1);  // 仅保留 lane_id 之前的位
    int num_ones_before = __popc(mask_shifted);  // 计算位为1的数量

    // 如果当前线程对应的位为1，加载数据，否则加载0
    if (mask & (1U << lane_id)) {
        return startpos[num_ones_before];  // 读取实际数据，按照之前1的数量偏移
    } else {
        return __float2half(0.0f);  // 如果不需要加载，则返回0
    }
}
// __device__ __forceinline__ half maskloading(uint32_t mask, const half* __restrict__ startpos, int lane_id)
// {
//     // 计算当前线程的 lane_id 之前的位数中有多少位为1
//     uint32_t mask_shifted = mask & ((1U << lane_id) - 1);  // 仅保留 lane_id 之前的位
//     int num_ones_before = __popc(mask_shifted);  // 计算位为1的数量

//     // 如果当前线程对应的位为1，加载数据，否则加载0
//     if (mask & (1U << lane_id)) {
//         // return startpos[lane_id/2];  // 读取实际数据，按照之前1的数量偏移
//         return startpos[num_ones_before];
//         // return __float2half(float(num_ones_before));  // 读取实际数据，按照之前1的数量偏移
//     } else {
//         return __float2half(0.0f);  // 如果不需要加载，则返回0
//     }
// }
__device__ __forceinline__ void SpMM_LoadFragAwithBitmap(uint32_t __restrict__ a[][4],
                                                         const int*   __restrict__ TileOffsetsWarp,
                                                         const half* __restrict__ Compressed_A,
                                                         const uint64_t* __restrict__ SharedBitmapWarp,
                                                         int TileOffsetsWarp_stride,
                                                         bool        Pred = true)
{
    int lane_id = threadIdx.x % 32;
    if (Pred == true) {
    // 每个warp循环
    // #pragma unroll
    // for (int i = 0; i < 8; i++) {
    //     // #pragma unroll
    //     for (int j = 0; j < 8; j++) {
    //         uint64_t bitmap = SharedBitmapWarp[i * 8 + j];

    //         // bitmap拆分成高位32位 mask1 和 低位32位 mask2

    //         uint32_t mask2 = static_cast<uint32_t>(bitmap >> 32);  // 高位32位 (奇数列)
    //         uint32_t mask1 = static_cast<uint32_t>(bitmap & 0xFFFFFFFF);  // 低位32位 (偶数列)
    //         // 基于当前的tile偏移，设置start_pos1为mask1对应的起始位置
    //         const half* start_pos1 = Compressed_A + TileOffsetsWarp[j];
            
    //         // 计算mask1中位为1的数量，来确定start_pos2的位置
    //         // int num_ones_mask1 = __popc(mask1);  // 计算mask1中位为1的数量
    //         int num_ones_mask1 = 16;  // 计算mask1中位为1的数量

    //         const half* start_pos2 = start_pos1 + num_ones_mask1;  // mask2的start_pos基于mask1的偏移
            
    //         // 根据mask1加载半精度值
    //         half val1 = maskloading(mask1, start_pos1, lane_id);
            
    //         // 根据mask2加载半精度值
    //         half val2 = maskloading(mask2, start_pos2, lane_id);
            
    //         // 将加载的两个half值组合成half2，并存储到 a[][] 中
    //         a[(j/2)*4 + i/2][(j%2)*2 + (i%2)] = *reinterpret_cast<const uint32_t*>(&__halves2half2(val1, val2));
    //     }
    //     // 跳到下一个 Warp 的偏移
    //     TileOffsetsWarp += TileOffsetsWarp_stride;
    // }
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t bitmap = SharedBitmapWarp[i * 8 + j];

            // bitmap拆分成高位32位 mask1 和 低位32位 mask2

            uint32_t mask2 = static_cast<uint32_t>(bitmap >> 32);  // 高位32位 (奇数列)
            uint32_t mask1 = static_cast<uint32_t>(bitmap & 0xFFFFFFFF);  // 低位32位 (偶数列)
            // 基于当前的tile偏移，设置start_pos1为mask1对应的起始位置
            const half* start_pos1 = Compressed_A + TileOffsetsWarp[j];
            
            // 计算mask1中位为1的数量，来确定start_pos2的位置
            int num_ones_mask1 = __popc(mask1);  // 计算mask1中位为1的数量
            // int num_ones_mask1 = 16;  // 计算mask1中位为1的数量

            const half* start_pos2 = start_pos1 + num_ones_mask1;  // mask2的start_pos基于mask1的偏移
            
            // 根据mask1加载半精度值
            half val1 = maskloading(mask1, start_pos1, lane_id);
            
            // 根据mask2加载半精度值
            half val2 = maskloading(mask2, start_pos2, lane_id);
            
            // 将加载的两个half值组合成half2，并存储到 a[][] 中
            a[(j/2)][(j%2)*2 + (i%2)] = *reinterpret_cast<const uint32_t*>(&__halves2half2(val1, val2));
        }
        // 跳到下一个 Warp 的偏移
        TileOffsetsWarp += TileOffsetsWarp_stride;
    }
    }
}

template<typename TilingConfig, typename SparseKernelConfig>
__global__ void SpMM_Kernel_bitmap(const half*  A,
                            const half* Compressed_A,
                            const int*   TileOffsets,
                            const uint64_t*   bitmap,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K)
{
//     //
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;

//     //
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
    uint64_t* smem_Bitmap = reinterpret_cast<uint64_t*>(&smem[(TILE_K * TilingConfig::TILE_N)*2]);
    // Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_Bitmap = y * TilingConfig::TILE_BITMAP_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    
    const int* TileOffsets_ThisWarp = nullptr;
    // TileOffsets_ThisWarp = TileOffsets + ((K_Global / TILE_K) * 8 * 32) * y + Warp_i * ((K_Global / TILE_K) * 8 * 8) + BatchID * AverageNumKBlock * 8;
    TileOffsets_ThisWarp = TileOffsets + ((K_Global / TILE_K) * 8 * 2) * y + Warp_i * ((K_Global / TILE_K) * 8 * 2) + BatchID * AverageNumKBlock * 8;

    const uint64_t* smem_Bitmap_ThisWarp = smem_Bitmap + Warp_i * 64;
    // if (threadIdx.x % 32 == 0){
    //     printf("Warp_i: %d, Warp_j: %d, TileOffsets_ThisWarp: %d \n", Warp_i, Warp_j, TileOffsets_ThisWarp[0]);
    // }

    uint32_t __restrict__ a[WARP_ROW_TENSORS * BLOCK_K_TENSORS][4]; //4*4个mma指令的寄存器全部加载（64个寄存器）
    // uint32_t __restrict__ a[WARP_ROW_TENSORS][4]; //4*4个mma指令的寄存器全部加载（64个寄存器）
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
   
   
    // copying B tile from GlobalMemory to SharedMemory
    const half* BTileGlobalPTR =
        B + Tile_Start_N * K_Global
        + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
    const uint64_t* BitmapTileGlobalPTR =
        bitmap + Tile_Start_Bitmap * (K_Global/8)
        + BatchID * AverageNumKBlock * TILE_BITMAP_K;  // Address for matrix bitmap, taking SplitK into consideration




    // CopyTileFromGlobalToShared_X_8<TilingConfig::TILE_BITMAP_M, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR, K_Global/8);  // 将32*8的bitmap加载在double buffer B shared tile之后
    CopyTileFromGlobalToShared_X_8_2<TilingConfig::TILE_BITMAP_M, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR, K_Global/8);  // 将32*8的bitmap加载在double buffer B shared tile之后

    cp_async_group_commit();
    
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(smem, BTileGlobalPTR, K_Global); //将B加载到shared mem
    cp_async_group_commit();
    
    // Initilazing C Matrix to Zeros
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    //
    cp_async_wait_group<1>(); //bitmap加载完成
    __syncthreads();
    // if (threadIdx.x == 0){
    //     for (int j = 0; j < 256; j++){
    //         printf("i: %d, bitmap val: %"PRIu64"\n", j, smem_Bitmap[j]);
    //     }
    // }
    
    SpMM_LoadFragAwithBitmap(a, TileOffsets_ThisWarp, Compressed_A, smem_Bitmap_ThisWarp, K_Global/8); //a load 完成

    cp_async_wait_group<0>(); // B加载完成
    __syncthreads();

// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first
#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {

        // copying A&B tile from GlobalMemory to SharedMemory (next tile)
        BTileGlobalPTR = B + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K);
        BitmapTileGlobalPTR = bitmap + Tile_Start_Bitmap * (K_Global/8) + BatchID * AverageNumKBlock * TILE_BITMAP_K + ((tile_id_k + 1) * TILE_BITMAP_K); 
        
        
        // double buffer
        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TILE_K * TilingConfig::TILE_N); //当前B写入的地址
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TILE_K * TilingConfig::TILE_N); //当前读取B的地址

        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        
        // Copying Bitmap Tile
        // CopyTileFromGlobalToShared_X_8<TilingConfig::TILE_BITMAP_M, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR, K_Global/8, GlobalCopy);  // 将32*8的bitmap加载在double buffer B shared tile之后
        CopyTileFromGlobalToShared_X_8_2<TilingConfig::TILE_BITMAP_M, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR, K_Global/8, GlobalCopy);  // 将32*8的bitmap加载在double buffer B shared tile之后
        
        cp_async_group_commit();

        // Copying B Tile
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();

        PipelinedCoreComputationsBitmap<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col);
        // PipelinedCoreComputationsBitmapV2<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col, TileOffsets_ThisWarp, Compressed_A, smem_Bitmap_ThisWarp, K_Global/8);

        //

        cp_async_wait_group<1>();
        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet

        TileOffsets_ThisWarp = TileOffsets_ThisWarp + 8;
        // //load A to reg a:
        SpMM_LoadFragAwithBitmap(a, TileOffsets_ThisWarp, Compressed_A, smem_Bitmap_ThisWarp, K_Global/8, GlobalCopy); //a load 完成


        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}



template<typename TilingConfig, typename SparseKernelConfig>
__global__ void SpMM_Kernelv1(const half*  A,
                            const uint32_t* MetaE,
                            const uint4* Compressed_A,
                            const int*   TileOffsets,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K)
{
    //
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);  // split k维度的ID. 总数：(M_Global / TilingConfig::TILE_M) * SPILT_K  //当前block是在第几个split
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;   // N维度的ID
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M); // M维度的ID
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock; // K_iter_num
    //
    const int* TileOffsets_ThisBlock1 = nullptr;
    const int* TileOffsets_ThisBlock2 = nullptr;
    if (TilingConfig::TILE_M == 256) {
        TileOffsets_ThisBlock1 =
            TileOffsets + K_Global / TILE_K * y * 2  // A tile 的ID, (K_Global / TILE_K)是一整行中的Block的数量。y是第几行。 BatchID * AverageNumKBlock。整个是block level的unstructured_A的起始位置（循环的第一个位置）。
            + BatchID * AverageNumKBlock;  // Address for matrix A, taking SplitK into consideration
        TileOffsets_ThisBlock2 =
            TileOffsets + K_Global / TILE_K * (y * 2 + 1)
            + BatchID * AverageNumKBlock;  // Address for matrix A, taking SplitK into consideration
    }
    else {
        TileOffsets_ThisBlock1 = TileOffsets + K_Global / TILE_K * y + BatchID * AverageNumKBlock;
        TileOffsets_ThisBlock2 = TileOffsets_ThisBlock1;  // otherwise will cause problem when passing
                                                          // TileOffsets_ThisBlock2[0] to SpMM_CopyFromGlobalToReg()
    }
    //
    uint32_t Registers_GlobalToShared[SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL];
    uint32_t NNZ_ThreadLocal1 = 0;
    uint32_t NNZ_ThreadLocal2 = 0;
    //
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
    uint32_t* smem_MetaE = reinterpret_cast<uint32_t*>(&smem[(TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N)*2]);
    // Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_MetaE = y * TilingConfig::TILE_MetaE;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4];
    uint32_t __restrict__ e[WARP_ROW_TENSORS];
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][8];
    // copying B tile from GlobalMemory to SharedMemory
    const half* BTileGlobalPTR =
        B + Tile_Start_N * K_Global
        + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
    const half* ATileGlobalPTR =
        A + Tile_Start_M * (K_Global/2)
        + BatchID * AverageNumKBlock * TILE_K_HALF;  // Address for matrix A, taking SplitK into consideration
    const uint32_t* ETileGlobalPTR =
        MetaE + Tile_Start_MetaE * (K_Global/2)
        + BatchID * AverageNumKBlock * TILE_K_HALF;  // Address for matrix E, taking SplitK into consideration
    //
    int NNZ_ThisTile1 = TileOffsets_ThisBlock1[1] - TileOffsets_ThisBlock1[0];
    int NNZ_ThisTile2 = 0;
    if (TilingConfig::TILE_M == 256)
        NNZ_ThisTile2 = TileOffsets_ThisBlock2[1] - TileOffsets_ThisBlock2[0];
    
    SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_GlobalToShared,
                                                               &NNZ_ThreadLocal1,
                                                               Compressed_A + TileOffsets_ThisBlock1[0],
                                                               NNZ_ThisTile1,
                                                               Registers_GlobalToShared
                                                                   + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
                                                               &NNZ_ThreadLocal2,
                                                               Compressed_A + TileOffsets_ThisBlock2[0],
                                                               NNZ_ThisTile2);
    SpMM_InitSharedMemory<TilingConfig>(smem);  
    cp_async_group_commit();  // group 1
    CopyTileFromGlobalToShared_X_32<TilingConfig::TILE_M, TilingConfig>(
        smem + TilingConfig::TILE_M * TILE_K_HALF, ATileGlobalPTR, K_Global/2);  // 将256*32 加载在smem + TilingConfig::TILE_M * TILE_K / 2 和 smem + TilingConfig::TILE_M * TILE_K之间。 
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global);  // 将64*16 加载在smem + TilingConfig::TILE_M * TILE_K之后。
    // CopyTileFromGlobalToShared_X_32<TilingConfig::TILE_MetaE, TilingConfig>(
    //     smem_MetaE, ETileGlobalPTR, K_Global/2);  // 将16*32 uint32 加载在smem_MetaE之后。
    CopyTileFromGlobalToShared_X_32_1<TilingConfig::TILE_MetaE, TilingConfig>(
            smem_MetaE, ETileGlobalPTR, K_Global/2);  // 将16*32 uint32 加载在smem_MetaE之后。 for 1 warp的情况 
    cp_async_group_commit();  // group 2
    // Initilazing C Matrix to Zeros
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    //
    cp_async_wait_group<1>();  // 使得执行wait group指令的线程等到还有1个group没有完成。总共两个group，也就是group 1异步操作要完成，group 2可以不完成。
    __syncthreads();
    SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(  // 将A_unstructured从寄存器解压到smem  -  smem + TilingConfig::TILE_M * TILE_K_HALF
        smem,
        Registers_GlobalToShared,
        NNZ_ThreadLocal1,
        smem + TilingConfig::TILE_M * TILE_K_HALF / 2,
        Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
        NNZ_ThreadLocal2);
    //
    cp_async_wait_group<0>();
    __syncthreads();
    // Prefetch to reduce stall_long_sb
    int StartIndex_SparseTiles_Prefetch1 = TileOffsets_ThisBlock1[0 + 1];
    int NNZ_ThisTile_Prefetch1           = TileOffsets_ThisBlock1[0 + 2] - TileOffsets_ThisBlock1[0 + 1];
    int StartIndex_SparseTiles_Prefetch2 = 0;
    int NNZ_ThisTile_Prefetch2           = 0;
    if (TilingConfig::TILE_M == 256) {
        StartIndex_SparseTiles_Prefetch2 = TileOffsets_ThisBlock2[0 + 1];
        NNZ_ThisTile_Prefetch2           = TileOffsets_ThisBlock2[0 + 2] - TileOffsets_ThisBlock2[0 + 1];
    }

#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        // Using the previous prefetched value
        int StartIndex_SparseTiles1 = StartIndex_SparseTiles_Prefetch1;
        int NNZ_ThisTile1           = NNZ_ThisTile_Prefetch1;
        int StartIndex_SparseTiles2 = 0;
        int NNZ_ThisTile2           = 0;
        if (TilingConfig::TILE_M == 256) {
            StartIndex_SparseTiles2 = StartIndex_SparseTiles_Prefetch2;
            NNZ_ThisTile2           = NNZ_ThisTile_Prefetch2;
        }
        //
        StartIndex_SparseTiles_Prefetch1 = TileOffsets_ThisBlock1[tile_id_k + 1 + 1];
        NNZ_ThisTile_Prefetch1 = TileOffsets_ThisBlock1[tile_id_k + 1 + 2] - TileOffsets_ThisBlock1[tile_id_k + 1 + 1];
        if (TilingConfig::TILE_M == 256) {
            StartIndex_SparseTiles_Prefetch2 = TileOffsets_ThisBlock2[tile_id_k + 1 + 1];
            NNZ_ThisTile_Prefetch2 =
                TileOffsets_ThisBlock2[tile_id_k + 1 + 2] - TileOffsets_ThisBlock2[tile_id_k + 1 + 1];
        }
        // copying B tile from GlobalMemory to SharedMemory
        BTileGlobalPTR = B + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K);
        ATileGlobalPTR = A + Tile_Start_M * (K_Global/2) + BatchID * AverageNumKBlock * TILE_K_HALF + ((tile_id_k + 1) * TILE_K_HALF);
        ETileGlobalPTR = MetaE + Tile_Start_MetaE * (K_Global/2) + BatchID * AverageNumKBlock * TILE_K_HALF + ((tile_id_k + 1) * TILE_K_HALF);

        // double buffer
        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        uint32_t* __restrict__ smem_MetaE_write_PTR = smem_MetaE;
        uint32_t* __restrict__ smem_MetaE_read_PTR  = smem_MetaE;
        smem_MetaE_write_PTR = smem_MetaE + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_MetaE * TILE_K_HALF);
        smem_MetaE_read_PTR  = smem_MetaE + ((tile_id_k) % 2) * (TilingConfig::TILE_MetaE * TILE_K_HALF);
        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        //
        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR);
        cp_async_group_commit();   // group 1
        SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(
            Registers_GlobalToShared,
            &NNZ_ThreadLocal1,
            Compressed_A + StartIndex_SparseTiles1,
            NNZ_ThisTile1,
            Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
            &NNZ_ThreadLocal2,
            Compressed_A + StartIndex_SparseTiles2,
            NNZ_ThisTile2);
        // Copying B Tile
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy);  //double buffer的地址。
        CopyTileFromGlobalToShared_X_32<TilingConfig::TILE_M, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K_HALF, ATileGlobalPTR, K_Global/2, GlobalCopy);  // 将256*32 加载在smem + TilingConfig::TILE_M * TILE_K / 2 和 smem + TilingConfig::TILE_M * TILE_K之间。
        // CopyTileFromGlobalToShared_X_32<TilingConfig::TILE_MetaE, TilingConfig>(
        //     smem_MetaE_write_PTR, ETileGlobalPTR, K_Global/2, GlobalCopy);  // 将16*32 加载在smem_MetaE_write_PTR。
        CopyTileFromGlobalToShared_X_32_1<TilingConfig::TILE_MetaE, TilingConfig>(
                smem_MetaE_write_PTR, ETileGlobalPTR, K_Global/2, GlobalCopy);  // 将16*32 加载在smem_MetaE_write_PTR。
        cp_async_group_commit();   // group 2
        //
        PipelinedCoreComputationsSparse<TilingConfig>(c, a, b, e, smem_read_PTR, smem_MetaE_read_PTR, warp_start_row, warp_start_col);
        //
        cp_async_wait_group<1>();    //使得执行wait group指令的线程等到还有1个group没有完成。总共两个group，也就是group 1异步操作要完成，group 2可以不完成。
        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet
        SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
            smem_write_PTR,
            Registers_GlobalToShared,
            NNZ_ThreadLocal1,
            smem_write_PTR + TilingConfig::TILE_M * TILE_K / 4,
            Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
            NNZ_ThreadLocal2);
        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}


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
#ifndef MatMulUtilities_H
#define MatMulUtilities_H
// C = A*B
// C: col major
// A: row major
// B: col major

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "AsyncCopy_PTX.cuh"
#include "MMA_PTX.cuh"
#include "TilingConfig.h"

int cuda_CheckError()
{
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
}

// New features: Copy size is X * 64, X can be any multiple to 8
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_64(half* __restrict__ SharedPTR,
                                                                const half* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 8;
    int row1          = lane_id / 8;
    int row2          = lane_id / 8 + 4;
    int store_column1 = col ^ row1;
    int store_column2 = col ^ row2;
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / COPY_UNIT_FP16_ROWS;
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;
//
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id);
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
        const half* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * GlobalStride;
        half* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * TILE_K;
        cp_async<16>(SharedPTR_Unit + store_column1 * HALF_PER_128B + row1 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row1 * GlobalStride,
                     AsyncCopyPredictor);
        cp_async<16>(SharedPTR_Unit + store_column2 * HALF_PER_128B + row2 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row2 * GlobalStride,
                     AsyncCopyPredictor);
    }
}


// New features: Copy size is X * 32, X can be any multiple to 16
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_32(half* __restrict__ SharedPTR,
                                                                const half* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 4;
    int row1          = lane_id / 4;    //将threads组织成 8*4    
    int row2          = lane_id / 4 + 8;
    int store_column1 = col ^ (row1/2);
    int store_column2 = col ^ (row1/2);
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / COPY_UNIT_FP16_ROWS_16;  // 256 / 16 = 16  一个warp一次iter, 访问16*32, 4个warp要拷贝256 行。 每个warp进行4个iter。
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;  //1 注意开了4个warp来做这个事情。(16 - 1) / 4 + 1 = 4
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id); // 4个warp连续访存，每个warp的号
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
        const half* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS_16 * GlobalStride;  // warp读取global的地址。
        half* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS_16 * TILE_K_HALF; // warp写入shared的地址。
        cp_async<16>(SharedPTR_Unit + store_column1 * HALF_PER_128B + row1 * TILE_K_HALF, // 线程写入shared的地址。
                     GlobalPTR_Unit + col * HALF_PER_128B + row1 * GlobalStride,   // 线程读取global的地址。
                     AsyncCopyPredictor);
        cp_async<16>(SharedPTR_Unit + store_column2 * HALF_PER_128B + row2 * TILE_K_HALF,
                     GlobalPTR_Unit + col * HALF_PER_128B + row2 * GlobalStride,
                     AsyncCopyPredictor);
    }
}

// New features: Copy size is X * 32 Uint32, X can be any multiple to 4 // for MetaE
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_32(uint32_t* __restrict__ SharedPTR,
                                                                const uint32_t* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 8;
    int row          = lane_id / 8;    //将threads组织成 4*8    
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / 4;  // 16 / 4 = 4  一个warp一次iter, 访问4*32, 4个warp要拷贝16行。 每个warp进行1个iter。
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;  //1 注意开了4个warp来做这个事情。(4 - 1) / 4 + 1 = 1
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id); // 4个warp连续访存，每个warp的号
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
        const uint32_t* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * 4 * GlobalStride;  // warp读取global的地址。
        uint32_t* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * 4 * TILE_K_HALF; // warp写入shared的地址。
        cp_async<16>(SharedPTR_Unit + col * UINT32_PER_128B + row * TILE_K_HALF, // 线程写入shared的地址。
                     GlobalPTR_Unit + col * UINT32_PER_128B + row * GlobalStride,   // 线程读取global的地址。
                     AsyncCopyPredictor);
    }
}
// New features: Copy size is X * 32 Uint32, X can be any multiple to 2 // for MetaE
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_32_2(uint32_t* __restrict__ SharedPTR,
                                                                const uint32_t* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 16;
    int row          = lane_id / 16;    //将threads组织成 4*8    
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / 2;  // 16 / 4 = 4  一个warp一次iter, 访问4*32, 4个warp要拷贝16行。 每个warp进行1个iter。
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;  //1 注意开了4个warp来做这个事情。(4 - 1) / 4 + 1 = 1
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id); // 4个warp连续访存，每个warp的号
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
        const uint32_t* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * 2 * GlobalStride;  // warp读取global的地址。
        uint32_t* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * 2 * TILE_K_HALF; // warp写入shared的地址。
        cp_async_8<8>(SharedPTR_Unit + col * UINT32_PER_64B + row * TILE_K_HALF, // 线程写入shared的地址。
                     GlobalPTR_Unit + col * UINT32_PER_64B + row * GlobalStride,   // 线程读取global的地址。
                     AsyncCopyPredictor);
    }
}
// New features: Copy size is X * 32 Uint32, X can be any multiple to 1 // for MetaE
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_32_1(uint32_t* __restrict__ SharedPTR,
                                                                const uint32_t* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 32;
    int row          = lane_id / 32;    //将threads组织成 4*8    
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy;  // 16 / 4 = 4  一个warp一次iter, 访问4*32, 4个warp要拷贝16行。 每个warp进行1个iter。
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;  //1 注意开了4个warp来做这个事情。(4 - 1) / 4 + 1 = 1
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id); // 4个warp连续访存，每个warp的号
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
        const uint32_t* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * GlobalStride;  // warp读取global的地址。
        uint32_t* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * TILE_K_HALF; // warp写入shared的地址。
        cp_async_8<4>(SharedPTR_Unit + col * UINT32_PER_64B + row * TILE_K_HALF, // 线程写入shared的地址。
                     GlobalPTR_Unit + col * UINT32_PER_64B + row * GlobalStride,   // 线程读取global的地址。
                     AsyncCopyPredictor);
    }
}

// New features: Copy size is X * 8 Uint64, X can be any multiple to 8 // for Bitmap
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_8(uint64_t* __restrict__ SharedPTR,
                                                               const uint64_t* GlobalPTR,
                                                               const int   GlobalStride,
                                                               bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 4;
    int row          = lane_id / 4;    //将threads组织成 8*4    
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / 8;  // 32 / 8 = 4  一个warp一次iter, 访问8*8, 4个warp要拷贝32行。 每个warp进行1个iter。
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;  //1 注意开了4个warp来做这个事情。(4 - 1) / 4 + 1 = 1
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id); // 4个warp连续访存，每个warp的号
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
        const uint64_t* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * 8 * GlobalStride;  // warp读取global的地址。
        uint64_t* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * 8 * TILE_BITMAP_K; // warp写入shared的地址。
        cp_async<16>(SharedPTR_Unit + col * UINT64_PER_128B + row * TILE_BITMAP_K, // 线程写入shared的地址。
                     GlobalPTR_Unit + col * UINT64_PER_128B + row * GlobalStride,   // 线程读取global的地址。
                     AsyncCopyPredictor);
    }
}
// New features: Copy size is X * 8 Uint64, X can be any multiple to 8 // for Bitmap
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_8_2(uint64_t* __restrict__ SharedPTR,
                                                               const uint64_t* GlobalPTR,
                                                               const int   GlobalStride,
                                                               bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 4;
    int row          = lane_id / 4;    //将threads组织成 8*4    
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy;  // 32 / 8 = 4  一个warp一次iter, 访问8*8, 4个warp要拷贝32行。 每个warp进行1个iter。
    bool AsyncCopyPredictor = row < TotalNumOfCopyUnit && Pred && warp_id == 0;  ///// Bug, too hard to find this bug, 5555
    const uint64_t* GlobalPTR_Unit        = GlobalPTR;  // warp读取global的地址。
    uint64_t* __restrict__ SharedPTR_Unit = SharedPTR; // warp写入shared的地址。
    cp_async<16>(SharedPTR_Unit + col * UINT64_PER_128B + row * TILE_BITMAP_K, // 线程写入shared的地址。
                     GlobalPTR_Unit + col * UINT64_PER_128B + row * GlobalStride,   // 线程读取global的地址。
                     AsyncCopyPredictor);
}
// New features: Copy size is X * 16 Uint64, X can be any multiple to 1 // for BitmapV1
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_16_1(uint64_t* __restrict__ SharedPTR,
                                                               const uint64_t* GlobalPTR,
                                                               const int   GlobalStride,
                                                               bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 8;
    int row          = lane_id / 8;    //将threads组织成 4*8    
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy;  // 32 / 8 = 4  一个warp一次iter, 访问8*8, 4个warp要拷贝32行。 每个warp进行1个iter。
    bool AsyncCopyPredictor = row < TotalNumOfCopyUnit && Pred && warp_id == 0;  ///// Bug, too hard to find this bug, 5555
    const uint64_t* GlobalPTR_Unit        = GlobalPTR;  // warp读取global的地址。
    uint64_t* __restrict__ SharedPTR_Unit = SharedPTR; // warp写入shared的地址。
    cp_async<16>(SharedPTR_Unit + col * UINT64_PER_128B + row * TilingConfig::TILE_BITMAP_K_V1, // 线程写入shared的地址。
                     GlobalPTR_Unit + col * UINT64_PER_128B + row * GlobalStride,   // 线程读取global的地址。
                     AsyncCopyPredictor);
}
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_16_1_V3(uint64_t* __restrict__ SharedPTR,
                                                               const uint64_t* GlobalPTR,
                                                               const int   GlobalStride,
                                                               bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 8;
    int row          = lane_id / 8;    //将threads组织成 4*8    
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy;  // 32 / 8 = 4  一个warp一次iter, 访问8*8, 4个warp要拷贝32行。 每个warp进行1个iter。
    bool AsyncCopyPredictor = row < TotalNumOfCopyUnit && Pred && warp_id == 0;  ///// Bug, too hard to find this bug, 5555
    const uint64_t* GlobalPTR_Unit        = GlobalPTR;  // warp读取global的地址。
    uint64_t* __restrict__ SharedPTR_Unit = SharedPTR; // warp写入shared的地址。
    cp_async<16>(SharedPTR_Unit + col * UINT64_PER_128B + row * TilingConfig::TILE_BITMAP_K_V3, // 线程写入shared的地址。
                     GlobalPTR_Unit + col * UINT64_PER_128B + row * GlobalStride,   // 线程读取global的地址。
                     AsyncCopyPredictor);
}
// New features: Copy size is X * 64 Uint64, X can be  1 // for BitmapV2
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_Bitmap_1_64(uint64_t* __restrict__ SharedPTR,
                                                               const uint64_t* GlobalPTR,
                                                               bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy;  // 32 / 8 = 4  一个warp一次iter, 访问8*8, 4个warp要拷贝32行。 每个warp进行1个iter。
    bool AsyncCopyPredictor = warp_id < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
    const uint64_t* GlobalPTR_Unit        = GlobalPTR;  // warp读取global的地址。
    uint64_t* __restrict__ SharedPTR_Unit = SharedPTR; // warp写入shared的地址。
    cp_async<16>(SharedPTR_Unit + lane_id * UINT64_PER_128B, // 线程写入shared的地址。
                     GlobalPTR_Unit + lane_id * UINT64_PER_128B,   // 线程读取global的地址。
                     AsyncCopyPredictor);
}


template<typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_Sparse(half* __restrict__ SharedPTR,
                                                               const half* GlobalPTR,
                                                               const int   NNZ,
                                                               bool        Pred = true)
{
    if(Pred) {
    int threadPerBlock = blockDim.x;
    int NNZ_8 = (NNZ>>3);
    for(int i = threadIdx.x; i < NNZ_8; i+= threadPerBlock) {
        const half* GlobalPTR_Unit        =  GlobalPTR + i * 8;  // warp读取global的地址。
        half* __restrict__ SharedPTR_Unit = SharedPTR + i * 8; // warp写入shared的地址。
        cp_async<16>(SharedPTR_Unit, // 线程写入shared的地址。
                     GlobalPTR_Unit,   // 线程读取global的地址。
                     Pred);
    }
    }
}


template<typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputations(float c[][REG_PER_C_TENSOR_16_16],
                                                          uint32_t __restrict__ a[][4],
                                                          uint32_t __restrict__ b[][4],
                                                          half* __restrict__ SharedMemoryPTR,
                                                          int warp_start_row,
                                                          int warp_start_col)
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    // First Register Loading
    FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a, SharedMemoryPTR, warp_start_row, 0);
    B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K, warp_start_col, 0);
// Sencond loading & first computation, so on
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) {
        uint32_t __restrict__(*a_read)[4]  = a;
        uint32_t __restrict__(*b_read)[4]  = b;
        uint32_t __restrict__(*a_write)[4] = a;
        uint32_t __restrict__(*b_write)[4] = b;
        a_read += ((k) % 2) * WARP_ROW_TENSORS;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        a_write += ((k + 1) % 2) * WARP_ROW_TENSORS;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS) {
            FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a_write, SharedMemoryPTR, warp_start_row, (k + 1) * MMA_K);
            B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
                b_write, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K, warp_start_col, (k + 1) * MMA_K);
        }
// computations
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++)
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
                MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS], a_read[i], b_read[j]);
                if (!TilingConfig::N8)
                    MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a_read[i], b_read[j] + 2);  // c+4; b+2
            }
    }
}

// template<typename TilingConfig>
// __device__ __forceinline__ void PipelinedCoreComputationsBitmap(float c[][REG_PER_C_TENSOR_16_16],
//                                                           uint32_t __restrict__ a[][4],
//                                                           uint32_t __restrict__ b[][4],
//                                                           half* __restrict__ SharedMemoryPTR,
//                                                           int warp_start_row,
//                                                           int warp_start_col)
// {
//     uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
//     B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
//         b, SharedMemoryPTR, warp_start_col, 0);
// // Sencond loading & first computation, so on
// #pragma unroll
//     for (int k = 0; k < BLOCK_K_TENSORS; k++) {
//         uint32_t __restrict__(*b_read)[4]  = b;
//         uint32_t __restrict__(*b_write)[4] = b;
//         b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
//         b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
//         // data loading
//         if (k + 1 < BLOCK_K_TENSORS) {
//             B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
//                 b_write, SharedMemoryPTR, warp_start_col, (k + 1) * MMA_K);
//         }
// // computations
// #pragma unroll
//         for (int i = 0; i < WARP_ROW_TENSORS; i++)
// #pragma unroll
//             for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
//                 // MMA_FP16_M16N16K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );
//                 MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS], a[k*4 + i], b_read[j]);
//                 if (!TilingConfig::N8)
//                     MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a[k*4 + i], b_read[j] + 2);  // c+4; b+2
//             }
//     }
// }
template<typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputationsBitmap(float c[][REG_PER_C_TENSOR_16_16],
                                                          uint32_t __restrict__ a[][4],
                                                          uint32_t __restrict__ b[][4],
                                                          half* __restrict__ SharedMemoryPTR,
                                                          int warp_start_row,
                                                          int warp_start_col)
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b, SharedMemoryPTR, warp_start_col, 0);
// Sencond loading & first computation, so on
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) {
        uint32_t __restrict__(*b_read)[4]  = b;
        uint32_t __restrict__(*b_write)[4] = b;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS) {
            B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
                b_write, SharedMemoryPTR, warp_start_col, (k + 1) * MMA_K);
        }
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
                    // for (int i = 0; i < 2; i++){
                    //    printf("tid: %d, %d ",  threadIdx.x, b_read[j][i]);
                    //    printf("\n");
                    // }
                // MMA_FP16_M16N16K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );
                MMA_FP16_M16N8K16(c_uint32_t[j * WARP_ROW_TENSORS_BITMAP_V1], a[k], b_read[j]);
                if (!TilingConfig::N8)
                    MMA_FP16_M16N8K16(c_uint32_t[j * WARP_ROW_TENSORS_BITMAP_V1] + 4, a[k], b_read[j] + 2);  // c+4; b+2
            }
    }
}
__device__ __forceinline__ half2 maskloadingv2(uint64_t bitmap, const half* __restrict__ startpos, int lane_id) {
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
__device__ __forceinline__ void SpMM_LoadFragAwithBitmapFromShemv2(uint32_t __restrict__ a[][4],
                                                         const half* __restrict__ ShemVal,
                                                         const uint64_t* __restrict__ SharedBitmap,
                                                         int* start_pos,
                                                         int bit_offset)
{
    int lane_id = threadIdx.x % 32;
    const uint64_t* SharedBitmapStart = SharedBitmap + bit_offset;
    // #pragma unroll
    for (int i = 0; i < 4; i++) {
            // #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t bitmap = SharedBitmapStart[i * 4 + j];
                // if(threadIdx.x == 0){
                //     printf("i: %d, j: %d, startpos: %d \n", i,j,*start_pos);
                // }
                // 根据mask1加载半精度值
                half2 val = maskloadingv2(bitmap, ShemVal+*start_pos, lane_id);
                // 将加载的两个half值组合成half2，并存储到 a[][] 中
                a[i][j] = *reinterpret_cast<const uint32_t*>(&val);
                // 计算mask1中位为1的数量，来确定start_pos2的位置
                *start_pos += __popcll(bitmap);  // 计算mask1中位为1的数量
            }
    }
}
template<typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputationsBitmapV2(float c[][REG_PER_C_TENSOR_16_16],
                                                          uint32_t __restrict__ a[][4],
                                                          uint32_t __restrict__ b[][4],
                                                          half* __restrict__ ShemAVal,
                                                          uint64_t* __restrict__ SharedMemoryBitmapPTR,
                                                          half* __restrict__ SharedMemoryPTR,
                                                          int warp_start_row,
                                                          int warp_start_col)
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    int start_pos = 0;
    SpMM_LoadFragAwithBitmapFromShemv2(a, ShemAVal, SharedMemoryBitmapPTR, &start_pos, 0);
    B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b, SharedMemoryPTR, warp_start_col, 0);
// Sencond loading & first computation, so on
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) {
        uint32_t __restrict__(*a_read)[4]  = a;
        uint32_t __restrict__(*b_read)[4]  = b;
        uint32_t __restrict__(*a_write)[4] = a;
        uint32_t __restrict__(*b_write)[4] = b;
        a_read += ((k) % 2) * WARP_ROW_TENSORS_BITMAP_V2;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        a_write += ((k + 1) % 2) * WARP_ROW_TENSORS_BITMAP_V2;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS) {
            SpMM_LoadFragAwithBitmapFromShemv2(a_write, ShemAVal, SharedMemoryBitmapPTR, &start_pos, (k+1)*16);
            B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
                b_write, SharedMemoryPTR, warp_start_col, (k + 1) * MMA_K);
        }
// computations
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V2; i++)
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
                    // for (int kk = 0; kk < 2; kk++){
                    //    printf("v2 tid: %d, %d ",  threadIdx.x, b_read[j][kk]);
                    //    printf("\n");
                    // }
                MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS_BITMAP_V2], a_read[i], b_read[j]);
                if (!TilingConfig::N8)
                    MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS_BITMAP_V2] + 4, a_read[i], b_read[j] + 2);  // c+4; b+2
            }
    }
}

// __device__ __forceinline__ half maskloading(uint32_t mask, const half* __restrict__ startpos, int lane_id)
// {
//     // 计算当前线程的 lane_id 之前的位数中有多少位为1
//     uint32_t mask_shifted = mask & ((1U << lane_id) - 1);  // 仅保留 lane_id 之前的位
//     int num_ones_before = __popc(mask_shifted);  // 计算位为1的数量

//     // 如果当前线程对应的位为1，加载数据，否则加载0
//     if (mask & (1U << lane_id)) {
//         return startpos[num_ones_before];  // 读取实际数据，按照之前1的数量偏移
//     } else {
//         return __float2half(0.0f);  // 如果不需要加载，则返回0
//     }
// }
// template<typename TilingConfig>
// __device__ __forceinline__ void PipelinedCoreComputationsBitmapV2(float c[][REG_PER_C_TENSOR_16_16],
//                                                           uint32_t __restrict__ a[][4],
//                                                           uint32_t __restrict__ b[][4],
//                                                           half* __restrict__ SharedMemoryPTR,
//                                                           int warp_start_row,
//                                                           int warp_start_col,

//                                                           const int*   __restrict__ TileOffsetsWarp,
//                                                           const half* __restrict__ Compressed_A,
//                                                           const uint64_t* __restrict__ SharedBitmapWarp,
//                                                           int TileOffsetsWarp_stride,
//                                                           bool        Pred = true)
// {
//     int lane_id = threadIdx.x % 32;
//     uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
//     B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
//         b, SharedMemoryPTR, warp_start_col, 0);
// // Sencond loading & first computation, so on
// // #pragma unroll
//     for (int k = 0; k < BLOCK_K_TENSORS; k++) {
//         uint32_t __restrict__(*b_read)[4]  = b;
//         uint32_t __restrict__(*b_write)[4] = b;
//         b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
//         b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
//         // data loading
//         if (k + 1 < BLOCK_K_TENSORS) {
//             B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
//                 b_write, SharedMemoryPTR, warp_start_col, (k + 1) * MMA_K);
//         }
//         const int*   TileOffsetsWarp_c = TileOffsetsWarp;
//         // #pragma unroll
//         for (int i = 0; i < 8; i++) {
//             // #pragma unroll
//             for (int j = 2*k; j < 2*k+2; j++) {
//                 uint64_t bitmap = SharedBitmapWarp[i * 8 + j];
    
//                 // bitmap拆分成高位32位 mask1 和 低位32位 mask2
//                 // uint32_t mask1 = static_cast<uint32_t>(bitmap >> 32);  // 高位32位
//                 // uint32_t mask2 = static_cast<uint32_t>(bitmap & 0xFFFFFFFF);  // 低位32位
//                 uint32_t mask2 = static_cast<uint32_t>(bitmap >> 32);  // 高位32位 (奇数列)
//                 uint32_t mask1 = static_cast<uint32_t>(bitmap & 0xFFFFFFFF);  // 低位32位 (偶数列)
//                 // 基于当前的tile偏移，设置start_pos1为mask1对应的起始位置
//                 const half* start_pos1 = Compressed_A + TileOffsetsWarp_c[j];
                
//                 // 计算mask1中位为1的数量，来确定start_pos2的位置
//                 int num_ones_mask1 = __popc(mask1);  // 计算mask1中位为1的数量
//                 const half* start_pos2 = start_pos1 + num_ones_mask1;  // mask2的start_pos基于mask1的偏移
                
//                 // 根据mask1加载半精度值
//                 half val1 = maskloading(mask1, start_pos1, lane_id);
                
//                 // 根据mask2加载半精度值
//                 half val2 = maskloading(mask2, start_pos2, lane_id);
                
//                 // 将加载的两个half值组合成half2，并存储到 a[][] 中
//                 a[i/2][(j%2)*2 + (i%2)] = *reinterpret_cast<const uint32_t*>(&__halves2half2(val1, val2));
//             }
//             // 跳到下一个 Warp 的偏移
//             TileOffsetsWarp_c += TileOffsetsWarp_stride;
//         }



// // computations
// #pragma unroll
//         for (int i = 0; i < WARP_ROW_TENSORS; i++)
// #pragma unroll
//             for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
//                 // MMA_FP16_M16N16K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );
//                 MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS], a[i], b_read[j]);
//                 if (!TilingConfig::N8)
//                     MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a[i], b_read[j] + 2);  // c+4; b+2
//             }
//     }
// }

template<typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputationsSparse(float c[][REG_PER_C_TENSOR_16_16],
                                                          uint32_t __restrict__ a[][4],
                                                          uint32_t __restrict__ b[][8],
                                                          uint32_t __restrict__ e[],
                                                          half* __restrict__ SharedMemoryPTR,
                                                          uint32_t* __restrict__ SharedMemoryPTRE,
                                                          int warp_start_row,
                                                          int warp_start_col)
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    // First Register Loading
    A_FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K / 2, warp_start_row, 0);  //一个warp先加载64*16的structured部分到寄存器a
    B_FragLoadFromSharedToRegisters_double<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K, warp_start_col, 0);
    E_FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(e, SharedMemoryPTRE, warp_start_row/16);
    // Sencond loading & first computation, so on
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS_HALF; k++) {
        uint32_t __restrict__(*a_read)[4]  = a;
        uint32_t __restrict__(*b_read)[8]  = b;
        uint32_t __restrict__(*a_write)[4] = (a + WARP_ROW_TENSORS);
        uint32_t __restrict__(*b_write)[8] = b;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS_HALF) {
            B_FragLoadFromSharedToRegisters_double<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(b_write, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K, warp_start_col, (k + 1) * 2 * MMA_K);
        }
        A_FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a_write, SharedMemoryPTR, warp_start_row, k * MMA_K);  //一个warp先加载64*16的unstructured部分到寄存器a_write
        // 计算structured部分
        if (k == 0) {
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++) // 4
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) { // 1 
                MMA_SP_FP16_M16N8K32(c_uint32_t[i + j * WARP_ROW_TENSORS], a_read[i], b_read[j], b_read[j] + 4, e[i]);
                if (!TilingConfig::N8)
                    MMA_SP_FP16_M16N8K32(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a_read[i], b_read[j] + 2,  b_read[j] + 6, e[i]);  // c+4; b+2
            }
        } else {
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++) // 4
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) { // 1 
                MMA_SP_FP16_M16N8K32_1(c_uint32_t[i + j * WARP_ROW_TENSORS], a_read[i], b_read[j], b_read[j] + 4, e[i]);
                if (!TilingConfig::N8)
                  MMA_SP_FP16_M16N8K32_1(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a_read[i], b_read[j] + 2,  b_read[j] + 6, e[i]);  // c+4; b+2
            }           
        }
        a_read += WARP_ROW_TENSORS;
        a_write = a;  
        if (k + 1 < BLOCK_K_TENSORS_HALF) {
            A_FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a_write, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K / 2, warp_start_row, (k + 1) * MMA_K);  //一个warp先加载64*16的structured部分到寄存器a
        }
        // 计算unstructured部分
        if (k == 0) {
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++) // 4
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) { // 1 
                MMA_SP_FP16_M16N8K32(c_uint32_t[i + j * WARP_ROW_TENSORS], a_read[i], b_read[j], b_read[j] + 4, ~e[i]);
                if (!TilingConfig::N8)
                    MMA_SP_FP16_M16N8K32(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a_read[i], b_read[j] + 2,  b_read[j] + 6, ~e[i]);  // c+4; b+4
            }
        } else {
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++) // 4
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) { // 1 
                MMA_SP_FP16_M16N8K32_1(c_uint32_t[i + j * WARP_ROW_TENSORS], a_read[i], b_read[j], b_read[j] + 4, ~e[i]);
                if (!TilingConfig::N8)
                  MMA_SP_FP16_M16N8K32_1(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a_read[i], b_read[j] + 2,  b_read[j] + 6, ~e[i]);  // c+4; b+4
            }            
        }           
    }
    
}

template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegister(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}


template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegisterBitmapV1(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS_BITMAP_V1);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V1; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS_BITMAP_V1;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegisterBitmapV2(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS_BITMAP_V2);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V2; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS_BITMAP_V2;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}
template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegisterBitmapV3(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS_BITMAP_V3);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS_BITMAP_V3;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}

#endif
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "fps_gpu.h"


__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2){
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>  // 因为下面定义数组的时候要用这个变量所以这里必须定义成模板函数
__global__ void farthest_point_sampling_kernel(int b, int n, int m,
    const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
    // dataset: (B, N, 3) 输入点
    // temp: (B, N)
    // output:
    //      idx: (B, M)  每个batch采样m个点

    if (m <= 0) return;

    // block_size == n_thread
    // 每个线程有一个用来存放结果的位置
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    // 每个block处理一个batch
    int batch_index = blockIdx.x;

    // 直接加上offset把指针推到该block的batch对应的数据处
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    // 不随机取初始点而是直接取第0个点
    int old = 0;
    if (threadIdx.x == 0)
        idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
        int besti = 0;
        float best = -1;
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];

        // 一共有block_size个线程在并行所以每个线程只需要每stride=block_size个点处理一个
        for (int k = tid; k < n; k += stride) {
            float x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];
            // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            // if (mag <= 1e-3)
            // continue;

            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            
            // 如果到其他采样点的距离更近就采用更近的距离
            float d2 = min(d, temp[k]);
            temp[k] = d2;

            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        // 整理数据，将最大的距离和最大距离对应点的idx逐步存储到dists[0]和dists_i[0]中
        // 每轮只需要一半的线程整理数据
        if (block_size >= 1024) {
            if (tid < 512) {
                __update(dists, dists_i, tid, tid + 512);
            }
            __syncthreads();
        }

        if (block_size >= 512) {
            if (tid < 256) {
                __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
                __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
                __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) {
                __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
                __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
                __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
                __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
                __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) {
                __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        // 将最大距离对应点的idx存到idxs中
        // 只需要一个线程来做就可以了
        old = dists_i[0];
        if (tid == 0)
            idxs[j] = old;
    }
}

void farthest_point_sampling_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs, cudaStream_t stream) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads) {
    case 1024:
        farthest_point_sampling_kernel<1024><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    case 512:
        farthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    case 256:
        farthest_point_sampling_kernel<256><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    case 128:
        farthest_point_sampling_kernel<128><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    case 64:
        farthest_point_sampling_kernel<64><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    case 32:
        farthest_point_sampling_kernel<32><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    case 16:
        farthest_point_sampling_kernel<16><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    case 8:
        farthest_point_sampling_kernel<8><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    case 4:
        farthest_point_sampling_kernel<4><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    case 2:
        farthest_point_sampling_kernel<2><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    case 1:
        farthest_point_sampling_kernel<1><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
    default:
        farthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

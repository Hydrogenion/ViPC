#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <THC/THC.h>

#include "fps_gpu.h"


int farthest_point_sampling_wrapper(int b, int n, int m,
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    farthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx, stream);
    return 1;
}

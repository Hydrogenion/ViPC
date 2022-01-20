#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "fps_gpu.h"
#include "ball_query_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sampling_wrapper", &farthest_point_sampling_wrapper, "farthest_point_sampling_wrapper");
    m.def("ball_query_wrapper", &ball_query_wrapper, "ball_query_wrapper");
}

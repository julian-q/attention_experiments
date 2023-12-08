#include <torch/extension.h>
#include "matmul.h"
#include "attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("simple_matmul", &simple_matmul);
    m.def("tiled_matmul", &tiled_matmul);
    m.def("simple_self_attention", &simple_self_attention);
    m.def("vectorized_self_attention", &vectorized_self_attention);
}

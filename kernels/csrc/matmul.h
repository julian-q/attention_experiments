#include <torch/extension.h>

void
simple_matmul(
    torch::Tensor C,
    torch::Tensor A,
    torch::Tensor B
);

void
tiled_matmul(
    torch::Tensor C,
    torch::Tensor A,
    torch::Tensor B
);

#include <torch/extension.h>

torch::Tensor
simple_self_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);

torch::Tensor
vectorized_self_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);

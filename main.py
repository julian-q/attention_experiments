import torch
import torch.nn.functional as F
import math
import kernel_pack
import time
torch.manual_seed(0)

M = 1024
N = 1024
K = 1024

A = torch.rand(M, K).cuda()
B = torch.rand(K, N).cuda()
C = torch.zeros(M, N).cuda()
C_torch = A @ B
kernel_pack.tiled_matmul(C, A, B)
assert torch.allclose(C_torch, C)

B = 8
H = 8
N = 32
D = 16 # note: must match compile-time constant D_ in attention.cu

trials = 10
for N in [24, 32, 33, 64, 100, 128, 256, 512, 1024, 2048, 4096, 8192]:
    Q = torch.rand(B, H, N, D).cuda()
    K = torch.rand(B, H, N, D).cuda()
    V = torch.rand(B, H, N, D).cuda()
    Q_ = Q.view(B*H, N, D)
    K_ = K.view(B*H, N, D)
    V_ = V.view(B*H, N, D)

    custom_time = 0
    torch_time = 0
    for _ in range(trials):
        start = time.time()
        O_custom = kernel_pack.simple_self_attention(Q, K, V)
        stop = time.time()
        custom_time += stop - start

        start = time.time()
        QK_t = (Q_ @ K_.transpose(-1, -2)) / math.sqrt(D)
        A = F.softmax(QK_t, dim=-1)
        O_torch = A @ V_
        stop = time.time()
        O_torch = O_torch.view(B, H, N, D)
        torch_time += stop - start

        print(f"N={N} max difference between O_torch and O_custom: {(O_torch - O_custom).abs().max()}")
        assert torch.allclose(O_torch, O_custom, atol=5e-2)
    custom_time /= trials
    torch_time /= trials
    print("custom time:", custom_time)
    print("torch time:", torch_time)
    print("speedup:", torch_time / custom_time)

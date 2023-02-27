import pandas as pd
import torch
from torch import nn

from deepctr_torch.layers import AutoDisLayer

print(torch.__version__)
torch.manual_seed(123)
field_num = 5
embedding_dim = 10
batch_size = 16
input = torch.randn(batch_size, field_num)

# att_size = 20
#
# trans_Q = nn.Parameter(torch.randn(embedding_dim, att_size))
# trans_K = nn.Parameter(torch.randn(embedding_dim, att_size))
# trans_V = nn.Parameter(torch.randn(embedding_dim, att_size))
# projection = nn.Linear(embedding_dim, att_size)
#
# Q = torch.matmul(input, trans_Q)
# K = torch.matmul(input, trans_K)
# V = torch.matmul(input, trans_V)

# attention = torch.matmul(Q, K.permute(0, 2, 1))
# w_h = nn.Linear(1, 10, bias=False)
# x = input[:, 0].unsqueeze(-1)
# w = w_h(x)
# input =torch.tensor([[1], [2], [10], [1]], dtype=torch.float32)
# autodis = AutoDisLayer(1, 3)
# autodis.train()
# out = autodis(input)
i = torch.unsqueeze(input, dim=1)

dict = nn.ParameterList([nn.Parameter(torch.randn(1,4)) for _ in range(4)])
print("end")




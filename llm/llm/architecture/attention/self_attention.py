import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    '''
    This class implements the self-attention mechanism as described in 
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    '''

    def __init__(self, d_in: int, d_out, qvk_bias: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qvk_bias)

    def forward(self, x: torch.Tensor):
        keys: torch.Tensor = self.W_key(x)
        queries: torch.Tensor = self.W_query(x)
        values: torch.Tensor = self.W_value(x)

        attention_score: torch.Tensor = torch.matmul(queries, keys.T)
        attention_weights = torch.softmax(attention_score / keys.shape[-1]**5,
                                          dim=-1)
        context_vector = torch.matmul(attention_weights, values)
        return context_vector

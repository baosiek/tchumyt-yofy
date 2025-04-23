''' DOCSTRING '''
import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module

'''
Code from:
@inproceedings{Feng2024WereRA,
    title   = {Were RNNs All We Needed?},
    author  = {Leo Feng and Frederick Tung and Mohamed Osama Ahmed and Yoshua
    Bengio and Hossein Hajimirsadegh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273025630}
}
'''


class minGRUBi(Module):
    def __init__(
            self,
            dim: int,
            expansion_factor: float = 1.0,
            proj_out: float = None,
            bidirectional: bool = False
    ):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        proj_out = default(proj_out, expansion_factor != 1.0)

        self.to_hidden_and_gate: Linear = Linear(
            dim,
            dim_inner * 2,
            bias=False
        )

        self.to_out: Linear = Linear(dim_inner, dim, bias=False) \
            if proj_out else Identity()
        
        self.bidirectional: bool = bidirectional

    def forward(
            self,
            x: torch.Tensor,
            prev_hidden: torch.Tensor = None,
            return_next_prev_hidden: torch.Tensor = False
    ):

        out_forward, next_prev_hidden_forward = self.forward_gru(x, prev_hidden, return_next_prev_hidden)

        out_backward: torch.Tensor = torch.zeros(size=out_forward.size()).to(device=x.device)
        next_prev_hidden_backward: torch.Tensor = torch.zeros(size=next_prev_hidden_forward.size()).to(device=x.device)

        if self.bidirectional:
            flipped_x = torch.flip(x, dims=[1])
            out_backward, next_prev_hidden_backward = self.forward_gru(flipped_x, prev_hidden, return_next_prev_hidden)

        if not return_next_prev_hidden:
            return out_forward + out_backward        

        return out_forward + out_backward, next_prev_hidden_forward + next_prev_hidden_backward
    
    def forward_gru(self,
            x: torch.Tensor,
            prev_hidden: torch.Tensor = None,
            return_next_prev_hidden: torch.Tensor = False):
        ''' DOCSTRING '''
        
        seq_len: int = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if seq_len == 1:
            # handle sequential

            hidden: torch.Tensor = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) \
                if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel

            log_coefficients = -F.softplus(gate)
            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim=1)
                log_coefficients = F.pad(log_coefficients, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coefficients, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]
        out = self.to_out(out)

        return out, next_prev_hidden
        



def exists(v):
    ''' DOCSTRING '''
    return v is not None


def default(v, d):
    ''' DOCSTRING '''
    return v if exists(v) else d


def heinsen_associative_scan_log(
        log_coefficients: torch.Tensor,
        log_values: torch.Tensor
) -> torch.Tensor:
    ''' DOCSTRING '''
    a_star: torch.Tensor = log_coefficients.cumsum(dim=1)
    log_h0_plus_b_star: torch.Tensor = \
        (log_values - a_star).logcumsumexp(dim=1)

    log_h: torch.Tensor = a_star + log_h0_plus_b_star

    return log_h.exp()


def g(x):
    ''' DOCSTRING '''
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    ''' DOCSTRING '''
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

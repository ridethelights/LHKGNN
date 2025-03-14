import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import math

# Check if torch.special.iv is available (PyTorch 1.9+)
HAS_TORCH_IV = hasattr(torch.special, "iv")

def bessel_iv_approx(k, x, terms=10):
    """
    Approximate the modified Bessel function I_k(x) using its Taylor series expansion.
    This function uses torch.lgamma to compute factorials (i.e. factorial(n) = exp(lgamma(n+1)))
    to avoid overflow.
    
    Args:
      k (int): Order of the Bessel function.
      x (Tensor): Input tensor.
      terms (int): Number of terms in the Taylor expansion.
      
    Returns:
      Tensor: Approximation of I_k(x) with the same shape as x.
    """
    result = torch.zeros_like(x)
    for m in range(terms):
        fact_m = torch.exp(torch.lgamma(torch.tensor(m, dtype=torch.float32, device=x.device) + 1))
        fact_mk = torch.exp(torch.lgamma(torch.tensor(m + k, dtype=torch.float32, device=x.device) + 1))
        denom = fact_m * fact_mk
        # Note: we use (x)^(2*m+k)/(2^(2*m+k)) to match the standard series expansion.
        term = (x ** (2 * m + k)) / (denom * (2 ** (2 * m + k)))
        result = result + term
    return result


def truncated_normal(size, mean=1.5, std=0.5, min_val=0.0, max_val=3.0, device="cuda"):
    t = torch.randn(size, device=device) * std + mean  # Sample from Normal(mean, std)
    return torch.clamp(t, min_val, max_val)  # Ensure values stay within [min_val, max_val]
    
class NodeLevelHeatDiffusionWithGamma(MessagePassing):
    """
    Node-Level Heat Diffusion approximation based on the Laplacian L using a Chebyshev expansion,
    with per-node learnable parameters gamma0 and gamma1.
    
    We start with the symmetric normalized Laplacian:
      L = I - A_norm,
    where A_norm is the symmetric normalized adjacency matrix (assumed to have eigenvalues in [0,1]).
    
    For the normalized Laplacian, λ_max(L)=2. We then define the scaled Laplacian as:
      L_tilde = (2/λ_max)*L - I. For λ_max=2, we have L_tilde = L - I.
    Note that, since L = I - A_norm, it follows that:
      L_tilde = (I - A_norm) - I = -A_norm.
      
    Thus, the heat kernel is given by:
      e^{-tL} = e^{-t(I-A_norm)} = e^{-t} e^{tA_norm} = e^{-t} e^{-tL_tilde},
    and we approximate e^{-tL_tilde} using the Chebyshev expansion:
      e^{-tL_tilde} ≈ I_0(t) + 2∑_{k=1}^K (-1)^k I_k(t) T_k(L_tilde),
    where T_k(L_tilde) is computed recursively:
      T_0(x) = x,  T_1(x) = L_tilde x,  T_k(x) = 2 L_tilde T_{k-1}(x) - T_{k-2}(x).
    
    To obtain node-level heat diffusion, we assign each node i its own diffusion parameter t_i,
    and two learnable parameters gamma0_i and gamma1_i. The output for node i is then:
    
      y_i = gamma0_i * x_i + gamma1_i * (e^{-t_iL} x)_i,
      
    where
      (e^{-t_iL} x)_i ≈ e^{-t_i} [ I_0(t_i)x_i + 2∑_{k=1}^K (-1)^k I_k(t_i) (T_k(L_tilde)x)_i ].
      
    Note: The impulse (or identity) is already embedded in T_0(x)=x.
    """
    def __init__(self, K=10, node_t_init=2.5, dp=0.5,num=None,gamma0_init=1.0, gamma1_init=1.0):
        super().__init__(aggr='add')
        self.K = K
        self.node_t_init = node_t_init
        self.gamma0_init = gamma0_init
        self.gamma1_init = gamma1_init
        self.dp=dp
        self.N=num
        
        self.node_t = nn.Parameter(torch.rand(self.N) * 2 * self.node_t_init)   # Per-node diffusion parameter, shape: [N]
        self.gamma0 = nn.Parameter(torch.ones(self.N) * self.gamma0_init)    # Per-node parameter gamma0, shape: [N]
        self.gamma1 = nn.Parameter(torch.ones(self.N) * self.gamma1_init)    # Per-node parameter gamma1, shape: [N]
        
        




    def forward(self, x, edge_index, edge_weight=None):
        N, d = x.size()
        device = x.device

        # Initialize per-node parameters if not already set.
        if self.node_t is None or self.node_t.numel() != N:
            self.node_t = nn.Parameter(torch.rand(N, device=x.device) * 2 * self.node_t_init)
        if self.gamma0 is None or self.gamma0.numel() != N:
            self.gamma0 = nn.Parameter(torch.ones(N, device=device) * self.gamma0_init)
        if self.gamma1 is None or self.gamma1.numel() != N:
            self.gamma1 = nn.Parameter(torch.ones(N, device=device) * self.gamma1_init)
        
        # Compute the symmetric normalized adjacency matrix A_norm.
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device)
        row, col = edge_index
        deg = degree(row, num_nodes=N, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        A_norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        A_sparse = torch.sparse_coo_tensor(edge_index, A_norm_values, (N, N))
        
        # Compute the Laplacian: L = I - A_norm.
        I = torch.eye(N, device=device)
        L = I - A_sparse.to_dense()
        
        # Compute the scaled Laplacian: L_tilde = L - I. Since L = I - A_norm,
        # we have L_tilde = (I - A_norm) - I = -A_norm.
        L_tilde = L - I

        # Ensure per-node diffusion parameters are non-negative.
        t = F.relu(self.node_t)
        # We directly use t as the expansion parameter in our Chebyshev series.
        # Compute per-node coefficients:
        # For k = 0: c0(i) = exp(-t_i) * I_0(t_i)
        # For k >= 1: c_k(i) = 2 * exp(-t_i) * (-1)^k I_k(t_i)
        c_list = []
        c0 = torch.exp(-t) * (torch.special.i0(t) if HAS_TORCH_IV else bessel_iv_approx(0, t))
        c_list.append(c0)
        for k in range(1, self.K + 1):
            sign = (-1) ** k
            ck = 2 * torch.exp(-t) * sign * (torch.special.iv(k, t) if HAS_TORCH_IV else bessel_iv_approx(k, t))
            c_list.append(ck)

        # Compute the Chebyshev polynomials T_k(L_tilde)x.
        # T_0(x) = x.
        T_list = [x]
        # T_1(x) = L_tilde x.
        T1 = torch.matmul(L_tilde, x)
        T_list.append(T1)
        for k in range(2, self.K + 1):
            T_k = 2 * torch.matmul(L_tilde, T_list[k - 1]) - T_list[k - 2]
            T_list.append(T_k)
        
        # Combine the Chebyshev terms with per-node coefficients to approximate e^{-t_i L}x.
        diffusion = torch.zeros_like(x)
        for k in range(self.K + 1):
            # Each coefficient is a vector of shape [N]. We unsqueeze(-1) to match x's shape [N, d].
            diffusion = diffusion + T_list[k] * c_list[k].unsqueeze(-1)
        
        # Final output: y_i = gamma0_i * x_i + gamma1_i * (diffusion)_i.
        out = self.gamma0.unsqueeze(-1) * x + self.gamma1.unsqueeze(-1) * diffusion
        #out=torch.tanh(out)
        return out

    def message(self, x_j, norm):
        # Not used because we compute via dense matrix multiplication.
        return x_j

    def update(self, aggr_out):
        return aggr_out

class LHKGNN(nn.Module):
    """
    LHKGNN model with node-level heat diffusion and per-node learnable parameters β0 and β1.
    
    The workflow:
      1. Transform input features via an MLP.
      2. Apply node-level heat diffusion (based on the Laplacian L and a Chebyshev expansion)
         with per-node diffusion parameter t_i, and per-node parameters β0 and β1.
      3. Output the node classification scores via log_softmax.
    """
    def __init__(self, in_dim, hidden, out_dim, K=10, node_t_init=0.1,dp=0.5, num=None,gamma0_init=1, gamma1_init=1):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dp
        self.diffusion = NodeLevelHeatDiffusionWithGamma(K=K, node_t_init=node_t_init,num=num,
                                                        gamma0_init=gamma0_init, gamma1_init=gamma1_init)
    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.diffusion(x, edge_index, edge_weight)
      
        return F.log_softmax(x, dim=1)

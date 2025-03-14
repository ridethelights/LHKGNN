#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import get_laplacian
from torch_scatter import scatter_add
from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.nn import GCNConv, ChebConv, APPNP, SGConv 
import torch.nn as nn
from torch_sparse import spmm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops,remove_self_loops, degree
from torch_scatter import scatter_add
from torch_geometric.nn import GATConv, SAGEConv, GraphConv
from torch_geometric.nn import JumpingKnowledge
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_layer = nn.Dropout(p=0.5)
    def forward(self, x, edge_index):
        x=self.dropout_layer(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10,
                 alpha=0.1, dropout=0.5, Init="Random", Gamma=None):
        super(GPRGNN, self).__init__()
        self.K = K                   # Number of propagation steps
        self.alpha = alpha           # Teleport probability
        self.dropout = dropout       # Dropout rate
        self.Init = Init             # Initialization type
        self.Gamma = Gamma           # Predefined Gamma for WS initialization

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop_weights = nn.Parameter(torch.ones(K + 1))  # K+1 propagation weights
        self.reset_parameters()      # Initialize propagation weights

    def reset_parameters(self):
        """Initialize propagation weights based on the chosen method."""
        if self.Init == 'SGC':
            weights = torch.zeros(self.K + 1)
            # Here, 'alpha' is assumed to indicate the index for the one-hot peak.
            weights[int(self.alpha)] = 1.0  
        elif self.Init == 'PPR':
            weights = torch.tensor([self.alpha * (1 - self.alpha) ** k for k in range(self.K + 1)])
            weights[-1] = (1 - self.alpha) ** self.K  # Special handling for last weight
        elif self.Init == 'NPPR':
            weights = torch.tensor([self.alpha ** k for k in range(self.K + 1)])
            weights = weights / weights.abs().sum()
        elif self.Init == 'Random':
            bound = (3 / (self.K + 1)) ** 0.5
            weights = torch.empty(self.K + 1).uniform_(-bound, bound)
            weights = weights / weights.abs().sum()
        elif self.Init == 'WS':
            if self.Gamma is None:
                raise ValueError("Gamma must be provided for WS initialization.")
            weights = torch.tensor(self.Gamma)
        else:
            raise ValueError(f"Unknown initialization method: {self.Init}")

        with torch.no_grad():
            self.prop_weights.copy_(weights)

    def forward(self, x, edge_index):
        # Apply dropout, activation, and linear layers
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # Compute normalized adjacency
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # GPR propagation
        H = x
        out = self.prop_weights[0] * H
        for k in range(1, self.K + 1):
            H = self.propagate(edge_index, H, norm)
            out = out + self.prop_weights[k] * H

        return out

    def propagate(self, edge_index, x, norm):
        # Basic message passing: aggregate features using normalized weights
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, col, norm.view(-1, 1) * x[row])
        return out




class GraphHeat(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, s=0.1, K=10):

        super(GraphHeat, self).__init__()
        
        self.theta_direct = nn.Parameter(torch.randn(in_channels, hidden_channels))
        self.theta_heat1  = nn.Parameter(torch.randn(in_channels, hidden_channels))
        self.theta_hidden = nn.Parameter(torch.randn(hidden_channels, out_channels))
        self.theta_heat2  = nn.Parameter(torch.randn(hidden_channels, out_channels))
        

        self.t = nn.Parameter(torch.tensor(s))
        self.K = K

    def forward(self, x, edge_index):

        num_nodes = x.size(0)
        

        lap_index, lap_weight = get_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
        L = torch.sparse.FloatTensor(lap_index, lap_weight, torch.Size([num_nodes, num_nodes])).to(x.device)
        

        x_heat = self._heat_kernel_approx(x, L, self.t, self.K)
        
        hidden = F.relu(x @ self.theta_direct + x_heat @ self.theta_heat1)
        
        hidden_heat = self._heat_kernel_approx(hidden, L, self.t, self.K)
        
        out = hidden @ self.theta_hidden + hidden_heat @ self.theta_heat2
        
        return F.log_softmax(out, dim=1)

    def _heat_kernel_approx(self, x, L, t, K):

        num_nodes = x.size(0)
        I = torch.eye(num_nodes, device=x.device)
        I_sparse = I.to_sparse()
        tilde_L = L - I_sparse

        # T_0 = x
        T0 = x
        out = torch.special.iv(0, t).unsqueeze(0) * T0  
        
        if K > 1:

            T1 = torch.sparse.mm(tilde_L, x)
            out = out + 2 * ((-1)**1) * torch.special.iv(1, t).unsqueeze(0) * T1

        Tkm2 = T0
        if K > 1:
            Tkm1 = T1
        for k in range(2, K):
            T_k = 2 * torch.sparse.mm(tilde_L, Tkm1) - Tkm2
            coeff = 2 * ((-1)**k) * torch.special.iv(float(k), t).unsqueeze(0)
            out = out + coeff * T_k
            Tkm2, Tkm1 = Tkm1, T_k

        return out







class APPNPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1, dropout=0.5):

        super(APPNPNet, self).__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        self.prop = APPNP(K=K, alpha=alpha, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)


def compute_diffusion_matrix(A, diffusion_type="ppr", alpha=0.05, t=1.0, K=10):

    N = A.size(0)

    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
  
    indices = torch.stack([torch.arange(N), torch.arange(N)], dim=0).to(A.device)
    D_inv = torch.sparse.FloatTensor(indices, deg_inv, torch.Size([N, N]))
    
 
    P = torch.sparse.mm(D_inv, A)
    

    if diffusion_type == "ppr":
        theta = lambda k: alpha * ((1 - alpha) ** k)
    elif diffusion_type == "heat":
        theta = lambda k: math.exp(-t) * (t ** k) / math.factorial(k)
    else:
        raise ValueError("error")
    
  
    I_indices = torch.stack([torch.arange(N), torch.arange(N)], dim=0).to(A.device)
    I_values = torch.ones(N, device=A.device)
    I_sparse = torch.sparse.FloatTensor(I_indices, I_values, torch.Size([N, N]))
    S = theta(0) * I_sparse
    

    Pk = I_sparse  # P^0
    for k in range(1, K+1):
        Pk = torch.sparse.mm(Pk, P)
        S = S + theta(k) * Pk
    return S

class GDCModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 diffusion_type="ppr", alpha=0.05, t=1.0, K=10, dropout=0.5):

        super(GDCModel, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.diffusion_type = diffusion_type
        self.alpha = alpha
        self.t = t
        self.K = K
        self.dropout = dropout
        
    def forward(self, x, edge_index, num_nodes):


        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        N = num_nodes
        device = x.device

        values = torch.ones(edge_index.size(1), device=device)
        A = torch.sparse.FloatTensor(edge_index, values, torch.Size([N, N]))
        

        S = compute_diffusion_matrix(A,
                                     diffusion_type=self.diffusion_type,
                                     alpha=self.alpha,
                                     t=self.t,
                                     K=self.K)

        x_diffused = torch.sparse.mm(S, x)

        x = F.dropout(x_diffused, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class ReactionNet(MessagePassing):
   # This code is adapted from the HiD-Net repository:
   #   https://github.com/BUPT-GAMMA/HiD-Net
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, args, in_channels: int, out_channels: int, bias: bool = False,
                cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.k = args.k
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.sigma1 = args.sigma1
        self.sigma2 = args.sigma2
        self.drop = args.drop
        self.dropout = args.dropout
        self.calg = 'g3'
        if args.dataset == 'pubmed':
            self.calg = 'g4'
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.lin1 = Linear(in_channels, args.hidden, bias=False, weight_initializer='glorot')
        self.lin2 = Linear(args.hidden, out_channels, bias=False, weight_initializer='glorot')
        self.relu = ReLU()
        self.reg_params = list(self.lin1.parameters())
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            edgei = edge_index
            edgew = edge_weight
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edgei, edgew, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
                edge_index2, edge_weight2 = gcn_norm(  # yapf: disable
                    edgei, edgew, x.size(self.node_dim), False,
                    False, dtype=x.dtype)

                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]
            ew = edge_weight.view(-1, 1)
            ew2 = edge_weight2.view(-1, 1)



        # preprocess
        if self.drop == 'True':
            x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin2(x)

        h = x
        for k in range(self.k):

            if self.calg == 'g3' or self.calg == 'cal_gradient_2':  # TODO
                g = cal_g_gradient3(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
            elif self.calg == 'g1':
                g = cal_g_gradient1(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
            elif self.calg == 'g2':
                g = cal_g_gradient2(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
            elif self.calg == 'g4':
                g = cal_g_gradient4(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
            elif self.calg == 'g5':
                g = cal_g_gradient5(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
            elif self.calg == 'ggat':
                g = cal_g_gradient_gat(edge_index2, x, self.gat1, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)

            adj = torch.sparse_coo_tensor(edge_index, edge_weight, [x.size(0), x.size(0)])
            Ax = torch.spmm(adj, x)
            Gx = torch.spmm(adj, g)
            x = self.alpha * h + (1 - self.alpha - self.beta) * x  \
                + self.beta * Ax \
                + self.beta * self.gamma * Gx

        out = F.log_softmax(x, dim=-1)

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # return edge_weight
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.k}, alpha={self.alpha})'

class HeatKernelLayer(nn.Module):

    def __init__(self, in_channels, out_channels, t=1.0, use_chebyshev=False, K=10):

        super(HeatKernelLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = t  
        self.use_chebyshev = use_chebyshev
        self.K = K
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):

        N = x.size(0)
        device = x.device
        

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device)
        

        adj = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([N, N])).to(device)
        

        deg = torch.sparse.sum(adj, dim=1).to_dense()

        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        

        A_dense = adj.to_dense()
        A_norm = D_inv_sqrt @ A_dense @ D_inv_sqrt
   
        L = torch.eye(N, device=device) - A_norm
        

        if not self.use_chebyshev:

            H_kernel = torch.matrix_exp(-self.t * L)
        else:

            L_max = 2.0
            L_tilde = (2.0 / L_max) * L - torch.eye(N, device=device)

            T0 = torch.eye(N, device=device)
            T1 = L_tilde
            H_kernel = self.cheb_coeff(0) * T0 + self.cheb_coeff(1) * T1
            Tkm2 = T0
            Tkm1 = T1
            for k in range(2, self.K):
                Tk = 2 * L_tilde @ Tkm1 - Tkm2
                H_kernel = H_kernel + self.cheb_coeff(k) * Tk
                Tkm2, Tkm1 = Tkm1, Tk
        

        x_diffused = H_kernel @ x

        out = self.linear(x_diffused)
        return out

    def cheb_coeff(self, k):

        return ((-self.t) ** k) / torch.exp(torch.lgamma(torch.tensor(k + 1.0, device=self.linear.weight.device)))


class HKGCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, t=1.0, dropout=0.5, use_chebyshev=False, K=10):

        super(HeatKernelGCN, self).__init__()
        self.dropout = dropout
        
        self.layers = nn.ModuleList()

        self.layers.append(HeatKernelLayer(in_channels, hidden_channels, t=t, 
                                           use_chebyshev=use_chebyshev, K=K))

        for _ in range(num_layers - 2):
            self.layers.append(HeatKernelLayer(hidden_channels, hidden_channels, t=t, 
                                               use_chebyshev=use_chebyshev, K=K))

        self.layers.append(HeatKernelLayer(hidden_channels, out_channels, t=t, 
                                           use_chebyshev=use_chebyshev, K=K))
        
    def forward(self, x, edge_index, edge_weight=None):


        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


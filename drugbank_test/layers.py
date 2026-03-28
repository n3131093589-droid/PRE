import math
import datetime

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv,SAGPooling,global_add_pool,GATConv



class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores
        return attentions

class RESCAL(nn.Module):
    """根据药物head, 药物tail和关系rel计算作用分值"""

    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    
    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)
      
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        
        rels = rels.view(-1, self.n_features, self.n_features)
        # print(heads.size(),rels.size(),tails.size())
        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
          scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))
       
        return scores 
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"

class IntraGraphAttention(nn.Module):
    """包含单层GAT, 对分子Graph进行学习"""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim,32,2)
    
    def forward(self,data):
        input_feature,edge_index = data.x, data.edge_index
        input_feature = F.elu(input_feature)
        intra_rep = self.intra(input_feature,edge_index)
        return intra_rep

class InterGraphAttention(nn.Module):
    """包含单层GAT, 对两个药物的Bipartite Graph进行学习"""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv(input_dim,32,2,dropout=0.3)

    def _get_local_edge_index(self, edge_index, start, end):
        mask = (edge_index[0] >= start) & (edge_index[0] < end)
        return edge_index[:, mask] - start

    def _get_virtual_node_feature(self, node_feature, node_type):
        if node_type is not None:
            mol_mask = node_type == 2
            if mol_mask.any():
                return node_feature[mol_mask].mean(dim=0, keepdim=True)
        return node_feature.mean(dim=0, keepdim=True)
    
    def forward(self,h_data,t_data,b_graph):
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        device = h_input.device
        h_ptr = h_data.ptr
        t_ptr = t_data.ptr

        fused_x_parts = []
        fused_edge_parts = []
        h_sizes = []
        t_sizes = []
        node_offset = 0

        for graph_idx in range(h_ptr.numel() - 1):
            h_start = int(h_ptr[graph_idx].item())
            h_end = int(h_ptr[graph_idx + 1].item())
            t_start = int(t_ptr[graph_idx].item())
            t_end = int(t_ptr[graph_idx + 1].item())

            h_feature = h_input[h_start:h_end]
            t_feature = t_input[t_start:t_end]
            h_type = h_data.y[h_start:h_end] if hasattr(h_data, 'y') else None
            t_type = t_data.y[t_start:t_end] if hasattr(t_data, 'y') else None

            v_feature = (
                self._get_virtual_node_feature(h_feature, h_type)
                + self._get_virtual_node_feature(t_feature, t_type)
            ) / 2

            fused_x = torch.cat([h_feature, t_feature, v_feature], dim=0)
            virtual_index = h_feature.size(0) + t_feature.size(0)
            drug_nodes = torch.arange(virtual_index, device=device)
            virtual_nodes = torch.full((virtual_index,), virtual_index, dtype=torch.long, device=device)
            virtual_edge_index = torch.stack([
                torch.cat([drug_nodes, virtual_nodes]),
                torch.cat([virtual_nodes, drug_nodes]),
            ], dim=0)

            h_edge_index = self._get_local_edge_index(h_data.edge_index, h_start, h_end)
            t_edge_index = self._get_local_edge_index(t_data.edge_index, t_start, t_end) + h_feature.size(0)
            fused_edge_index = torch.cat([h_edge_index, t_edge_index, virtual_edge_index], dim=1)

            fused_x_parts.append(fused_x)
            fused_edge_parts.append(fused_edge_index + node_offset)
            h_sizes.append(h_feature.size(0))
            t_sizes.append(t_feature.size(0))
            node_offset += fused_x.size(0)

        fused_x = torch.cat(fused_x_parts, dim=0)
        fused_edge_index = torch.cat(fused_edge_parts, dim=1)
        fused_rep = self.inter(fused_x, fused_edge_index)

        h_x_inter_parts = []
        h_y_inter_parts = []
        node_offset = 0
        for h_size, t_size in zip(h_sizes, t_sizes):
            h_x_inter_parts.append(fused_rep[node_offset:node_offset + h_size])
            node_offset += h_size
            h_y_inter_parts.append(fused_rep[node_offset:node_offset + t_size])
            node_offset += t_size + 1

        h_x_inter = torch.cat(h_x_inter_parts, dim=0)
        h_y_inter = torch.cat(h_y_inter_parts, dim=0)
        return h_x_inter,h_y_inter




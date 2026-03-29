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
        self.heads = 2
        self.head_out_feats = 32
        self.dropout = 0.3
        self.inter = GATConv(input_dim, self.head_out_feats, self.heads, dropout=self.dropout)

    def _segment_mean(self, node_feature, batch, n_graphs):
        pooled = node_feature.new_zeros((n_graphs, node_feature.size(-1)))
        pooled.index_add_(0, batch, node_feature)
        counts = node_feature.new_zeros((n_graphs, 1))
        counts.index_add_(0, batch, node_feature.new_ones((node_feature.size(0), 1)))
        return pooled / counts.clamp_min(1.0)

    def _get_virtual_node_feature(self, node_feature, batch, node_type, n_graphs):
        graph_mean = self._segment_mean(node_feature, batch, n_graphs)
        if node_type is None:
            return graph_mean

        mol_mask = node_type == 2
        if not mol_mask.any():
            return graph_mean

        mol_feature = node_feature[mol_mask]
        mol_batch = batch[mol_mask]
        mol_mean = self._segment_mean(mol_feature, mol_batch, n_graphs)
        mol_counts = node_feature.new_zeros((n_graphs, 1))
        mol_counts.index_add_(0, mol_batch, node_feature.new_ones((mol_feature.size(0), 1)))
        return torch.where(mol_counts > 0, mol_mean, graph_mean)

    def forward(self,h_data,t_data,b_graph):
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        h_num_nodes = h_input.size(0)
        t_num_nodes = t_input.size(0)
        n_graphs = int(torch.maximum(h_data.batch.max(), t_data.batch.max()).item()) + 1

        h_type = h_data.y if hasattr(h_data, 'y') else None
        t_type = t_data.y if hasattr(t_data, 'y') else None
        h_virtual = self._get_virtual_node_feature(h_input, h_data.batch, h_type, n_graphs)
        t_virtual = self._get_virtual_node_feature(t_input, t_data.batch, t_type, n_graphs)
        virtual_feature = (h_virtual + t_virtual) / 2

        virtual_offset = h_num_nodes + t_num_nodes
        virtual_indices = torch.arange(n_graphs, device=h_input.device) + virtual_offset

        h_node_indices = torch.arange(h_num_nodes, device=h_input.device)
        t_node_indices = torch.arange(t_num_nodes, device=t_input.device) + h_num_nodes

        h_virtual_edges = torch.stack([
            torch.cat([h_node_indices, virtual_indices[h_data.batch]]),
            torch.cat([virtual_indices[h_data.batch], h_node_indices]),
        ], dim=0)
        t_virtual_edges = torch.stack([
            torch.cat([t_node_indices, virtual_indices[t_data.batch]]),
            torch.cat([virtual_indices[t_data.batch], t_node_indices]),
        ], dim=0)

        fused_x = torch.cat([h_input, t_input, virtual_feature], dim=0)
        fused_edge_index = torch.cat([
            h_data.edge_index,
            t_data.edge_index + h_num_nodes,
            h_virtual_edges,
            t_virtual_edges,
        ], dim=1)
        fused_rep = self.inter(fused_x, fused_edge_index)

        h_x_inter = fused_rep[:h_num_nodes]
        h_y_inter = fused_rep[h_num_nodes:virtual_offset]
        return h_x_inter,h_y_inter




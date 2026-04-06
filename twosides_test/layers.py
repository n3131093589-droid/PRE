import math
import datetime

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv,SAGPooling,global_add_pool,GATConv
from torch_geometric.utils import softmax


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
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores
        return attentions


class RESCAL(nn.Module):
    """根据药物head, 药物tail和关系rel计算作用分值"""

    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * 2)
        self.rel_proj = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.n_features * 2, self.n_features * 2),
            nn.ELU(),
            nn.Linear(self.n_features * 2, self.n_features),
        )
        nn.init.xavier_uniform_(self.rel_emb.weight)

    
    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)
        rels = self.rel_proj(rels)
      
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        pair = (heads.unsqueeze(-3) * tails.unsqueeze(-2)).unsqueeze(-2)
       
        rels = rels.view(-1,1,1,self.n_features,1)
        scores = ((torch.matmul(pair,rels)).squeeze(-1)).squeeze(-1)

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


class PairConditionedNodeGate(nn.Module):
    def __init__(self, input_dim, target_type=1, use_relation=False):
        super().__init__()
        self.input_dim = input_dim
        self.target_type = target_type
        self.use_relation = use_relation
        gate_input_dim = input_dim * 3 if self.use_relation else input_dim * 2
        self.gate = nn.Linear(gate_input_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _segment_mean(self, node_feature, batch, n_graphs):
        pooled = node_feature.new_zeros((n_graphs, node_feature.size(-1)))
        pooled.index_add_(0, batch, node_feature)
        counts = node_feature.new_zeros((n_graphs, 1))
        counts.index_add_(0, batch, node_feature.new_ones((node_feature.size(0), 1)))
        return pooled / counts.clamp_min(1.0)

    def _get_context(self, node_feature, batch, node_type, n_graphs):
        graph_mean = self._segment_mean(node_feature, batch, n_graphs)
        if node_type is None:
            return graph_mean

        target_mask = node_type == self.target_type
        if not target_mask.any():
            return graph_mean

        target_feature = node_feature[target_mask]
        target_batch = batch[target_mask]
        target_mean = self._segment_mean(target_feature, target_batch, n_graphs)
        target_counts = node_feature.new_zeros((n_graphs, 1))
        target_counts.index_add_(0, target_batch, node_feature.new_ones((target_feature.size(0), 1)))
        return torch.where(target_counts > 0, target_mean, graph_mean)

    def _apply_gate(self, node_feature, batch, node_type, partner_context, relation_context=None):
        gate_inputs = [node_feature, partner_context[batch]]
        if relation_context is not None:
            gate_inputs.append(relation_context[batch])
        gate_input = torch.cat(gate_inputs, dim=-1)
        gate_scale = 2.0 * torch.sigmoid(self.gate(gate_input))
        if node_type is None:
            return node_feature * gate_scale

        target_mask = node_type == self.target_type
        if not target_mask.any():
            return node_feature

        full_scale = torch.ones_like(gate_scale)
        full_scale[target_mask] = gate_scale[target_mask]
        return node_feature * full_scale

    def forward(self, h_feature, h_batch, h_type, t_feature, t_batch, t_type, relation_context=None):
        n_graphs = int(torch.maximum(h_batch.max(), t_batch.max()).item()) + 1
        h_context = self._get_context(h_feature, h_batch, h_type, n_graphs)
        t_context = self._get_context(t_feature, t_batch, t_type, n_graphs)
        h_feature = self._apply_gate(h_feature, h_batch, h_type, t_context, relation_context=relation_context)
        t_feature = self._apply_gate(t_feature, t_batch, t_type, h_context, relation_context=relation_context)
        return h_feature, t_feature


class InterGraphAttention(nn.Module):
    """包含单层GAT, 对两个药物的Bipartite Graph进行学习"""

    def __init__(self, input_dim, ablation_mode=2):
        super().__init__()
        self.input_dim = input_dim
        self.heads = 2
        self.head_out_feats = 32
        self.out_dim = self.heads * self.head_out_feats
        self.dropout = 0.3
        self.ablation_mode = ablation_mode

        self.inter = GATConv((input_dim, input_dim), self.head_out_feats, self.heads, dropout=self.dropout)
        self.node_key_proj = nn.Linear(input_dim, self.out_dim, bias=False)
        self.node_value_proj = nn.Linear(input_dim, self.out_dim, bias=False)
        self.virtual_query_proj = nn.Linear(input_dim, self.out_dim, bias=False)
        self.virtual_residual_proj = nn.Linear(input_dim, self.out_dim, bias=False)
        self.node_query_proj = nn.Linear(input_dim, self.out_dim, bias=False)
        self.virtual_key_proj = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.virtual_value_proj = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.h_aux_gate = nn.Linear(self.out_dim * 2, self.out_dim)
        self.t_aux_gate = nn.Linear(self.out_dim * 2, self.out_dim)
        nn.init.constant_(self.h_aux_gate.bias, -2.0)
        nn.init.constant_(self.t_aux_gate.bias, -2.0)

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

    def _compute_virtual_message(self, all_nodes, pair_ids, virtual_feature):
        node_key = self.node_key_proj(all_nodes).view(-1, self.heads, self.head_out_feats)
        node_value = self.node_value_proj(all_nodes).view(-1, self.heads, self.head_out_feats)
        virtual_query = self.virtual_query_proj(virtual_feature).view(-1, self.heads, self.head_out_feats)

        scores = (node_key * virtual_query[pair_ids]).sum(dim=-1) / math.sqrt(self.head_out_feats)
        attention = softmax(scores, pair_ids)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        virtual_context = all_nodes.new_zeros((virtual_feature.size(0), self.heads, self.head_out_feats))
        virtual_context.index_add_(0, pair_ids, node_value * attention.unsqueeze(-1))
        virtual_state = self.virtual_residual_proj(virtual_feature).view(-1, self.heads, self.head_out_feats)
        virtual_state = F.elu(virtual_state + virtual_context).reshape(-1, self.out_dim)

        node_query = self.node_query_proj(all_nodes).view(-1, self.heads, self.head_out_feats)
        virtual_key = self.virtual_key_proj(virtual_state).view(-1, self.heads, self.head_out_feats)
        virtual_value = self.virtual_value_proj(virtual_state).view(-1, self.heads, self.head_out_feats)
        gate = torch.sigmoid((node_query * virtual_key[pair_ids]).sum(dim=-1) / math.sqrt(self.head_out_feats))
        gate = F.dropout(gate, p=self.dropout, training=self.training)
        return (gate.unsqueeze(-1) * virtual_value[pair_ids]).reshape(-1, self.out_dim)

    def forward(self,h_data,t_data,b_graph):
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        n_graphs = int(torch.maximum(h_data.batch.max(), t_data.batch.max()).item()) + 1

        edge_index = b_graph.edge_index
        h_main = self.inter((t_input, h_input), edge_index[[1, 0]])
        t_main = self.inter((h_input, t_input), edge_index)

        h_type = h_data.y if hasattr(h_data, 'y') else None
        t_type = t_data.y if hasattr(t_data, 'y') else None
        h_virtual = self._get_virtual_node_feature(h_input, h_data.batch, h_type, n_graphs)
        t_virtual = self._get_virtual_node_feature(t_input, t_data.batch, t_type, n_graphs)
        virtual_feature = (h_virtual + t_virtual) / 2

        all_nodes = torch.cat([h_input, t_input], dim=0)
        pair_ids = torch.cat([h_data.batch, t_data.batch], dim=0)
        virtual_message = self._compute_virtual_message(all_nodes, pair_ids, virtual_feature)

        h_virtual_message = virtual_message[:h_input.size(0)]
        t_virtual_message = virtual_message[h_input.size(0):]

        if self.ablation_mode == 1:
            return h_main, t_main

        if self.ablation_mode == 3:
            return h_main + h_virtual_message, t_main + t_virtual_message

        if self.ablation_mode == 4:
            return h_virtual_message, t_virtual_message

        h_aux_weight = torch.sigmoid(self.h_aux_gate(torch.cat([h_main, h_virtual_message], dim=-1)))
        t_aux_weight = torch.sigmoid(self.t_aux_gate(torch.cat([t_main, t_virtual_message], dim=-1)))
        return h_main + h_aux_weight * h_virtual_message, t_main + t_aux_weight * t_virtual_message

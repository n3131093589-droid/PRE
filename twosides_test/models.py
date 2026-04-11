import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (
                                GATConv,
                                LayerNorm,
                                )

from layers import (
                    CoAttentionLayer, 
                    PairConditionedNodeGate,
                    RESCAL, 
                    IntraGraphAttention,
                    InterGraphAttention,
                    )

def get_node(node_rep, batch, type, needed_type):
    emb_dim = node_rep.shape[-1]
    node_rep = node_rep.masked_select((type==needed_type).unsqueeze(-1))
    batch = batch.masked_select(type==needed_type)
    node_rep = node_rep.reshape(-1, emb_dim)
    return node_rep, batch


class RelationAwareResidualUpdate(nn.Module):
    def __init__(self, feature_dim, relation_dim):
        super().__init__()
        self.rel_proj = nn.Linear(relation_dim, feature_dim, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, fused_feature, batch, relation_context):
        if relation_context is None:
            return fused_feature
        relation_feature = self.rel_proj(relation_context)[batch]
        residual = self.residual(torch.cat([fused_feature, relation_feature], dim=-1))
        return fused_feature + residual


class BidirectionalRelationAwareResidualUpdate(nn.Module):
    def __init__(self, feature_dim, relation_dim):
        super().__init__()
        self.rel_proj = nn.Linear(relation_dim, feature_dim, bias=False)
        self.intra_residual = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.inter_residual = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        nn.init.zeros_(self.intra_residual[-1].weight)
        nn.init.zeros_(self.intra_residual[-1].bias)
        nn.init.zeros_(self.inter_residual[-1].weight)
        nn.init.zeros_(self.inter_residual[-1].bias)

    def forward(self, intra_feature, inter_feature, batch, relation_context):
        if relation_context is None:
            return intra_feature, inter_feature
        relation_feature = self.rel_proj(relation_context)[batch]
        intra_residual = self.intra_residual(torch.cat([intra_feature, inter_feature, relation_feature], dim=-1))
        inter_residual = self.inter_residual(torch.cat([inter_feature, intra_feature, relation_feature], dim=-1))
        return intra_feature + intra_residual, inter_feature + inter_residual

class HDN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params, ablation_mode=2, node_gate_mode=0, update_mode=0):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)
        self.ablation_mode = ablation_mode
        self.node_gate_mode = node_gate_mode
        self.update_mode = update_mode
        
        self.initial_norm = LayerNorm(self.in_features)
        self.relation_context_emb = nn.Embedding(self.rel_total, self.kge_dim) if self.node_gate_mode == 2 or self.update_mode in (1, 2) else None
        if self.relation_context_emb is not None:
            nn.init.xavier_uniform_(self.relation_context_emb.weight)
        self.blocks = []
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = HDN_DDI_Block(
                n_heads,
                in_features,
                head_out_feats,
                final_out_feats=self.hidd_dim,
                ablation_mode=self.ablation_mode,
                node_gate_mode=self.node_gate_mode,
                update_mode=self.update_mode,
                relation_dim=self.kge_dim,
            )
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            in_features = head_out_feats * n_heads
        
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)
        self.initial_conv = GATConv(self.in_features, heads_out_feat_params[0], blocks_params[0])

    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples
        relation_context = self.relation_context_emb(rels.view(-1)) if self.relation_context_emb is not None else None

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        h_data.x = self.initial_conv(h_data.x, h_data.edge_index)
        t_data.x = self.initial_conv(t_data.x, t_data.edge_index)
        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph, relation_context)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        kge_heads = repr_h
        kge_tails = repr_t
        attentions = self.co_attention(kge_heads, kge_tails)
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        return scores     

class HDN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats, ablation_mode=2, node_gate_mode=0, update_mode=0, relation_dim=None):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.node_gate_mode = node_gate_mode
        self.update_mode = update_mode
        fusion_dim = n_heads * head_out_feats
        branch_dim = fusion_dim // 2

        self.intraAtt = IntraGraphAttention(head_out_feats*n_heads)
        self.interAtt = InterGraphAttention(head_out_feats*n_heads, ablation_mode=ablation_mode)
        self.nodeGate = PairConditionedNodeGate(head_out_feats * n_heads, use_relation=self.node_gate_mode == 2) if self.node_gate_mode in (1, 2) else None
        self.nodeGateRelProj = nn.Linear(relation_dim, head_out_feats * n_heads, bias=False) if self.node_gate_mode == 2 and relation_dim is not None else None
        self.hRelationUpdate = RelationAwareResidualUpdate(fusion_dim, relation_dim) if self.update_mode == 1 and relation_dim is not None else None
        self.tRelationUpdate = RelationAwareResidualUpdate(fusion_dim, relation_dim) if self.update_mode == 1 and relation_dim is not None else None
        self.hBidirectionalRelationUpdate = BidirectionalRelationAwareResidualUpdate(branch_dim, relation_dim) if self.update_mode == 2 and relation_dim is not None else None
        self.tBidirectionalRelationUpdate = BidirectionalRelationAwareResidualUpdate(branch_dim, relation_dim) if self.update_mode == 2 and relation_dim is not None else None
        self.pool = GATConv(n_heads*head_out_feats, head_out_feats, n_heads)
        self.norm = LayerNorm(n_heads*head_out_feats)
    
    def forward(self, h_data, t_data, b_graph, relation_context=None):
   
        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)
        
        h_x_inter,h_y_inter = self.interAtt(h_data,t_data,b_graph)
        if self.hBidirectionalRelationUpdate is not None and relation_context is not None:
            h_intraRep, h_x_inter = self.hBidirectionalRelationUpdate(h_intraRep, h_x_inter, h_data.batch, relation_context)
            t_intraRep, h_y_inter = self.tBidirectionalRelationUpdate(t_intraRep, h_y_inter, t_data.batch, relation_context)
        
        h_rep = torch.cat([h_intraRep,h_x_inter],1)
        t_rep = torch.cat([t_intraRep,h_y_inter],1)
        if self.hRelationUpdate is not None and relation_context is not None:
            h_rep = self.hRelationUpdate(h_rep, h_data.batch, relation_context)
            t_rep = self.tRelationUpdate(t_rep, t_data.batch, relation_context)
        h_data.x = F.elu(self.norm(h_rep, h_data.batch))
        t_data.x = F.elu(self.norm(t_rep, t_data.batch))

        if self.nodeGate is not None:
            gate_relation_context = self.nodeGateRelProj(relation_context) if self.nodeGateRelProj is not None and relation_context is not None else None
            h_data.x, t_data.x = self.nodeGate(
                h_data.x,
                h_data.batch,
                getattr(h_data, 'y', None),
                t_data.x,
                t_data.batch,
                getattr(t_data, 'y', None),
                relation_context=gate_relation_context,
            )

        h_data.x = self.pool(h_data.x, h_data.edge_index)
        t_data.x = self.pool(t_data.x, t_data.edge_index)
        h_global_graph_emb = get_node(h_data.x, h_data.batch, h_data.y, 2)[0]
        t_global_graph_emb = get_node(t_data.x, t_data.batch, t_data.y, 2)[0]

        return h_data,t_data, h_global_graph_emb,t_global_graph_emb
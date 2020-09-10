import torch
import torch.nn as nn
from sg2im.data.base_dataset import ORIGINAL_EDGE, TRANSITIVE_EDGE, SYMMETRIC_EDGE, ANTI_SYMMETRIC_EDGE
from sg2im.layers import build_mlp

"""
PyTorch modules for dealing with graphs.
"""


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self, obj_input_dim, object_output_dim, predicate_input_dim, predicate_output_dim, hidden_dim,
                 num_attributes, pooling='avg', mlp_normalization='none', predicates_transitive_weights=None,
                 return_new_p_vecs=True):
        super(GraphTripleConv, self).__init__()

        self.return_new_p_vecs = return_new_p_vecs
        self.hidden_dim = hidden_dim
        self.num_attributes = num_attributes
        self.predicate_output_dim = predicate_output_dim
        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [2 * obj_input_dim + predicate_input_dim, hidden_dim,
                       2 * hidden_dim + self.predicate_output_dim]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization, final_nonlinearity='relu')
        self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, hidden_dim, object_output_dim]

        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization, final_nonlinearity='relu')
        self.net2.apply(_init_weights)
        self.predicates_transitive_weights = predicates_transitive_weights

    def forward(self, obj_vecs, pred_vecs, edges, pred_indicators, triplet_type, predicate_ids):
        """
        Inputs:
        - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
        - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
        - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

        Outputs:
        - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
        - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        B, O, T = obj_vecs.size(0), obj_vecs.size(1), pred_vecs.size(1)

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, :, 0].contiguous()
        o_idx = edges[:, :, 1].contiguous()

        cur_s_vecs = torch.stack([obj_vecs[b, s_idx[b], :] for b in range(B)]) # [B, N, d]
        cur_o_vecs = torch.stack([obj_vecs[b, o_idx[b], :] for b in range(B)])  # [B, N, d]

        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=-1)  # [B, N, 3d]
        new_t_vecs = self.net1(cur_t_vecs)  # [B, T, d]

        triplet_type = triplet_type.type(dtype).to(device)
        predicate_transitive_prediction = torch.sigmoid(self.predicates_transitive_weights)

        # 0 - original - 1 - transitive, 2
        triplet_confidence = (triplet_type == ORIGINAL_EDGE).type(dtype) + \
                             (triplet_type == TRANSITIVE_EDGE).type(dtype) * predicate_transitive_prediction[predicate_ids]

        triplet_confidence_expaned = triplet_confidence.view(triplet_confidence.size(0), triplet_confidence.size(1), 1).expand(triplet_confidence.size(0), triplet_confidence.size(1), new_t_vecs.size(2)) # B, T, d
        new_t_vecs = new_t_vecs*triplet_confidence_expaned

        new_s_vecs = new_t_vecs[:, :, :self.hidden_dim]
        new_p_vecs = new_t_vecs[:, :, self.hidden_dim:(self.hidden_dim + self.predicate_output_dim)]
        new_o_vecs = new_t_vecs[:, :, (self.hidden_dim + self.predicate_output_dim):]

        pooled_obj_vecs_batches = []
        # important. for each batch, we mask the redundant triplets and don't add them up to avg object representation
        for b in range(B):

            sample_predicates_indicator = pred_indicators[b]
            sample_s_idx = s_idx[b][sample_predicates_indicator]
            sample_o_idx = o_idx[b][sample_predicates_indicator]
            sample_new_s_vecs = new_s_vecs[b][sample_predicates_indicator]
            sample_new_o_vecs = new_o_vecs[b][sample_predicates_indicator]
            sample_triplet_condifence = triplet_confidence[b][sample_predicates_indicator]

            s_idx_exp = sample_s_idx.view(-1, 1).expand_as(sample_new_s_vecs)
            o_idx_exp = sample_o_idx.view(-1, 1).expand_as(sample_new_o_vecs)

            pooled_obj_vecs = torch.zeros(O, self.hidden_dim, dtype=dtype, device=device)
            pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, sample_new_s_vecs)
            pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, sample_new_o_vecs)

            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, sample_s_idx, sample_triplet_condifence)
            obj_counts = obj_counts.scatter_add(0, sample_o_idx, sample_triplet_condifence)

            obj_mask = obj_counts > 0
            pooled_obj_vecs[obj_mask] = pooled_obj_vecs[obj_mask] / obj_counts[obj_mask].view(-1, 1)
            pooled_obj_vecs_batches.append(pooled_obj_vecs)

        pooled_obj_vecs_batches = torch.stack(pooled_obj_vecs_batches, dim=0)
        new_obj_vecs = self.net2(pooled_obj_vecs_batches)
        if not(self.return_new_p_vecs):
            new_p_vecs = pred_vecs
        return new_obj_vecs, new_p_vecs

def get_predicates_weights(num_preds, learned_init):
    if learned_init == 'uniform':
        predicates_weights = torch.nn.Parameter(torch.zeros(num_preds), requires_grad=True)
        predicates_weights.data.uniform_(-1, 1)
    elif learned_init == '-4':
        predicates_weights = torch.nn.Parameter(-4 * torch.ones(num_preds), requires_grad=True)
    elif learned_init == '0':
        predicates_weights = torch.nn.Parameter(torch.zeros(num_preds), requires_grad=True)
    elif learned_init == '4':
        predicates_weights = torch.nn.Parameter(4 * torch.ones(num_preds), requires_grad=True)
    else:
        raise ValueError()
    return predicates_weights

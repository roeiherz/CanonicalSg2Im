import torch
import torch.nn as nn
from sg2im.attribute_embed import AttributeEmbeddings
from sg2im.graph import GraphTripleConv, get_predicates_weights
from sg2im.layers import build_mlp, Interpolate


def get_conv_converse(model):
    if isinstance(model, dict):
        converse_candidates_weights_base = model["sg_to_layout.module.converse_candidates_weights"]
    else:
        converse_candidates_weights_base = model.sg_to_layout.module.converse_candidates_weights
    triu = torch.triu(converse_candidates_weights_base, diagonal=0)
    converse_candidates_weights = triu + triu.t()
    return converse_candidates_weights


class Sg2LayoutModel(nn.Module):
    def __init__(self, opt):
        super(Sg2LayoutModel, self).__init__()
        args = vars(opt)
        self.vocab = args["vocab"]
        self.image_size = args["image_size"]
        self.layout_noise_dim = args["layout_noise_dim"]
        self.mask_noise_dim = args.get("mask_noise_dim")
        self.args = args
        self.attribute_embedding = AttributeEmbeddings(self.vocab['attributes'], args["embedding_dim"])
        num_preds = len(self.vocab['pred_idx_to_name'])
        self.pred_embeddings = nn.Embedding(num_preds, args["embedding_dim"])
        num_attributes = len(self.vocab['attributes'].keys())

        self.trans_candidates_weights = get_predicates_weights(num_preds, opt.learned_init)
        self.converse_candidates_weights = get_predicates_weights((num_preds, num_preds), opt.learned_init)

        obj_input_dim = len(self.vocab['attributes'].keys()) * args["embedding_dim"]
        first_graph_conv_layer = {
            "obj_input_dim": obj_input_dim,
            "object_output_dim": args["gconv_dim"],
            "predicate_input_dim": args["embedding_dim"],
            "predicate_output_dim": args["gconv_dim"],
            "hidden_dim": args["gconv_hidden_dim"],
            "num_attributes": num_attributes,
            "mlp_normalization": args["mlp_normalization"],
            "pooling": args["gconv_pooling"],
            "predicates_transitive_weights": self.trans_candidates_weights # learned softly
        }
        general_graph_conv_layer = first_graph_conv_layer.copy()
        general_graph_conv_layer.update(
            {"obj_input_dim": first_graph_conv_layer["object_output_dim"], "predicate_input_dim": args["gconv_dim"]})
        layers = [first_graph_conv_layer] + [general_graph_conv_layer] * (args["gconv_num_layers"] - 1)

        self.gconvs = nn.ModuleList()

        for layer in layers:
            self.gconvs.append(GraphTripleConv(**layer))

        object_output_dim = layers[-1]["object_output_dim"]
        box_net_dim = 4
        box_net_layers = [object_output_dim, args["gconv_hidden_dim"], box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=args["mlp_normalization"], final_nonlinearity=None)

        # masks generation
        self.mask_net = None
        if args["mask_size"] is not None and args["mask_size"] > 0:
            self.mask_net = self._build_mask_net(args['g_mask_dim'], args["mask_size"])

    def _build_mask_net(self, dim, mask_size):
        output_dim = 1
        layers, cur_size = [], 1
        while cur_size < mask_size:
            layers.append(Interpolate(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(dim))
            layers.append(nn.ReLU())
            cur_size *= 2
        if cur_size != mask_size:
            raise ValueError('Mask size must be a power of 2')
        layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def create_mask_vecs(self, objs, obj_vecs):
        B = objs.size(0)
        O = objs.size(1)
        mask_vecs = obj_vecs
        layout_noise = torch.randn((1, self.mask_noise_dim), dtype=mask_vecs.dtype, device=mask_vecs.device).repeat(
            (B, O, 1)).view(B, O, self.mask_noise_dim)
        mask_vecs = torch.cat([mask_vecs, layout_noise], dim=-1)
        return mask_vecs

    def forward(self, objs, triplets, triplet_type, boxes_gt=None, masks_gt=None):
        """
        Required Inputs:
        - objs: LongTensor of shape (O,) giving categories for all objects
        - triplets: LongTensor of shape (T, 3) where triplets[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])

        Optional Inputs:
        - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
          means that objects[o] is an object in image i. If not given then
          all objects are assumed to belong to the same image.
        - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
          the spatial layout; if not given then use predicted boxes.
        """
        s, p, o = triplets.chunk(3, dim=-1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=-1)  # Shape is (T, 2)
        pred_indicators = p != self.vocab["pred_name_to_idx"]["__padding__"]
        obj_vecs = self.attribute_embedding.forward(objs)  # [B, N, d']
        pred_vecs = self.pred_embeddings(p)  # [B, T, d']

        for i in range(len(self.gconvs)):
            obj_vecs, pred_vecs = self.gconvs[i](obj_vecs, pred_vecs, edges, pred_indicators, triplet_type, p)

        # Generate Boxes
        boxes_pred = self.box_net(obj_vecs)

        # Generate Masks
        masks_pred = None
        if self.args["mask_size"] > 0:
            mask_vecs = self.create_mask_vecs(objs, obj_vecs)
            mask_scores = self.mask_net(mask_vecs.view(objs.size(0) * objs.size(1), -1, 1, 1))
            mask_scores = mask_scores.view(objs.size(0), objs.size(1), mask_scores.size(2), mask_scores.size(3))
            masks_pred = mask_scores.sigmoid()
        return obj_vecs, boxes_pred, masks_pred

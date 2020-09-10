import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sg2im.attribute_embed import AttributeEmbeddings
from sg2im.bilinear import crop_bbox_batch, crop_bbox
from sg2im.layers import build_cnn, GlobalAvgPool, build_mlp, get_norm_layer
from sg2im.layout import boxes_to_layout, masks_to_layout
from sg2im.utils import remove_dummy_objects
from spade.models.networks.base_network import BaseNetwork
from spade.models.networks.normalization import get_nonspade_norm_layer


class VectorPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.vectors = {}

    def query(self, objs, vectors):
        if self.pool_size == 0:
            return vectors
        return_vectors = []
        for obj, vector in zip(objs, vectors):
            obj = obj.item()
            vector = vector.cpu().clone().detach()
            if obj not in self.vectors:
                self.vectors[obj] = []
            obj_pool_size = len(self.vectors[obj])
            if obj_pool_size == 0:
                return_vectors.append(vector)
                self.vectors[obj].append(vector)
            elif obj_pool_size < self.pool_size:
                random_id = random.randint(0, obj_pool_size - 1)
                self.vectors[obj].append(vector)
                return_vectors.append(self.vectors[obj][random_id])
            else:
                random_id = random.randint(0, obj_pool_size - 1)
                tmp = self.vectors[obj][random_id]
                self.vectors[obj][random_id] = vector
                return_vectors.append(tmp)
        return_vectors = torch.stack(return_vectors).to(vectors.device)
        return return_vectors


class AppearanceEncoder(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 padding='same', vecs_size=1024, pooling='avg'):
        super(AppearanceEncoder, self).__init__()
        self.vocab = vocab

        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        cnn, channels = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(channels, vecs_size))

    def forward(self, crops):
        return self.cnn(crops)


class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.attribute_embedding = AttributeEmbeddings(self.opt.vocab['attributes'], self.opt.embedding_dim,
                                                       use_attr_fc_gen=True)

        self.repr_input = opt.g_mask_dim
        rep_hidden_size = 64
        repr_layers = [self.repr_input, rep_hidden_size, opt.rep_size]
        self.repr_net = build_mlp(repr_layers, batch_norm=opt.mlp_normalization)
        appearance_encoder_kwargs = {
            'vocab': self.opt.vocab,
            'arch': 'C4-64-2,C4-128-2,C4-256-2',
            'normalization': opt.appearance_normalization,
            'activation': opt.a_activation,
            'padding': 'valid',
            'vecs_size': opt.g_mask_dim
        }
        self.image_encoder = AppearanceEncoder(**appearance_encoder_kwargs)  # Ignore
        self.fake_pool = VectorPool(opt.pool_size)  # Ignore
        for i in range(opt.num_D):
            subnetD = NLayerDiscriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, img, objs, layout_boxes, layout_masks=None, gt_train=True, fool=False):
        obj_vecs = self.attribute_embedding.forward(objs)  # [B, N, d']

        # Masks Layout
        seg_batches = []
        for b in range(obj_vecs.size(0)):
            mask = remove_dummy_objects(objs[b], self.opt.vocab)
            objs_vecs_batch = obj_vecs[b][mask]
            layout_boxes_batch = layout_boxes[b][mask]

            # Masks Layout
            if layout_masks is not None:
                layout_masks_batch = layout_masks[b][mask]
                seg = masks_to_layout(objs_vecs_batch, layout_boxes_batch, layout_masks_batch, self.opt.image_size[0],
                                      self.opt.image_size[0], test_mode=False)  # test mode always false in disc.
            else:
                # Boxes Layout
                seg = boxes_to_layout(objs_vecs_batch, layout_boxes_batch, self.opt.image_size[0],
                                      self.opt.image_size[0])
            seg_batches.append(seg)

        # layout = torch.cat(layout_batches, dim=0)  # [B, N, d']
        seg = torch.cat(seg_batches, dim=0)
        input = torch.cat([img, seg], dim=1)

        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            if name.startswith('discriminator'):
                out = D(input)
                if not get_intermediate_features:
                    out = [out]
                result.append(out)
                input = self.downsample(input)
        return result

    def get_fake_pool(self, fool, gt_train, img, layout_boxes, layout_masks, obj_vecs, objs):
        objs_batch_all = []
        objs_repr_all = []
        for b in range(obj_vecs.size(0)):
            mask = remove_dummy_objects(objs[b], self.opt.vocab)
            objs_batch = objs[b][mask]
            objs_vecs_batch = obj_vecs[b][mask]
            layout_boxes_batch = layout_boxes[b][mask]
            layout_masks_batch = layout_masks[b][mask]
            O = objs_vecs_batch.size(0)
            img_exp = img[b].repeat(O, 1, 1, 1)
            if gt_train:
                # create encoding
                crops = crop_bbox(img_exp, layout_boxes_batch, 64)
                obj_repr = self.repr_net(self.image_encoder(crops))
            else:
                obj_repr = self.repr_net(layout_masks_batch)

            objs_repr_all.append(obj_repr)
            objs_batch_all.append(objs_batch)
        objs_batch_all = torch.cat(objs_batch_all, dim=0)
        objs_repr_all = torch.cat(objs_repr_all, dim=0)

        # Create fool layout
        fake_pool = None
        if fool:
            fake_pool = self.fake_pool.query(objs_batch_all, objs_repr_all)
        return fake_pool, objs_repr_all


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self):
        return self.opt.semantic_nc + 3

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class AcDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 padding='same', pooling='avg'):
        super(AcDiscriminator, self).__init__()
        self.vocab = vocab

        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        cnn, D = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
        # num_objects = len(vocab['object_name_to_idx'])
        num_objects = max(vocab['object_name_to_idx'].values()) + 1

        self.real_classifier = nn.Linear(1024, 1)
        self.obj_classifier = nn.Linear(1024, num_objects)

    def forward(self, x, y):
        if x.dim() == 3:
            x = x[:, None]
        vecs = self.cnn(x)
        real_scores = self.real_classifier(vecs)
        obj_scores = self.obj_classifier(vecs)
        ac_loss = F.cross_entropy(obj_scores, y)
        return real_scores, ac_loss


class AcCropDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 object_size=64, padding='same', pooling='avg'):
        super(AcCropDiscriminator, self).__init__()
        self.vocab = vocab
        self.discriminator = AcDiscriminator(vocab, arch, normalization, activation, padding, pooling)
        self.object_size = object_size

    def forward(self, imgs, objs, boxes):
        crops = crop_bbox_batch(imgs, objs, boxes, self.object_size, vocab=self.vocab)

        N = objs.size(0)
        new_objs = []
        for i in range(N):
            mask = remove_dummy_objects(objs[i], self.vocab)

            curr_objs = objs[i][mask]
            new_objs.append(curr_objs)

        objs = torch.cat(new_objs, dim=0).squeeze(1)  # [N]
        real_scores, ac_loss = self.discriminator(crops, objs)
        return real_scores, ac_loss, crops


class MultiscaleMaskDiscriminator2(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        for i in range(opt.num_D):
            subnetD = NLayerMaskDiscriminator2(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, objs, layout_masks, gt_train=True):
        layout_batches = []
        for b in range(layout_masks.size(0)):
            mask = remove_dummy_objects(objs[b], self.opt.vocab)
            objs_batch = objs[b][mask]
            # Masks Layout
            layout_masks_batch = layout_masks[b][mask]  # [N, 32, 32]
            new_layout_masks = layout_masks_batch.unsqueeze(1)  # [N, 1, 32, 32]
            O = objs_batch.size(0)
            M = layout_masks_batch.size(1)
            # create one-hot vector for label map
            one_hot_size = (O, max(self.opt.vocab['object_name_to_idx'].values()) + 1)
            one_hot_obj = torch.zeros(one_hot_size, dtype=layout_masks_batch.dtype, device=layout_masks_batch.device)
            one_hot_obj = one_hot_obj.scatter_(1, objs_batch.view(-1, 1).long(), 1.0)
            one_hot_obj = one_hot_obj.view(O, -1, 1, 1).expand(-1, -1, M, M)

            layout_vecs = torch.cat([one_hot_obj, new_layout_masks], dim=1)
            layout_batches.append(layout_vecs)

        input = torch.cat(layout_batches, dim=0).float()  # [N, d', M, M]

        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            if name.startswith('discriminator'):
                out = D(input)
                if not get_intermediate_features:
                    out = [out]
                result.append(out)
                input = self.downsample(input)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerMaskDiscriminator2(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self):
        return max(self.opt.vocab['object_name_to_idx'].values()) + 2
        # return self.opt.semantic_nc
        # return self.opt.semantic_nc + 3

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

#
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm2d') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#
#
# class MultiscaleMaskDiscriminator2(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3,
#                  num_objects=None):
#         super(MultiscaleMaskDiscriminator2, self).__init__()
#         self.num_D = num_D
#         self.n_layers = n_layers
#
#         for i in range(num_D):
#             netD = NLayerMaskDiscriminator2(input_nc, ndf, n_layers, norm_layer, use_sigmoid, num_objects)
#             for j in range(n_layers + 2):
#                 setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
#
#         self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
#
#     def singleD_forward(self, model, input, cond):
#         result = [input]
#         for i in range(len(model) - 2):
#             # print(result[-1].shape)
#             result.append(model[i](result[-1]))
#
#         a, b, c, d = result[-1].shape
#         cond = cond.view(a, -1, 1, 1).expand(-1, -1, c, d)
#         concat = torch.cat([result[-1], cond], dim=1)
#         result.append(model[len(model) - 2](concat))
#         result.append(model[len(model) - 1](result[-1]))
#         return result[1:]
#
#     def forward(self, objs, layout_masks):
#
#         layout_masks_batch = []
#         for b in range(layout_masks.size(0)):
#             mask = remove_dummy_objects(objs[b], self.opt.vocab)
#             # Masks Layout
#             layout_masks_batch = layout_masks[b][mask]
#
#         input = torch.cat(layout_masks_batch, dim=0)
#         O, _, mask_size = input.shape
#         one_hot_size = (O, max(self.opt.vocab['object_name_to_idx'].values()) + 1)
#         one_hot_obj = torch.zeros(one_hot_size, dtype=input.dtype, device=input.device)
#         cond = one_hot_obj.scatter_(1, objs.view(-1, 1).long(), 1.0)
#
#         num_D = self.num_D
#         result = []
#         input_downsampled = input
#         for i in range(num_D):
#             model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
#                      range(self.n_layers + 2)]
#             result.append(self.singleD_forward(model, input_downsampled, cond))
#             if i != (num_D - 1):
#                 input_downsampled = self.downsample(input_downsampled)
#         return result
#
#
# # Defines the PatchGAN discriminator with the specified arguments.
# class NLayerMaskDiscriminator2(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
#                  num_objects=None):
#         super(NLayerMaskDiscriminator2, self).__init__()
#         self.n_layers = n_layers
#
#         kw = 3
#         padw = int(np.ceil((kw - 1.0) / 2))
#         sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
#
#         nf = ndf
#         for n in range(1, n_layers):
#             nf_prev = nf
#             nf = min(nf * 2, 512)
#             sequence += [[
#                 nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
#                 norm_layer(nf), nn.LeakyReLU(0.2, True)
#             ]]
#
#         nf_prev = nf
#         nf = min(nf * 2, 512)
#         nf_prev += num_objects
#         sequence += [[
#             nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
#             norm_layer(nf),
#             nn.LeakyReLU(0.2, True)
#         ]]
#
#         sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
#
#         if use_sigmoid:
#             sequence += [[nn.Sigmoid()]]
#
#         for n in range(len(sequence)):
#             setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
#
#     def forward(self, input):
#         res = [input]
#         for n in range(self.n_layers + 2):
#             model = getattr(self, 'model' + str(n))
#             res.append(model(res[-1]))
#         return res[1:]

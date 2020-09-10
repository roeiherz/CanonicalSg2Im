import torch
import torch.nn as nn
import torch.nn.functional as F
from sg2im.attribute_embed import AttributeEmbeddings
from sg2im.bilinear import crop_bbox
from sg2im.layers import build_cnn, GlobalAvgPool, build_mlp
from sg2im.layout import boxes_to_layout, masks_to_layout
from sg2im.utils import remove_dummy_objects
from spade.models.networks.base_network import BaseNetwork
from spade.models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock


class SPADEGenerator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.attribute_embedding = AttributeEmbeddings(opt.vocab['attributes'], opt.embedding_dim)
        self.opt = opt
        nf = opt.ngf
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

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
        self.image_encoder = AppearanceEncoder(**appearance_encoder_kwargs)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.image_size[0] // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, objs, layout_boxes, layout_masks, test_mode=False):
        obj_vecs = self.attribute_embedding.forward(objs)  # [B, N, d']
        seg_batches = []
        for b in range(obj_vecs.size(0)):
            mask = remove_dummy_objects(objs[b], self.opt.vocab)
            objs_vecs_batch = obj_vecs[b][mask]
            layout_boxes_batch = layout_boxes[b][mask]
            # Masks Layout
            if layout_masks is not None:
                layout_masks_batch = layout_masks[b][mask]
                seg = masks_to_layout(objs_vecs_batch, layout_boxes_batch, layout_masks_batch, self.opt.image_size[0],
                                      self.opt.image_size[0], test_mode=test_mode)
            else:
                # Boxes Layout
                seg = boxes_to_layout(objs_vecs_batch, layout_boxes_batch, self.opt.image_size[0],
                                      self.opt.image_size[0])
            seg_batches.append(seg)
        seg = torch.cat(seg_batches, dim=0)

        # we downsample segmap and run convolution
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)
        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


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

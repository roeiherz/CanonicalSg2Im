"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import spade.models.networks as networks
import torch.nn.functional as F
from sg2im.losses import get_gan_losses


class Pix2PixModel(torch.nn.Module):

    def __init__(self, opt, discriminator, netE=None):
        super().__init__()
        self.opt = opt
        device = torch.device("cuda:{gpu}".format(gpu=self.opt.gpu_ids[0]) if self.opt.use_cuda else "cpu")
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor
        self.discriminator = discriminator
        if hasattr(discriminator, 'img_discriminator'):
            self.netD_img = discriminator.img_discriminator

        if hasattr(opt, 'use_img_disc') and not opt.use_img_disc:
            if hasattr(discriminator, 'obj_discriminator'):
                self.netD_obj = discriminator.obj_discriminator
            if hasattr(discriminator, 'mask_discriminator'):
                self.netD_mask = discriminator.mask_discriminator

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.gan_g_loss, self.gan_d_loss = get_gan_losses(opt.gan_loss_type)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids).to(device)

            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, batch, model_out):

        imgs, objs, boxes, triplets, _, _, masks, _ = batch
        imgs_pred, boxes_pred, masks_pred = model_out
        G_losses = {}

        if not self.opt.skip_graph_model:
            flattened_bbox_pred = F.smooth_l1_loss(boxes_pred.view(-1, 4), boxes.view(-1, 4), reduction='none') * self.opt.bbox_pred_loss_weight
            flattened_objs = objs.view(-1, objs.size(-1))
            if objs.size(-1) > 1:
                # Objs contain multiple attributes (CLEVR), dummy are [0, 0...0]
                object_mask = (flattened_objs.sum(1, keepdim=True) != 0)
            else:
                # Objs contain only single attribute (VG/COCO), dummy is 0
                object_mask = (flattened_objs != 0)

            is_real_object_mask = object_mask.type(torch.FloatTensor).to(flattened_objs.device)
            masked_bbox_pred_loss = flattened_bbox_pred * is_real_object_mask

            G_losses["bbox_pred_all"] = masked_bbox_pred_loss.view(boxes.shape).sum(dim=[1,2])/is_real_object_mask.view(boxes.shape[0], boxes.shape[1]).sum(dim=1)
            G_losses["bbox_pred"] = G_losses["bbox_pred_all"].mean()

            # Masks cross-entropy prediction
            if masks is not None:
                flattened_masks = masks.view(-1, masks.size(2), masks.size(3)).float()
                flattened_masks_pred = masks_pred.view(-1, masks_pred.size(2), masks_pred.size(3))
                masks_loss = F.binary_cross_entropy(flattened_masks_pred, flattened_masks, reduction='none').mean(dim=(1,2))
                G_losses["masks_pred"] = (masks_loss[object_mask.nonzero()[:, 0]] * self.opt.mask_pred_loss_weight).mean()

        if not self.opt.skip_generation:
            # Img disc
            pred_img_fake = self.netD_img(imgs_pred, objs, boxes, layout_masks=masks, gt_train=True, fool=False)
            pred_img_fake_loss = self.criterionGAN(pred_img_fake, True, for_discriminator=False).squeeze(0)
            G_losses['GAN_Img'] = pred_img_fake_loss * self.opt.discriminator_img_loss_weight
            if not self.opt.no_ganFeat_loss:
                pred_img_real = self.netD_img(imgs, objs, boxes, layout_masks=masks, gt_train=True, fool=False)
                num_D = len(pred_img_fake)
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_img_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionFeat(pred_img_fake[i][j], pred_img_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss.squeeze(0)

            # VGG feature matching loss
            if not self.opt.no_vgg_loss:
                G_losses['VGG'] = self.criterionVGG(imgs_pred, imgs) * self.opt.lambda_vgg

            if not self.opt.use_img_disc:
                # Obj disc
                scores_fake, ac_loss, g_fake_crops = self.netD_obj(imgs_pred, objs, boxes)
                G_losses['GAN_Obj'] = self.gan_g_loss(scores_fake) * self.opt.lambda_obj
                G_losses['GAN_Obj'] = self.criterionGAN(scores_fake, True, for_discriminator=False).squeeze(
                    0) * self.opt.discriminator_obj_loss_weight
                G_losses['GAN_Ac'] = ac_loss * self.opt.ac_loss_weight

                # Mask disc
                if self.netD_mask is not None and self.opt.mask_size > 0 and masks_pred is not None:
                    scores_fake = self.netD_mask(objs, masks_pred)
                    mask_loss = self.criterionGAN(scores_fake, True, for_discriminator=False).squeeze(0)
                    G_losses['GAN_Mask'] = mask_loss * self.opt.discriminator_img_loss_weight
                    if not self.opt.no_ganFeat_loss:
                        # Images should not be used
                        scores_real = self.netD_mask(objs, masks)
                        num_D = len(scores_fake)
                        GAN_Mask_Feat_loss = self.FloatTensor(1).fill_(0)
                        for i in range(num_D):  # for each discriminator
                            # last output is the final prediction, so we exclude it
                            num_intermediate_outputs = len(scores_fake[i]) - 1
                            for j in range(num_intermediate_outputs):  # for each layer output
                                unweighted_loss = self.criterionFeat(scores_fake[i][j], scores_real[i][j].detach())
                                GAN_Mask_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                        G_losses['GAN_Mask_Feat'] = GAN_Mask_Feat_loss.squeeze(0)

        scalar_losses = [k for k in G_losses.keys() if k != "bbox_pred_all"]
        G_losses['total_loss'] = torch.stack([G_losses[k] for k in scalar_losses], dim=0).sum()
        return G_losses

    def compute_discriminator_loss(self, batch, model_out):
        imgs, objs, boxes, _, _, _, masks, _ = batch
        imgs_pred, boxes_pred, masks_pred = model_out

        # Detach
        imgs_pred = imgs_pred.detach()
        if boxes_pred is not None:
            boxes_pred = boxes_pred.detach()
        if masks_pred is not None:
            masks_pred = masks_pred.detach()

        # Img disc
        D_img_losses = {}
        # Fake images; Real layout
        pred_fake = self.netD_img(imgs_pred, objs, boxes, layout_masks=masks, gt_train=True, fool=False)
        # Real images; Real layout
        gt_real = self.netD_img(imgs, objs, boxes, layout_masks=masks, gt_train=True, fool=False)
        # Update Loss
        D_img_losses.update(
            {"D_img_fake": self.criterionGAN(pred_fake, False, for_discriminator=True)})
        D_img_losses.update({"D_img_real": self.criterionGAN(gt_real, True, for_discriminator=True)})
        D_img_losses.update({"total_img_loss": torch.stack(list(D_img_losses.values()), dim=0).sum()})

        if not self.opt.use_img_disc:
            # Real images; Wrong Layout
            pred_wrong = self.netD_img(imgs, objs, boxes, layout_masks=masks, gt_train=True, fool=True)
            D_img_losses.update(
                {"D_img_wrong": self.criterionGAN(pred_wrong, False, for_discriminator=True) * (1 / 2) * (.5)})

        if not self.opt.use_img_disc:
            # Obj disc
            D_obj_losses = {}
            # real objects score
            scores_real, ac_loss_real, self.d_real_crops = self.netD_obj(imgs, objs, boxes)
            # fake objects score
            scores_fake, ac_loss_fake, self.d_fake_crops = self.netD_obj(imgs_pred, objs, boxes)
            # Update Loss
            D_obj_losses.update({"D_obj": self.gan_d_loss(scores_real, scores_fake) * 0.5})
            D_obj_losses.update({"D_ac_real": ac_loss_real})
            D_obj_losses.update({"D_ac_fake": ac_loss_fake})
            D_obj_losses.update({"total_obj_loss": torch.stack(list(D_obj_losses.values()), dim=0).sum()})

            # Mask disc
            D_mask_losses = {}
            if self.opt.mask_size > 0 and masks_pred is not None:
                # Images should not be used
                scores_fake = self.netD_mask(objs, masks_pred)
                scores_real = self.netD_mask(objs, masks)
                # Update Loss
                D_mask_losses.update({"D_mask_fake": self.criterionGAN(scores_fake, False, for_discriminator=True) * 0.5})
                D_mask_losses.update({"D_mask_real": self.criterionGAN(scores_real, True, for_discriminator=True) * 0.5})
                D_mask_losses.update({"total_mask_loss": torch.stack(list(D_mask_losses.values()), dim=0).sum()})

            # Merge
            D_losses = {**D_img_losses, **D_obj_losses, **D_mask_losses}
            return D_losses

        return D_img_losses

    # Given fake and real image, return the prediction of discriminator for each fake and real image.
    # Take the prediction of fake and real images from the combined batch
    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def forward(self, batch, model_out, mode):

        if mode == "compute_discriminator_loss":
            return self.compute_discriminator_loss(batch, model_out)

        if mode == "compute_generator_loss":
            return self.compute_generator_loss(batch, model_out)

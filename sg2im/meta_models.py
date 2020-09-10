import torch
import torch.nn as nn
from sg2im.model import Sg2LayoutModel
from spade.models.networks import SPADEGenerator, AcCropDiscriminator, \
    MultiscaleDiscriminator, MultiscaleMaskDiscriminator2
from spade.models.networks.sync_batchnorm import DataParallelWithCallback


class MetaGeneratorModel(nn.Module):
    def __init__(self, opt, device):
        super(MetaGeneratorModel, self).__init__()
        self.args = vars(opt)
        self.vocab = self.args["vocab"]

        # Graph Model
        if not self.args['skip_graph_model']:
            self.sg_to_layout = DataParallelWithCallback(Sg2LayoutModel(opt), device_ids=self.args['gpu_ids']).to(device)

        # SPADE Generator
        if not self.args['skip_generation']:
            self.layout_to_image_model = SPADEGenerator(opt)
            self.layout_to_image_model = DataParallelWithCallback(self.layout_to_image_model,
                                                                  device_ids=self.args['gpu_ids']).to(device)

    def forward(self, objs, triplets, triplet_type, boxes_gt=None, masks_gt=None, test_mode=False):
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

        boxes_pred = None
        masks_pred = None
        if not self.args['skip_graph_model']:
            obj_vecs, boxes_pred, masks_pred = self.sg_to_layout(objs, triplets, triplet_type, boxes_gt)

        img = None
        if not self.args["skip_generation"]:
            layout_boxes = boxes_pred if boxes_gt is None else boxes_gt
            layout_masks = masks_pred if masks_gt is None else masks_gt
            img = self.layout_to_image_model(objs, layout_boxes, layout_masks, test_mode=test_mode)

        return img, boxes_pred, masks_pred


class MetaDiscriminatorModel(nn.Module):
    def __init__(self, opt):
        super(MetaDiscriminatorModel, self).__init__()
        self.args = vars(opt)
        self.init_img_discriminator(opt)
        if not opt.use_img_disc:
            self.init_obj_discriminator(opt)
            self.init_mask_discriminator(opt)

    def init_img_discriminator(self, opt):
        self.img_discriminator = MultiscaleDiscriminator(opt)
        self.img_discriminator.type(torch.cuda.FloatTensor)
        self.img_discriminator.train()
        self.optimizer_d_img = torch.optim.Adam(list(self.img_discriminator.parameters()),
                                                lr=opt.img_learning_rate,
                                                betas=(opt.beta1, 0.999))

    def init_obj_discriminator(self, opt):
        self.obj_discriminator = AcCropDiscriminator(vocab=opt.vocab,
                                                     arch=opt.d_obj_arch,
                                                     normalization=opt.d_normalization,
                                                     activation=opt.d_activation,
                                                     padding=opt.d_padding,
                                                     object_size=opt.crop_size)
        self.obj_discriminator.type(torch.cuda.FloatTensor)
        self.obj_discriminator.train()
        self.optimizer_d_obj = torch.optim.Adam(list(self.obj_discriminator.parameters()),
                                                lr=opt.learning_rate,
                                                betas=(opt.beta1, 0.999))

    def init_mask_discriminator(self, opt):
        self.mask_discriminator = MultiscaleMaskDiscriminator2(opt)
        self.mask_discriminator.type(torch.cuda.FloatTensor)
        self.mask_discriminator.train()
        self.optimizer_d_mask = torch.optim.Adam(list(self.mask_discriminator.parameters()),
                                                 lr=opt.mask_learning_rate,
                                                 betas=(opt.beta1, 0.999))
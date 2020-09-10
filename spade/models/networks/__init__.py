"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from spade.models.networks.base_network import BaseNetwork
from spade.models.networks.loss import *
from spade.models.networks.discriminator import *
from spade.models.networks.generator import *
from spade.models.networks.encoder import *


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt)


def define_E(opt):
    netE_cls = find_network_using_name('conv', 'encoder')
    return create_network(netE_cls, opt)

networks = {

}
def find_network_using_name(name):
    return networks[name]
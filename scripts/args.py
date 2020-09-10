import os
import argparse
import datetime
import torch
from sg2im.utils import int_tuple, bool_flag

__author__ = 'roeiherz'

VG_DIR = os.path.expanduser('datasets/vg')
COCO_DIR = os.path.expanduser('datasets/coco')
CLEVR_DIALOG_DIR = os.path.expanduser('Datasets/CLEVR_Dialog')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco',
                    choices=['vg', 'clevr', 'coco', 'synthetic', 'packed_coco', 'packed_vg', 'packed_clevr'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--mask_learning_rate', default=1e-5, type=float)
parser.add_argument('--img_learning_rate', default=1e-4, type=float)
parser.add_argument('--beta1', default=0.5, type=float)

# Dataset options common to both CLEVR, VG and COCO
parser.add_argument('--img_deprocess', default='decode_img', type=str)
parser.add_argument('--image_size', default='256,256', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

# Generator options
parser.add_argument('--mask_size', default=0, type=int)  # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=32, type=int)
parser.add_argument('--graph_model', default='jj', type=str)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--g_mask_dim', default=128 + 64, type=int)
parser.add_argument('--mask_noise_dim', default=64, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_pooling', default='avg', type=str)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--heads', default=1, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)
parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='normal',
                    help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first generator conv layer')
parser.add_argument('--netD_subarch', type=str, default='n_layer',
                    help='architecture of each discriminator')
parser.add_argument('--num_D', type=int, default=2,
                    help='number of discriminators to be used in multiscale')
parser.add_argument('--n_layers_D', type=int, default=4,
                    help='# layers in each discriminator')
parser.add_argument('--aspect_ratio', type=float, default=1.0,
                    help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
parser.add_argument('--isTrain', default=1, type=int)
parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')

# Output options
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--output_dir',
                    default=os.path.join(os.getcwd(), 'output', "{:%Y%m%dT%H%M}_%s".format(datetime.datetime.now())))
parser.add_argument('--run_name', default="debug")
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_gan_name', default='checkpoint')
parser.add_argument('--checkpoint_graph_name', default='checkpoint')

# parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_checkpoint', default=0, type=int)
parser.add_argument('--freeze', default=0, type=int)
parser.add_argument('--freeze_options', default=None, help='choose "fine_tuned" or "predict"')
parser.add_argument('--use_cuda', action='store_true', help="to use GPUs")
parser.add_argument('--name', type=str, default='label2coco',
                    help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3',
                    help='instance normalization or batch normalization')
parser.add_argument('--norm_D', type=str, default='spectralinstance',
                    help='instance normalization or batch normalization')
parser.add_argument('--norm_E', type=str, default='spectralinstance',
                    help='instance normalization or batch normalization')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

# input/output sizes
parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop',
                    help='scaling and cropping of images at load time.',
                    choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop",
                             "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
parser.add_argument('--label_nc', type=int, default=182,
                    help='# of input label classes without unknown class. If you have unknown class as class label, '
                         'specify --contain_dopntcare_label.')
parser.add_argument('--contain_dontcare_label', action='store_true',
                    help='if the label map contains dontcare label (dontcare=255)')
parser.add_argument('--output_nc', type=int, default=131, help='# of output image channels')

# for setting inputs
parser.add_argument('--dataroot', type=str, default='./datasets')
parser.add_argument('--no_flip', action='store_true',
                    help='if specified, do not flip the images for data argumentation')
parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
parser.add_argument('--load_from_opt_file', action='store_true',
                    help='load the options from checkpoints and use that as default')
parser.add_argument('--cache_filelist_write', action='store_true',
                    help='saves the current filelist into a text file, so that it loads faster')
parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')

# for displays
parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

# for generator
parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
parser.add_argument('--init_type', type=str, default='xavier',
                    help='network initialization [normal|xavier|kaiming|orthogonal]')
parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
parser.add_argument('--z_dim', type=int, default=256, help="dimension of the latent z vector")

# for instance-wise features
parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')

# spade train options
parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

# for training
parser.add_argument('--niter', type=int, default=50,
                    help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--D_steps_per_G', type=int, default=1,
                    help='number of discriminator iterations per generator iterations.')

# Discriminators losses
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--no_ganFeat_loss', action='store_true',
                    help='if specified, do *not* use discriminator feature matching loss')
parser.add_argument('--no_vgg_loss', action='store_true',
                    help='if specified, do *not* use VGG feature matching loss')
parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
parser.add_argument('--lambda_kld', type=float, default=0.05)

# Appearance Generator options
parser.add_argument('--rep_size', default=32, type=int)
parser.add_argument('--appearance_normalization', default='batch')
parser.add_argument('--a_activation', default='leakyrelu-0.2')

# Lambda losses
parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
parser.add_argument('--lambda_obj', default=0.1, type=float)
parser.add_argument('--discriminator_img_loss_weight', default=1.0, type=float)
parser.add_argument('--discriminator_obj_loss_weight', default=0.1, type=float)
parser.add_argument('--discriminator_mask_loss_weight', default=1.0, type=float)
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
parser.add_argument('--mask_pred_loss_weight', default=0, type=float)
parser.add_argument('--pool_size', default=100, type=int)

# Object discriminator
parser.add_argument('--gan_loss_type', default='gan')
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')
parser.add_argument('--d_obj_arch', default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=32, type=int)
parser.add_argument('--ac_loss_weight', default=0.1, type=float)

# Mask Discriminator
parser.add_argument('--ndf_mask', default=64, type=int)
parser.add_argument('--num_D_mask', default=1, type=int)
parser.add_argument('--norm_D_mask', default='instance', type=str)
parser.add_argument('--n_layers_D_mask', default=2, type=int)

#
parser.add_argument('--skip_generation', type=int, default=0)
parser.add_argument('--skip_graph_model', type=int, default=0)
parser.add_argument('--use_transitivity', type=int, default=0)
parser.add_argument('--use_img_disc', type=int, default=0)
parser.add_argument('--dense_scenes', type=int, default=0)
parser.add_argument('--min_objects', type=int)
parser.add_argument('--max_objects', type=int)
parser.add_argument('--max_objects_val', type=int)
parser.add_argument('--min_object_size', type=float, default=0.02)
parser.add_argument('--transformer_hidden_dim', type=int, default=32)
parser.add_argument('--use_attributes', type=int, default=1)
parser.add_argument('--resolution', type=int, default=256, choices=[64, 128, 256])
parser.add_argument('--full_test', type=int, default=1000000)
parser.add_argument('--include_dummies', type=int, default=0)
parser.add_argument('--learned_transitivity', type=int, default=0)
parser.add_argument('--all_transitive_baseline', type=int, default=0)
parser.add_argument('--use_all_relations', default=0, type=int)
parser.add_argument('--learned_init', type=str, default='uniform', choices=['uniform', '0', '-4']),
parser.add_argument('--learned_symmetry', type=int, default=0)
parser.add_argument('--learned_converse', type=int, default=0)
parser.add_argument('--use_converse', type=int, default=0)


def init_args(args):
    try:
        args.output_dir = args.output_dir % args.run_name
    except Exception as e:
        pass

    if args.debug:
        args.num_val_samples = 100

    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    assert len(args.gpu_ids) == 0 or args.batch_size % (len(args.gpu_ids)) == 0, \
        "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
        % (args.batch_size, len(args.gpu_ids))

    args.semantic_nc = len(args.vocab['attributes']) * args.embedding_dim


def print_args(args):
    print("Config Parameters:")
    for name, value in vars(args).items():
        if name == "vocab":
            for voc_name, voc_val in value.items():
                print("  > Vocab > {key}: {val}".format(key=voc_name, val=voc_val))
        else:
            print(" > {key}: {val}".format(key=name, val=value))


def get_args():
    args = parser.parse_args()
    return args

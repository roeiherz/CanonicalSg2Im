from functools import partial
import cv2
import numpy as np
import matplotlib
from scripts.train import update_loader_params
matplotlib.use('Agg')
from sg2im.data.dataset_params import get_dataset, get_collate_fn
from evaluation.inception import InceptionScore
from sg2im.data import deprocess_batch, imagenet_deprocess
from sg2im.data.utils import decode_image
from sg2im.utils import batch_to
from sg2im.meta_models import MetaGeneratorModel
from torch.utils.data import DataLoader
import argparse, os
import torch
from types import SimpleNamespace
import json

# CHECKPOINTS_PATH = '/specific/netapp5_2/gamir/DER-Roei/qa2im/output/'
CHECKPOINTS_PATH = '/home/roeiherz/qa2im/output/'


def get_data_loader(opt):
    val_dset = get_dataset(opt.dataset, 'test', opt)
    collate = get_collate_fn(opt)
    loader_kwargs = {'batch_size': opt.batch_size,
                     'num_workers': opt.loader_num_workers,
                     'shuffle': False,
                     'collate_fn': partial(collate, opt.vocab)}
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return val_loader


def check_model_layout(loader, model, inception_score_gt, inception_score_pred, output_dir, deprocess_func):
    inception_score_gt.clean()
    inception_score_pred.clean()
    with torch.no_grad():
        batch_ind = 0
        for batch in loader:
            try:
                print("Iteration {}".format(batch_ind))

                batch = batch_to(batch)
                samples = {}
                imgs, objs, boxes, triplets, _, triplet_type, masks, image_ids = batch

                samples['gt_img'] = imgs
                samples['gt_box_gt_mask'] = model(objs, triplets, triplet_type, boxes_gt=boxes, masks_gt=masks, test_mode=True)[0]
                samples['pred_box_pred_mask'] = model(objs, triplets, triplet_type, test_mode=True)[0]

                # Calc Inception score
                inception_score_gt(samples['gt_box_gt_mask'])
                inception_score_pred(samples['pred_box_pred_mask'])
                # Save images
                draw_datasets(samples, output_dir, deprocess_func, image_ids)
                batch_ind += 1

            except Exception as e:
                print("Exception in iter {} - {}".format(batch_ind, e))

        inception_mean, inception_std = inception_score_gt.compute_score(splits=5)
        print(' >> GT inception_mean: %.4f' % inception_mean)
        print(' >> GT inception_std: %.4f' % inception_std)
        inception_mean, inception_std = inception_score_pred.compute_score(splits=5)
        print(' >> PRED inception_mean: %.4f' % inception_mean)
        print(' >> PRED inception_std: %.4f' % inception_std)


def draw_datasets(samples, output_dir, deprocess_func, image_ids):
    for k, v in samples.items():
        samples[k] = np.transpose(deprocess_batch(v, deprocess_func=deprocess_func).cpu().numpy(),
                                  [0, 2, 3, 1])
    for k, v in samples.items():
        # Set the output path
        if k == 'gt_img':
            path = os.path.join(output_dir, "gt")
        else:
            path = os.path.join(output_dir, "generation", k)

        os.makedirs(path, exist_ok=True)
        for i in range(v.shape[0]):
            RGB_img_i = cv2.cvtColor(v[i], cv2.COLOR_BGR2RGB)
            cv2.imwrite("{}/{}.jpg".format(path, image_ids[i]), RGB_img_i)


def main(args):
    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    if not os.path.isdir(args.output_dir):
        print('Output directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(args.output_dir)

    if args.gpu_ids == 'cpu':
        device = torch.device('cpu')
    elif args.gpu_ids == 'gpu':
        device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            print('WARNING: CUDA not available; falling back to CPU')
            device = torch.device('cpu')
    else:
        device = torch.device('cuda:{gpu}'.format(gpu=args.gpu_ids[0]))
        if not torch.cuda.is_available():
            print('WARNING: CUDA not available; falling back to CPU')
            device = torch.device('cpu')

    # Load the model, with a bit of care in case there are no GPUs
    map_location = 'cpu' if device == torch.device('cpu') else device
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    trained_args = json.load(open(os.path.join(os.path.dirname(args.checkpoint), 'run_args.json'), 'rb'))
    set_args(args, trained_args)

    # Model
    opt = SimpleNamespace(**trained_args)
    # opt.batch_size = 2
    model = MetaGeneratorModel(opt, device)
    # Load pre-trained weights for generation
    model.load_state_dict(checkpoint['model_state'])
    # Eval
    model.eval()
    # Put on device
    model.to(device)

    # Init Inception Score
    inception_score_gt = InceptionScore(device, batch_size=opt.batch_size, resize=True)
    inception_score_pred = InceptionScore(device, batch_size=opt.batch_size, resize=True)
    # Get the data
    data_loader = get_data_loader(opt)
    # data_loader = build_test_dsets(opt)
    update_loader_params(data_loader.dataset, model.sg_to_layout.module.converse_candidates_weights, model.sg_to_layout.module.trans_candidates_weights)
    check_model_layout(data_loader, model, inception_score_gt, inception_score_pred, args.output_dir,
                       opt.deprocess_func)
    print(" >> Dataset generated in {}".format(args.output_dir))


def set_args(args, trained_args):
    trained_args['gpu_ids'] = args.gpu_ids

    # Define img_deprocess
    if trained_args['img_deprocess'] == "imagenet":
        trained_args['deprocess_func'] = imagenet_deprocess
    elif trained_args['img_deprocess'] == "decode_img":
        trained_args['deprocess_func'] = decode_image
    else:
        print("Error: No deprocess function was found. decode_image was chosen")
        trained_args['deprocess_func'] = decode_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='{}/coco/jj128_fixcoco_3layers_128emb_5gpu_35batch_refactor_wolf_taotie/itr_140000.pt'.format(CHECKPOINTS_PATH))
    parser.add_argument('--output_dir', default='{}/coco/jj128_fixcoco_3layers_128emb_5gpu_35batch_refactor_wolf_taotie/generation/itr_140000'.format(CHECKPOINTS_PATH))
    parser.add_argument('--dataset', default='coco', choices=['vg', 'clevr', 'coco'])
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    args = parser.parse_args()

    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    main(args)

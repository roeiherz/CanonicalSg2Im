import argparse
import json
from argparse import Namespace
import matplotlib
matplotlib.use('Agg')
from evaluation.inception import InceptionScore
from scripts.train import check_model, build_test_dsets, update_loader_params
from sg2im.model import get_conv_converse
from sg2im.pix2pix_model import Pix2PixModel
from spade.models.networks.sync_batchnorm import DataParallelWithCallback
import numpy as np
from sg2im.meta_models import MetaGeneratorModel
import os
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

__author__ = 'roeiherz'


def main(args):
    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda:%s' % args.gpu_ids[0])
        if not torch.cuda.is_available():
            print('WARNING: CUDA not available; falling back to CPU')
            device = torch.device('cpu')
    else:
        device = torch.device('cuda:{gpu}'.format(gpu=args.device))
        if not torch.cuda.is_available():
            print('WARNING: CUDA not available; falling back to CPU')
            device = torch.device('cpu')

    # Load the model, with a bit of care in case there are no GPUs
    map_location = 'cpu' if device == torch.device('cpu') else device
    print("loading: %s" % args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    val_loader, vocab = build_test_dsets(args)
    # Model
    model = MetaGeneratorModel(args, device)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()
    model.to(device)
    conv_weights_mat = get_conv_converse(model)
    update_loader_params(val_loader.dataset, conv_weights_mat, None)

    gans_model = Pix2PixModel(args, discriminator=None)
    gans_model = DataParallelWithCallback(gans_model, device_ids=args.gpu_ids).to(device)
    inception_score = InceptionScore(device, batch_size=args.batch_size, resize=True)
    mean_losses, _, res_df = check_model(args, val_loader, model, gans_model, inception_score, use_gt=False,
                                         full_test=False)
    if args.verbose:
        print("Avg. IOU is {}, IOU 0.3 is {}, IOU 0.5 is {}".format(mean_losses['avg_iou'], mean_losses['total_iou_03'], mean_losses['total_iou_05']))
    if res_df is not None:
        res_df.to_csv(os.path.join(os.path.dirname(args.checkpoint), 'results_%s.csv' % args.dataset), index=False)
    return mean_losses['avg_iou'], mean_losses['total_iou_03'], mean_losses['total_iou_05']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='+', type=str)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--use_dataset', default=None)
    args = parser.parse_args()
    all_avg, all_iou3, all_iou5 = [], [], []
    for checkpoint in args.checkpoint:

        if os.path.isdir(checkpoint):
            event_acc = EventAccumulator(checkpoint)
            event_acc.Reload()
            # Show all tags in the log file

            # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'

            w_times, step_nums, vals = zip(*event_acc.Scalars('val/loss/avg_iou'))
            best_itr = step_nums[np.argmax(list(vals))]
            checkpoint = os.path.join(checkpoint, 'itr_%s.pt' % best_itr)

        train_args = json.load(open(os.path.join(os.path.dirname(checkpoint), 'run_args.json'), 'rb'))
        train_args.update(vars(args))
        train_args["gpu_ids"] = [int(k) for k in train_args["gpu_ids"].split(',')]
        train_args["checkpoint"] = checkpoint
        train_args["shuffle_val"] = False
        train_args["verbose"] = args.verbose
        train_args["learned_transitivity"] = train_args.get("learned_transitivity", 0)
        train_args["learned_symmetry"] = train_args.get("learned_symmetry", 0)
        train_args["learned_converse"] = train_args.get("learned_converse", 0)
        train_args["use_converse"] = train_args.get("use_converse", 0)
        train_args["use_transitivity"] = train_args.get("use_transitivity", 0)
        train_args["image_size"] = train_args.get("image_size")
        train_args["mask_size"] = train_args.get("mask_size", 0)
        train_args["include_dummies"] = train_args.get("include_dummies", 0)

        if args.use_dataset is not None:
            print("Overriding default dataset with: %s" % args.use_dataset)
            train_args["dataset"] = args.use_dataset

        avg, iou3, iou5 = main(Namespace(**train_args))

        all_avg.append("%.1f" % (avg * 100))
        all_iou3.append("%.1f" % (iou3 * 100))
        all_iou5.append("%.1f" % (iou5 * 100))

    all_model_iou = all_avg + all_iou3 + all_iou5
    print(' & '.join(all_model_iou))

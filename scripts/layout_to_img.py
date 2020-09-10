import argparse
import json
from argparse import Namespace

import cv2
import numpy as np
import matplotlib
from PIL import Image

from scripts.args import init_args
from sg2im.data import deprocess_batch
from sg2im.data.utils import decode_image
from sg2im.meta_models import MetaGeneratorModel
from spade.models.networks import SPADEGenerator
matplotlib.use('Agg')
import os
import torch
import pandas as pd

def main(args):
    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda:%s'%args.gpu_ids[0])
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
    print("loading: %s"%args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Model
    args.skip_graph_model = 1
    args.skip_generation = 0
    model = MetaGeneratorModel(args, device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval()
    model.to(device)
    df = pd.read_csv(os.path.join(args.base_dir, "results_objs.csv"))
    run_args = json.load(open(os.path.join(args.base_dir, "run_args.json"), "r"))
    vocab = run_args["vocab"]

    gen_run_args = json.load(open(os.path.join(os.path.dirname(args.checkpoint), "run_args.json"), "r"))
    gen_vocab = gen_run_args["vocab"]

    with torch.no_grad():
        for i, row in df.iterrows():

            # bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
            # bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
            # bbox = np.concatenate([bbox, [[-0.6, -0.6, 0.5, 0.5]]], axis=0)
            bbox = np.array(eval(row['predicted_boxes']))
            bbox = torch.FloatTensor(bbox).unsqueeze(0)

            object_class = eval(row['class'])
            labels = torch.LongTensor([gen_vocab["object_name_to_idx"][c] for c in object_class if c != "__image__"]).unsqueeze(0).unsqueeze(-1)

            model_out = model.layout_to_image_model.forward(None, labels, bbox, None, test_mode=True)
            img_pred = model_out
            image = deprocess_batch(img_pred, deprocess_func=decode_image)[0]
            image = np.transpose(image.cpu().numpy(), [1, 2, 0])
            Image.fromarray(image).save(os.path.join(args.base_dir, 'samples_roei', os.path.basename(row['image_id'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default="/specific/netapp5_2/gamir/DER-Roei/qa2im/output/vg/jj128_fixvg_3layers_128emb_5gpu_15batch_refactor_wolf_nian/itr_230000.pt")
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--base_dir', type=str, default="/specific/netapp5_2/gamir/DER-Roei/bamir/Deployment/output/packed_vg/sanity_check/packed_vg_learnt_0/")

    args = parser.parse_args()
    train_args = json.load(open(os.path.join(os.path.dirname(args.checkpoint), 'run_args.json'), 'rb'))
    train_args.update(vars(args))
    args = Namespace(**train_args)
    init_args(args)
    main(args)

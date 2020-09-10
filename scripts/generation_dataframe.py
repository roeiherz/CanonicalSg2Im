from evaluation.inception import InceptionScore
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib
from sg2im.data import deprocess_batch, imagenet_deprocess
from sg2im.data.utils import decode_image
matplotlib.use('Agg')
from sg2im.meta_models import MetaGeneratorModel
import argparse, os
import torch
from types import SimpleNamespace
import json


def check_model_layout(df, model, inception_score, output_dir, deprocess_func, vocab, mode="gt"):
    inception_score.clean()
    model.args['skip_graph_model'] = True
    with torch.no_grad():
        img_ind = 0
        for i, row in df.iterrows():
            try:
                print("Iteration {}".format(img_ind))

                # Get boxes
                if mode == 'pred':
                    bbox = np.array(eval(row['predicted_boxes']))
                else:
                    bbox = np.array(eval(row['gt_boxes']))
                # bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
                # bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
                # bbox = np.concatenate([bbox, [[0, 0, 1, 1]]], axis=0)

                # Get labels
                object_class = eval(row['class'])
                indx = np.where(np.array(object_class) != '__image__')[0]
                label_one_hot = [c for c in object_class if c != "__image__"]
                label_one_hot = np.array([vocab["object_name_to_idx"][c] if c in vocab["object_name_to_idx"] else 180 for c in label_one_hot])
                # label_one_hot.append(vocab['object_name_to_idx']['__image__'])
                # print(label_one_hot)

                bbox = bbox[indx]
                label_one_hot = label_one_hot[indx]

                # Get Image_id
                if args.dataset == "vg":
                    obj_mask = np.array(label_one_hot) < 179
                    label_one_hot = np.array(label_one_hot)[obj_mask]
                    bbox = bbox[obj_mask]
                    # image_id = row['image_id'].split('/')[-1].split('.')[0]
                    image_id = re.findall(r'\d+', row['image_id'])[0]
                else:
                    # image_id = row['image_id']
                    image_id = re.findall(r'\d+', row['image_id'])[0]
                image_ids = [image_id]

                boxes = torch.FloatTensor(bbox).unsqueeze(0)
                labels = torch.LongTensor(label_one_hot)
                objs = labels.long().unsqueeze(0).unsqueeze(-1).cuda()

                samples = {}
                triplets, triplet_type, masks = None, None, None
                name = '{mode}_box_{mode}_mask'.format(mode=mode)
                samples[name] = model(objs, triplets, triplet_type, boxes_gt=boxes, masks_gt=masks, test_mode=True)[0]

                # Calc Inception score
                inception_score(samples[name])
                # Save images
                draw_datasets(samples, output_dir, deprocess_func, image_ids)
                img_ind += 1

            except Exception as e:
                print("Exception in iter {} - {}".format(img_ind, e))

        inception_mean, inception_std = inception_score.compute_score(splits=5)
        print(' >> ' + str(mode.upper()) + ' inception_mean: %.4f' % inception_mean)
        print(' >> ' + str(mode.upper()) + ' inception_std: %.4f' % inception_std)


def draw_datasets(samples, output_dir, deprocess_func, image_ids):
    for k, v in samples.items():
        samples[k] = np.transpose(deprocess_batch(v, deprocess_func=deprocess_func).cpu().numpy(), [0, 2, 3, 1])
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
    opt.skip_graph_model = True
    args.dataset = opt.dataset if not args.dataset else args.dataset
    model = MetaGeneratorModel(opt, device)
    # Load pre-trained weights for generation
    model.load_state_dict(checkpoint['model_state'], strict=False)
    # Eval
    model.eval()
    # Put on device
    model.to(device)

    # Init Inception Score
    inception_score = InceptionScore(device, batch_size=opt.batch_size, resize=True)
    # Get the data
    df = pd.read_csv(args.data_frame)
    check_model_layout(df, model, inception_score, args.output_dir, opt.deprocess_func, opt.vocab, mode=args.mode)
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
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--data_frame', default='')
    parser.add_argument('--mode', default='pred')
    parser.add_argument('--dataset', default='', choices=['vg', 'clevr', 'coco'])
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


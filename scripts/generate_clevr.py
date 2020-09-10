import copy
import pickle
from functools import partial
import cv2
import matplotlib
from PIL import Image
from torch.utils.data import DataLoader
from evaluation.inception import InceptionScore
from scripts.train import update_loader_params
from sg2im.data.dataset_params import get_dataset
from sg2im.data.packed_clevr_dialog import packed_clevr_inference_collate_fn, packed_sync_clevr_collate_fn, \
    packed_clevr_collate_fn
from sg2im.meta_models import MetaGeneratorModel
from sg2im.metrics import jaccard
from sg2im.model import get_conv_converse
from sg2im.utils import batch_to, remove_dummies_and_padding

matplotlib.use('Agg')
from sg2im.data.clevr_dialog import extract_objs, extract_triplets, clevr_collate_inference_fn
import numpy as np
from matplotlib import pyplot as plt
import argparse, os
from imageio import imwrite
import torch
from sg2im.data.utils import deprocess_batch, imagenet_deprocess, decode_image
import sg2im.vis as vis
from types import SimpleNamespace
import json

SORT_IDS = [0, 1, 5]


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


def draw_results(model, output_dir, scene_graphs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_objs, all_triplets, all_obj_to_img = encode_scene_graphs_list(model, scene_graphs)
    # Run the model forward
    with torch.no_grad():
        imgs_pred, boxes_pred, _, _, obj_to_img = model(all_objs, all_triplets, all_obj_to_img)
    imgs_pred = deprocess_batch(imgs_pred)
    boxes_pred = boxes_pred.cpu()
    obj_to_img = obj_to_img.cpu()
    # Save the generated images
    draw_predictions(scene_graphs, imgs_pred, boxes_pred, obj_to_img, output_dir)

    for i, sg in enumerate(scene_graphs):
        sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
        sg_img_path = os.path.join(output_dir, 'img_%06d_sg.png' % i)
        imwrite(sg_img_path, sg_img)
    plt.close("all")


def encode_scene_graphs_list(model, scene_graphs):
    batch = []
    for g in scene_graphs:
        objs = extract_objs(g, model.vocab)
        triplets = extract_triplets(g, model.vocab)
        batch.append((objs, triplets))
    all_objs, all_triplets, all_obj_to_img, all_triplets_to_img = clevr_collate_inference_fn(batch)
    return all_objs, all_triplets, all_obj_to_img, all_triplets_to_img


def draw_predictions(scene_graphs, imgs, boxes_pred, obj_to_img, output_dir):
    for i in range(imgs.shape[0]):
        # draw all objects, but skip __image__.
        img_np = imgs[i].numpy().transpose(1, 2, 0)
        plt.figure()
        _ = vis.draw_reversed_item(boxes_pred[np.where(obj_to_img == i)[0]][:-1], img=img_np,
                                   image_size=img_np.shape[:2],
                                   text=[', '.join(item.values()) for item in scene_graphs[i]['objects']])
        scene_layout = os.path.join(output_dir, 'img_%06d_layout.png' % i)
        plt.savefig(scene_layout)
        plt.close()
        img_path = os.path.join(output_dir, 'img_%06d_generated.png' % i)
        imwrite(img_path, img_np)


def draw_datasets(imgs_pred, output_dir, deprocess_func, image_ids, boxes_pred, boxes):
    samples = {'gt_box_pred_mask': imgs_pred}
    for k, v in samples.items():
        samples[k] = np.transpose(deprocess_batch(v, deprocess_func=deprocess_func).cpu().numpy(),
                                  [0, 2, 3, 1])
    for k, v in samples.items():
        # Set the output path
        if k == 'gt_img':
            path = os.path.join(output_dir, "gt")
        elif k == 'gt_box_pred_mask':
            path = os.path.join(output_dir, "generation")
        else:
            raise Exception

        os.makedirs(path, exist_ok=True)
        for i in range(v.shape[0]):
            RGB_img_i = cv2.cvtColor(v[i], cv2.COLOR_BGR2RGB)
            cv2.imwrite("{}/{}.jpg".format(path, image_ids[i]), RGB_img_i)

            # # Print boxes
            # boxess = boxes_pred[i].cpu().numpy()
            # x0 = boxess[:, 0]
            # y0 = boxess[:, 1]
            # x1 = x0 + boxess[:, 2]
            # y1 = y0 + boxess[:, 2]
            # boxess = np.stack([x0, y0, x1, y1], axis=1)
            # coords = (boxess * RGB_img_i.shape[0]).astype(int)
            # for box in coords:
            #     top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            #     image = cv2.rectangle(RGB_img_i, tuple(top_left), tuple(bottom_right), (0, 0, 255), 1)
            # cv2.imwrite("{}/{}_boxes.jpg".format(path, image_ids[i]), image)


def set_model(args, device, layout_checkpoint, trained_args):
    # Model
    opt = SimpleNamespace(**trained_args)
    opt.batch_size = args.batch_size
    # Support old replication
    if 'use_antisymmetry' in trained_args:
        opt.use_converse = trained_args['use_antisymmetry']
    if 'learned_antisymmetry' in trained_args:
        opt.learned_converse = trained_args['learned_antisymmetry']
    opt.debug = args.debug
    opt.max_samples = args.max_samples
    opt.max_objects = args.max_objects
    opt.min_objects = args.min_objects
    # opt.use_transitivity_test = 0
    opt.dataroot = '/specific/netapp5_2/gamir/DER-Roei/datasets'
    opt.dataset = "packed_clevr_syn"
    model = MetaGeneratorModel(opt, device)
    # Load pre-trained weights for generation
    layout_checkpoint['model_state'].update(layout_checkpoint['model_state'])
    model.load_state_dict(layout_checkpoint['model_state'], strict=False)
    # Eval
    model.eval()
    # Put on device
    model.to(device)
    return model, opt


def get_data_loader(opt):
    val_dset = get_dataset(opt.dataset, 'val', opt)
    collate = packed_clevr_inference_collate_fn
    # collate = packed_sync_clevr_collate_fn
    loader_kwargs = {'batch_size': opt.batch_size,
                     'num_workers': opt.loader_num_workers,
                     'shuffle': False,
                     'collate_fn': partial(collate, opt.vocab)}
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return val_loader


def draw_scene_graphs(scene_graphs, output_dir, vocab):
    for i, sg in enumerate(scene_graphs):
        sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'], vocab)
        sg_img_path = os.path.join(output_dir, 'img_%06d_sg.png' % i)
        imwrite(sg_img_path, sg_img)


def list_to_str(lst):
    str_ids = lst.split(',')
    lst = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            lst.append(id)
    return lst


def main(args):
    print(' >> Layout Not-learned checkpoint: {}'.format(args.layout_not_learned_checkpoint))
    print(' >> Layout learned checkpoint: {}'.format(args.layout_learned_checkpoint))

    if not os.path.isfile(args.layout_learned_checkpoint) or not os.path.isfile(args.layout_not_learned_checkpoint):
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
    layout_not_learned_checkpoint = torch.load(args.layout_not_learned_checkpoint, map_location=device)
    layout_learned_checkpoint = torch.load(args.layout_learned_checkpoint, map_location=device)
    trained_layout_not_learned_args = json.load(
        open(os.path.join(os.path.dirname(args.layout_not_learned_checkpoint), 'run_args.json'), 'rb'))
    trained_layout_learned_args = json.load(
        open(os.path.join(os.path.dirname(args.layout_learned_checkpoint), 'run_args.json'), 'rb'))
    set_args(args, trained_layout_not_learned_args)
    set_args(args, trained_layout_learned_args)

    model_not_learned, opt_not_learned = set_model(args, device, layout_not_learned_checkpoint,
                                                   trained_layout_not_learned_args)
    conv_weights_mat_not_learned = get_conv_converse(model_not_learned)
    model_learned, opt_learned = set_model(args, device, layout_learned_checkpoint,
                                           trained_layout_learned_args)
    conv_weights_mat_learned = get_conv_converse(model_learned)

    # Get the data
    data_loader_not_learned = get_data_loader(opt_not_learned)
    update_loader_params(data_loader_not_learned.dataset, conv_weights_mat_not_learned, None)
    data_loader_learned = get_data_loader(opt_learned)
    update_loader_params(data_loader_learned.dataset, conv_weights_mat_learned, None)
    data_loader_learned.dataset.data = copy.deepcopy(data_loader_not_learned.dataset.data)
    # Loag SGs
    load_sgs(data_loader_learned, data_loader_not_learned)

    inception_score = InceptionScore(device, batch_size=args.batch_size, resize=True)
    for data_loader, model, opt, name in [(data_loader_not_learned, model_not_learned, opt_not_learned, "not_learned"),
                                          (data_loader_learned, model_learned, opt_learned, "learned")]:
        print("Name: {}".format(name))
        output_dir = os.path.join(args.output_dir, name)
        check_model_layout(data_loader, model, inception_score, output_dir, opt.deprocess_func, opt.vocab)
    print("End")


def load_sgs(data_loader_learned, data_loader_not_learned):
    sgs = pickle.load(open('../sg2im/data/scene_graphs.pkl', "rb"))
    dataa = [sgs[ii] for ii in range(len(sgs))]
    data_loader_not_learned.dataset.data = copy.deepcopy(dataa)
    data_loader_learned.dataset.data = copy.deepcopy(dataa)


def check_model_layout(loader, model, inception_score, output_dir, deprocess_func, vocab):
    total_iou = 0.
    total_iou_05 = 0.
    total_iou_03 = 0.
    total_boxes = 0
    inception_score.clean()
    with torch.no_grad():
        batch_ind = 0
        for batch in loader:
            try:
                batch = batch_to(batch)
                objs, boxes, triplets, conv_counts, triplet_type, masks, image_ids = batch

                # Run the model with pred boxes
                model_out = model(objs, triplets, triplet_type, test_mode=True)
                imgs_pred, boxes_pred, masks_pred = model_out
                boxes_pred = torch.clamp(boxes_pred, 0., 1.)

                # Save images
                draw_datasets(imgs_pred, output_dir, deprocess_func, image_ids, boxes_pred, boxes)

                # Get results
                if imgs_pred is not None:
                    inception_score(imgs_pred)

                for i in range(boxes.size(0)):
                    boxes_sample = boxes[i]
                    boxes_pred_sample = boxes_pred[i]
                    boxes_pred_sample, boxes_sample = \
                        remove_dummies_and_padding(boxes_sample, objs[i], vocab, [boxes_pred_sample, boxes_sample])
                    iou, iou05, iou03 = jaccard(boxes_pred_sample, boxes_sample)
                    total_iou += iou.sum()
                    total_iou_05 += iou05.sum()
                    total_iou_03 += iou03.sum()
                    total_boxes += float(iou.shape[0])

                batch_ind += 1

            except Exception as e:
                print("Exception in iter {} - {}".format(batch_ind, e))

        if total_boxes != 0:
            iou = total_iou / total_boxes
            total_iou_05 = total_iou_05 / total_boxes
            total_iou_03 = total_iou_03 / total_boxes

        inception_mean, inception_std = inception_score.compute_score(splits=5)

    print(' >> avg_iou: %.4f' % iou)
    print(' >> total_iou_05: %.4f' % total_iou_05)
    print(' >> total_iou_03: %.4f' % total_iou_03)
    print(' >> inception_mean: %.4f' % inception_mean)
    print(' >> inception_std: %.4f' % inception_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout_not_learned_checkpoint', default='')
    parser.add_argument('--layout_learned_checkpoint', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--draw_scene_graphs', type=int, default=1)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--min_objects', type=int, default=15)
    parser.add_argument('--max_objects', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--debug', action='store_true',
                        help='only do one epoch and displays at each iteration')
    args = parser.parse_args()

    # set gpu ids
    args.gpu_ids = list_to_str(args.gpu_ids)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    main(args)

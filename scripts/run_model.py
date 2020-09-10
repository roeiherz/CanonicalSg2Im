import copy

import matplotlib

matplotlib.use('Agg')
from sg2im.data.clevr_dialog import extract_objs, extract_triplets, clevr_collate_inference_fn
import numpy as np
from matplotlib import pyplot as plt
import argparse, os
from imageio import imwrite
import torch
from sg2im.model import Sg2LayoutModel
from sg2im.data.utils import deprocess_batch
import sg2im.vis as vis
from types import SimpleNamespace
import json

# based on sample
# graphs = [
#     # just like val sample
#     {"objects": [{"shape": "cube", "color": "brown", "material": "metal", "size": "large"},
#                  {"shape": "cube", "color": "green", "material": "rubber", "size": "large"},
#                  {"shape": "sphere", "color": "gray", "material": "rubber", "size": "small"}],
#
#      "relationships": {
#          "right": [[], [0, 2], [0]],
#          "behind": [[], [0], [0, 1]],
#          "front": [[1, 2], [2], []],
#          "left": [[1, 2], [], [1]]}
#      },
#     # sparse    sample
#     {"objects": [{"shape": "cube", "color": "brown", "material": "metal", "size": "large"},
#                  {"shape": "cube", "color": "green", "material": "rubber", "size": "large"},
#                  {"shape": "sphere", "color": "gray", "material": "rubber", "size": "small"}],
#
#      "relationships": {
#          "right": [[], [2], [0]],
#          "behind": [[], [0], [1]],
#          "front": [[], [], []],
#          "left": [[], [], []]}
#      }
# ]

sg_template = {"objects": [],
               "relationships": {
                   "right": [],
                   "behind": [],
                   "front": [],
                   "left": []}
               }

obj_template = {"shape": "cylinder", "color": "brown", "material": "rubber", "size": "large"}
colors = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']


def auto_create_graphs(num_objs):
    sg_sparse = copy.deepcopy(sg_template)
    sg_dense = copy.deepcopy(sg_template)
    sg_hyper = copy.deepcopy(sg_template)

    clrs = colors[:num_objs]
    for color in clrs:
        obj = copy.deepcopy(obj_template)
        obj.update({'color': color})

        all_previous = list(range(len(sg_sparse["objects"])))

        # build dense
        sg_dense["relationships"]["front"].append(all_previous)
        sg_dense["relationships"]["right"].append(all_previous)
        sg_dense["objects"].append(obj)
        sg_dense["relationships"]["behind"].append([])
        sg_dense["relationships"]["left"].append([])

        # build sparse
        if len(all_previous) == 0:
            sg_sparse["relationships"]["front"].append([])
            sg_sparse["relationships"]["right"].append([])
        else:
            sg_sparse["relationships"]["front"].append([all_previous[-1]])
            sg_sparse["relationships"]["right"].append([all_previous[-1]])

        sg_sparse["relationships"]["behind"].append([])
        sg_sparse["relationships"]["left"].append([])
        sg_sparse["objects"].append(obj)

    sg_hyper = copy.deepcopy(sg_dense)
    all_objects = list(range(len(sg_sparse["objects"])))
    for i in all_objects:
        # build dense
        remains = list(set(all_objects) - set(sg_dense["relationships"]["front"][i]))
        sg_hyper["relationships"]["behind"][i].extend(remains)
        sg_hyper["relationships"]["left"][i].extend(remains)

    return sg_hyper, sg_sparse, sg_dense


scene_graphs = []
for i in range(3, 7):
    sg_hyper, sg_sparse, sg_dense = auto_create_graphs(i)
    scene_graphs.append(sg_sparse)
    scene_graphs.append(sg_dense)
    scene_graphs.append(sg_hyper)


def main(args):
    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    if not os.path.isdir(args.output_dir):
        print('Output directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(args.output_dir)

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda:0')
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
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    checkpoint.update(json.load(open(os.path.join(os.path.dirname(args.checkpoint), 'run_args.json'), 'rb')))
    model = Sg2LayoutModel(SimpleNamespace(**checkpoint))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)

    output_dir = args.output_dir
    draw_results(model, output_dir)


def draw_results(model, output_dir):
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


def encode_sg(model, scene_graphs):
    batch = []
    for g in scene_graphs:
        objs = extract_objs(g, model.vocab)
        triplets = extract_triplets(g, model.vocab)
        batch.append((objs, triplets))
    all_objs, all_triplets, all_obj_to_img, _ = clevr_collate_inference_fn(batch)
    return all_obj_to_img, all_objs, all_triplets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='/specific/netapp5_2/gamir/DER-Roei/bamir/Deployment/output/20190805T0026_default_config/iou_0.48873225_itr_10000.pt')
    parser.add_argument('--output_dir',
                        default='/specific/netapp5_2/gamir/DER-Roei/qa2im/output/bamir')
    parser.add_argument('--draw_scene_graphs',
                        type=int,
                        default=1)
    parser.add_argument('--mapping_output',
                        type=int,
                        default=0)
    parser.add_argument('--device',
                        default='gpu',
                        choices=['cpu', 'gpu', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    args = parser.parse_args()

    if not args.mapping_output:
        main(args)
    else:
        # Run multiple predictions
        print("Multiple Predictions:")
        # Main path for the current checkpoints and output dirs
        main_path = "/specific/netapp5_2/gamir/DER-Roei/bamir/Deployment/output/20190814T1102_debug_spade"
        # Mapping between checkpoint to output dir
        mapping = {
            "iou_0.462347_itr_5000.pt": "itr_00005000",
            "iou_0.46957305_itr_10000.pt": "itr_00010000",
            "iou_0.47982028_itr_15000.pt": "itr_00015000",
            "iou_0.4496151_itr_20000.pt": "itr_00020000",
            "iou_0.46003357_itr_25000.pt": "itr_00025000",
            "iou_0.50615066_itr_33500.pt": "itr_00033500",
            "iou_0.0_itr_34000.pt": "itr_00034000",
            "iou_0.0_itr_50000.pt": "itr_00050000",
        }
        print(" > Path: {}".format(main_path))
        print(" > Mapping:")
        for k, v in mapping.items():
            print(" >> ", k, ":", v)

        for checkpoint, output_dir in mapping.items():
            checkpoint_path = os.path.join(main_path, checkpoint)
            output_dir_path = os.path.join(main_path, output_dir)
            args.checkpoint = checkpoint_path
            args.output_dir = output_dir_path

            main(args)
            print(" > Saving checkpoint: {checkpoint}, output_dir: {output_dir}".format(checkpoint=checkpoint_path,
                                                                                        output_dir=output_dir_path))

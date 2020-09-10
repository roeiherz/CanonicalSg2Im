import json
import os
import random
import cv2
import torch
from sg2im.data.base_dataset import BaseDataset
import torchvision.transforms as T
import numpy as np
import h5py
import PIL
from sg2im.data.utils import Resize, deprocess_batch, encode_image, decode_image
from sg2im.vis import draw_item
from matplotlib import pyplot as plt


class VGSceneGraphDataset(BaseDataset):
    def __init__(self, h5_path, base_path, image_size=(256, 256), mask_size=0,
                 normalize_images=True, max_objects=10,
                 max_samples=None, include_relationships=True, use_orphaned_objects=True, debug=False,
                 use_transitivity=False, learned_transitivity=False, include_dummies=True, min_objects=0,
                 learned_symmetry=False, learned_converse=False, use_converse=False):
        super(VGSceneGraphDataset, self).__init__()
        self.include_dummies = include_dummies
        self.learned_transitivity = learned_transitivity
        self.learned_symmetry = learned_symmetry
        self.learned_converse = learned_converse
        self.image_dir = base_path
        self.image_size = image_size
        self.vocab = json.load(open(os.path.join(base_path, 'vocab.json')))
        self.num_objects = len(self.vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships
        self.use_transitivity = use_transitivity
        self.min_objects = min_objects

        transform = [Resize(image_size), T.ToTensor()]
        if normalize_images:
            # transform.append(imagenet_preprocess())
            transform.append(encode_image())
        self.transform = T.Compose(transform)

        if debug:
            h5_path = 'val.h5'

        # Load pretrained data
        self.data = {}
        with h5py.File(os.path.join(base_path, h5_path), 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

        if self.min_objects > 0:
            col_len = len(self.data["objects_per_image"])
            objects_mask = (self.data['objects_per_image'] >= self.min_objects).nonzero()[:, 0]
            cols = [col for col in self.data.keys() if len(self.data[col]) == col_len]
            for col in cols:
                self.data[col] = self.data[col][objects_mask]
            self.image_paths = np.array(self.image_paths)[objects_mask]

        self.vocab["attributes"] = {}
        self.vocab["attributes"]['objects'] = self.vocab['object_name_to_idx']
        self.vocab["reverse_attributes"] = {}
        for attr in self.vocab["attributes"].keys():
            self.vocab["reverse_attributes"][attr] = {v: k for k, v in self.vocab["attributes"][attr].items()}

        self.vocab["pred_name_to_idx"]['__padding__'] = len(self.vocab["pred_name_to_idx"].values())
        self.vocab["pred_idx_to_name"].append("__padding__")

        if use_converse:
            raise ValueError("not implemented")

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triplets: LongTensor of shape (T, 3) where triplets[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        img_path = os.path.join(self.image_dir, self.image_paths[index])
        image_id = int(self.image_paths[index].split('/')[-1].split('.')[0])
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size

        # Figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects - 1:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
        if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
            num_to_add = self.max_objects - 1 - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)

        O = len(obj_idxs)
        if self.include_dummies:
            O += 1

        objs = torch.LongTensor(O).fill_(-1)

        boxes = torch.FloatTensor([[-1, -1, -1, -1]]).repeat(O, 1)
        obj_idx_mapping = {}

        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            x0 = float(x) / WW
            y0 = float(y) / HH
            boxes[i] = torch.FloatTensor([x0, y0, float(w) / WW, float(h) / HH])
            obj_idx_mapping[obj_idx] = i

        # The last object will be the special __image__ object
        if self.include_dummies:
            objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

        triplets = []
        for r_idx in range(self.data['relationships_per_image'][index].item()):
            if not self.include_relationships:
                break
            s = self.data['relationship_subjects'][index, r_idx].item()
            p = self.data['relationship_predicates'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triplets.append([s, p, o])

        self.add_dummy_triplets(objs, triplets)
        triplets, conv_counts, triplet_type = self.add_learnt_triplets(triplets, objs.size(0))
        return image, {"objects": objs}, boxes, torch.LongTensor(triplets), torch.LongTensor(conv_counts), \
               torch.LongTensor(triplet_type), None, image_id


def vg_collate_fn(vocab, batch):
    """
    Collate function to be used when wrapping a CLEVRDialogDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triplets: FloatTensor of shape (T, 3) giving all triplets, where
    triplets[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triplets to images;
    triple_to_img[t] = n means that triplets[t] belongs to imgs[n].
    """
    # batch is a list, and each element is (image, objs, boxes, triplets)
    all_imgs, all_boxes, all_triplets, all_conv_counts, all_triplet_type = [], [], [], [], []
    all_objs = []
    all_masks = None
    all_image_ids = []

    max_triplets = 0
    max_objects = 0

    for i, (img, objs, boxes, triplets, conv_counts, triplet_type, _, image_id) in enumerate(batch):
        O = objs[list(objs.keys())[0]].size(0)
        T = triplets.size(0)

        if max_objects < O:
            max_objects = O

        if max_triplets < T:
            max_triplets = T

    for i, (img, objs, boxes, triplets, conv_counts, triplet_type, _, image_id) in enumerate(batch):
        all_imgs.append(img[None])
        all_image_ids.append(image_id)
        O, T = objs[list(objs.keys())[0]].size(0), triplets.size(0)

        # Padded objs
        attributes = list(objs.keys())
        sorted(attributes)
        attributes_to_index = {attributes[i]: i for i in range(len(attributes))}
        attributes_objects = torch.zeros(len(attributes), max_objects, dtype=torch.long)

        for k, v in objs.items():
            # Padded objects
            if max_objects - O > 0:
                zeros_v = torch.zeros(max_objects - O, dtype=torch.long)
                padd_v = torch.cat([v, zeros_v])
            else:
                padd_v = v
            attributes_objects[attributes_to_index[k], :] = padd_v
        attributes_objects = attributes_objects.transpose(1, 0)

        # Padded boxes
        if max_objects - O > 0:
            padded_boxes = torch.FloatTensor([[-1, -1, -1, -1]]).repeat(max_objects - O, 1)
            boxes = torch.cat([boxes, padded_boxes])

        # Padded triplets
        if max_triplets - T > 0:
            padded_triplets = torch.LongTensor([[0, vocab["pred_name_to_idx"]["__padding__"], 0]]).repeat(
                max_triplets - T, 1)
            triplets = torch.cat([triplets, padded_triplets])
            triplet_type = torch.cat([triplet_type, torch.LongTensor([0] * (max_triplets - T))])

        all_objs.append(attributes_objects)
        all_boxes.append(boxes)
        all_triplets.append(triplets)
        all_triplet_type.append(triplet_type)
        all_conv_counts.append(conv_counts)

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.stack(all_objs, dim=0)
    all_boxes = torch.stack(all_boxes, dim=0)
    all_triplets = torch.stack(all_triplets, dim=0)
    all_triplet_type = torch.stack(all_triplet_type, dim=0)
    all_conv_counts = torch.stack(all_conv_counts, dim=0).to(torch.float32)
    all_image_ids = torch.LongTensor(all_image_ids)

    out = (all_imgs, all_objs, all_boxes, all_triplets, all_conv_counts, all_triplet_type, all_masks, all_image_ids)
    return out


if __name__ == "__main__":
    dset = VGSceneGraphDataset(h5_path='val.h5',
                               base_path='/specific/netapp5_2/gamir/DER-Roei/datasets/VisualGenome',
                               image_size=(256, 256),
                               max_objects=10)
    # idx = np.where(np.array(dset.data['image_ids']) ==2411298)[0][0]
    item = dset[15]
    idx = 3408
    # idx = np.random.randint(0, len(dset))
    item = dset[idx]
    image, objs, boxes, triplets = item
    image = deprocess_batch(torch.unsqueeze(image, 0), deprocess_func=decode_image)[0]
    cv2.imwrite('img.png', np.transpose(image.cpu().numpy(), [1, 2, 0]))
    objs_text = np.array(dset.vocab['object_idx_to_name'])[objs['object_names']]
    # objs_text = obj_names_list
    # objs_text.append("object")
    mask = dset.data['object_names'][idx] != -1
    image_objects = dset.data['object_names'][idx][mask]

    draw_item(item, image_size=dset.image_size, text=objs_text)
    plt.figure()
    # plt.imshow(draw_scene_graph(dset.clevr_data['scenes'][idx]['objects'],
    #                             triplets=dset.clevr_data['scenes'][idx]['relationships'], vocab=dset.vocab))
    plt.savefig('sg.png')

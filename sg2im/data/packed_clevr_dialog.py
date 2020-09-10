import random
import re

from matplotlib import pyplot as plt
import json
import os
import pickle
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import PIL
from scripts.graphs_utils import get_minimal_and_transitive_triplets, get_symmetric_triplets, \
    get_current_and_transitive_triplets, get_edge_converse_triplets
from sg2im.data.base_dataset import BaseDataset
from sg2im.data.utils import Resize, deprocess_batch, encode_image, imagenet_preprocess
from sg2im.vis import draw_item, draw_scene_graph


def extract_bounding_boxes(scene):
    """
    Get for each scene the bounding box
    :param scene:
    :return:
    """
    objs = scene['objects']
    rotation = scene['directions']['right']

    x_list = []
    y_list = []
    h_list = []
    w_list = []

    for i, obj in enumerate(objs):
        [x, y, z] = obj['pixel_coords']

        [x1, y1, z1] = obj['3d_coords']

        cos_theta, sin_theta, _ = rotation

        x1 = x1 * cos_theta + y1 * sin_theta
        y1 = x1 * -sin_theta + y1 * cos_theta

        height_d = 6.9 * z1 * (15 - y1) / 2.0
        height_u = height_d
        width_l = height_d
        width_r = height_d

        if obj['shape'] == 'cylinder':
            d = 9.4 + y1
            h = 6.4
            s = z1

            height_u *= (s * (h / d + 1)) / ((s * (h / d + 1)) - (s * (h - s) / d))
            height_d = height_u * (h - s + d) / (h + s + d)

            width_l *= 11 / (10 + y1)
            width_r = width_l

        if obj['shape'] == 'cube':
            height_u *= 1.3 * 10 / (10 + y1)
            height_d = height_u
            width_l = height_u
            width_r = height_u

        y_min_coord = (y - height_d) / 320.
        y_max_coord = (y + height_u) / 320.
        x_man_coord = (x + width_r) / 480.
        x_min_coord = (x - width_l) / 480.

        x_list.append(x_min_coord)
        y_list.append(y_min_coord)
        h_list.append(y_max_coord - y_min_coord)
        w_list.append(x_man_coord - x_min_coord)

    return x_list, y_list, w_list, h_list


ORIGINAL_EDGE = 0
TRANSITIVE_EDGE = 1
SYMMETRIC_EDGE = 2
ANTI_SYMMETRIC_EDGE = 3


class PackedCLEVRDialogDataset(BaseDataset):
    def __init__(self, h5_path, base_path, mode, image_size=(64, 64), mask_size=0,
                 normalize_images=True, min_objects=10, max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True, debug=False,
                 dense_scenes=False, learned_transitivity=False, include_dummies=True, use_transitivity=False,
                 use_all_relations=False, use_converse=False, learned_symmetry=False,
                 learned_converse=False):
        super(PackedCLEVRDialogDataset, self).__init__()

        self.image_dir = os.path.join(base_path, 'images')
        self.image_size = image_size
        self.mask_size = mask_size
        self.learned_transitivity = learned_transitivity
        self.learned_symmetry = learned_symmetry
        self.learned_converse = learned_converse
        self.include_dummies = include_dummies
        self.use_transitivity = use_transitivity
        self.use_all_relations = use_all_relations
        self.use_converse = use_converse
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.min_objects = min_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships
        self.mode = mode

        # objects
        self.vocab = {}
        self.vocab["use_object_embedding"] = False

        # predicates
        self.register_augmented_relations()

        # attributes, currently ignored.
        self.vocab["attributes"] = {}
        self.vocab["attributes"]['shape'] = {'__image__': 0, 'cube': 1, 'sphere': 2, 'cylinder': 3}
        self.vocab["attributes"]["color"] = {'__image__': 0, 'gray': 1, 'red': 2, 'blue': 3, 'green': 4, 'brown': 5,
                                             'purple': 6, 'cyan': 7, 'yellow': 8}
        self.vocab["attributes"]["material"] = {'__image__': 0, 'rubber': 1, 'metal': 2}
        self.vocab["attributes"]["size"] = {'__image__': 0, 'small': 1, 'large': 2}
        self.vocab["reverse_attributes"] = {}
        for attr in self.vocab["attributes"].keys():
            self.vocab["reverse_attributes"][attr] = {v: k for k, v in self.vocab["attributes"][attr].items()}

        self.vocab['object_name_to_idx'] = {}
        ind = 0
        for attr in self.vocab["attributes"].keys():
            for attr_label in self.vocab["attributes"][attr].keys():
                if ind != 0:
                    keyy = "{}_{}".format(attr_label, ind)
                    self.vocab['object_name_to_idx'][keyy] = ind
                else:
                    # __image__
                    self.vocab['object_name_to_idx'][attr_label] = ind
                ind += 1
        self.vocab['object_idx_to_name'] = {}
        for k, v in self.vocab['object_name_to_idx'].items():
            self.vocab['object_idx_to_name'][v] = k

        self.image_paths = []
        transform = [Resize(image_size), T.ToTensor()]
        if normalize_images:
            # transform.append(imagenet_preprocess())
            transform.append(encode_image())
        self.transform = T.Compose(transform)

        if debug:
            self.clevr_data = pickle.load(open("clevr_data_sample.pkl", 'rb'))
            self.dialog_data = pickle.load(open("dialog_data_sample.pkl", 'rb'))
        else:
            self.clevr_data = json.load(
                open(os.path.join(base_path, 'scenes/CLEVR_{mode}_scenes.json'.format(mode=mode)), 'rb'))
            self.dialog_data = json.load(open(os.path.join(base_path, h5_path), 'rb'))

        # CLEVR doesn't have any dense scenes; number of objects 3-10
        if dense_scenes:
            self.keep_dense_scenes()

    def __len__(self):
        return len(self.dialog_data)

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - shapes: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j] means that (shapes[i], p, shapes[j]) is a triple.
        """

        # Get image
        entry = self.dialog_data[index]
        img_path = os.path.join(self.image_dir, entry['split'], entry['image_filename'])
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                image = self.transform(image.convert('RGB'))

        sg = self.clevr_data['scenes'][index]
        image_id = sg['image_index']
        objs = self.extract_objs(sg)

        # Get boxes
        x, y, w, h = extract_bounding_boxes(sg)
        boxes = list(zip(x, y, w, h))

        # Compute centers of all objects
        obj_centers = []
        for i, obj_idx in enumerate(boxes):
            x0, y0, w, h = boxes[i]
            mean_x = x0 + 0.5 * w
            mean_y = y0 + 0.5 * h
            obj_centers.append([mean_x, mean_y])
        obj_centers = torch.FloatTensor(obj_centers)

        if self.include_dummies:
            # Add dummy __image__ box
            boxes.append([-1, -1, -1, -1])
        boxes = torch.FloatTensor(boxes)

        # Add triplets
        triplets = []
        self.add_location_triplets(boxes, obj_centers, objs['shape'], triplets)
        self.add_dummy_triplets(objs['shape'], triplets)
        triplets, conv_counts, triplet_type = self.add_learnt_triplets(triplets, boxes.size(0))

        # Add masks
        masks = None

        return image, objs, boxes, torch.LongTensor(triplets), torch.LongTensor(conv_counts), \
               torch.LongTensor(triplet_type), masks, image_id

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def keep_dense_scenes(self):
        new_clv_data = []
        new_clv_dial_data = []
        for ind in range(len(self.clevr_data['scenes'])):
            num_obj = len(self.clevr_data['scenes'][ind]['objects'])
            if self.min_objects < num_obj < self.max_objects:
                new_clv_data.append(self.clevr_data['scenes'][ind])
                new_clv_dial_data.append(self.dialog_data[ind])

        if len(new_clv_data) == 0 or len(new_clv_dial_data) == 0:
            print("No data has been selected in dense scenes mode")

        # Replace to dense scenes
        self.clevr_data['scenes'] = new_clv_data
        self.dialog_data = new_clv_dial_data

    def extract_objs(self, sg):
        objs = {}
        for attr in self.vocab["attributes"].keys():
            attr_list = [self.vocab["attributes"][attr][obj[attr]] for obj in sg['objects']]
            # Add dummy __image__ object
            if self.include_dummies:
                attr_list.append(self.vocab["attributes"][attr]['__image__'])
            objs[attr] = torch.LongTensor(attr_list)
        return objs


def packed_clevr_collate_fn(vocab, batch):
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
    all_imgs, all_boxes, all_triplets, all_triplet_type, all_conv_counts = [], [], [], [], []
    all_objs = []
    all_masks = None
    all_image_ids = []

    max_triplets = 0
    max_objects = 0
    for i, (img, objs, boxes, triplets, conv_counts, triplet_type, _, _) in enumerate(batch):
        O = boxes.size(0)
        T = triplets.size(0)

        if max_objects < O:
            max_objects = O

        if max_triplets < T:
            max_triplets = T

    for i, (img, objs, boxes, triplets, conv_counts, triplet_type, _, image_id) in enumerate(batch):
        all_imgs.append(img[None])
        O, T = boxes.size(0), triplets.size(0)

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
        all_image_ids.append(image_id)

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.stack(all_objs, dim=0)
    all_boxes = torch.stack(all_boxes, dim=0)
    all_triplets = torch.stack(all_triplets, dim=0)
    all_triplet_type = torch.stack(all_triplet_type, dim=0)
    all_conv_counts = torch.stack(all_conv_counts, dim=0).to(torch.float32)
    all_image_ids = torch.LongTensor(all_image_ids)

    out = (all_imgs, all_objs, all_boxes, all_triplets, all_conv_counts, all_triplet_type, all_masks, all_image_ids)
    return out


class PackedGenCLEVRDataset(BaseDataset):
    def __init__(self, base_path, mode, image_size=(64, 64), mask_size=0,
                 normalize_images=True, min_objects=10, max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True, debug=False,
                 learned_transitivity=False, include_dummies=True, use_transitivity=False,
                 use_all_relations=False, use_converse=False, learned_symmetry=False,
                 learned_converse=False):
        super(PackedGenCLEVRDataset, self).__init__()

        self.image_dir = os.path.join(base_path, 'images')
        self.image_size = image_size
        self.mask_size = mask_size
        self.learned_transitivity = learned_transitivity
        self.learned_symmetry = learned_symmetry
        self.learned_converse = learned_converse
        self.include_dummies = include_dummies
        self.use_transitivity = use_transitivity
        self.use_all_relations = use_all_relations
        self.use_converse = use_converse
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.min_objects = min_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships
        self.mode = mode

        # objects
        self.vocab = {}
        self.vocab["use_object_embedding"] = False

        # predicates
        self.register_augmented_relations()

        # attributes, currently ignored.
        self.vocab["attributes"] = {}
        self.vocab["attributes"]['shape'] = {'__image__': 0, 'cube': 1, 'sphere': 2, 'cylinder': 3}
        self.vocab["attributes"]["color"] = {'__image__': 0, 'gray': 1, 'red': 2, 'blue': 3, 'green': 4, 'brown': 5,
                                             'purple': 6, 'cyan': 7, 'yellow': 8}
        self.vocab["attributes"]["material"] = {'__image__': 0, 'rubber': 1, 'metal': 2}
        self.vocab["attributes"]["size"] = {'__image__': 0, 'small': 1, 'large': 2}
        self.vocab["reverse_attributes"] = {}
        for attr in self.vocab["attributes"].keys():
            self.vocab["reverse_attributes"][attr] = {v: k for k, v in self.vocab["attributes"][attr].items()}

        self.vocab['object_name_to_idx'] = {}
        ind = 0
        for attr in self.vocab["attributes"].keys():
            for attr_label in self.vocab["attributes"][attr].keys():
                if ind != 0:
                    keyy = "{}_{}".format(attr_label, ind)
                    self.vocab['object_name_to_idx'][keyy] = ind
                else:
                    # __image__
                    self.vocab['object_name_to_idx'][attr_label] = ind
                ind += 1
        self.vocab['object_idx_to_name'] = {}
        for k, v in self.vocab['object_name_to_idx'].items():
            self.vocab['object_idx_to_name'][v] = k

        self.image_paths = []
        transform = [Resize(image_size), T.ToTensor()]
        if normalize_images:
            # transform.append(imagenet_preprocess())
            transform.append(encode_image())
        self.transform = T.Compose(transform)

        # Load data
        if debug:
            self.data = self.create_packed_sgs()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - shapes: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j] means that (shapes[i], p, shapes[j]) is a triple.
        """

        # Get image
        sg = self.data[index]
        image_id = sg['image_index']
        objs = self.extract_objs(sg)

        # Get boxes
        boxes = sg['boxes']

        # Compute centers of all objects
        obj_centers = []
        for i, obj_idx in enumerate(boxes):
            x0, y0, w, h = boxes[i]
            mean_x = x0 + 0.5 * w
            mean_y = y0 + 0.5 * h
            obj_centers.append([mean_x, mean_y])
        obj_centers = torch.FloatTensor(obj_centers)

        if self.include_dummies:
            # Add dummy __image__ box
            boxes.append([-1, -1, -1, -1])
        boxes = torch.FloatTensor(boxes)

        # Add triplets
        triplets = []
        self.add_location_triplets(boxes, obj_centers, objs['shape'], triplets)
        self.add_dummy_triplets(objs['shape'], triplets)
        triplets, conv_counts, triplet_type = self.add_learnt_triplets(triplets, boxes.size(0))

        # Add masks
        masks = None
        image = None

        return image, objs, boxes, torch.LongTensor(triplets), torch.LongTensor(conv_counts), \
               torch.LongTensor(triplet_type), masks, image_id

    def extract_objs(self, sg):
        objs = {}
        for attr in self.vocab["attributes"].keys():
            attr_list = [self.vocab["attributes"][attr][obj[attr]] for obj in sg['objects']]
            # Add dummy __image__ object
            if self.include_dummies:
                attr_list.append(self.vocab["attributes"][attr]['__image__'])
            objs[attr] = torch.LongTensor(attr_list)
        return objs

    def create_packed_sgs(self):
        scene_graphs = []
        for j in range(self.max_samples):

            # if j not in [5, 9, 11, 46, 65, 85, 161]:
            #     continue

            objects = []
            # Generate random number of objects
            max_objs = random.randint(self.min_objects, self.max_objects)
            # Generate objects, boxes and centers
            for i in range(0, max_objs):
                color_ind = random.randint(1, len(self.vocab["attributes"]["color"]) - 1)
                color = self.vocab["reverse_attributes"]["color"][color_ind]
                size_ind = random.randint(1, len(self.vocab["attributes"]["size"]) - 1)
                size = self.vocab["reverse_attributes"]["size"][size_ind]
                shape_ind = random.randint(1, len(self.vocab["attributes"]["shape"]) - 1)
                shape = self.vocab["reverse_attributes"]["shape"][shape_ind]
                material_ind = random.randint(1, len(self.vocab["attributes"]["material"]) - 1)
                material = self.vocab["reverse_attributes"]["material"][material_ind]
                object = {'color': color, 'size': size, 'shape': shape, 'material': material}
                objects.append(object)

            # Generate boxes and centers
            boxes = []
            for i in range(0, max_objs):
                size = objects[i]['size']
                if size == "small":
                    obj_size = 0.1
                else:
                    obj_size = 0.2

                x0, y0 = np.random.uniform(0, 1 - obj_size, size=2)
                h, w = obj_size, obj_size
                boxes.append([x0, y0, w, h])

            # Compute centers of all objects
            obj_centers = []
            for i, obj_idx in enumerate(boxes):
                x0, y0, w, h = boxes[i]
                mean_x = x0 + 0.5 * w
                mean_y = y0 + 0.5 * h
                obj_centers.append([mean_x, mean_y])

            # Get relations
            triplets = []
            self.add_location_triplets(torch.FloatTensor(boxes), torch.FloatTensor(obj_centers), None, triplets)

            sg = {'objects': objects, 'boxes': boxes, 'image_index': str(j), 'obj_centers': obj_centers,
                  'relationships': triplets}
            scene_graphs.append(sg)
        return scene_graphs


class PackedSynCLEVRDataset(BaseDataset):
    def __init__(self, base_path, mode, image_size=(64, 64), mask_size=0,
                 min_objects=16, max_objects=32, max_samples=10,
                 include_relationships=True, use_orphaned_objects=True, debug=False,
                 dense_scenes=False, learned_transitivity=False, include_dummies=True, use_transitivity=False,
                 use_all_relations=False, use_converse=False, learned_symmetry=False,
                 learned_converse=False):
        super(PackedSynCLEVRDataset, self).__init__()

        self.image_dir = os.path.join(base_path, 'images')
        self.image_size = image_size
        self.mask_size = mask_size
        self.learned_transitivity = learned_transitivity
        self.learned_symmetry = learned_symmetry
        self.learned_converse = learned_converse
        self.include_dummies = include_dummies
        self.use_transitivity = use_transitivity
        self.use_all_relations = use_all_relations
        self.use_converse = use_converse
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.min_objects = min_objects
        self.max_samples = 10 if debug else max_samples
        self.include_relationships = include_relationships
        self.mode = mode

        # objects
        self.vocab = {}
        self.vocab["use_object_embedding"] = False

        # predicates
        self.register_augmented_relations()

        # attributes
        self.vocab["attributes"] = {}
        self.vocab["attributes"]['shape'] = {'__image__': 0, 'cube': 1, 'sphere': 2, 'cylinder': 3}
        self.vocab["attributes"]["color"] = {'__image__': 0, 'gray': 1, 'red': 2, 'blue': 3, 'green': 4, 'brown': 5,
                                             'purple': 6, 'cyan': 7, 'yellow': 8}
        self.vocab["attributes"]["material"] = {'__image__': 0, 'rubber': 1, 'metal': 2}
        self.vocab["attributes"]["size"] = {'__image__': 0, 'small': 1, 'large': 2}
        self.vocab["reverse_attributes"] = {}
        for attr in self.vocab["attributes"].keys():
            self.vocab["reverse_attributes"][attr] = {v: k for k, v in self.vocab["attributes"][attr].items()}

        self.vocab['object_name_to_idx'] = {}
        ind = 0
        for attr in self.vocab["attributes"].keys():
            for attr_label in self.vocab["attributes"][attr].keys():
                if ind != 0:
                    keyy = "{}_{}".format(attr_label, ind)
                    self.vocab['object_name_to_idx'][keyy] = ind
                else:
                    # __image__
                    self.vocab['object_name_to_idx'][attr_label] = ind
                ind += 1

        # Load data
        self.data = self.create_packed_sgs()

    def create_packed_sgs(self):
        scene_graphs = []
        for j in range(self.max_samples):

            # if j not in [5, 9, 11, 46, 65, 85, 161]:
            #     continue

            objects = []
            # Generate random number of objects
            max_objs = random.randint(self.min_objects, self.max_objects)
            # Generate objects, boxes and centers
            for i in range(0, max_objs):
                color_ind = random.randint(1, len(self.vocab["attributes"]["color"]) - 1)
                color = self.vocab["reverse_attributes"]["color"][color_ind]
                size_ind = random.randint(1, len(self.vocab["attributes"]["size"]) - 1)
                size = self.vocab["reverse_attributes"]["size"][size_ind]
                shape_ind = random.randint(1, len(self.vocab["attributes"]["shape"]) - 1)
                shape = self.vocab["reverse_attributes"]["shape"][shape_ind]
                material_ind = random.randint(1, len(self.vocab["attributes"]["material"]) - 1)
                material = self.vocab["reverse_attributes"]["material"][material_ind]
                object = {'color': color, 'size': size, 'shape': shape, 'material': material}
                objects.append(object)

            # Generate boxes and centers
            boxes = []
            for i in range(0, max_objs):
                size = objects[i]['size']
                if size == "small":
                    obj_size = 0.1
                else:
                    obj_size = 0.2

                x0, y0 = np.random.uniform(0, 1 - obj_size, size=2)
                h, w = obj_size, obj_size
                boxes.append([x0, y0, w, h])

            # Compute centers of all objects
            obj_centers = []
            for i, obj_idx in enumerate(boxes):
                x0, y0, w, h = boxes[i]
                mean_x = x0 + 0.5 * w
                mean_y = y0 + 0.5 * h
                obj_centers.append([mean_x, mean_y])

            # Get relations
            triplets = []
            self.add_location_triplets(torch.FloatTensor(boxes), torch.FloatTensor(obj_centers), None, triplets)

            sg = {'objects': objects, 'boxes': boxes, 'image_index': str(j), 'obj_centers': obj_centers,
                  'relationships': triplets}
            scene_graphs.append(sg)
        return scene_graphs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - shapes: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j] means that (shapes[i], p, shapes[j]) is a triple.
        """

        # Get Scene Graph
        sg = self.data[index]
        objs = self.extract_objs(sg)

        # Get boxes
        boxes = sg['boxes']

        # Compute centers of all objects
        obj_centers = []
        for i, obj_idx in enumerate(boxes):
            x0, y0, w, h = boxes[i]
            mean_x = x0 + 0.5 * w
            mean_y = y0 + 0.5 * h
            obj_centers.append([mean_x, mean_y])
        obj_centers = torch.FloatTensor(obj_centers)

        # Add dummy __image__ box
        if self.include_dummies:
            boxes.append([-1, -1, -1, -1])
        boxes = torch.FloatTensor(boxes)

        # Add triplets
        triplets = []
        self.add_location_triplets(boxes, obj_centers, objs, triplets)
        self.add_dummy_triplets(objs, triplets)
        triplets, triplet_type, source_edges = self.add_learnt_triplets(triplets, boxes.size(0))

        return objs, boxes, torch.LongTensor(triplets), torch.LongTensor(triplet_type), \
               torch.LongTensor(source_edges), sg

    def add_location_triplets(self, boxes, obj_centers, objs, triplets):
        O = boxes.size(0)
        __image__ = self.vocab['object_name_to_idx']['__image__']
        real_objs = []
        if O > 1:
            real_objs = (boxes != -1).any(dim=-1).nonzero().squeeze(1)
        for cur in real_objs:
            choices = [obj for obj in real_objs if obj != cur]
            for other in choices:
                s, o = int(cur), int(other)
                sx0, sy0, sw, sh = boxes[s]
                sx1, sy1 = sx0 + sw / 2, sy0 + sh / 2
                ox0, oy0, ow, oh = boxes[o]
                ox1, oy1 = ox0 + ow / 2, oy0 + oh / 2

                d = obj_centers[s] - obj_centers[o]
                if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                    p = '__surrounding__'
                    p = self.vocab['pred_name_to_idx'][p]
                    triplets.append([s, p, o])

                elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                    p = '__inside__'
                    p = self.vocab['pred_name_to_idx'][p]
                    triplets.append([s, p, o])
                else:
                    if d[0] > 0:
                        p = '__right of__'
                        p = self.vocab['pred_name_to_idx'][p]
                        triplets.append([s, p, o])

                    elif d[0] < 0:
                        p = '__left of__'
                        p = self.vocab['pred_name_to_idx'][p]
                        triplets.append([s, p, o])

                    if d[1] > 0:
                        p = '__below__'
                        p = self.vocab['pred_name_to_idx'][p]
                        triplets.append([s, p, o])

                    elif d[1] < 0:
                        p = '__above__'
                        p = self.vocab['pred_name_to_idx'][p]
                        triplets.append([s, p, o])

    def add_learnt_triplets(self, triplets, O):
        triplets = np.unique(triplets, axis=0)
        triplet_type = []
        new_triplets = []
        source_edges = []
        triplets = np.array(triplets)

        meta_relations = [self.vocab['pred_name_to_idx'][p] for p in self.meta_relations]
        non_meta_relations = set(self.vocab['pred_name_to_idx'].values()) - set(meta_relations)
        for rel in non_meta_relations:
            rel_new_triplets = []
            rel_triplets = triplets[triplets[:, 1] == rel].copy()
            if len(rel_triplets) == 0:
                continue

            if self.vocab['pred_idx_to_name'][rel] in self.augmented_relations:
                minimal_triplets, transitive_triplets = get_minimal_and_transitive_triplets(rel_triplets)
            else:
                minimal_triplets, transitive_triplets = get_current_and_transitive_triplets(rel_triplets)
            rel_new_triplets.extend(minimal_triplets)
            triplet_type.extend([ORIGINAL_EDGE] * len(minimal_triplets))

            if self.learned_transitivity:
                rel_new_triplets.extend(transitive_triplets)
                triplet_type.extend([TRANSITIVE_EDGE] * len(transitive_triplets))

            if self.learned_symmetry:
                symmetric_triplets = get_symmetric_triplets(minimal_triplets)
                rel_new_triplets.extend(symmetric_triplets)
                triplet_type.extend([SYMMETRIC_EDGE] * len(symmetric_triplets))

            if self.learned_converse:
                converse_triplets = get_edge_converse_triplets(minimal_triplets, self.vocab)
                rel_new_triplets.extend(converse_triplets)
                triplet_type.extend([ANTI_SYMMETRIC_EDGE] * len(converse_triplets))

            source_edges.extend([rel] * len(rel_new_triplets))
            new_triplets.extend(rel_new_triplets)

        for rel in meta_relations:
            rel_triplets = triplets[triplets[:, 1] == rel].copy()
            new_triplets.extend(rel_triplets)
            triplet_type.extend([ORIGINAL_EDGE] * len(rel_triplets))
            source_edges.extend([rel] * len(rel_triplets))

        triplet_type = np.array(triplet_type)
        source_edges = np.array(source_edges)
        triplets = np.array(new_triplets)
        return triplets, triplet_type, source_edges

    def add_dummy_triplets(self, objs, triplets):
        # Add dummy __in_image__ relationships for all objects
        if self.include_dummies:
            O = objs[list(objs.keys())[0]].size(0)
            __image__ = O - 1
            in_image = self.vocab['pred_name_to_idx']['__in_image__']
            for i in range(O):
                if i == __image__:
                    continue
                triplets.append([i, in_image, __image__])

    def extract_objs(self, sg):
        objs = {}
        for attr in self.vocab["attributes"].keys():
            attr_list = [self.vocab["attributes"][attr][obj[attr]] for obj in sg['objects']]
            # Add dummy __image__ object
            if self.include_dummies:
                attr_list.append(self.vocab["attributes"][attr]['__image__'])
            objs[attr] = torch.LongTensor(attr_list)
        return objs


def packed_clevr_inference_collate_fn(vocab, batch):
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
    all_imgs, all_boxes, all_triplets, all_triplet_type, all_conv_counts = [], [], [], [], []
    all_objs = []
    all_masks = None
    all_image_ids = []

    max_triplets = 0
    max_objects = 0
    for i, (img, objs, boxes, triplets, conv_counts, triplet_type, _, _) in enumerate(batch):
        O = boxes.size(0)
        T = triplets.size(0)

        if max_objects < O:
            max_objects = O

        if max_triplets < T:
            max_triplets = T

    for i, (img, objs, boxes, triplets, conv_counts, triplet_type, _, image_id) in enumerate(batch):
        O, T = boxes.size(0), triplets.size(0)

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
        all_image_ids.append(image_id)

    all_objs = torch.stack(all_objs, dim=0)
    all_boxes = torch.stack(all_boxes, dim=0)
    all_triplets = torch.stack(all_triplets, dim=0)
    all_triplet_type = torch.stack(all_triplet_type, dim=0)
    all_conv_counts = torch.stack(all_conv_counts, dim=0).to(torch.float32)

    out = (all_objs, all_boxes, all_triplets, all_conv_counts, all_triplet_type, all_masks, all_image_ids)
    return out


def packed_sync_clevr_collate_fn(vocab, batch):
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
    all_imgs, all_boxes, all_triplets, all_triplet_type, all_source_edges = [], [], [], [], []
    all_objs = []
    all_image_ids = []

    max_triplets = 0
    max_objects = 0
    for i, (objs, boxes, triplets, triplet_type, source_edges, sg) in enumerate(batch):
        O = boxes.size(0)
        T = triplets.size(0)

        if max_objects < O:
            max_objects = O

        if max_triplets < T:
            max_triplets = T

    for i, (objs, boxes, triplets, triplet_type, source_edges, sg) in enumerate(batch):
        all_image_ids.append(int(re.findall(r'\d+', sg['image_index'])[0]))
        O, T = boxes.size(0), triplets.size(0)

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
            source_edges = torch.cat(
                [source_edges, torch.LongTensor([vocab["pred_name_to_idx"]["__padding__"]] * (max_triplets - T))])

        all_objs.append(attributes_objects)
        all_boxes.append(boxes)
        all_triplets.append(triplets)
        all_triplet_type.append(triplet_type)
        all_source_edges.append(source_edges)

    all_objs = torch.stack(all_objs, dim=0)
    all_boxes = torch.stack(all_boxes, dim=0)
    all_triplets = torch.stack(all_triplets, dim=0)
    all_triplet_type = torch.stack(all_triplet_type, dim=0)
    all_source_edges = torch.stack(all_source_edges, dim=0)
    all_image_ids = torch.LongTensor(all_image_ids)

    out = (all_objs, all_boxes, all_triplets, all_triplet_type, all_source_edges, all_image_ids)
    return out


if __name__ == "__main__":

    dset = PackedCLEVRDialogDataset(h5_path='clevr_dialog_train_raw.json',
                                    base_path='/specific/netapp5_2/gamir/DER-Roei/datasets/CLEVR/CLEVR_Dialog',
                                    mode="train",
                                    image_size=(256, 256),
                                    debug=True,
                                    max_objects=5,
                                    dense_scenes=False)

    # ss = 0
    # ii = []
    # for scene in dset.clevr_data['scenes']:
    #     ss += len(scene['objects'])
    #     ii.append(len(scene['objects']))
    # ss /= len(dset.clevr_data['scenes'])
    #
    # import matplotlib.pyplot as plt
    #
    # ii = np.array(ii)
    # _ = plt.hist(ii, bins='auto')  # arguments are passed to np.histogram
    # plt.savefig("hist.png")

    it = dset[2]
    while True:
        # idx = 5149
        idx = np.random.randint(0, len(dset))
        item = dset[idx]
        image, objs, boxes, triplets = item
        image = deprocess_batch(torch.unsqueeze(image, 0))[0]
        cv2.imwrite('/tmp/img.png', np.transpose(image.cpu().numpy(), [1, 2, 0]))
        draw_item(item, image_size=dset.image_size)  # dset.clevr_data['scenes'][index]
        plt.figure()
        plt.imshow(draw_scene_graph(dset.clevr_data['scenes'][idx]['objects'],
                                    triplets=dset.clevr_data['scenes'][idx]['relationships'], vocab=dset.vocab))
        plt.savefig('/tmp/sg.png')

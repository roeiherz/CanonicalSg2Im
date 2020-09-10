import torch
from torch.utils.data import Dataset
import numpy as np
from scripts.graphs_utils import get_edge_converse_triplets, get_current_and_transitive_triplets, \
    triplets_to_minimal

ORIGINAL_EDGE = 0
TRANSITIVE_EDGE = 1
SYMMETRIC_EDGE = 2
ANTI_SYMMETRIC_EDGE = 3


class BaseDataset(Dataset):
    meta_relations = ["__padding__", "__in_image__"]
    augmented_relations = ['__below__', '__above__', '__left of__', '__right of__', '__inside__', '__surrounding__']

    def __init__(self):

        self.max_samples = None
        self.data = None
        self.include_dummies = None
        self.vocab = None
        self.learned_transitivity = None
        self.learned_symmetry = None
        self.learned_converse = None
        self.converse_candidates_weights = None
        self.trans_candidates_weights = None

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def add_location_triplets(self, boxes, obj_centers, objs, triplets):
        new_triplets = []
        O = objs.size(0)
        __image__ = self.vocab['object_name_to_idx']['__image__']
        real_objs = []
        if O > 1:
            real_objs = (objs != __image__).nonzero().squeeze(1)
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
                    new_triplets.append([s, p, o])

                elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                    p = '__inside__'
                    p = self.vocab['pred_name_to_idx'][p]
                    new_triplets.append([s, p, o])
                else:
                    if d[0] > 0:
                        p = '__right of__'
                        p = self.vocab['pred_name_to_idx'][p]
                        new_triplets.append([s, p, o])

                    elif d[0] < 0:
                        p = '__left of__'
                        p = self.vocab['pred_name_to_idx'][p]
                        new_triplets.append([s, p, o])

                    if d[1] > 0:
                        p = '__below__'
                        p = self.vocab['pred_name_to_idx'][p]
                        new_triplets.append([s, p, o])

                    elif d[1] < 0:
                        p = '__above__'
                        p = self.vocab['pred_name_to_idx'][p]
                        new_triplets.append([s, p, o])

        new_triplets = np.array(new_triplets)
        for p in self.augmented_relations:
            p = self.vocab["pred_name_to_idx"][p]
            rel_triplets = new_triplets[new_triplets[:,1] == p]
            new_rel_triplets = triplets_to_minimal(rel_triplets)
            triplets.extend(new_rel_triplets)

    def add_learnt_triplets(self, triplets, O):
        triplets = np.unique(triplets, axis=0).astype('int')
        new_triplets = []
        N_REL = len(self.vocab['pred_name_to_idx'].values())
        conv_counts = np.zeros((N_REL, N_REL + 1))
        triplets = np.array(triplets)

        meta_relations = [self.vocab['pred_name_to_idx'][p] for p in self.meta_relations]
        non_meta_relations = set(self.vocab['pred_name_to_idx'].values())-set(meta_relations)
        for rel in non_meta_relations:
            rel_new_triplets = []
            rel_triplets = triplets[triplets[:, 1] == rel].copy()
            if len(rel_triplets) == 0:
                continue

            rel_new_triplets.extend(rel_triplets)
            if self.learned_converse:
                converse_triplets, conv_counts = get_edge_converse_triplets(rel_triplets, non_meta_relations - {rel}, self.converse_candidates_weights, conv_counts)
                rel_new_triplets.extend(converse_triplets)

            new_triplets.extend(rel_new_triplets)

        all_transitive_triplets = []
        if self.learned_transitivity:
            new_triplets = np.array(new_triplets)
            for rel in non_meta_relations:
                if not len(new_triplets):
                    continue
                rel_triplets = new_triplets[new_triplets[:, 1] == rel].copy()
                if not len(rel_triplets):
                    continue
                _, transitive_triplets = get_current_and_transitive_triplets(rel_triplets)
                all_transitive_triplets.extend(transitive_triplets)

        new_triplets = list(new_triplets)
        for rel in meta_relations:
            rel_triplets = triplets[triplets[:, 1] == rel].copy()
            new_triplets.extend(rel_triplets)

        # triplets = np.array(new_triplets)
        triplets_w_extras = np.array(new_triplets)
        triplets = np.unique(triplets_w_extras, axis=0)
        # if len(triplets_w_extras) > len(triplets):
        #     print("removed redundant edges")

        triplet_type = [ORIGINAL_EDGE]*len(triplets)
        if len(all_transitive_triplets) > 0:
            triplet_type = triplet_type + [TRANSITIVE_EDGE]*len(all_transitive_triplets)
            triplets = np.concatenate([triplets, all_transitive_triplets], axis=0)

        return triplets, conv_counts, triplet_type

    def add_dummy_triplets(self, objs, triplets):
        # Add dummy __in_image__ relationships for all objects
        if self.include_dummies:
            __image__ = int((objs == self.vocab['attributes'][list(self.vocab['attributes'].keys())[0]]["__image__"]).nonzero().squeeze())
            in_image = self.vocab['pred_name_to_idx']['__in_image__']
            O = objs.size(0)
            for i in range(O):
                if i == __image__:
                    continue
                triplets.append([i, in_image, __image__])

    def register_augmented_relations(self):
        if "pred_name_to_idx" not in self.vocab:
            self.vocab["pred_name_to_idx"] = {}
        if "pred_idx_to_name" not in self.vocab:
            self.vocab["pred_idx_to_name"] = []

        for p in self.meta_relations+self.augmented_relations:
            if p not in self.vocab["pred_name_to_idx"]:
                self.vocab["pred_name_to_idx"][p] = max(list(self.vocab["pred_name_to_idx"].values())+[-1]) + 1
                self.vocab["pred_idx_to_name"].append(p)


def collate_fn(vocab, batch):
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
    for i, (img, objs, boxes, triplets, triplet_type, source_edges, masks, image_ids) in enumerate(batch):
        O = objs[list(objs.keys())[0]].size(0)
        T = triplets.size(0)

        if max_objects < O:
            max_objects = O

        if max_triplets < T:
            max_triplets = T

    for i, (img, objs, boxes, triplets, triplet_type, source_edges, image_ids) in enumerate(batch):
        all_imgs.append(img[None])
        all_image_ids.append(image_ids)
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
            padded_triplets = torch.LongTensor([[0, vocab["pred_name_to_idx"]["__padding__"], 0]]).repeat(max_triplets - T, 1)
            triplets = torch.cat([triplets, padded_triplets])
            triplet_type = torch.cat([triplet_type, torch.LongTensor([0]*(max_triplets - T))])
            source_edges = torch.cat([source_edges, torch.LongTensor([vocab["pred_name_to_idx"]["__padding__"]]*(max_triplets - T))])

        all_objs.append(attributes_objects)
        all_boxes.append(boxes)
        all_triplets.append(triplets)
        all_triplet_type.append(triplet_type)
        all_source_edges.append(source_edges)

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.stack(all_objs, dim=0)
    all_boxes = torch.stack(all_boxes, dim=0)
    all_triplets = torch.stack(all_triplets, dim=0)
    all_triplet_type = torch.stack(all_triplet_type, dim=0)
    all_source_edges = torch.stack(all_source_edges, dim=0)
    out = (all_imgs, all_objs, all_boxes, all_triplets, all_triplet_type, all_source_edges, all_image_ids)
    return out

import PIL
import torch
import torchvision.transforms as T
from scripts.graphs_utils import calc_prob

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def encode_image():
    return T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def decode_image(rescale_image=False):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=[2.0, 2.0, 2.0]),
        T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def imagenet_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
        T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def deprocess_batch(imgs, rescale=True, deprocess_func=imagenet_deprocess):
    """
  Input:
  - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

  Output:
  - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
    in the range [0, 255]
  """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = deprocess_func(rescale_image=rescale)
    imgs_de = []
    for i in range(imgs.size(0)):
        img_de = deprocess_fn(imgs[i])[None]
        img_de = img_de.mul(255).clamp(0, 255).byte()
        imgs_de.append(img_de)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple) or isinstance(size, list):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


def unpack_var(v):
    if isinstance(v, torch.autograd.Variable):
        return v.data
    return v


def split_graph_batch(triplets, obj_data, obj_to_img, triple_to_img):
    triplets = unpack_var(triplets)
    obj_data = [unpack_var(o) for o in obj_data]
    obj_to_img = unpack_var(obj_to_img)
    triple_to_img = unpack_var(triple_to_img)

    triplets_out = []
    obj_data_out = [[] for _ in obj_data]
    obj_offset = 0
    N = obj_to_img.max() + 1
    for i in range(N):
        o_idxs = (obj_to_img == i).nonzero().view(-1)
        t_idxs = (triple_to_img == i).nonzero().view(-1)

        cur_triplets = triplets[t_idxs].clone()
        cur_triplets[:, 0] -= obj_offset
        cur_triplets[:, 2] -= obj_offset
        triplets_out.append(cur_triplets)

        for j, o_data in enumerate(obj_data):
            cur_o_data = None
            if o_data is not None:
                cur_o_data = o_data[o_idxs]
            obj_data_out[j].append(cur_o_data)

        obj_offset += o_idxs.size(0)

    return triplets_out, obj_data_out


def compute_transitive_edges(entry, transitive_weights, vocab):
    for i in range(transitive_weights.size(0)):
        rel = vocab["pred_idx_to_name"][i]
        if '__padding__' == rel or "__in_image__" == rel:
            continue
        val = float(transitive_weights[i].cpu().numpy())
        entry[f"trans_{rel}"] = round(val, 3)
    return entry


def compute_converse_edges(entry, converse_weights, vocab):
    converse_weights = converse_weights.cpu().numpy()
    for i in range(len(vocab["pred_idx_to_name"])):
        if vocab["pred_idx_to_name"][i] in ["__padding__", "__in_image__"]:
            continue
        rel_dict = {}
        for j in range(len(vocab["pred_idx_to_name"]) + 1):
            if j == len(vocab["pred_idx_to_name"]):
                rel_dict["No Edge"] = converse_weights[i, j]
                continue

            if vocab["pred_idx_to_name"][j] in ["__padding__", "__in_image__"] or i == j:
                continue

            rel_dict[vocab["pred_idx_to_name"][j]] = converse_weights[i, j]

        entry[vocab["pred_idx_to_name"][i]] = rel_dict
    return entry


def print_compute_converse_edges(entry, converse_weights, vocab, non_meta_relations):
    prob_mat = calc_prob(converse_weights, non_meta_relations)
    entry = compute_converse_edges(entry, prob_mat, vocab)
    for k, v in entry.items():
        print(f"{k}: {v}")


def print_compute_transitive_edges(entry, transitive_weights, vocab):
    entry = compute_transitive_edges(entry, transitive_weights, vocab)
    for k, v in entry.items():
        print(f"{k}: {v}")

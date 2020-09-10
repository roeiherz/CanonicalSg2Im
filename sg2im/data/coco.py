import json, os, random, math
from collections import defaultdict
import numpy as np
import torch
import torchvision.transforms as T
import PIL
import cv2
import pycocotools.mask as mask_utils
from scripts.graphs_utils import get_current_and_transitive_triplets, get_minimal_and_transitive_triplets
from sg2im.data import imagenet_preprocess, deprocess_batch
from sg2im.data.base_dataset import BaseDataset
from sg2im.data.utils import decode_image
from sg2im.vis import draw_item


class CocoSceneGraphDataset(BaseDataset):
    val_image_ids = [252219, 87038, 174482, 6818, 480985, 331352, 502136, 184791, 348881, 289393, 522713, 143931,
                     460347, 322864, 226111, 153299, 456496, 41888, 565778, 297343, 219578, 555705, 443303, 500663,
                     418281, 25560, 403817, 239274, 286994, 511321, 314294, 475779, 185250, 572517, 270244, 125211,
                     360661, 382088, 266409, 430961, 80671, 577539, 104612, 476258, 448365, 35197, 349860, 400573,
                     109798, 370677, 502737, 515445, 173383, 180560, 321214, 474028, 66523, 355257, 142092, 63154,
                     199551, 239347, 514508, 228144, 78915, 544519, 96493, 434230, 428454, 399462, 168330, 383384,
                     342006, 217285, 236412, 95786, 231339, 550426, 368294, 329219, 34873, 263796, 365387, 487583,
                     504711, 311295, 166391, 48153, 459153, 223130, 198960, 344059, 410428, 87875, 450758, 460160,
                     458109, 30675, 338428, 269314, 476415, 360137, 122046, 512836, 8021, 84477, 562243, 181859, 288042,
                     113403, 375015, 334719, 134112, 283520, 31269, 319721, 165351, 347265, 414170, 389381, 118921, 785,
                     300842, 105014, 261982, 34205, 314709, 339442, 409475, 464786, 378605, 578545, 372577, 212166,
                     294831, 84431, 100582, 4495, 9483, 398237, 507223, 31050, 340930, 11813, 281414, 284282, 321333,
                     521282, 108026, 177935, 397327, 555050, 376442, 356347, 293044, 560279, 190756, 199977, 442480,
                     384350, 383621, 189828, 537153, 361103, 338560, 264535, 295231, 154947, 212559, 458755, 104782,
                     315257, 130599, 227187, 151662, 523811, 456559, 101068, 544605, 385190, 53994, 314034, 291490,
                     24919, 79837, 337055, 110638, 34139, 83113, 173033, 72813, 545129, 546011, 121031, 172547, 369081,
                     464089, 177714, 459887, 155179, 396274, 29640, 308430, 43314, 273715, 406611, 466567, 15079,
                     296284, 366611, 263969, 551439, 159458, 554735, 99428, 311394, 312237, 221291, 199310, 323151,
                     219283, 471869, 111179, 57238, 502732, 8277, 173044, 168458, 512194, 533958, 202228, 403565,
                     441586, 530099, 312278, 97679, 564127, 251065, 3845, 138819, 205834, 348708, 99054, 22969, 570539,
                     278353, 158548, 176606, 44699, 268996, 11197, 448810, 724, 51961, 131131, 98839, 402992, 465675,
                     240754, 148730, 186873, 82180, 446522, 552902, 125405, 110211, 16010, 64462, 314182, 248980, 68387,
                     429281, 352900, 118367, 113235, 311303, 370999, 1490, 88269, 193494, 252776, 201072, 337498,
                     521405, 201646, 234607, 323895, 384670, 50326, 205542, 217957, 162035, 46252, 182021, 90284,
                     488736, 383386, 450686, 5060, 286523, 120420, 579655, 322844, 223747, 458992, 164602, 101762,
                     557501, 203317, 368940, 144798, 284623, 520301, 127987, 488270, 67180, 150265, 216739, 354829,
                     525155, 163314, 259571, 561679, 236166, 253835, 34071, 36861, 569565, 205647, 123131, 334006,
                     229858, 174004, 9769, 205776, 163257, 85478, 318080, 361551, 236784, 92839, 42296, 560266, 486479,
                     127955, 307658, 417465, 342971, 11760, 70158, 176634, 281447, 361919, 138115, 114871, 374369,
                     123213, 123321, 15278, 357742, 465836, 414385, 131556, 322724, 320664, 109916, 276434, 295316,
                     115898, 329542, 223959, 560011, 38576, 579307, 154425, 432898, 404923, 130586, 163057, 7511, 67406,
                     290179, 248752, 494427, 311180, 91654, 111951, 103585, 352584, 327601, 255749, 8762, 535578,
                     580757, 165039, 148719, 108440, 579818, 423229, 323828, 166287, 101420, 196759, 411665, 526751,
                     24021, 47828, 183716, 271997, 8532, 94336, 11051, 360960, 360097, 421455, 504589, 464522, 454750,
                     509735, 23034, 141671, 45728, 424551, 341719, 72795, 417285, 43816, 455555, 535306, 30504, 473118,
                     283113, 226130, 97278, 532493, 430056, 441286, 885, 378284, 229849, 56344, 193348, 83172, 205105,
                     176446, 308531, 455352, 232684, 415238, 290843, 519522, 144784, 392228, 80057, 570169, 163562,
                     102707, 51598, 520910, 131273, 206411, 472375, 481404, 471991, 17436, 165518, 459467, 134886,
                     485895, 577182, 289222, 372819, 310072, 430875, 60347, 42070, 453584, 296224, 122606, 311909,
                     579893, 284296, 221017, 315001, 439715, 284991, 78843, 122927, 225532, 153568, 395633, 419096,
                     361268, 508101, 253386, 222991, 138492, 263463, 378454, 20059, 227686, 476215, 297698, 26690,
                     176901, 334767, 301563, 86755, 194471, 420281, 99810, 89670, 482275, 425702, 47740, 77460, 14439,
                     447314, 181303, 58350, 26465, 76731, 433980, 561366, 517687, 213035, 349837, 350002, 131431,
                     507235, 481573, 123480, 274687, 164637, 178028, 493286, 348216, 345027, 102644, 581615, 230008,
                     284698, 102356, 323709, 525322, 33114, 381639, 217614, 468124, 273198, 95843, 417779, 447342,
                     166563, 490125, 561009, 183675, 290248, 532058, 578093, 429011, 301061, 105264, 25393, 471087,
                     106757, 49269, 79144, 519688, 431727, 215245, 91921, 218424, 473974, 235784, 521540, 119445,
                     507015, 173830, 356498, 18575, 373315, 227765, 13546, 67310, 414034, 450488, 99182, 232489, 351823,
                     65736, 379842, 185890, 261097, 410510, 214224, 255483, 222455, 187271, 544565, 369771, 289516,
                     452084, 413552, 17379, 176778, 104572, 90108, 86220, 508602, 17178, 314177, 313182, 149406, 180383,
                     402433, 449996, 168619, 103548, 15338, 512564, 336658, 260266, 106048, 479099, 269196, 315450,
                     171050, 243867, 263594, 272364, 138979, 519491, 100283, 563653, 113051, 286708, 475732, 108244,
                     121153, 23230, 86483, 521141, 61268, 493566, 191471, 198510, 126592, 416269, 521052, 332318,
                     415990, 200667, 77595, 190140, 476810, 540280, 126216, 407960, 422836, 493613, 217948, 317024,
                     463522, 547886, 124975, 69356, 162415, 377113, 79651, 430377, 512776, 95155, 184978, 199055,
                     128699, 121591, 424521, 254016, 73946, 230819, 82715, 85195, 435299, 50828, 27696, 541123, 409630,
                     343706, 199395, 514586, 279774, 474078, 872, 32038, 261732, 12120, 534601, 288391, 531771, 113867,
                     349678, 384136, 549136, 57672, 138639, 110884, 523175, 59920, 343934, 221754, 522751, 503841,
                     127092, 482477, 369675, 151857, 505638, 539143, 316054, 231169, 488664, 444879, 297022, 407083,
                     212226, 220858, 244411, 453166, 6894, 133631, 279927, 161032, 318908, 460927, 139883, 348243,
                     317433, 132408, 191288, 260106, 100510, 441442, 140270, 553990, 326542, 250766, 23781, 327306,
                     567825, 485972, 94944, 151051, 93717, 394510, 18193, 92177, 1425, 234779, 485130, 403584, 40757,
                     35062, 503755, 320642, 494759, 409542, 427055, 119995, 369323, 179653, 400922, 226408, 152771,
                     309391, 480944, 568690, 382122, 234807, 60835, 213935, 502599, 224119, 417043, 393282, 78032,
                     472030, 537964, 542423, 344909, 140556, 189698, 407518, 310622, 206994, 523241, 258793, 469067,
                     559099, 311789, 201025, 549390, 401991, 545730, 364102, 13291, 440336, 148783, 325306, 488251,
                     437205, 62554, 138954, 289417, 776, 470121, 309467, 327605, 451084, 22479, 243148, 249786, 581062,
                     185950, 44195, 499109, 478136, 451150, 148957, 251119, 364126, 354307, 483531, 170191, 445999,
                     426836, 557916, 99024, 305309, 311928, 384949, 196141, 522889, 37988, 418961, 154644, 17714,
                     474039, 153782, 21465, 84664, 461751, 466986, 170893, 465179, 567197, 498286, 437392, 153217,
                     394611, 211069, 455085, 282046, 542856, 12062, 427500, 213605, 270705, 544811, 436551, 389532,
                     39670, 429718, 563604, 162366, 124636, 551815, 78565, 492992, 107851, 105923, 325031, 166642,
                     225670, 575357, 186422, 252716, 205324, 447313, 172977, 560880, 449406, 491613, 61584, 244099,
                     460682, 481159, 28809, 50679, 10764, 537827, 329827, 161820, 116362, 500270, 308165, 125245,
                     544444, 378244, 258388, 358195, 3553, 253819, 187990, 51938, 396729, 347174, 439525, 302990,
                     262587, 117719, 127476, 499313, 511760, 361730, 427077, 203931, 69795, 577735, 325527, 346968,
                     229216, 199681, 294163, 176232, 407403, 562843, 58705, 248334, 53505, 78823, 527220, 178982,
                     332455, 152120, 517523, 5477, 391375, 442661, 69138, 548267, 315187, 76417, 138550, 244019, 439623,
                     343315, 46804, 340451, 480842, 546556, 98497, 338718, 228771, 50165, 158956, 311883, 240767,
                     431896, 442456, 79229, 188439, 46872, 532575, 489014, 257896, 28449, 393115, 370813, 442746,
                     236592, 116589, 369541, 122969, 381971, 236730, 576052, 51712, 480275, 476770, 220764, 493799,
                     312720, 568981, 546626, 532855, 38210, 563349, 10583, 74457, 177357, 185292, 493864, 102820,
                     381360, 304812, 333237, 55950, 175251, 368212, 190637, 239318, 1503, 284743, 65798, 312192, 154705,
                     192904, 384661, 341828, 45596, 232649, 172330, 123585]

    def __init__(self, image_dir, instances_json, stuff_json=None,
                 stuff_only=True, image_size=(64, 64), mask_size=32, normalize_images=True, max_samples=None,
                 include_relationships=True, min_object_size=0.02, min_objects=3, max_objects=8,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None, learned_transitivity=False,
                 include_dummies=True, use_transitivity=False, use_converse=False, learned_symmetry=False,
                 learned_converse=False):

        """
    A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
    them to scene graphs on the fly.

    Inputs:
    - image_dir: Path to a directory where images are held
    - instances_json: Path to a JSON file giving COCO annotations
    - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
    - stuff_only: (optional, default True) If True then only iterate over
      images which appear in stuff_json; if False then iterate over all images
      in instances_json.
    - image_size: Size (H, W) at which to load images. Default (64, 64).
    - mask_size: Size M for object segmentation masks; default 16.
    - normalize_image: If True then normalize images by subtracting ImageNet
      mean pixel and dividing by ImageNet std pixel.
    - max_samples: If None use all images. Other wise only use images in the
      range [0, max_samples). Default None.
    - include_relationships: If True then include spatial relationships; if
      False then only include the trivial __in_image__ relationship.
    - min_object_size: Ignore objects whose bounding box takes up less than
      this fraction of the image.
    - min_objects_per_image: Ignore images which have fewer than this many
      object annotations.
    - max_objects_per_image: Ignore images which have more than this many
      object annotations.
    - include_other: If True, include COCO-Stuff annotations which have category
      "other". Default is False, because I found that these were really noisy
      and pretty much impossible for the system to model.
    - instance_whitelist: None means use all instance categories. Otherwise a
      list giving a whitelist of instance category names to use.
    - stuff_whitelist: None means use all stuff categories. Otherwise a list
      giving a whitelist of stuff category names to use.
    """
        super(CocoSceneGraphDataset, self).__init__()
        self.use_converse = use_converse
        self.learned_transitivity = learned_transitivity
        self.learned_symmetry = learned_symmetry
        self.learned_converse = learned_converse
        self.include_dummies = include_dummies
        self.image_dir = image_dir
        # self.mask_size = image_size[0]
        self.mask_size = mask_size
        self.masks = True
        if self.mask_size == 0:
            self.masks = False
            self.mask_size = 32

        self.max_samples = max_samples
        self.normalize_images = normalize_images
        self.include_relationships = include_relationships
        self.set_image_size(image_size)
        self.use_transitivity = use_transitivity

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        with open(stuff_json, 'r') as f:
            stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {}
        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        all_stuff_categories = []

        for category_data in stuff_data['categories']:
            category_name = category_data['name']
            category_id = category_data['id']
            all_stuff_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id

        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data)

        # Add object data from stuff
        image_ids_with_stuff = set()
        for object_data in stuff_data['annotations']:
            image_id = object_data['image_id']
            image_ids_with_stuff.add(image_id)
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data)

        new_image_ids = []
        for image_id in self.image_ids:
            if image_id in image_ids_with_stuff:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        all_image_ids = set(self.image_id_to_filename.keys())
        image_ids_to_remove = all_image_ids - image_ids_with_stuff
        for image_id in image_ids_to_remove:
            self.image_id_to_filename.pop(image_id, None)
            self.image_id_to_size.pop(image_id, None)
            self.image_id_to_objects.pop(image_id, None)

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects <= num_objs <= max_objects:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        self.register_augmented_relations()

        self.vocab["attributes"] = {}
        self.vocab["attributes"]['objects'] = self.vocab['object_name_to_idx']
        self.vocab["reverse_attributes"] = {}
        for attr in self.vocab["attributes"].keys():
            self.vocab["reverse_attributes"][attr] = {v: k for k, v in self.vocab["attributes"][attr].items()}

    def set_image_size(self, image_size):
        # print('called set_image_size', image_size)
        transform = [T.Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def __getitem__(self, index):
        """
    Get the pixels of an image, and a random synthetic scene graph for that
    image constructed on-the-fly from its COCO object annotations. We assume
    that the image will have height H, width W, C channels; there will be O
    object annotations, each of which will have both a bounding box and a
    segmentation mask of shape (M, M). There will be T triplets in the scene
    graph.

    Returns a tuple of:
    - image: FloatTensor of shape (C, H, W)
    - objs: LongTensor of shape (O,)
    - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
      (x0, y0, x1, y1) format, in a [0, 1] coordinate system
    - masks: LongTensor of shape (O, M, M) giving segmentation masks for
      objects, where 0 is background and 1 is object.
    - triplets: LongTensor of shape (T, 3) where triplets[t] = [i, p, j]
      means that (objs[i], p, objs[j]) is a triple.
    """
        image_id = self.image_ids[index]

        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size
        objs, boxes, masks = [], [], []
        for object_data in self.image_id_to_objects[image_id]:
            objs.append(object_data['category_id'])
            x, y, w, h = object_data['bbox']
            x0 = x / WW
            y0 = y / HH
            boxes.append(torch.FloatTensor([x0, y0, w / WW, h / HH]))

            # This will give a numpy array of shape (HH, WW)
            mask = seg_to_mask(object_data['segmentation'], WW, HH)

            # Crop the mask according to the bounding box, being careful to
            # ensure that we don't crop a zero-area region
            mx0, mx1 = int(round(x)), int(round(x + w))
            my0, my1 = int(round(y)), int(round(y + h))
            mx1 = max(mx0 + 1, mx1)
            my1 = max(my0 + 1, my1)
            mask = mask[my0:my1, mx0:mx1]
            mask = cv2.resize(255.0 * mask, (self.mask_size, self.mask_size), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy((mask > 128).astype(np.int64))
            masks.append(mask)

        if self.include_dummies:
            # Add dummy __image__ object
            objs.append(self.vocab['object_name_to_idx']['__image__'])
            boxes.append(torch.FloatTensor([-1, -1, -1, -1]))
            masks.append(torch.ones(self.mask_size, self.mask_size).long())

        objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)
        masks = torch.stack(masks, dim=0)

        # Compute centers of all objects
        obj_centers = []
        _, MH, MW = masks.size()
        for i, obj_idx in enumerate(objs):
            x0, y0, w, h = boxes[i]
            mask = (masks[i] == 1)
            xs = torch.linspace(x0, x0 + w, MW).view(1, MW).expand(MH, MW)
            ys = torch.linspace(y0, y0 + h, MH).view(MH, 1).expand(MH, MW)
            if mask.sum() == 0:
                mean_x = x0 + 0.5 * w
                mean_y = y0 + 0.5 * h
            else:
                mean_x = xs[mask].mean()
                mean_y = ys[mask].mean()
            obj_centers.append([mean_x, mean_y])
        obj_centers = torch.FloatTensor(obj_centers)

        # Add triplets
        triplets = []
        num_objs = objs.size(0)
        __image__ = self.vocab['object_name_to_idx']['__image__']
        real_objs = []
        if num_objs > 1:
            real_objs = (objs != __image__).nonzero().squeeze(1)
        for cur in real_objs:
            choices = [obj for obj in real_objs if obj != cur]
            if len(choices) == 0 or not self.include_relationships:
                break
            other = random.choice(choices)
            if random.random() > 0.5:
                s, o = cur, other
            else:
                s, o = other, cur
                # Check for inside / surrounding
            sx0, sy0, sw, sh = boxes[s]
            sx1, sy1 = sx0 + sw / 2, sy0 + sh / 2
            ox0, oy0, ow, oh = boxes[o]
            ox1, oy1 = ox0 + ow / 2, oy0 + oh / 2
            d = obj_centers[s] - obj_centers[o]
            theta = math.atan2(d[1], d[0])

            if not self.use_converse:
                if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                    p = '__surrounding__'
                elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                    p = '__inside__'
                elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                    p = '__left of__'
                elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                    p = '__above__'
                elif -math.pi / 4 <= theta < math.pi / 4:
                    p = '__right of__'
                elif math.pi / 4 <= theta < 3 * math.pi / 4:
                    p = '__below__'
                p = self.vocab['pred_name_to_idx'][p]
                triplets.append([s, p, o])
            else:
                if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                    p = '__surrounding__'
                elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                    p = '__surrounding__'
                    s, o = o, s
                elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                    p = '__left of__'
                elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                    p = '__above__'
                elif -math.pi / 4 <= theta < math.pi / 4:
                    p = '__left of__'
                    s, o = o, s
                elif math.pi / 4 <= theta < 3 * math.pi / 4:
                    p = '__above__'
                    s, o = o, s
                p = self.vocab['pred_name_to_idx'][p]
                triplets.append([s, p, o])

        self.add_dummy_triplets(objs, triplets)

        if not self.masks:
            masks = None

        triplets, conv_counts, triplet_type = self.add_learnt_triplets(triplets, objs.size(0))
        return image, {"objects": objs}, boxes, torch.LongTensor(triplets), torch.LongTensor(conv_counts), \
               torch.LongTensor(triplet_type), masks, image_id

    def __len__(self):
        if self.max_samples is None:
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)


def seg_to_mask(seg, width=1.0, height=1.0):
    """
  Tiny utility for decoding segmentation masks using the pycocotools API.
  """
    if type(seg) == list:
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif type(seg['counts']) == list:
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle)


def coco_collate_fn(vocab, batch):
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
    all_masks = []
    all_image_ids = []

    max_triplets = 0
    max_objects = 0

    for i, (img, objs, boxes, triplets, conv_counts, triplet_type, masks, image_id) in enumerate(batch):
        O = objs[list(objs.keys())[0]].size(0)
        T = triplets.size(0)

        if max_objects < O:
            max_objects = O

        if max_triplets < T:
            max_triplets = T

    for i, (img, objs, boxes, triplets, conv_counts, triplet_type, masks, image_id) in enumerate(batch):
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

        # Padded masks
        if masks is not None and max_objects - O > 0:
            padded_masks = torch.zeros([max_objects - O, masks.size(1), masks.size(2)]).type(torch.LongTensor)
            masks = torch.cat([masks, padded_masks])

        # Padded triplets
        if max_triplets - T > 0:
            padded_triplets = torch.LongTensor([[0, vocab["pred_name_to_idx"]["__padding__"], 0]]).repeat(
                max_triplets - T, 1)
            triplets = torch.cat([triplets, padded_triplets])
            triplet_type = torch.cat([triplet_type, torch.LongTensor([0] * (max_triplets - T))])

        all_objs.append(attributes_objects)
        all_boxes.append(boxes)
        all_triplets.append(triplets)
        if masks is not None:
            all_masks.append(masks)
        else:
            all_masks = None
        all_triplet_type.append(triplet_type)
        all_conv_counts.append(conv_counts)

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.stack(all_objs, dim=0)
    all_boxes = torch.stack(all_boxes, dim=0)
    if all_masks is not None:
        all_masks = torch.stack(all_masks, dim=0)
    all_triplets = torch.stack(all_triplets, dim=0)
    all_image_ids = torch.LongTensor(all_image_ids)
    all_conv_counts = torch.stack(all_conv_counts, dim=0).to(torch.float32)
    all_triplet_type = torch.stack(all_triplet_type, dim=0)

    out = (all_imgs, all_objs, all_boxes, all_triplets, all_conv_counts, all_triplet_type, all_masks, all_image_ids)
    return out


def coco_collate_fn_inferece(vocab, batch):
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
    all_masks = []
    all_image_ids = []

    max_triplets = 0
    max_objects = 0
    for i, (img, objs, boxes, triplets, triplet_type, source_edges, masks, image_id) in enumerate(batch):
        O = objs[list(objs.keys())[0]].size(0)
        T = triplets.size(0)

        if max_objects < O:
            max_objects = O

        if max_triplets < T:
            max_triplets = T

    for i, (img, objs, boxes, triplets, triplet_type, source_edges, masks, image_id) in enumerate(batch):
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

        # Padded masks
        if masks is not None and max_objects - O > 0:
            padded_masks = torch.zeros([max_objects - O, masks.size(1), masks.size(2)]).type(torch.LongTensor)
            masks = torch.cat([masks, padded_masks])

        # Padded triplets
        if max_triplets - T > 0:
            padded_triplets = torch.LongTensor([[0, vocab["pred_name_to_idx"]["__padding__"], 0]]).repeat(
                max_triplets - T, 1)
            triplets = torch.cat([triplets, padded_triplets])
            triplet_type = torch.cat([triplet_type, torch.LongTensor([0]*(max_triplets - T))])
            source_edges = torch.cat([source_edges, torch.LongTensor([vocab["pred_name_to_idx"]["__padding__"]]*(max_triplets - T))])

        all_objs.append(attributes_objects)
        all_boxes.append(boxes)
        all_triplets.append(triplets)
        if masks is not None:
            all_masks.append(masks)
        else:
            all_masks = None
        all_triplet_type.append(triplet_type)
        all_source_edges.append(source_edges)

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.stack(all_objs, dim=0)
    all_boxes = torch.stack(all_boxes, dim=0)
    if all_masks is not None:
        all_masks = torch.stack(all_masks, dim=0)
    all_triplets = torch.stack(all_triplets, dim=0)
    # all_image_ids = torch.LongTensor(all_image_ids)
    all_triplet_type = torch.stack(all_triplet_type, dim=0)
    all_source_edges = torch.stack(all_source_edges, dim=0)
    all_image_ids = torch.LongTensor(all_image_ids)

    out = (all_imgs, all_objs, all_boxes, all_triplets, all_triplet_type, all_source_edges, all_masks, all_image_ids)
    return out


if __name__ == "__main__":
    dset = CocoSceneGraphDataset(
        image_dir="/home/roeiherz/Datasets/MSCoco/images/val2017",
        instances_json="/home/roeiherz/Datasets/MSCoco/annotations/instances_val2017.json",
        stuff_json="/home/roeiherz/Datasets/MSCoco/annotations/stuff_val2017.json",
        stuff_only=True, image_size=(256, 256), normalize_images=True, max_samples=None,
        include_relationships=True, min_object_size=0.02, min_objects=3,
        max_objects=8, include_other=False, instance_whitelist=None, stuff_whitelist=None)
    idx = 100
    item = dset[idx]
    image, objs, boxes, triplets = item
    image = deprocess_batch(torch.unsqueeze(image, 0), deprocess_func=decode_image)[0]
    cv2.imwrite('img.png', np.transpose(image.cpu().numpy(), [1, 2, 0]))
    objs_text = [dset.vocab['object_idx_to_name'][k] for k in objs['objects'].cpu().numpy()]
    draw_item(item, image_size=dset.image_size, text=objs_text)
    print(item)

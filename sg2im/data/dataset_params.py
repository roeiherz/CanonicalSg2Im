from sg2im.data.base_dataset import collate_fn
from sg2im.data.coco import CocoSceneGraphDataset
from sg2im.data.packed_clevr_dialog import packed_clevr_collate_fn, packed_sync_clevr_collate_fn, \
    PackedGenCLEVRDataset
from sg2im.data.clevr_dialog import CLEVRDialogDataset
from sg2im.data.packed_clevr_dialog import PackedCLEVRDialogDataset
from sg2im.data.packed_coco import PackedCocoSceneGraphDataset, coco_collate_fn
from sg2im.data.packed_vg import PackedVGSceneGraphDataset, vg_collate_fn
from sg2im.data.vg import VGSceneGraphDataset


def get_dataset(name, partition, args):
    config = {
        "image_size": args.image_size,
        "mask_size": args.mask_size,
        "use_transitivity": args.use_transitivity,
        "use_converse": args.use_converse,
        "learned_transitivity": args.learned_transitivity,
        "learned_symmetry": args.learned_symmetry,
        "learned_converse": args.learned_converse,
        "include_dummies": args.include_dummies,
    }

    if name == 'clevr':
        dataset_config = {
            "common": {
                "base_path": '{}/CLEVR/CLEVR_Dialog'.format(args.dataroot),
                "max_objects": args.max_objects,
            },
            "train": {
                "h5_path": 'clevr_dialog_train_raw.json',
                "mode": "train",
            },
            "val": {
                "h5_path": 'clevr_dialog_val_raw.json',
                "mode": "val",
            },
            "class": CLEVRDialogDataset
        }
    elif name == 'packed_vg':
        dataset_config = {
            "common": {
                "max_objects": 100,
                "min_objects": 16,
                "base_path": '{}/vg',
            },
            "train": {
                "h5_path": 'train.h5',
                "mode": "train"
            },
            "val": {
                "h5_path": 'val.h5',
                "mode": "val"
            },
            "test": {
                "h5_path": 'test.h5',
                "mode": "val"
            },
            "class": PackedVGSceneGraphDataset
        }
    elif name == 'packed_coco':
        dataset_config = {
            "common": {
                "stuff_only": True,
                "normalize_images": True,
                "max_samples": None,
                "include_relationships": True,
                "min_object_size": 0.02,
                "min_objects": 16,
                "max_objects": 1000,  # default in coco is 3, 8
                "include_other": False,
                "instance_whitelist": None,
                "stuff_whitelist": None,
            },
            "train": {
                "image_dir": "{}/MSCoco/images/train2017".format(args.dataroot),
                "instances_json": "{}/MSCoco/annotations/instances_train2017.json".format(args.dataroot),
                "stuff_json": "{}/MSCoco/annotations/stuff_train2017.json".format(args.dataroot),
            },
            "val": {
                "image_dir": "{}/MSCoco/images/val2017".format(args.dataroot),
                "instances_json": "{}/MSCoco/annotations/instances_val2017.json".format(args.dataroot),
                "stuff_json": "{}/MSCoco/annotations/stuff_val2017.json".format(args.dataroot),
            },
            "test": {
                "image_dir": "{}/MSCoco/images/val2017".format(args.dataroot),
                "instances_json": "{}/MSCoco/annotations/instances_val2017.json".format(args.dataroot),
                "stuff_json": "{}/MSCoco/annotations/stuff_val2017.json".format(args.dataroot),
            },
            "class": PackedCocoSceneGraphDataset
        }
    elif name == 'packed_clevr':
        dataset_config = {
            "common": {
                "base_path": '{}/CLEVR/CLEVR_Dialog'.format(args.dataroot),
                "max_objects": 1000,
                "min_objects": 0,
                "debug": args.debug,
            },
            "train": {
                "h5_path": 'clevr_dialog_train_raw.json',
                "mode": "train",
            },
            "val": {
                "h5_path": 'clevr_dialog_val_raw.json',
                "mode": "val",
            },
            "class": PackedCLEVRDialogDataset
        }
    elif name == 'vg':
        dataset_config = {
            "common": {
                "base_path": '{}/VisualGenome'.format(args.dataroot),
                "max_objects": args.max_objects if args.max_objects else 10,
                "min_objects": args.min_objects if args.min_objects else 3
            },
            "train": {
                "h5_path": 'train.h5',
            },
            "val": {
                "h5_path": 'val.h5',
            },
            "test": {
                "h5_path": 'test.h5',
            },
            "class": VGSceneGraphDataset
        }
    elif name == 'coco':
        dataset_config = {
            "common": {
                "min_object_size": 0.02,
                "max_objects": args.max_objects if args.max_objects else 8,
                "min_objects": args.min_objects if args.min_objects else 3,
                "include_relationships": True,
                "stuff_only": True,
                "normalize_images": True,
                "max_samples": None,
                "include_other": False,
                "instance_whitelist": None,
                "stuff_whitelist": None,
            },
            "train": {
                "image_dir": "{}/MSCoco/images/train2017".format(args.dataroot),
                "instances_json": "{}/MSCoco/annotations/instances_train2017.json".format(args.dataroot),
                "stuff_json": "{}/MSCoco/annotations/stuff_train2017.json".format(args.dataroot),

            },
            "val": {
                "image_dir": "{}/MSCoco/images/val2017".format(args.dataroot),
                "instances_json": "{}/MSCoco/annotations/instances_val2017.json".format(args.dataroot),
                "stuff_json": "{}/MSCoco/annotations/stuff_val2017.json".format(args.dataroot),
            },
            "test": {
                "image_dir": "{}/MSCoco/images/val2017".format(args.dataroot),
                "instances_json": "{}/MSCoco/annotations/instances_val2017.json".format(args.dataroot),
                "stuff_json": "{}/MSCoco/annotations/stuff_val2017.json".format(args.dataroot),
            },
            "class": CocoSceneGraphDataset
        }
    elif name == 'packed_clevr_syn':
        dataset_config = {
            "common": {
                "base_path": '{}/CLEVR/CLEVR_Dialog'.format(args.dataroot),
                "max_objects": args.max_objects,
                "min_objects": args.min_objects,
                "max_samples": args.max_samples,
                "debug": args.debug,
            },
            "train": {
                "mode": "train",
            },
            "val": {
                "mode": "val",
            },
            "class": PackedGenCLEVRDataset
        }
    else:
        raise ValueError("Wrong config.")

    config.update(dataset_config["common"])
    if args.debug:
        print("######### RUNNING IN DEBUG MODE, LOADING VALIDATION TO SAVE TIME #########")
        partition = 'val'
    config.update(dataset_config[partition])
    ds_instance = dataset_config['class'](**config)

    if name == 'coco':
        if partition == 'val':
            ds_instance.image_ids = list(set(ds_instance.val_image_ids) & set(ds_instance.image_ids))
        elif partition == 'test':
            ds_instance.image_ids = list(set(ds_instance.image_ids) - set(ds_instance.val_image_ids))

    # print number of objects
    print("######### Dataset - Max Objects: {} #########".format(dataset_config["common"]['max_objects']))
    print("######### Dataset - Min Objects: {} #########".format(dataset_config["common"]['min_objects']))
    return ds_instance


def get_collate_fn(args):
    if args.dataset == "coco" or args.dataset == "packed_coco" or args.dataset == 'canonical_packed_coco':
        return coco_collate_fn
    elif args.dataset == "vg" or args.dataset == "packed_vg":
        return vg_collate_fn
    elif args.dataset == "packed_clevr":
        return packed_clevr_collate_fn
    elif args.dataset == "packed_clevr_inf":
        return packed_sync_clevr_collate_fn
    else:
        return collate_fn

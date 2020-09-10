import logging
from functools import partial
import cv2
import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from evaluation.inception import InceptionScore
from sg2im.data.dataset_params import get_dataset, get_collate_fn
from scripts.args import get_args, print_args, init_args
from scripts.graphs_utils import calc_log_p
from sg2im.data.utils import decode_image, imagenet_deprocess, print_compute_converse_edges, \
    print_compute_transitive_edges
from sg2im.meta_models import MetaGeneratorModel, MetaDiscriminatorModel
from sg2im.model import get_conv_converse
from sg2im.pix2pix_model import Pix2PixModel
from sg2im.data import deprocess_batch
from sg2im.metrics import jaccard
from sg2im.utils import batch_to, log_scalar_dict, remove_dummies_and_padding
from spade.models.networks.sync_batchnorm import DataParallelWithCallback

torch.backends.cudnn.benchmark = True


def restore_checkpoint(args, model, gans_model, discriminator, optimizer, device):
    try:
        if args.checkpoint_name is None:
            raise Exception('You should pre-train the model on your training data first')

        img_discriminator, obj_discriminator = discriminator.img_discriminator, discriminator.obj_discriminator,
        optimizer_d_img, optimizer_d_obj = discriminator.optimizer_d_img, discriminator.optimizer_d_obj

        # Load pre-trained weights for fine-tune
        checkpoint = torch.load(args.checkpoint_name, map_location=device)

        model.load_state_dict(checkpoint['model_state'])
        gans_model.load_state_dict(checkpoint['gans_model_state'])
        img_discriminator.load_state_dict(checkpoint['d_img_state'])
        obj_discriminator.load_state_dict(checkpoint['d_obj_state'])

        # Load Optimizers
        try:
            optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])
            optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])
            optimizer.load_state_dict(checkpoint['optim_state'])
        except Exception as e:
            print("Could not load optimizers state:", e)

        # Load Epoch and Iteration num.
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']

    except Exception as e:
        raise NotImplementedError(
            'Could not restore weights for checkpoint {} because `{}`'.format(args.checkpoint_name, e))
    return epoch, t


def restore_checkpoints(args, model, gans_model, discriminator, optimizer, device):
    try:
        if args.checkpoint_name is None:
            raise Exception('You should pre-train the model on your training data first')

        img_discriminator, obj_discriminator = discriminator.img_discriminator, discriminator.obj_discriminator,
        optimizer_d_img, optimizer_d_obj = discriminator.optimizer_d_img, discriminator.optimizer_d_obj

        # Load pre-trained weights for fine-tune
        checkpoint_gan = torch.load(args.checkpoint_gan_name, map_location=device)
        checkpoint_graph = torch.load(args.checkpoint_graph_name, map_location=device)
        checkpoint_gan['model_state'].update(checkpoint_graph['model_state'])
        model.load_state_dict(checkpoint_gan['model_state'], strict=False)
        checkpoint_gan['gans_model_state'].pop(
            'module.discriminator.mask_discriminator.discriminator_0.model0.0.weight')
        checkpoint_gan['gans_model_state'].pop(
            'module.discriminator.mask_discriminator.discriminator_1.model0.0.weight')
        checkpoint_gan['gans_model_state'].pop('module.netD_mask.discriminator_0.model0.0.weight')
        checkpoint_gan['gans_model_state'].pop('module.netD_mask.discriminator_1.model0.0.weight')
        gans_model.load_state_dict(checkpoint_gan['gans_model_state'], strict=False)
        img_discriminator.load_state_dict(checkpoint_gan['d_img_state'])
        obj_discriminator.load_state_dict(checkpoint_gan['d_obj_state'])
    except Exception as e:
        raise NotImplementedError(
            'Could not restore weights for checkpoint {} because `{}`'.format(args.checkpoint_name, e))

    # Load Optimizers
    try:
        optimizer_d_img.load_state_dict(checkpoint_gan['d_img_optim_state'])
        optimizer_d_obj.load_state_dict(checkpoint_gan['d_obj_optim_state'])
        optimizer.load_state_dict(checkpoint_gan['optim_state'])
    except Exception as e:
        print("Could not load optimizers state:", e)

    # Load Epoch and Iteration num.
    t = 0
    epoch = 0

    return epoch, t


def freeze_weights(model, discriminator, module):
    print(" >> Freeze Weights:")
    if module == 'generation':
        print(" >> Freeze Layout to image module")
        if hasattr(model, 'layout_to_image_model'):
            for param in model.layout_to_image_model.parameters():
                param.requires_grad = False

        for param in discriminator.parameters():
            param.requires_grad = False

    else:
        raise NotImplementedError('Unrecognized option, you can freeze either graph module or I3D module')
    pass


def add_loss(curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()


def build_test_dsets(args):
    test_dset = get_dataset(args.dataset, 'test', args)
    vocab = test_dset.vocab
    collate_fn = get_collate_fn(args)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': False,
        'collate_fn': partial(collate_fn, vocab),
    }

    test_loader = DataLoader(test_dset, **loader_kwargs)
    return test_loader, test_dset.vocab


def build_train_val_loaders(args):
    train_dset = get_dataset(args.dataset, 'train', args)
    val_dset = get_dataset(args.dataset, 'val', args)
    assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(train_dset.vocab))
    collate = get_collate_fn(args)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'collate_fn': partial(collate, vocab),
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args.shuffle_val
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, val_loader


def check_model(args, loader, model, gans_model, inception_score, use_gt=True, full_test=False):
    model.eval()
    num_samples = 0
    all_losses = defaultdict(list)
    total_iou = 0.
    total_iou_masks = 0.
    total_iou_05 = 0.
    total_iou_03 = 0.
    total_boxes = 0.
    inception_score.clean()
    image_df = {
        'image_id': [],
        'avg_iou': [],
        'iou03': [],
        'iou05': [],
        "predicted_boxes": [],
        "gt_boxes": [],
        "number_of_objects": [],
        "class": []
    }
    with torch.no_grad():
        for batch in loader:
            try:
                batch = batch_to(batch)
                imgs, objs, boxes, triplets, _, triplet_type, masks, image_ids = batch

                # Run the model as it has been run during training
                if use_gt:
                    model_out = model(objs, triplets, triplet_type, boxes_gt=boxes, masks_gt=masks, test_mode=True)
                else:
                    model_out = model(objs, triplets, triplet_type, test_mode=True)
                imgs_pred, boxes_pred, masks_pred = model_out
                G_losses = gans_model(batch, model_out, mode='compute_generator_loss')

                if boxes_pred is not None:
                    boxes_pred = torch.clamp(boxes_pred, 0., 1.)
                if imgs_pred is not None:
                    inception_score(imgs_pred)

                if not args.skip_graph_model:
                    image_df['image_id'].extend(image_ids)

                    for i in range(boxes.size(0)):
                        # masks_sample = masks[i]
                        # masks_pred_sample = masks_pred[i]
                        boxes_sample = boxes[i]
                        boxes_pred_sample = boxes_pred[i]
                        boxes_pred_sample, boxes_sample = \
                            remove_dummies_and_padding(boxes_sample, objs[i], args.vocab,
                                                       [boxes_pred_sample, boxes_sample])
                        iou, iou05, iou03 = jaccard(boxes_pred_sample, boxes_sample)
                        # iou_masks = jaccard_masks(masks_pred_sample, masks_sample)
                        total_iou += iou.sum()
                        # total_iou_masks += iou_masks.sum()
                        total_iou_05 += iou05.sum()
                        total_iou_03 += iou03.sum()
                        total_boxes += float(iou.shape[0])

                        image_df['avg_iou'].append(np.mean(iou))
                        image_df['iou03'].append(np.mean(iou03))
                        image_df['iou05'].append(np.mean(iou03))
                        image_df['predicted_boxes'].append(str(boxes_pred_sample.cpu().numpy().tolist()))
                        image_df['gt_boxes'].append(str(boxes_sample.cpu().numpy().tolist()))
                        image_df["number_of_objects"].append(len(objs[i]))
                        if objs.shape[-1] == 1:
                            image_df["class"].append(
                                str([args.vocab["object_idx_to_name"][obj_index] for obj_index in objs[i]]))
                        else:
                            image_df["class"].append(str(
                                [args.vocab["reverse_attributes"]['shape'][str(int(objs[i][obj_index][2]))] for
                                 obj_index in range(objs[i].shape[0])]))

                for loss_name, loss_val in G_losses.items():
                    all_losses[loss_name].append(loss_val)

                num_samples += imgs.size(0)
                if not full_test and args.num_val_samples and num_samples >= args.num_val_samples:
                    break
            except Exception as e:
                print("Error in {}".format(str(e)))

        samples = {}
        if not args.skip_generation and not args.skip_graph_model:
            samples['pred_box_pred_mask'] = model(objs, triplets, triplet_type, test_mode=True)[0]
            samples['pred_box_gt_mask'] = model(objs, triplets, triplet_type, masks_gt=masks, test_mode=True)[0]

        if not args.skip_generation:
            samples['gt_img'] = imgs
            samples['gt_box_gt_mask'] = \
                model(objs, triplets, triplet_type, boxes_gt=boxes, masks_gt=masks, test_mode=True)[0]
            samples['gt_box_pred_mask'] = model(objs, triplets, triplet_type, boxes_gt=boxes, test_mode=True)[0]

            for k, v in samples.items():
                samples[k] = np.transpose(deprocess_batch(v, deprocess_func=args.deprocess_func).cpu().numpy(),
                                          [0, 2, 3, 1])

        mean_losses = {k: torch.stack(v).mean() for k, v in all_losses.items() if k != 'bbox_pred_all'}
        if not args.skip_graph_model:
            mean_losses.update({'avg_iou': total_iou / total_boxes,
                                'total_iou_05': total_iou_05 / total_boxes,
                                'total_iou_03': total_iou_03 / total_boxes})
            mean_losses.update({'inception_mean': 0.0})
            mean_losses.update({'inception_std': 0.0})

        if not args.skip_generation:
            inception_mean, inception_std = inception_score.compute_score(splits=5)
            mean_losses.update({'inception_mean': inception_mean})
            mean_losses.update({'inception_std': inception_std})

    model.train()
    return mean_losses, samples, pd.DataFrame.from_dict(image_df)


def update_loader_params(dset, w_conv, w_trans):
    if w_conv is not None:
        dset.converse_candidates_weights = w_conv.detach().cpu().numpy()
    if w_trans is not None:
        dset.trans_candidates_weights = torch.sigmoid(w_trans).detach().cpu().numpy()


def main(args):
    logger = logging.getLogger(__name__)
    args.vocab, train_loader, val_loader = build_train_val_loaders(args)
    init_args(args)
    learning_rate = args.learning_rate
    print_args(args)

    if not os.path.isdir(args.output_dir):
        print('Checkpoints directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(args.output_dir)
    json.dump(vars(args), open(os.path.join(args.output_dir, 'run_args.json'), 'w'))
    writer = SummaryWriter(args.output_dir)
    float_dtype = torch.cuda.FloatTensor

    # Define img_deprocess
    if args.img_deprocess == "imagenet":
        args.deprocess_func = imagenet_deprocess
    elif args.img_deprocess == "decode_img":
        args.deprocess_func = decode_image
    else:
        print("Error: No deprocess function was found. decode_image was chosen")
        args.deprocess_func = decode_image

    # setup device - CPU or GPU
    device = torch.device("cuda:{gpu}".format(gpu=args.gpu_ids[0]) if args.use_cuda else "cpu")
    print(" > Active GPU ids: {}".format(args.gpu_ids))
    print(" > Using device: {}".format(device.type))

    model = MetaGeneratorModel(args, device)
    model.type(float_dtype)
    conv_weights_mat = get_conv_converse(model)
    update_loader_params(train_loader.dataset, conv_weights_mat, None)
    update_loader_params(val_loader.dataset, conv_weights_mat, None)
    converse_list = [
        'sg_to_layout.module.converse_candidates_weights']  # 'sg_to_layout.module.trans_candidates_weights'
    trans_list = ['sg_to_layout.module.trans_candidates_weights']  # 'sg_to_layout.module.trans_candidates_weights'
    learned_converse_params = [kv[1] for kv in model.named_parameters() if kv[0] in converse_list]
    learned_transitivity_params = [kv[1] for kv in model.named_parameters() if kv[0] in trans_list]
    all_special_params = converse_list + trans_list
    base_params = [kv[1] for kv in model.named_parameters() if kv[0] not in all_special_params]
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': learning_rate},
                                  {'params': learned_transitivity_params, 'lr': 1e-2}])
    optimizer_converse = torch.optim.Adam([{'params': learned_converse_params, 'lr': 1e-2}])
    print(model)

    discriminator = MetaDiscriminatorModel(args)
    print(discriminator)
    gans_model = Pix2PixModel(args, discriminator=discriminator)
    gans_model = DataParallelWithCallback(gans_model, device_ids=args.gpu_ids).to(device)

    epoch, t = 0, 0
    # Restore checkpoint
    if args.restore_checkpoint:
        epoch, t = restore_checkpoint(args, model, gans_model, discriminator, optimizer, device)

    # Freeze weights
    if args.freeze:
        freeze_weights(model, discriminator, args.freeze_options)

    # Init Inception Score
    inception_score = InceptionScore(device, batch_size=args.batch_size, resize=True)
    # Run Epoch
    meta_relations = [args.vocab['pred_name_to_idx'][p] for p in train_loader.dataset.meta_relations]
    non_meta_relations = set(args.vocab['pred_name_to_idx'].values()) - set(meta_relations)
    eps = np.finfo(np.float32).eps.item()
    while True:
        if t >= args.num_iterations:
            break
        epoch += 1
        print('Starting epoch %d' % epoch)

        # Run Batch
        for batch in train_loader:
            try:
                t += 1
                batch = batch_to(batch)
                imgs, objs, boxes, triplets, conv_counts, triplet_type, masks, image_ids = batch
                model_out = model(objs, triplets, triplet_type, boxes_gt=boxes, masks_gt=masks, test_mode=False)

                # non gan losses
                G_losses = gans_model(batch, model_out, mode="compute_generator_loss")
                r = G_losses["bbox_pred_all"].detach()
                G_losses = {k: v.mean() for k, v in G_losses.items()}
                log_scalar_dict(writer, G_losses, 'train/loss', t)

                optimizer.zero_grad()
                G_losses["total_loss"].backward()
                optimizer.step()

                # Update SRC params
                if args.learned_converse:
                    batch_size = batch[0].shape[0]
                    if batch_size > 1:
                        r = (r - r.mean()) / (r.std() + eps)
                    conv_weights_mat = get_conv_converse(model)
                    log_prob = calc_log_p(conv_weights_mat, non_meta_relations, conv_counts)
                    loss_conv = torch.mean(r * log_prob)

                    optimizer_converse.zero_grad()
                    loss_conv.backward()
                    optimizer_converse.step()

                    conv_weights_mat = get_conv_converse(model)
                    update_loader_params(train_loader.dataset, conv_weights_mat, None)
                    update_loader_params(val_loader.dataset, conv_weights_mat, None)

                # Update GAN discriminators losses
                D_losses = {}
                if not args.skip_generation and args.freeze_options != "generation":
                    D_losses = gans_model(batch, model_out, mode="compute_discriminator_loss")
                    D_losses = {k: v.mean() for k, v in D_losses.items()}
                    log_scalar_dict(writer, D_losses, 'train/loss', t)
                    set_d_gans_loss(D_losses, args, discriminator)

                # Logger
                if t % args.print_every == 0:
                    print('t = %d / %d' % (t, args.num_iterations))
                    for name, val in G_losses.items():
                        print(' G [%s]: %.4f' % (name, val))
                    for name, val in D_losses.items():
                        print(' D [%s]: %.4f' % (name, val))

                # Save checkpoint
                if t % args.checkpoint_every == 0:
                    conv_weights_mat = get_conv_converse(model)
                    print_compute_converse_edges({}, conv_weights_mat.detach(), args.vocab, non_meta_relations)
                    print_compute_transitive_edges({}, torch.sigmoid(
                        model.sg_to_layout.module.trans_candidates_weights).detach(), args.vocab)

                    # GT Boxes; GT Masks
                    print('checking: input box/mask as GT')
                    gt_val_losses, gt_val_samples, _ = check_model(args, val_loader, model, gans_model, inception_score,
                                                                   use_gt=True, full_test=False)
                    log_scalar_dict(writer, gt_val_losses, 'gt_val/loss', t)
                    log_results(gt_val_losses, t, prefix='GT VAL')

                    # Pred Boxes; Pred Masks
                    print('checking: input box/mask as PRED')
                    use_gt = True if args.skip_graph_model else False  # if skip graph then use gt
                    val_losses, val_samples, _ = check_model(args, val_loader, model, gans_model,
                                                             inception_score, use_gt=use_gt, full_test=False)
                    log_scalar_dict(writer, val_losses, 'val/loss', t)
                    log_results(val_losses, t, prefix='VAL')
                    save_images(args, t, val_samples, writer)

                    # Save checkpoint
                    checkpoint_path = os.path.join(args.output_dir, 'itr_%s.pt' % t)
                    print('Saving checkpoint to ', checkpoint_path)
                    save_checkpoint(args, checkpoint_path, discriminator, epoch, gans_model, model, optimizer, t)

                # Full test
                if t % args.full_test == 0:
                    print('checking on full eval')
                    test_losses, test_samples, _ = check_model(args, val_loader, model, gans_model, inception_score,
                                                               use_gt=False, full_test=True)
                    log_scalar_dict(writer, test_losses, 'test/loss', t)
                    print('Iter: {},'.format(t) + ' TEST Inception mean: %.4f' % test_losses['inception_mean'])
                    print('Iter: {},'.format(t) + ' TEST Inception STD: %.4f' % test_losses['inception_std'])

            except Exception as e:
                logger.exception(e)

    writer.close()


def log_results(semi_val_losses, t, prefix=''):
    print('Iter: {}, '.format(t) + prefix + ' avg_iou: %.4f' % semi_val_losses.get('avg_iou', 0.0))
    print('Iter: {}, '.format(t) + prefix + ' total_iou_03: %.4f' % semi_val_losses.get('total_iou_03', 0.0))
    print('Iter: {}, '.format(t) + prefix + ' total_iou_05: %.4f' % semi_val_losses.get('total_iou_05', 0.0))
    print('Iter: {}, '.format(t) + prefix + ' Inception mean: %.4f' % semi_val_losses.get('inception_mean', 0.0))
    print('Iter: {}, '.format(t) + prefix + ' Inception STD: %.4f' % semi_val_losses.get('inception_std', 0.0))


def save_images(args, t, val_samples, writer, dir_name='val'):
    for k, v in val_samples.items():
        if isinstance(v, list):
            for i in range(len(v)):
                writer.add_figure('val_%s/%s' % (k, i), v, global_step=t)
        else:
            path = os.path.join(args.output_dir, dir_name, str(t), k)
            os.makedirs(path)
            for i in range(v.shape[0]):
                writer.add_images('val_%s/%s' % (k, i), v[i], global_step=t, dataformats='HWC')
                RGB_img_i = cv2.cvtColor(v[i], cv2.COLOR_BGR2RGB)
                cv2.imwrite("{}/{}.jpg".format(path, i), RGB_img_i)


def set_d_gans_loss(D_losses, args, discriminator):
    if args.use_img_disc:
        discriminator.optimizer_d_img.zero_grad()
        D_losses["total_img_loss"].backward()
        discriminator.optimizer_d_img.step()
    else:
        discriminator.optimizer_d_img.zero_grad()
        D_losses["total_img_loss"].backward()
        discriminator.optimizer_d_img.step()

        discriminator.optimizer_d_obj.zero_grad()
        D_losses["total_obj_loss"].backward()
        discriminator.optimizer_d_obj.step()

        if args.mask_size > 0 and "total_mask_loss" in D_losses:
            discriminator.optimizer_d_mask.zero_grad()
            D_losses["total_mask_loss"].backward()
            discriminator.optimizer_d_mask.step()


def save_checkpoint(args, checkpoint_path, discriminator, epoch, gans_model, model, optimizer, t):
    if args.use_img_disc:
        checkpoint_dict = {
            'model_state': model.state_dict(),
            'gans_model_state': gans_model.state_dict(),
            'd_img_state': discriminator.img_discriminator.state_dict(),
            'd_img_optim_state': discriminator.optimizer_d_img.state_dict(),
            'optim_state': optimizer.state_dict(),
            'vocab': args.vocab,
            'counters': {
                't': t,
                'epoch': epoch,
            }
        }
    else:
        checkpoint_dict = {
            'model_state': model.state_dict(),
            'gans_model_state': gans_model.state_dict(),
            'd_img_state': discriminator.img_discriminator.state_dict(),
            'd_obj_state': discriminator.obj_discriminator.state_dict(),
            'd_mask_state': discriminator.mask_discriminator.state_dict(),
            'd_img_optim_state': discriminator.optimizer_d_img.state_dict(),
            'd_obj_optim_state': discriminator.optimizer_d_obj.state_dict(),
            'd_mask_optim_state': discriminator.optimizer_d_mask.state_dict(),
            'optim_state': optimizer.state_dict(),
            'vocab': args.vocab,
            'counters': {
                't': t,
                'epoch': epoch,
            }
        }
    torch.save(checkpoint_dict, checkpoint_path)


if __name__ == '__main__':
    args = get_args()
    main(args)

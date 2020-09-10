import torch
from torch.tensor import Tensor


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def batch_to(batch, device='cuda'):
    output_batch = []
    for obj in batch:

        if obj is None:
            obj = None
        elif isinstance(obj, Tensor):
            if device == 'cuda':
                obj = obj.cuda()
            elif isinstance(device, torch.device):
                obj = obj.to(device)
            else:
                obj = obj.cpu().numpy()
        elif isinstance(obj, list):
            pass
        else:
            for k, v in obj.items():
                if device == 'cuda':
                    obj[k] = obj[k].cuda()
                elif isinstance(device, torch.device):
                    obj[k] = obj[k].to(device)
                else:
                    obj[k] = obj[k].cpu().numpy()
        output_batch.append(obj)
    return output_batch


def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def log_scalar_dict(writer, d, tag, itr, every=500):
    if itr%every==0:
        for k, v in d.items():
            writer.add_scalar('%s/%s' % (tag, k), v, itr)


def remove_dummy_objects(objs, vocab):
    # boxes = boxes.reshape((-1, 4))
    # isnotpadding_mask = (boxes != -1).any(dim=-1)
    isnotpadding_mask = (objs != 0)[:, 0]
    __image__ = vocab['object_name_to_idx']["__image__"]
    dummies_objs_mask = (objs != __image__)[:, 0]
    mask = dummies_objs_mask & isnotpadding_mask
    return mask


def remove_dummies_and_padding(boxes, objs, vocab, items_lst):
    isnotpadding_mask = (boxes != -1).any(dim=-1)
    __image__ = vocab['object_name_to_idx']["__image__"]
    dummies_objs_mask = (objs != __image__)[:, 0]
    new_mask = dummies_objs_mask & isnotpadding_mask
    return [item[new_mask] for item in items_lst]
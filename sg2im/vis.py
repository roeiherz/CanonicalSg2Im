import tempfile, os
import torch
import random
import colorsys
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from imageio import imread
from sg2im.data import imagenet_deprocess
from sg2im.metrics import xywh_to_points

"""
Utilities for making visualizations.
"""


def draw_box(box, color, text=None):
    """
  Draw a bounding box using pyplot, optionally with a text box label.

  Inputs:
  - box: Tensor or list with 4 elements: [x0, y0, x1, y1] in [0, W] x [0, H]
         coordinate system.
  - color: pyplot color to use for the box.
  - text: (Optional) String; if provided then draw a label for this box.
  """
    TEXT_BOX_HEIGHT = 10
    if torch.is_tensor(box) and box.dim() == 2:
        box = box.view(-1)
        assert box.size(0) == 4
    x0, y0, x1, y1 = box
    assert y1 > y0, box
    assert x1 > x0, box
    w, h = x1 - x0, y1 - y0
    rect = Rectangle((x0, y0), w, h, fc='none', lw=2, ec=color)
    plt.gca().add_patch(rect)
    if text is not None:
        text_rect = Rectangle((x0, y0), w, TEXT_BOX_HEIGHT, fc=color, alpha=0.5)
        plt.gca().add_patch(text_rect)
        tx = 0.5 * (x0 + x1)
        ty = y0 + TEXT_BOX_HEIGHT / 2.0
        plt.text(tx, ty, text, va='center', ha='center')


def draw_scene_graph(objs, triplets, vocab=None, **kwargs):
    """
  Use GraphViz to draw a scene graph. If vocab is not passed then we assume
  that objs and triplets are python lists containing strings for object and
  relationship names.

  Using this requires that GraphViz is installed. On Ubuntu 16.04 this is easy:
  sudo apt-get install graphviz
  """
    output_filename = kwargs.pop('output_filename', 'graph.png')
    orientation = kwargs.pop('orientation', 'V')
    edge_width = kwargs.pop('edge_width', 6)
    arrow_size = kwargs.pop('arrow_size', 1.5)
    binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
    ignore_dummies = kwargs.pop('ignore_dummies', True)
    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    # General setup, and style for object nodes
    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]
    # Output nodes for objects
    num_objs = len(objs)
    for i, obj in enumerate(objs):
        lines.append('%d [label="%s"]' % (i, ', '.join(list(val for key, val in obj.items() if key in vocab['attributes'].keys()))))

    # Output relationships
    next_node_id = num_objs
    lines.append('node [fillcolor="lightblue1"]')
    for p in triplets:
        rel = triplets[p]
        for o in range(len(rel)):
            for s in rel[o]:
                lines += [
                    '%d [label="%s"]' % (next_node_id, p),
                    '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                        s, next_node_id, edge_width, arrow_size, binary_edge_weight),
                    '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                        next_node_id, o, edge_width, arrow_size, binary_edge_weight)
                ]
                next_node_id += 1
    lines.append('}')

    # Now it gets slightly hacky. Write the graphviz spec to a temporary
    # text file
    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

    # Shell out to invoke graphviz; this will save the resulting image to disk,
    # so we read it, delete it, then return it.
    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
    os.remove(dot_filename)
    img = imread(output_filename)
    os.remove(output_filename)

    return img


def draw_item(item, image_size, text=None):
    """
    :param image_size: tuple of the image size
    :param item: item is a CLEVRDialogDataset() entry
    :return:
    """
    plt.figure()
    reverse_prep = imagenet_deprocess()(item[0]).permute(1, 2, 0).cpu().numpy()
    draw_reversed_item(item[2], image_size, reverse_prep, text)
    scene_layout = os.path.join('img_layout.png')
    plt.savefig(scene_layout)
    plt.close()


def draw_reversed_item(boxes, image_size, img, text):
    '''
    :param boxes: in x0, y0, w, h
    :param image_size:
    :param img:
    :param text:
    :return:
    '''
    ax = plt.gca()
    ax.imshow(img)
    boxes = xywh_to_points(boxes)
    coords = boxes.cpu().numpy() * image_size[0]
    for i in range(len(coords)):
        try:
            sample_text = "object" if text is None else text[i]
            draw_box(coords[i], 'red', sample_text)
        except Exception as e:
            print(str(e))
    return ax


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
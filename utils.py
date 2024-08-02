import os
import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import cv2

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def plot_one_box(box, im, color=(128, 128, 128), txt_color=(255, 255, 255), label=None, line_width=3):

    # Plots one xyxy box on image im with label
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    lw = line_width or max(int(min(im.size) / 200), 2)  # line width

    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    
    cv2.rectangle(im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        txt_width, txt_height = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        c2 = c1[0] + txt_width, c1[1] - txt_height - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
    return im

def resize_and_pad(image, desired_size):
    old_size = image.shape[:2] 
    ratio = float(desired_size/max(old_size))
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    image = cv2.resize(image, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    
    pad = (delta_w, delta_h)
    
    color = [100, 100, 100]
    new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT,
        value=color)
        
    return new_im, pad
     
def get_image_tensor(img, max_size, debug=False):
    """
    Reshapes an input image into a square with sides max_size
    """
    if type(img) is str:
        img = cv2.imread(img)
    
    resized, pad = resize_and_pad(img, max_size)
    resized = resized.astype(np.float32)
    
    if debug:
        cv2.imwrite("intermediate.png", resized)

    # Normalise!
    resized /= 255.0
    
    return img, resized, pad


class Seperate_Output_Decoder:
    def __init__(self):
        self.b = None
        self.a = None
        self.conv = None
        self.strides = None
        self.anchors = None
        self.dims = None
        self.img_h = None
        self.img_w = None
        self.reg_max = None
        self.pos = None
        self.num_classes = None
        self.initialized = False
    
    def initialize(self, preds, img_shape):
        # use first prediction to calculate fixed values!
        self.num_classes = next((o.shape[2] for o in preds if o.shape[2] != 64), -1)
        assert self.num_classes != -1, 'cannot infer postprocessor inputs via output shape if there are 64 classes'
        self.pos = [
            i for i, _ in sorted(enumerate(preds),
                                 key=lambda x: (x[1].shape[2] if self.num_classes > 64 
                                                else -x[1].shape[2], -x[1].shape[1]))]
        x = np.transpose(
            np.concatenate([
                np.concatenate([preds[i] for i in self.pos[:len(self.pos) // 2]], axis=1),
                np.concatenate([preds[i] for i in self.pos[len(self.pos) // 2:]], axis=1)], axis=2), axes=(0, 2, 1))
        self.reg_max = (x.shape[1] - self.num_classes) // 4
        self.img_h, self.img_w = img_shape[-3], img_shape[-2]
        strides = []
        for p in self.pos:
            if preds[p].shape[2] != 64:
                strides.append(int(np.sqrt(self.img_h * self.img_w / preds[p].shape[1])))
        self.dims = [(self.img_h // s, self.img_w // s) for s in strides]
        fake_feats = [np.zeros((1, 1, h, w)) for h, w in self.dims]
        self.anchors, self.strides = (np.transpose(x, (1, 0))
                            for x in make_anchors(fake_feats, strides, 0.5))
        self.anchors = np.expand_dims(self.anchors, 0)
        """Initialize a convolutional layer with a given number of input channels."""
        self.conv = Conv2d(self.reg_max)
        param = np.arange(self.reg_max, dtype=np.float32)
        self.conv.weight[:] = param.reshape(1, self.reg_max, 1, 1)
        self.b, _, self.a = x.shape # batch, channels, anchors

        self.initialized=True

    def dfl(self, x):
        assert self.initialized
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        x_reshaped = x.reshape(self.b, 4, self.reg_max, self.a)

        # Transpose x to (b, reg_max, 4, a)
        x_transposed = x_reshaped.transpose(0, 2, 1, 3)

        # Apply softmax along axis 2 (originally axis 1 before transpose)
        x_softmax = softmax(x_transposed, axis=2)

        return self.conv.forward(x_softmax).reshape(self.b, 4, self.a)

    def decode_bbox(self, preds):
        x = np.transpose(
            np.concatenate([
                np.concatenate([preds[i] for i in self.pos[:len(self.pos) // 2]], axis=1),
                np.concatenate([preds[i] for i in self.pos[len(self.pos) // 2:]], axis=1)], axis=2), axes=(0, 2, 1))

        dbox = dist2bbox(self.dfl(x[:, :-self.num_classes, :]), self.anchors, xywh=True,
                         dim=1) * self.strides  # Placeholder for dist2bbox function
        # FIXME: Bis hier sind die Ergebnisse vergleichbar!!!
        for i, arg in enumerate(dbox.shape):
            print("dbox.shape[" + str(i) + ']: ' + str(arg))
        arg1=np.concatenate((dbox, 1 / (1 + np.exp(-x[:, -self.num_classes:, :]))), axis=1)
        for i, arg in enumerate(arg1.shape):
            print("return_result.shape[" + str(i) + ']: ' + str(arg))
        return np.concatenate((dbox, 1 / (1 + np.exp(-x[:, -self.num_classes:, :]))), axis=1)

def softmax(x, axis):
    """
    Compute the softmax of each element along the specified axis of x.

    Parameters:
    x (numpy.ndarray): Input array.
    axis (int): Axis along which to apply the softmax.

    Returns:
    numpy.ndarray: The array with softmax applied along the specified axis.
    """
    # Subtract the max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_e_x


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = np.arange(w) + grid_cell_offset  # shift x
        sy = np.arange(h) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(np.full((h * w, 1), stride))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb =np.array_split(distance, 2, axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), axis=dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), axis=dim)  # xyxy bbox


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x
    
def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})

class Conv2d:
    def __init__(self, in_channels):
        self.in_channels = in_channels
        # Initialize weights
        self.weight = np.zeros((1, in_channels, 1, 1), dtype=np.float32)

    def forward(self, x):
        # Naive implementation of forward pass
        batch_size, in_channels, height, width = x.shape
        output = np.zeros((batch_size, 1, height, width), dtype=np.float32)

        for b in range(batch_size):
            for i in range(height):
                for j in range(width):
                    for k in range(self.in_channels):
                        output[b, 1, i, j] += np.sum(
                            x[b, k, i:i+1, j:j+1] * self.weight[1, k])
        return output
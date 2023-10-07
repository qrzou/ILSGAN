import numpy as np
import os
# import cv2
import torch
import pickle
import PIL.Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#----------------------------------------------------------------------------

class Visualizer:
    def __init__(self, grid_z, grid_c, grid_size, run_dir):
        self.grid_z = grid_z
        self.grid_c = grid_c
        self.grid_size = grid_size
        self.run_dir = run_dir

    @torch.no_grad()
    def save_sample_image_mask_grid(self, G, filename, M=None):
        images, masks = [], []
        for z, c in zip(self.grid_z, self.grid_c):
            if M is not None:
                img, mask_logit = G(z, c, return_mask=True, M=M)
            else:
                img, mask_logit = G(z, c, return_mask=True)
            # mask = torch.nn.functional.softmax(mask, dim=1)
            images.append(img.cpu())
            masks.append(mask_logit.cpu())
        images = torch.cat(images).numpy()
        masks = torch.cat(masks).numpy()
        save_image_label_grid(images, masks, os.path.join(self.run_dir, filename),
                              drange=[-1, 1], grid_size=self.grid_size)

    @torch.no_grad()
    def save_sample_image_label_grid(self, G, A, filename):
        images, labels = [], []
        for z, c in zip(self.grid_z, self.grid_c):
            img, feat = G(z, c)
            lbl = A(feat)
            images.append(img.cpu())
            labels.append(lbl.cpu())
        images = torch.cat(images).numpy()
        labels = torch.cat(labels).numpy()
        save_image_label_grid(images, labels, os.path.join(self.run_dir, filename),
                              drange=[-1, 1], grid_size=self.grid_size)

    @torch.no_grad()
    def save_single_sample_image_label_with_pickle(self, G, A, filename, top_num=10):
        assert top_num > 0 and isinstance(top_num, int)

        count = 0
        for z, c in zip(self.grid_z, self.grid_c):
            assert z.shape[0] == 1
            count += 1
            img, feat = G(z, c)
            lbl = A(feat)

            image = img.cpu().numpy()
            label = lbl.cpu().numpy()

            save_single_image_grid_z_c(image, label, z, c, self.run_dir, count=count,
                                  drange=[-1, 1], img_size=None, mask_size=None)
            if count == top_num:
                break

#----------------------------------------------------------------------------

def setup_snapshot_image_label_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    im_h, im_w = training_set.image_shape[1], training_set.image_shape[2]
    gw = np.clip(7680 // im_w, 7, 32)
    gh = np.clip(4320 // im_h, 4, 32) // 2 # Half the number for label
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def setup_snapshot_grid(im_h, im_w):
    gw = np.clip(7680 // im_w, 7, 32)
    gh = np.clip(4320 // im_h, 4, 32) // 2 # Half the number for label
    return gw, gh

#----------------------------------------------------------------------------

def save_single_image_grid_z_c(img, label, z, c, dir, count, drange, img_size=None, mask_size=None):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    im_h, im_w = img.shape[2:]

    if label.ndim == 4:
        label = label.argmax(axis=1)

    if label.ndim == 3: # Colorize label of segmentation
        label = colorize_segmentation(label) # (n, h, w, c)

    if label.ndim == 1:
        n = label.shape[0]
        onehot = np.zeros((n, 10)) # TODO: configure the num_class
        onehot[np.arange(n), label] = 1.
        label = onehot

    if label.ndim == 2: # Plot categorical distribution for classification
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        label = plot_categorical_distribution(label, class_names, (im_h, im_w))

    # gw, gh = grid_size
    # _N, C, H, W = img.shape
    # assert C == 3, f'{C}'
    # img = img.reshape(gh, gw, C, H, W)
    # img = img.transpose(0, 3, 1, 4, 2)
    # label = label.reshape(gh, gw, H, W, C)
    # label = label.transpose(0, 2, 1, 3, 4)
    # img = np.concatenate([img, label], axis=1)
    # img = img.reshape(gh * H * 2, gw * W, C)

    _N, C, H, W = img.shape
    assert _N == 1
    img = img.squeeze(0).transpose(1, 2, 0)
    label = label.squeeze(0)

    if C == 3:
        img_fname = os.path.join(dir, 'sample-' + str(count) + '-gen_img.png')
        PIL.Image.fromarray(img, 'RGB').save(img_fname)
        mask_fname = os.path.join(dir, 'sample-' + str(count) + '-gen_mask.png')
        PIL.Image.fromarray(label, 'RGB').save(mask_fname)
        code_fname = os.path.join(dir, 'sample-' + str(count) + '-code.pkl')
        code_data = {
            'z': z,
            'c': c,
        }
        with open(code_fname, 'wb') as f:
            pickle.dump(code_data, f)

#----------------------------------------------------------------------------

def save_image_label_grid(img, label, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    im_h, im_w = img.shape[2:]

    if label.ndim == 4:
        label = label.argmax(axis=1)

    if label.ndim == 3: # Colorize label of segmentation
        label = colorize_segmentation(label) # (n, h, w, c)

    if label.ndim == 1:
        n = label.shape[0]
        onehot = np.zeros((n, 10)) # TODO: configure the num_class
        onehot[np.arange(n), label] = 1.
        label = onehot

    if label.ndim == 2: # Plot categorical distribution for classification
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        label = plot_categorical_distribution(label, class_names, (im_h, im_w))

    gw, gh = grid_size
    _N, C, H, W = img.shape
    assert C == 3, f'{C}'
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    label = label.reshape(gh, gw, H, W, C)
    label = label.transpose(0, 2, 1, 3, 4)
    img = np.concatenate([img, label], axis=1)
    img = img.reshape(gh * H * 2, gw * W, C)

    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def save_image_rec_label_grid(img, rec_img, label, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    rec_img = np.asarray(rec_img, dtype=np.float32)
    rec_img = (rec_img - lo) * (255 / (hi - lo))
    rec_img = np.rint(rec_img).clip(0, 255).astype(np.uint8)
    im_h, im_w = img.shape[2:]

    if label.ndim == 4:
        label = label.argmax(axis=1)

    if label.ndim == 3: # Colorize label of segmentation
        label = colorize_segmentation(label) # (n, h, w, c)

    if label.ndim == 1:
        n = label.shape[0]
        onehot = np.zeros((n, 10)) # TODO: configure the num_class
        onehot[np.arange(n), label] = 1.
        label = onehot

    if label.ndim == 2: # Plot categorical distribution for classification
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        label = plot_categorical_distribution(label, class_names, (im_h, im_w))

    gw, gh = grid_size
    _N, C, H, W = img.shape
    assert C == 3, f'{C}'
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)

    rec_img = rec_img.reshape(gh, gw, C, H, W)
    rec_img = rec_img.transpose(0, 3, 1, 4, 2)

    label = label.reshape(gh, gw, H, W, C)
    label = label.transpose(0, 2, 1, 3, 4)

    img = np.concatenate([img, rec_img, label], axis=1)
    img = img.reshape(gh * H * 3, gw * W, C)

    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def colorize_segmentation(segmentation):
    # Assume segmentation is (n, h, w)

    # Construct color map. Reference: https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    n = 256  # The number of total entries
    cmap = np.zeros((n, 3), dtype=np.uint8)

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])

    return cmap[segmentation] # (n, h, w, c)

#----------------------------------------------------------------------------

def plot_categorical_distribution(label, class_names, im_size):

    if label.shape[1] == 1: # Convert the label into onehot
        onehot = np.zeros((label.shape[0], len(class_names)))
        onehot[np.arange(label.shape[0]), label[:, 0]] = 1.
        label = onehot

    height, width = im_size
    assert height == width
    label_ims = []
    for y in label:
        fig = plt.figure(figsize=(8, 8), dpi=width // 8)
        canvas = FigureCanvas(fig)
        plt.barh(class_names, y)
        plt.xlim((0., 1.))
        plt.yticks(fontsize=48)
        plt.xticks(fontsize=24)
        plt.tight_layout()
        canvas.draw()
        im = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(width, width, 3)
        label_ims.append(im.copy())
        plt.close()

    return np.stack(label_ims) # (n, h, w, 3)

#----------------------------------------------------------------------------

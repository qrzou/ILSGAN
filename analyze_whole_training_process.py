# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from tqdm import tqdm
from typing import List, Optional
import time
import json
import shutil
import psutil
import pickle

import click
import dnnlib
import numpy as np
import pandas as pd
import PIL.Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as torch_tf
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import cv2
from torch_utils import misc
from torch_utils import training_stats
import legacy

#----------------------------------------------------------------------------

def save_layer(tensor, outdir, layer_name, filename):
    outfile = os.path.join(outdir, layer_name, filename)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    if tensor.size(0) == 1 and layer_name == 'mask':
        tensor = 2 * torch.cat([tensor, tensor, tensor], dim=0) - 1

    img = (tensor.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(outfile)

#----------------------------------------------------------------------------

def get_pkl(path, specified_pkl_idx=None, topk=None, sigma=None):
    if specified_pkl_idx is not None and topk is not None:
        raise Exception()
    if os.path.isfile(path) and path.endswith('.pkl'):
        return path

    assert os.path.isdir(path)
    assert os.path.exists(path)
    if os.path.exists(os.path.join(path, 'metric-fid10k_full.jsonl')):
        metric_jsonl = os.path.join(path, 'metric-fid10k_full.jsonl')
    else:
        metric_jsonl = os.path.join(path, 'metric-fid50k_full.jsonl')
    if specified_pkl_idx is None and topk is None:
        with open(metric_jsonl, 'rt') as f:
            metrics = [json.loads(line) for line in f]

        if sigma is not None:
            metrics = [metric for metric in metrics if float(metric['sigma']) == sigma]
            if len(metrics) == 0:
                raise FileExistsError()

        minfid_idx = np.asarray([entry['results']['fid50k_full'] for entry in metrics]).argmin()
        return os.path.join(path, metrics[minfid_idx]['snapshot_pkl'])

    elif topk is not None:
        with open(metric_jsonl, 'rt') as f:
            metrics = [json.loads(line) for line in f]

        if sigma is not None:
            metrics = [metric for metric in metrics if float(metric['sigma']) == sigma]
            if len(metrics) == 0:
                raise FileExistsError()

        # select top k
        bag = []
        for entry in metrics:
            if int(entry['snapshot_pkl'].split('-')[-1].split('.')[0]) <= topk:
                bag.append(entry)

        minfid_idx = np.asarray([entry['results']['fid50k_full'] for entry in bag]).argmin()
        return os.path.join(path, bag[minfid_idx]['snapshot_pkl'])

    else:  # specified pkl idx
        with open(metric_jsonl, 'rt') as f:
            exits = False
            for idx, line in enumerate(f):
                result = json.loads(line)
                if str(specified_pkl_idx) in result['snapshot_pkl']:
                    exits = True
                    break
        if not exits:
            raise FileExistsError('Specified pkl does not exist!')

        return os.path.join(path, result['snapshot_pkl'])

#----------------------------------------------------------------------------

def generate_data(path, network_pkl, n, truncation_psi, noise_mode, tag='analyze_training_process'):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    ckpt_num = network_pkl.split('-')[-1].split('.')[0]
    if tag is None:
        outdir = os.path.join(path, 'synthetic_data-' + ckpt_num)
    else:
        outdir = os.path.join(path, tag, 'synthetic_data-' + ckpt_num)

    if os.path.exists(outdir):
        # Check the integrity
        prev_pkl = os.path.join(outdir, 'network-snapshot.pkl')
        prev_img = os.path.join(outdir, 'img')
        if len(os.listdir(prev_img)) == n and os.path.exists(prev_pkl):
            print('Using existing synthetic data:', outdir)
            return outdir
        else:
            shutil.rmtree(outdir, ignore_errors=True)
            os.makedirs(outdir, exist_ok=True)
    else:
        os.makedirs(outdir, exist_ok=True)


    # Generate images.
    batch_gpu = 16 * 4
    all_z = []
    all_c = []
    for base_i in tqdm(range(0, n, batch_gpu)):
        this_batch = min(batch_gpu, n - base_i)
        z = torch.randn(this_batch, G.z_dim).to(device)
        c = None
        if G.c_dim > 0:
            c = torch.nn.functional.one_hot(torch.randint(G.c_dim, [this_batch, ]).to(device), G.c_dim)
        lyr = G(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode, return_layers=True)
        for lyr_name, lyr_tensor in lyr.items():
            for inst_i in range(this_batch):
                save_layer(lyr[lyr_name][inst_i], outdir, lyr_name, f'{base_i + inst_i + 1:06d}.png')
        all_z.append(z.cpu().numpy())
        if c is not None:
            all_c.append(c.cpu().numpy())

    # Save the z, c, truncation_psi, and noise_mode
    gen_in = {
        'z': np.concatenate(all_z, axis=0),
        'c': np.concatenate(all_c, axis=0) if all_c else None,
        'truncation_psi': truncation_psi,
        'noise_mode': noise_mode
    }
    with open(os.path.join(outdir, 'input.pkl'), 'wb') as f:
        pickle.dump(gen_in, f)

    # Save the network_pkl
    shutil.copyfile(network_pkl, os.path.join(outdir, 'network-snapshot.pkl'))
    print(f'Finish generating {n} image of', network_pkl)
    return outdir

#----------------------------------------------------------------------------

def get_select_pkls(path, topk, interval):
    if os.path.isfile(path) and path.endswith('.pkl'):
        raise Exception()
    assert os.path.isdir(path)
    assert os.path.exists(path)
    if os.path.exists(os.path.join(path, 'metric-fid10k_full.jsonl')):
        metric_jsonl = os.path.join(path, 'metric-fid10k_full.jsonl')
    else:
        metric_jsonl = os.path.join(path, 'metric-fid50k_full.jsonl')
    with open(metric_jsonl, 'rt') as f:
        metrics = [json.loads(line) for line in f]

    # select top k
    bag = []
    for entry in metrics:
        if int(entry['snapshot_pkl'].split('-')[-1].split('.')[0]) <= topk:
            bag.append(entry)

    snap_list = [entry['snapshot_pkl'] for entry in bag]
    pkl_path_list = sorted([os.path.join(path, snap) for snap in snap_list])
    select_pkl = []
    for i, elem in enumerate(pkl_path_list):
        if i % interval == 0 and i != 0:
            select_pkl.append(elem)
    return select_pkl

#----------------------------------------------------------------------------

class UNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, up=False, down=False, bilinear=True):
        super().__init__()
        self.up_scale = None
        self.down_scale = None
        assert not (up and down)
        if up:
            self.up_scale = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else \
                torch.nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        if down:
            self.down_scale = torch.nn.MaxPool2d(2)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x1, x2=None):
        x = x1

        if self.up_scale is not None:
            assert x2 is not None
            x1 = self.up_scale(x1)
            # pad x1 if the size does not match the size of x2
            dh = x2.size(2) - x1.size(2)
            dw = x2.size(3) - x1.size(3)
            x1 = torch.nn.functional.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x2, x1], dim=1)

        if self.down_scale is not None:
            x = self.down_scale(x1)

        x = torch.nn.functional.relu_(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu_(self.bn2(self.conv2(x)))
        return x

class UNet(torch.nn.Module):
    def __init__(self, input_channels=3, output_channels=2, bilinear=True, scale=64):
        super(UNet, self).__init__()
        self.scale = scale
        if scale == 64:
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.bilinear = bilinear

            self.inc = UNetBlock(self.input_channels, 64)

            self.down1 = UNetBlock(64, 128, down=True)
            self.down2 = UNetBlock(128, 256, down=True)
            self.down3 = UNetBlock(256, 512, down=True)
            self.down4 = UNetBlock(512, 512, down=True)

            self.up1 = UNetBlock(1024, 256, up=True, bilinear=bilinear)
            self.up2 = UNetBlock(512, 128, up=True, bilinear=bilinear)
            self.up3 = UNetBlock(256, 64, up=True, bilinear=bilinear)
            self.up4 = UNetBlock(128, 64, up=True, bilinear=bilinear)

            self.outc = torch.nn.Conv2d(64, self.output_channels, kernel_size=1)

        elif scale == 128:
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.bilinear = bilinear

            self.inc = UNetBlock(self.input_channels, 64)

            self.down1 = UNetBlock(64, 128, down=True)
            self.down2 = UNetBlock(128, 256, down=True)
            self.down3 = UNetBlock(256, 512, down=True)
            self.down4 = UNetBlock(512, 512, down=True)
            self.down5 = UNetBlock(512, 512, down=True)

            self.up0 = UNetBlock(1024, 512, up=True, bilinear=bilinear)
            self.up1 = UNetBlock(1024, 256, up=True, bilinear=bilinear)
            self.up2 = UNetBlock(512, 128, up=True, bilinear=bilinear)
            self.up3 = UNetBlock(256, 64, up=True, bilinear=bilinear)
            self.up4 = UNetBlock(128, 64, up=True, bilinear=bilinear)

            self.outc = torch.nn.Conv2d(64, self.output_channels, kernel_size=1)

        else:
            raise AssertionError()



    def forward(self, x):
        if self.scale == 64:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)

        elif self.scale == 128:
            x1 = self.inc(x)  # 64
            x2 = self.down1(x1)  # 128
            x3 = self.down2(x2)  # 256
            x4 = self.down3(x3)  # 512
            x5 = self.down4(x4)  # 512
            x6 = self.down5(x5)  # 512

            x = self.up0(x6, x5)
            x = self.up1(x, x5)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)

        else:
            raise AssertionError()

        return logits

#----------------------------------------------------------------------------

class SynSegDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 indices     =None,
                 transforms  =None,
                 size  =None,
                 ):
        from glob import glob
        filenames = sorted(glob(os.path.join(path, "img", "*.png")))
        if indices is None:
            indices = range(len(filenames))
        self.filenames = [filenames[i] for i in indices]
        self.maskfiles = [os.path.join(path, 'mask', os.path.basename(filename)) for filename in self.filenames]
        # self.maskfiles = [os.path.join(path, f'mask/{i:06d}.png') for i in indices]

        if transforms is None:
            if size is None:
                transforms = torch_tf.Compose([torch_tf.ToTensor(), torch_tf.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
            else:
                transforms = torch_tf.Compose([
                    torch_tf.Resize((int(size), int(size))),
                    torch_tf.ToTensor(),
                    torch_tf.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
        self.transforms = transforms
        self.resolution = self[0][0].size(1)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).convert("RGB")
        img_tensor = self.transforms(img)

        # standard.
        mask = Image.open(self.maskfiles[idx]).convert("RGB")
        mask_tensor = (self.transforms(mask)[0] > 0.).long()
        return img_tensor, mask_tensor

class CubDataset(Dataset):
    def __init__(
            self,
            root_dir,
            size,
            data_split=0,
            use_flip=False,
    ):
        super().__init__()
        self.data_split = data_split
        self.use_flip = use_flip

        self.transform = transforms.Compose([
            transforms.Resize((int(size), int(size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

        self.transform_seg = transforms.Compose([
            transforms.Resize((int(size), int(size))),
            transforms.ToTensor()])

        root_dir = os.path.abspath(root_dir)
        self.ROOT_DIR = root_dir
        self.IM_SIZE = size

        self.bbox_meta, self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return item

    def collect_meta(self):
        """ Returns a dictionary with image filename as
        'key' and its bounding box coordinates as 'value' """

        data_dir = self.ROOT_DIR

        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)

        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)

        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)

        splits = np.loadtxt(os.path.join(data_dir, 'train_val_test_split.txt'), int)

        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        filenames = [fname[:-4] for fname in filenames]

        if self.data_split == 0: # training split
            filenames = np.array(filenames)
            filenames = filenames[splits[:, 1] == 0]
            filename_bbox_ = {fname: filename_bbox[fname] for fname in filenames}
        elif self.data_split == 2: # testing split
            filenames = np.array(filenames)
            filenames = filenames[splits[:, 1] == 2]
            filename_bbox_ = {fname: filename_bbox[fname] for fname in filenames}
        elif self.data_split == -1: # all dataset
            filenames = filenames.copy()
            filename_bbox_ = filename_bbox

        print('Filtered filenames: ', len(filenames))
        return filename_bbox_, filenames

    def load_item(self, index):
        key = self.file_meta[index]
        bbox = self.bbox_meta[key]

        data_dir = self.ROOT_DIR

        img_path = '%s/images/%s.jpg' % (data_dir, key)
        img = self.load_imgs(img_path, bbox)

        seg_path = '%s/segmentations/%s.png' % (data_dir, key)
        seg = self.load_segs(seg_path, bbox)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])

        if seg.ndim > 2:
            seg = seg[0]

        # return img, seg, index
        return img, seg

    def load_imgs(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)

        cimg = img.crop([x1, y1, x2, y2])
        return self.transform(cimg)

    def load_segs(self, seg_path, bbox):
        # img = Image.open(seg_path).convert('1')

        img = Image.open(seg_path)

        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)

        cimg = img.crop([x1, y1, x2, y2])

        img_tensor = (self.transform_seg(cimg) > 0.5).long()

        if img_tensor.ndim > 2:
            img_tensor = img_tensor[0]

        # return self.transform_seg(cimg)
        return img_tensor

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item

class DogDataset(Dataset):
    def __init__(
            self,
            root_dir,
            size,
            data_split=0,
            use_flip=False,
    ):
        super().__init__()
        self.data_split = data_split
        self.use_flip = use_flip

        self.transform = transforms.Compose([
            transforms.Resize((int(size), int(size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

        self.transform_seg = transforms.Compose([
            transforms.Resize((int(size), int(size))),
            transforms.ToTensor()])

        root_dir = os.path.abspath(root_dir)
        self.ROOT_DIR = root_dir
        self.IM_SIZE = size

        self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return item

    def collect_meta(self):
        sel_indices_tr = np.load('{}/data_tr_sel.npy'.format(self.ROOT_DIR))
        sel_indices_te = np.load('{}/data_te_sel.npy'.format(self.ROOT_DIR))

        if self.data_split == 0: # training split
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr]
        elif self.data_split == 2: # testing split
            filenames = ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        elif self.data_split == -1: # all dataset
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr] \
                        + ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        return filenames

    def load_item(self, index):
        key = self.file_meta[index]

        data_dir = self.ROOT_DIR

        img_path = '%s/%s_resized.png' % (data_dir, key)
        img = self.load_imgs(img_path)

        seg_path = '%s/%s_maskresized.png' % (data_dir, key)
        seg = self.load_segs(seg_path)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])

        if seg.ndim > 2:
            seg = seg[0]

        # return img, seg, index
        return img, seg

    def load_imgs(self, img_path):
        img = cv2.imread(img_path)
        img = Image.fromarray(img)

        return self.transform(img)

    def load_segs(self, seg_path):
        img = Image.open(seg_path).convert('1')

        # return self.transform_seg(img)
        return (self.transform_seg(img) > 0.5).long()

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item

class CarDataset(Dataset):
    def __init__(
            self,
            root_dir,
            size,
            data_split=0,
            use_flip=False,
    ):
        super().__init__()
        self.data_split = data_split
        self.use_flip = use_flip

        self.transform = transforms.Compose([
            transforms.Resize((int(size), int(size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

        self.transform_seg = transforms.Compose([
            transforms.Resize((int(size), int(size))),
            transforms.ToTensor()])

        root_dir = os.path.abspath(root_dir)
        self.ROOT_DIR = root_dir
        self.IM_SIZE = size

        self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return item

    def collect_meta(self):
        sel_indices_tr = np.load('{}/data_mrcnn_train_select.npy'.format(self.ROOT_DIR))
        sel_indices_te = np.load('{}/data_mrcnn_test_select.npy'.format(self.ROOT_DIR))

        if self.data_split == 0: # training split
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr]
        elif self.data_split == 2: # testing split
            filenames = ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        elif self.data_split == -1: # all dataset
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr] \
                        + ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        return filenames

    def load_item(self, index):
        key = self.file_meta[index]

        data_dir = self.ROOT_DIR

        img_path = '%s/%s_resized.png' % (data_dir, key)
        img = self.load_imgs(img_path)

        seg_path = '%s/%s_maskresized.png' % (data_dir, key)
        seg = self.load_segs(seg_path)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])

        if seg.ndim > 2:
            seg = seg[0]

        # return img, seg, index
        return img, seg

    def load_imgs(self, img_path):
        img = cv2.imread(img_path)
        img = Image.fromarray(img)

        return self.transform(img)

    def load_segs(self, seg_path):
        img = Image.open(seg_path).convert('1')

        # return self.transform_seg(img)
        return (self.transform_seg(img) > 0.5).long()

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item

class LSUNCarSegTestset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir     = '../data/LSUN_car_lmdb/first_10k/',
                 size       = 128,
                 ):
        with open(os.path.join(root_dir, 'test.txt'), 'rt') as f:
            self.indices = [line.strip() for line in f.readlines()]
        self.rootdir = root_dir
        self.ROOT_DIR = root_dir

        self.transform = transforms.Compose([
            transforms.Resize((int(size), int(size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

        self.transform_seg = transforms.Compose([
            transforms.Resize((int(size), int(size))),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        filename = self.indices[idx]

        img = Image.open(os.path.join(self.rootdir, "images", filename[:-3] + 'jpg')).convert("RGB")
        img_tensor = self.transform(img)

        # parse the segmentation
        mask = Image.open(os.path.join(self.rootdir, 'masks', filename)).convert('1')
        mask_tensor = (self.transform_seg(mask) > 0.5).long()
        if mask_tensor.ndim > 2:
            mask_tensor = mask_tensor[0]
        return img_tensor, mask_tensor


#----------------------------------------------------------------------------

def calc_metrics(metrics, model, testloader, device):
    start_time = time.time()

    nC = model.output_channels
    C = np.zeros((nC, nC), dtype=np.int64) # Confusion matrix: [Pred x GT]

    iou_s = 0
    dice_s = 0
    cnt = 0

    model.eval().requires_grad_(False)
    for img, gt_mask in testloader:
        pred_logits = model(img.to(device))
        pred_mask = pred_logits.max(dim=1)[1]

        bs = pred_mask.shape[0]
        pred = pred_mask
        gt = gt_mask.to(device)

        pred_mask = pred_mask.cpu().numpy()
        gt_mask = gt_mask.cpu().numpy()
        C += np.bincount(
            nC * pred_mask.reshape(-1) + gt_mask.reshape(-1), # the value is one of [0, 1, 2, 3] which suggests tn, fn, fp, tp
            minlength=nC ** 2).reshape(nC, nC) # reshape to [Pred x GT]

        # metric code used in DRC
        iou = (pred * gt).view(bs, -1).sum(dim=-1) / \
              ((pred + gt) > 0).view(bs, -1).sum(dim=-1)

        dice = 2 * (pred * gt).view(bs, -1).sum(dim=-1) / \
               (pred.view(bs, -1).sum(dim=-1) + gt.view(bs, -1).sum(dim=-1))

        iou_s += iou.sum().item()
        dice_s += dice.sum().item()
        cnt += bs


    model.train().requires_grad_(True)

    assert all([metric in ['pACC', 'IoU', 'mIoU', 'DRC_IoU', 'DRC_DICE'] for metric in metrics])

    results = {}
    C = C.astype(np.float)
    if 'pACC' in metrics:
        # pACC = (tn + tp) / (tn + fn + fp + tp)
        results['pACC'] = C.diagonal().sum() / C.sum()
    if 'IoU' in metrics or 'mIoU' in metrics:
        # IoU = tp / (tp + fn + fp)
        union = C.sum(axis=1) + C.sum(axis=0) - C.diagonal()
        union[union == 0] = 1e-8
        iou_vals = C.diagonal() / union # (nC,)
        if 'IoU' in metrics:
            results['IoU'] = iou_vals[1]
        if 'mIoU' in metrics:
            results['mIoU'] = iou_vals.mean()

    results['DRC_IoU'] = iou_s / cnt
    results['DRC_DICE'] = dice_s / cnt


    total_time = time.time() - start_time

    return dict(
        results         = results,
        metrics         = metrics,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
    )

#----------------------------------------------------------------------------

def report_metrics(result_dict, run_dir=None, snapshot_pth=None, split='val'):
    if run_dir is not None and snapshot_pth is not None:
        snapshot_pth = os.path.relpath(snapshot_pth, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pth=snapshot_pth, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{split}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------

def resume(to_resume, snapshot_pth):
    print(f'Resume from {snapshot_pth}...')

    snapshot_data = torch.load(snapshot_pth)
    for name, module in to_resume.items():
        module.load_state_dict(snapshot_data[name])
    return snapshot_data.get('cur_iter', 0)

#----------------------------------------------------------------------------

def save_seg_grid(results, fname, grid_size):
    gw, gh = grid_size
    N = gw * gh

    def pre_reshape(x):
        lo, hi = (-1, 1) if x.ndim > 3 else (0, 1)
        x = np.asarray(x, dtype=np.float32)
        x = (x - lo) * (255 / (hi - lo))
        if x.ndim == 3:
            x = np.tile(x[:, None], (1, 3, 1, 1))
        x = np.rint(x).clip(0, 255).astype(np.uint8)
        return x

    results = [pre_reshape(torch.cat(x, dim=0).cpu()[:N]) for x in results]
    grid_im = np.stack(results, axis=3) # (N, C, H, 3, W)
    grid_im = grid_im.reshape(gh, gw, *grid_im.shape[1:]) # (gh, gw, C, H, 3, W)
    grid_im = grid_im.transpose(0, 3, 1, 4, 5, 2) # (gh, H, gw, 3, W, C)
    out_h = np.prod(grid_im.shape[:2])
    out_w = np.prod(grid_im.shape[2:5])
    grid_im = grid_im.reshape(out_h, out_w, -1)
    Image.fromarray(grid_im, 'RGB').save(fname)

#----------------------------------------------------------------------------

def augment_data(op, img, mask):
    if op is None:
        return img, mask

    to_hard_mask = False
    if mask.dim() == 3:
        # Convert to one-hot
        mask = F.one_hot(mask).permute(0, 3, 1, 2).to(torch.float32)
        to_hard_mask = True

    # Transform img
    img, params = op(img, is_mask=False)
    mask, _ = op(mask, params, is_mask=True)

    if to_hard_mask:
        mask = mask.max(dim=1).indices

    return img, mask

#----------------------------------------------------------------------------

def do_segmentation(
        run_dir,
        syn_data,
        real_data,
        real_data_abs_path      = None,  # using abs_path to avoid error when sshfs disconnected
        #
        aug                     = None,
        #
        resume_from             = None,
        metrics                 = ['pACC', 'IoU', 'mIoU', 'DRC_IoU', 'DRC_DICE'],
        #
        batch                   = 64,
        total_iter              = 6000,
        lr                      = 0.001,
        lr_steps                = 4000,
        lr_decay                = 0.2,
        #
        niter_per_tick          = 20,
        network_snapshot_ticks  = 50,
        image_snapshot_ticks    = 50,
        #
        random_seed             = 0,
        cudnn_benchmark         = True,
        scale                   = 64,
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda')
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.

    # Load training, validation (synthetic data), and test set (real data).
    assert os.path.exists(os.path.join(syn_data, 'img'))
    assert os.path.exists(os.path.join(syn_data, 'mask'))

    whole_dataset = SynSegDataset(syn_data, size=scale)
    num_samples = len(whole_dataset)
    del whole_dataset

    split_num = int(num_samples * 0.9)
    trainset = SynSegDataset(syn_data, range(split_num), size=scale)
    trainset_sampler = misc.InfiniteSampler(dataset=trainset, seed=random_seed)
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    trainset_iterator = iter(torch.utils.data.DataLoader(dataset=trainset, sampler=trainset_sampler,
                                                         batch_size=batch, **data_loader_kwargs))
    valset = SynSegDataset(syn_data, range(split_num, num_samples), size=scale)
    valset_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=batch, drop_last=False, **data_loader_kwargs)

    # prepare the RealDataset
    assert real_data in ['cub', 'dog', 'car', 'lsuncar']
    if real_data == 'cub':
        if real_data_abs_path is None:
            root_dir = '../../../datasets_local/DRC_processed/birds'
        else:
            root_dir = real_data_abs_path
        testset = CubDataset(
            root_dir=root_dir,
            size=scale,
            data_split=2,
            use_flip=False,
        )
    elif real_data == 'dog':
        if real_data_abs_path is None:
            root_dir = '../../../datasets_local/DRC_processed/dogs'
        else:
            root_dir = real_data_abs_path
        testset = DogDataset(
            root_dir=root_dir,
            size=scale,
            data_split=2,
            use_flip=False,
        )
    elif real_data == 'car':
        if real_data_abs_path is None:
            root_dir = '../../../datasets_local/DRC_processed/cars'
        else:
            root_dir = real_data_abs_path
        testset = CarDataset(
            root_dir=root_dir,
            size=scale,
            data_split=2,
            use_flip=False,
        )

    elif real_data == 'lsuncar':
        if real_data_abs_path is None:
            root_dir = '../../../datasets_local/LSUN_car_lmdb/first_10k/'
        else:
            root_dir = real_data_abs_path
        testset = LSUNCarSegTestset(
            root_dir=root_dir,
            size=scale,
        )
    else:
        raise NotImplementedError()


    print('Num of test images:  ', len(testset))
    testset_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch, drop_last=False, **data_loader_kwargs)


    # Construct networks.
    model = UNet(scale=64).to(device).eval().requires_grad_(False)
    # Construct augmentation.
    # augmentation from ELGANv2.0
    if aug is not None:
        aug_kwargs = {
            'geom': dict(xflip=0.5, scale=0.5, rotate=0.5, xfrac=0.5),
            'color': dict(brightness=0.5, contrast=0.5, lumaflip=0.5, hue=0.5, saturation=0.5),
            'gc': dict(xflip=0.5, scale=0.5, rotate=0.5, xfrac=0.5,
                       brightness=0.5, contrast=0.5, lumaflip=0.5, hue=0.5, saturation=0.5),
        }[aug]
        augment = dnnlib.util.construct_class_by_name(class_name='training.seg_augment.AugmentPipe', **aug_kwargs)
    else:
        augment = None

    # Print network summary tables.
    # misc.print_module_summary(model, [torch.empty([batch, 3, trainset.resolution, trainset.resolution], device=device)])

    # Setup training phases.
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, lr_steps, lr_decay)

    # Export sample images.
    print('Exporting segmentation visualization...')
    grid_size = (7, 32)
    grid_img, grid_gt = zip(*[next(trainset_iterator) for _ in range((np.prod(grid_size) - 1) // batch + 1)])
    grid_img = [x.to(device) for x in grid_img]
    grid_pred = [model(img.to(device)).max(dim=1)[1] for img in grid_img]
    save_seg_grid([grid_img, grid_pred, grid_gt], os.path.join(run_dir, 'seg-train_init.png'), grid_size=grid_size)

    # Visualize augmentation
    grid_augx, grid_augy = zip(*[augment_data(augment, img.to(device), mask.to(device)) for img, mask in zip(grid_img, grid_gt)])
    save_seg_grid([grid_img, grid_gt, grid_augx, grid_augy], os.path.join(run_dir, 'augment.png'), grid_size=grid_size)

    del grid_pred, grid_augx, grid_augy
    torch.cuda.empty_cache()

    # Initialize logs.
    print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_tfevents = None
    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
    try:
        import torch.utils.tensorboard as tensorboard
        stats_tfevents = tensorboard.SummaryWriter(run_dir)
    except ImportError as err:
        print('Skipping tfevents export:', err)

    # Resume from existing pth.
    cur_iter = 0
    if resume_from is not None:
        assert os.path.exists(resume_from) and os.path.isdir(resume_from)
        from glob import glob
        resume_pth = sorted(glob(os.path.join(resume_from, '*.pth')))[-1]
        cur_iter = resume(dict(model=model, opt=opt, lr_sch=lr_scheduler), resume_pth)

    # Train
    print(f'Training for {total_iter} iterations...')
    print(f'Start from {cur_iter} iterations ...')
    print()
    cur_tick = 0
    tick_start_iter = cur_iter
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    # do_train = not test_only
    do_train = True
    if do_train:
        model.train().requires_grad_(True)
    while do_train:
        # Fetch data
        img, target = next(trainset_iterator)
        img = img.to(device)
        target = target.to(device)

        # Forward and backward
        opt.zero_grad()
        img, target = augment_data(augment, img, target)
        pred = model(img)
        loss = torch.nn.functional.cross_entropy(pred, target)
        training_stats.report('Loss/loss', loss)
        loss.backward()
        opt.step()

        # Update state.
        cur_iter += 1

        # Perform maintenance tasks once per tick.
        done = (cur_iter >= total_iter)
        if (not done) and (cur_iter != 0) and (cur_iter < tick_start_iter + niter_per_tick):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"niter {training_stats.report0('Progress/niter', cur_iter):<8d}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/iter {training_stats.report0('Timing/sec_per_iter', (tick_end_time - tick_start_time) / (cur_iter - tick_start_iter)):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        # print(' '.join(fields))

        # Save segmentation snapshot.
        if (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            model.eval().requires_grad_(False)
            grid_pred = [model(img.to(device)).max(dim=1)[1] for img in grid_img]
            save_seg_grid([grid_img, grid_pred, grid_gt], os.path.join(run_dir, f'seg-train_{cur_iter:06d}.png'),
                          grid_size=grid_size)
            del grid_pred
            model.train().requires_grad_(True)

        # Save network snapshot.
        snapshot_data = None
        snapshot_pth = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            # snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            snapshot_data = {name: module.state_dict() for name, module in [('model', model), ('opt', opt), ('lr_sch', lr_scheduler)]}
            snapshot_data['cur_iter'] = cur_iter
            snapshot_pth = os.path.join(run_dir, f'network-snapshot-{cur_iter:06d}.pth')
            torch.save(snapshot_data, snapshot_pth)

        # Evaluate metrics.
        if (snapshot_data is not None) and len(metrics) > 0:
            print('Evaluating metrics...')
            result_dict = calc_metrics(metrics=metrics, model=model, testloader=valset_loader, device=device)
            report_metrics(result_dict, run_dir=run_dir, snapshot_pth=snapshot_pth, split='val')
            stats_metrics.update(result_dict["results"])
        del snapshot_data  # conserve memory

        # Collect statistics.
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = cur_iter
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()

        # Update state.
        cur_tick += 1
        tick_start_iter = cur_iter
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time

        if done:
            break

    # Resume from the best (maximum IoU)
    print('Conclude training, resume the snapshot of best validation performance ...')
    metric_jsonl = os.path.join(run_dir, 'metric-val.jsonl')
    with open(metric_jsonl, 'rt') as f:
        metric_results = [json.loads(line) for line in f]
    maxiou_idx = np.asarray([entry['results']['IoU'] for entry in metric_results]).argmax()
    ckpt_path_list = [os.path.join(run_dir, metric_results[maxiou_idx]['snapshot_pth'])]

    # Evaluation of testset
    # Evaluate and report on test split.
    print('\n\n Evaluating on resized dataset (default DRC processed)!\n')
    results = []
    for ckpt_path in ckpt_path_list:
        resume(dict(model=model, opt=opt, lr_sch=lr_scheduler), ckpt_path)
        result_dict = calc_metrics(metrics=metrics, model=model, testloader=testset_loader, device=device)
        results.append(result_dict)
        report_metrics(result_dict, run_dir=run_dir, snapshot_pth=metric_results[maxiou_idx]['snapshot_pth'], split='test')

        # Save segmentation snapshot.
        testset_iterator = iter(testset_loader)
        grid_img, grid_gt = zip(*[next(testset_iterator) for _ in range((np.prod(grid_size) - 1) // batch + 1)])
        grid_img = [x.to(device) for x in grid_img]
        model.eval().requires_grad_(False)
        grid_pred = [model(img.to(device)).max(dim=1)[1] for img in grid_img]
        model.train().requires_grad_(True)
        save_seg_grid([grid_img, grid_pred, grid_gt], os.path.join(run_dir, 'seg-test.png'), grid_size=grid_size)

    print('\nDone')
    return results


#----------------------------------------------------------------------------
@click.command()
@click.pass_context
@click.option('--path', help='Path to layered GAN training results.', required=True)
@click.option('--topk', help='only consider top k pkl to generate', type=int, metavar='INT')
@click.option('--interval', help='interval of analyze', type=int, metavar='INT')
@click.option('--n', help='How many number of images to be generated', type=int, metavar='INT')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--real-data', help='Which dataset to evaluate on', required=True, type=str)
@click.option('--aug', help='Which augmentation to be used', type=str)
@click.option('--scale', help='Test img scale [default: 64]', type=int, metavar='INT')
def do_analyze(
        ctx: click.Context,
        path: str,
        topk,
        interval,
        n,
        truncation_psi: float,
        noise_mode: str,

        real_data,
        aug,
        scale,
):
    """
    Select several ckpt from Layered GAN  training process, generate synthetic datasets
    and test their segmentation performance
    :return:
    """
    # [Assertion] + [Default]
    if path is None:
        raise Exception()
    if interval is None:
        interval = 3
    if topk is None:
        topk = 8000
    if n is None:
        n = 10000
    if scale is None:
        scale = 64
    if aug is not None:
        assert aug in ['geom', 'color', 'gc']
    assert isinstance(interval, int)
    assert real_data in ['cub', 'dog', 'car']

    # [Get select pkl]: prepare the ckpt list
    select_pkl = get_select_pkls(path, topk, interval)

    # [Generate synthetic data]
    start_time = time.time()
    synthetic_path = []
    for network_pkl in select_pkl:
        # Do generate.
        outdir = generate_data(path, network_pkl, n, truncation_psi, noise_mode)
        synthetic_path.append(outdir)
    total_time = time.time() - start_time
    print(f'total analyze time: {dnnlib.util.format_time(total_time)}')

    # [Segmentation]
    # prepare the argument
    # do the segmentation
    all_results = {}
    for syn_data in synthetic_path:
        # Pick output directory
        outdir = syn_data
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1

        if aug is not None:
            run_dir = os.path.join(outdir, f'{cur_run_id:05d}-Seg-DRCreal-{real_data}-{aug}_aug')
        else:
            run_dir = os.path.join(outdir, f'{cur_run_id:05d}-Seg-DRCreal-{real_data}')

        if 'scale' is not None:
            run_dir = run_dir + '-' + str(scale)

        assert not os.path.exists(run_dir)
        # Create output directory.
        print('Creating output directory...')
        os.makedirs(run_dir, exist_ok=True)

        dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
        results = do_segmentation(run_dir, syn_data, real_data, aug=aug, scale=scale)
        steps = syn_data.split('/')[-1].split('-')[-1]
        all_results[steps] = results[0]

    print('\nAll segmentation results:\n', all_results)
    with open(os.path.join(synthetic_path[0], '..', 'analyze_segmentation_results.json'), 'wt') as f:
        json.dump(all_results, f, indent=2)

    print('\n Segmentation analysis done.\n')



#----------------------------------------------------------------------------

if __name__ == "__main__":
    do_analyze()

#----------------------------------------------------------------------------

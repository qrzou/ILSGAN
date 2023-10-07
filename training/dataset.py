# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import cv2
import PIL.Image
from PIL import Image
import json
import torch
import dnnlib
import torchvision
import torchvision.transforms as transforms
import pandas as pd

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        if self.image_shape[1] == self.image_shape[2]:
            return self.image_shape[1]
        else:
            return max(self.image_shape[1], self.image_shape[2])

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

class ImageFolderCropDataset(Dataset):
    def __init__(self,
        data_name,
        path,                           # Path to directory or zip.
        size,                           # Ensure specific resolution, None = highest available.
        aspect_ratio,                   # H:W (1.0 or 0.75)
        **super_kwargs,                 # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self._size = (size, size)       # The size of output image (h, w)
        self._aspect_ratio = aspect_ratio
        self._crop = 'center'
        self._resize = None

        self._data_name = data_name

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = data_name + str(size)
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        # raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def set_random_resize_crop(self, resize=None):
        self._crop = 'random'
        self._resize_scale = resize

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                # image = pyspng.load(f.read())
                image = PIL.Image.fromarray(pyspng.load(f.read()))
            else:
                # image = np.array(PIL.Image.open(f))
                image = PIL.Image.open(f)

        # Crop the image
        i, j, h, w = self._get_crop_params(image) # top, left, height, width
        image = np.array(torchvision.transforms.functional.resized_crop(image, i, j, h, w, self._size))

        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        if image.shape[-1] == 1:
            image = np.concatenate([image, image, image], axis=-1)
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _get_crop_params(self, x):
        width, height = x.size

        if self._crop == 'center':
            #
            i, j = 0, 0
            if height > width * self._aspect_ratio: # height is longer than expected
                i = int((height - width * self._aspect_ratio) // 2)
                height = int(width * self._aspect_ratio)

            elif height < width * self._aspect_ratio:
                j = int((width - height / self._aspect_ratio) // 2)
                width = int(height / self._aspect_ratio)

        elif self._crop == 'random':
            scale = np.random.uniform(*self._resize_scale)
            ww, hh = int(scale * width), int(scale * height) # Scale
            ww = min(ww, hh / self._aspect_ratio) # Crop to aspect_ratio
            hh = min(hh, ww * self._aspect_ratio)
            i = int(np.random.uniform(0, height - hh))
            j = int(np.random.uniform(0, width -ww))
            height, width = hh, ww

        else:
            raise NotImplementedError

        return i, j, height, width

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

class DRCDataset(Dataset):
    def __init__(self,
                 data_name,                      # select cub, dog, car
                 size,                           # Ensure specific resolution, None = highest available.
                 path=None,
                 # aspect_ratio,                   # H:W (1.0 or 0.75)
                 **super_kwargs,                 # Additional arguments for the Dataset base class.
                 ):

        # self._path = path
        # self._aspect_ratio = aspect_ratio
        # self._crop = 'center'
        # self._resize = None

        assert data_name in ['cub', 'dog', 'car',]
        self._data_name = data_name
        if data_name == 'cub':
            if path is None:
                self.root_dir = '../../../datasets_local/DRC_processed/birds'
            else:
                self.root_dir = path
            self.data_split = 0
            self.bbox_meta, self.file_meta = self.collect_meta_cub()

        elif data_name == 'dog':
            if path is None:
                self.root_dir = '../../../datasets_local/DRC_processed/dogs'
            else:
                self.root_dir = path
            self.data_split = 0
            self.file_meta = self.collect_meta_dog()

        elif data_name == 'car':
            if path is None:
                self.root_dir = '../../../datasets_local/DRC_processed/cars'
            else:
                self.root_dir = path
            self.data_split = 0
            self.file_meta = self.collect_meta_car()

        else:
            raise NotImplementedError()

        # for car datset
        self._aspect_ratio = 1.0
        self._crop = 'center'
        # Convert relpath to abspath
        self.root_dir = os.path.abspath(self.root_dir)
        self._path = self.root_dir
        self._type = 'dir'
        self._zipfile = None
        self._size = size       # The size of output image (h, w)
        self._all_fnames = self.file_meta
        self._image_fnames = self._all_fnames

        name = 'DRC_' + data_name + str(size) + 'split' + str(self.data_split)
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)


    def collect_meta_cub(self):
        """ Returns a dictionary with image filename as
        'key' and its bounding box coordinates as 'value' """

        data_dir = self.root_dir

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

    def collect_meta_dog(self):
        sel_indices_tr = np.load('{}/data_tr_sel.npy'.format(self.root_dir))
        sel_indices_te = np.load('{}/data_te_sel.npy'.format(self.root_dir))

        if self.data_split == 0: # training split
            # filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr]
            filenames = ['train_orig/orig/{}'.format(token) for token in sel_indices_tr]
        elif self.data_split == 2: # testing split
            filenames = ['test_orig/orig/{}'.format(token) for token in sel_indices_te]
        elif self.data_split == -1: # all dataset
            filenames = ['train_orig/orig/{}'.format(token) for token in sel_indices_tr] \
                        + ['test_orig/orig/{}'.format(token) for token in sel_indices_te]
        return filenames

    def collect_meta_car(self):
        sel_indices_tr = np.load('{}/data_mrcnn_train_select.npy'.format(self.root_dir))
        sel_indices_te = np.load('{}/data_mrcnn_test_select.npy'.format(self.root_dir))

        if self.data_split != 0:
            raise Exception('Only support training split for original images!')

        if self.data_split == 0: # training split
            filenames = ['data_mrcnn/train/orig/{}'.format(token) for token in sel_indices_tr]
        elif self.data_split == 2: # testing split
            filenames = ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        elif self.data_split == -1: # all dataset
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr] \
                        + ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        return filenames




    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def set_random_resize_crop(self, resize=None):
        self._crop = 'random'
        self._resize_scale = resize

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):

        data_dir = self.root_dir

        if self._data_name == 'cub':
            key = self._image_fnames[raw_idx]
            bbox = self.bbox_meta[key]

            img_path = '%s/images/%s.jpg' % (data_dir, key)
            img = Image.open(img_path)

            # width, height = img.size
            #
            # if bbox is not None:
            #     r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            #     center_x = int((2 * bbox[0] + bbox[2]) / 2)
            #     center_y = int((2 * bbox[1] + bbox[3]) / 2)
            #     y1 = np.maximum(0, center_y - r)
            #     y2 = np.minimum(height, center_y + r)
            #     x1 = np.maximum(0, center_x - r)
            #     x2 = np.minimum(width, center_x + r)
            #
            # image = img.crop([x1, y1, x2, y2])
            # Crop the image

            image = img
            i, j, h, w = self._get_crop_params(image) # top, left, height, width
            image = torchvision.transforms.functional.resized_crop(image, i, j, h, w, self._size)

        elif self._data_name == 'dog':
            key = self.file_meta[raw_idx]
            img_path = '%s/%s_orig.png' % (data_dir, key)
            img = cv2.imread(img_path)
            image = Image.fromarray(img)
            # image = Image.open(img_path)

            # Crop the image
            i, j, h, w = self._get_crop_params(image) # top, left, height, width
            image = torchvision.transforms.functional.resized_crop(image, i, j, h, w, self._size)


        elif self._data_name == 'car':
            key = self.file_meta[raw_idx]
            img_path = '%s/%s_orig.png' % (data_dir, key)
            img = cv2.imread(img_path)
            image = Image.fromarray(img)
            # image = Image.open(img_path)

            # Crop the image
            i, j, h, w = self._get_crop_params(image) # top, left, height, width
            image = torchvision.transforms.functional.resized_crop(image, i, j, h, w, self._size)


        image = transforms.Resize((int(self._size), int(self._size)))(image)
        image = np.array(image)

        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        if image.shape[-1] == 1:
            image = np.concatenate([image, image, image], axis=-1)
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image


    def _get_crop_params(self, x):
        width, height = x.size

        if self._crop == 'center':
            #
            i, j = 0, 0
            if height > width * self._aspect_ratio: # height is longer than expected
                i = int((height - width * self._aspect_ratio) // 2)
                height = int(width * self._aspect_ratio)

            elif height < width * self._aspect_ratio:
                j = int((width - height / self._aspect_ratio) // 2)
                width = int(height / self._aspect_ratio)

        elif self._crop == 'random':
            scale = np.random.uniform(*self._resize_scale)
            ww, hh = int(scale * width), int(scale * height) # Scale
            ww = min(ww, hh / self._aspect_ratio) # Crop to aspect_ratio
            hh = min(hh, ww * self._aspect_ratio)
            i = int(np.random.uniform(0, height - hh))
            j = int(np.random.uniform(0, width -ww))
            height, width = hh, ww

        else:
            raise NotImplementedError

        return i, j, height, width


    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

if __name__ == '__main__':
    dataset_cub = DRCDataset(data_name='cub', size=64, path='cub')
    dataset_dog = DRCDataset(data_name='dog', size=64, path='dog')
    dataset_car = DRCDataset(data_name='car', size=64, path='car')
    dataset_1 = ImageFolderCropDataset(path='../data/fine_grained_categories/CUB_short170_train_10k.zip', size=64, aspect_ratio=1.0)
    t_1 = dataset_1[0]
    print('Debug')
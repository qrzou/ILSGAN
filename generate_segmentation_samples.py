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
from typing import List, Optional
import time
import json
import shutil
import pickle

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

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
    metric_jsonl = os.path.join(path, 'metric-fid10k_full.jsonl')
    if specified_pkl_idx is None and topk is None:
        with open(metric_jsonl, 'rt') as f:
            metrics = [json.loads(line) for line in f]

        if sigma is not None:
            metrics = [metric for metric in metrics if float(metric['sigma']) == sigma]
            if len(metrics) == 0:
                raise FileExistsError()

        minfid_idx = np.asarray([entry['results']['fid10k_full'] for entry in metrics]).argmin()
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

        minfid_idx = np.asarray([entry['results']['fid10k_full'] for entry in bag]).argmin()
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

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--pkl_idx', 'specified_pkl_idx', help='Specify the pkl to use', type=int, metavar='INT')
@click.option('--topk', help='only consider top k pkl to generate', type=int, metavar='INT')
@click.option('--sigma', help='Select the pkl with the sigma value', type=float)
@click.option('--n', help='How many number of images to be generated', type=int, metavar='INT')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    specified_pkl_idx,
    topk,
    sigma,
    n,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    start_time = time.time()

    network_pkl = get_pkl(network_pkl, specified_pkl_idx=specified_pkl_idx, topk=topk, sigma=sigma)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    ckpt_num = network_pkl.split('-')[-1].split('.')[0]

    if outdir is None:
        outdir = os.path.join(os.path.dirname(network_pkl), 'synthetic_data-' + ckpt_num)
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    if n is None:
        n = 10000
    if truncation_psi is None:
        truncation_psi = 1.0
    if noise_mode is None:
        noise_mode = 'const'

    # Generate images.
    batch_gpu = 16 * 4
    all_z = []
    all_c = []
    for base_i in range(0, n, batch_gpu):
        print(f'Finish generating {base_i} image ')
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

    total_time = time.time() - start_time
    print(f'Finish generating {n} image')
    print(f'total time: {dnnlib.util.format_time(total_time)}')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

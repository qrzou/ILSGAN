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
from analyze_whole_training_process import generate_data, do_segmentation


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


def auto_test(
    path,
    real_data,
    scale,
    specified_pkl_idx=None,
    topk=None,
    sigma=None,
    n=10000,
    truncation_psi=1.0,
    noise_mode='const',
    aug=None,
    real_data_abs_path=None,
):
    """
    Select several ckpt from Layered GAN  training process, generate synthetic datasets
    and test their segmentation performance
    :return:
    """
    # [Assertion]
    if path is None:
        raise Exception()
    if aug is not None:
        assert aug in ['geom', 'color', 'gc']
    assert real_data in ['cub', 'dog', 'car', 'lsuncar']

    # [Get select pkl]: prepare the ckpt list
    select_pkl = [get_pkl(path, specified_pkl_idx=specified_pkl_idx, topk=topk, sigma=sigma)]

    # [Generate synthetic data]
    start_time = time.time()
    synthetic_path = []
    for network_pkl in select_pkl:
        # Do generate.
        outdir = generate_data(path, network_pkl, n, truncation_psi, noise_mode, tag='auto_test')
        synthetic_path.append(outdir)
    total_time = time.time() - start_time
    print(f'total generating time: {dnnlib.util.format_time(total_time)}')

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
            run_dir = os.path.join(outdir, f'{cur_run_id:05d}-DRCreal-{real_data}-{aug}_aug')
        else:
            run_dir = os.path.join(outdir, f'{cur_run_id:05d}-DRCreal-{real_data}')

        if scale is not None:
            run_dir = run_dir + '-' + str(scale)

        if topk is not None:
            run_dir = run_dir + '-topk' + str(topk)

        assert not os.path.exists(run_dir)
        # Create output directory.
        print('Creating output directory...')
        os.makedirs(run_dir, exist_ok=True)

        dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
        results = do_segmentation(run_dir, syn_data, real_data, aug=aug, scale=scale, real_data_abs_path=real_data_abs_path)
        steps = syn_data.split('/')[-1].split('-')[-1]
        all_results[steps] = results[0]

    print('\nBest FID synthetic data segmentation results:\n', all_results)
    with open(os.path.join(synthetic_path[0], '..', 'best_fid_data_seg_results.json'), 'wt') as f:
        json.dump(all_results, f, indent=2)

    print('\n Auto test: generation and segmentation is done.\n')

    return synthetic_path[0]



#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--path', help='Network pickle filename', required=True)
@click.option('--real-data', help='Which dataset to evaluate on', required=True, type=str)
@click.option('--scale', help='Test img scale [default: 64]', type=int, metavar='INT')
@click.option('--pkl_idx', 'specified_pkl_idx', help='Specify the pkl to use', type=int, metavar='INT')
@click.option('--topk', help='only consider top k pkl to generate', type=int, metavar='INT')
@click.option('--sigma', help='Select the pkl with the sigma value', type=float)
@click.option('--n', help='How many number of images to be generated', type=int, metavar='INT')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--aug', help='Which augmentation to be used', type=str)
def call_auto_test(
        ctx: click.Context,
        path: str,
        real_data,
        scale,
        specified_pkl_idx,
        topk,
        sigma,
        n,
        truncation_psi: float,
        noise_mode: str,
        aug,
):
    if n is None:
        n = 10000
    if truncation_psi is None:
        truncation_psi = 1.0
    if noise_mode is None:
        noise_mode = 'const'


    print("\n=====> Start auto test <=====\n")
    auto_test(
        path=path,
        real_data=real_data,
        scale=scale,
        specified_pkl_idx=specified_pkl_idx,
        topk=topk,
        sigma=sigma,
        n=n,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
        aug=aug,
    )


#----------------------------------------------------------------------------

if __name__ == "__main__":
    call_auto_test()


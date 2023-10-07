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
import torchvision.transforms as torch_tf
from torchvision import utils
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import cv2
import psutil
import legacy
from torch_utils import misc, training_stats
from analyze_whole_training_process import generate_data, do_segmentation
from pixelcnnpp.utils import *
from pixelcnnpp.model import *


#----------------------------------------------------------------------------

class SynMIDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 indices=None,
                 size=None,
                 ):
        from glob import glob
        fg_filenames = sorted(glob(os.path.join(path, "fg", "*.png")))
        if indices is None:
            indices = range(len(fg_filenames))
        self.fg_filenames = [fg_filenames[i] for i in indices]
        self.bg_filenames = [os.path.join(path, 'bg', os.path.basename(filename)) for filename in self.fg_filenames]
        self.mask_filenames = [os.path.join(path, 'mask', os.path.basename(filename)) for filename in self.fg_filenames]

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
        return len(self.fg_filenames)

    def __getitem__(self, idx):
        assert os.path.basename(self.fg_filenames[idx]) == os.path.basename(self.bg_filenames[idx]) == os.path.basename(self.mask_filenames[idx])

        fg = Image.open(self.fg_filenames[idx]).convert("RGB")
        fg = self.transforms(fg)

        bg = Image.open(self.bg_filenames[idx]).convert("RGB")
        bg = self.transforms(bg)

        # standard.
        mask = Image.open(self.mask_filenames[idx]).convert("RGB")
        mask = self.transforms(mask)[0]  # softmask, just pick one channel
        mask = (mask + 1) / 2.0  # [-1,1] to [0,1]

        return fg, bg, mask


#----------------------------------------------------------------------------
def estimate_mi(
        run_dir,
        synthetic_path,

        # MI
        estimator           = 'club',

        # [ILS_form_config]
        # [two region]: also need two estimators
        do_separate     = True,
        sepmode     = 'vis2invis',  # vis2invis, invis2vis. default:vis2invis.

        # [single region]
        # config for do_mf mode.
        do_mf           = False,  # q(m*f|b)
        do_inv          = False,  # q((1-m)*b|f)
        mask_condition  = False,  # if True: q(s*x|y*(1-s))
        # config for layer to layer, will be used if do_separate and do_mf are False.
        layer2layer     = 'bg2fg',  # fg2bg, bg2fg. default: bg2fg

        # # Autoregressive Model
        obs                 = (3, 64, 64),  # input shape, default: (3, 64, 64)

        # Optimization
        batch               = 64,   # default: 64
        total_iter          = 100000,  # default: 100000
        lr                  = 0.0002, # default: 0.0002
        lr_decay            = 0.99,
        lr_step_ticks       = 1,  # after the num-interval's iters do the lr_scheduler.step()

        # Log and Save
        niter_per_tick          = 20,
        network_snapshot_ticks  = 2500,
        est_mi_trainset_ticks   = 20,
        est_mi_valset_ticks     = 20,

        # Others
        random_seed=0,
        cudnn_benchmark=True,
):
    assert not (do_mf and do_separate)
    # Print and Save arguments
    args = {
        'run_dir': run_dir,
        'synthetic_path': synthetic_path,

        'estimator': estimator,

        # [ILS_form_config]
        # [two region]: also need two estimators
        'do_separate': do_separate,
        'sepmode': sepmode,  # vis2invis, invis2vis. default:vis2invis.
        # [single region]
        # config for do_mf mode.
        'do_mf': do_mf,  # q(m*f|b)
        'do_inv': do_inv,  # q((1-m)*b|f)
        'mask_condition': mask_condition,  # if True: q(s*x|y*(1-s))
        # config for layer to layer, will be used if do_separate and do_mf are False.
        'layer2layer': layer2layer,  # fg2bg, bg2fg

        'obs': obs,

        'batch': batch,
        'total_iter': total_iter,
        'lr': lr,
        'lr_decay': lr_decay,
        'lr_step_ticks': lr_step_ticks,

        'niter_per_tick': niter_per_tick,
        'network_snapshot_ticks': network_snapshot_ticks,
        'est_mi_trainset_ticks': est_mi_trainset_ticks,
        'est_mi_valset_ticks': est_mi_valset_ticks,

        'random_seed': random_seed,
        'cudnn_benchmark': cudnn_benchmark,
    }
    print('\nOptions:\n', json.dumps(args, indent=2), '\n\n')

    ILS_form = dict(

        # [two region]: also need two estimators
        do_separate     = do_separate,
        sepmode     = sepmode,  # vis2invis, invis2vis. default:vis2invis.

        # [single region]
        # config for do_mf mode.
        do_mf           = do_mf,  # q(m*f|b)
        do_inv          = do_inv,  # q((1-m)*b|f)
        mask_condition  = mask_condition,  # if True: q(s*x|y*(1-s))
        # config for layer to layer, will be used if do_separate and do_mf are False.
        layer2layer     = layer2layer,  # fg2bg, bg2fg
    )

    with open(os.path.join(run_dir, f'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # train estimator on 9/10 dataset, validate on 1/10, test on all, 9/10 and 1/10.
    # [Initialization]
    start_time = time.time()
    device = torch.device('cuda')
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.


    # [Dataset]: prepare datasets
    whole_dataset = SynMIDataset(synthetic_path, size=obs[1])
    num_samples = len(whole_dataset)
    split_num = int(num_samples * 0.9)
    # MINE estimator doesn't need valset to prevent overfitting, so we use whole set to train.
    trainset = SynMIDataset(synthetic_path, range(num_samples), size=obs[1])
    trainset_sampler = misc.InfiniteSampler(dataset=trainset, seed=random_seed)
    valset = SynMIDataset(synthetic_path, range(split_num, num_samples), size=obs[1])

    # code for debug.
    # valset = SynMIDataset(synthetic_path, range(int(split_num * 0.1)), size=obs[1])

    data_loader_kwargs = dict(pin_memory=True, num_workers=6, prefetch_factor=4)
    trainset_4train_loader = torch.utils.data.DataLoader(dataset=trainset, sampler=trainset_sampler,
                                                         batch_size=batch, **data_loader_kwargs)
    trainset_4train_iterator = iter(torch.utils.data.DataLoader(dataset=trainset, sampler=trainset_sampler,
                                                         batch_size=batch, **data_loader_kwargs))

    # loaders for evaluation. bigger batch size could bring more accurate mi
    # [Enlarge the batch size for evaluation]
    trainset_4eval_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, drop_last=False, shuffle=False, **data_loader_kwargs)
    valset_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=128, drop_last=False, shuffle=False, **data_loader_kwargs)
    wholeset_loader = torch.utils.data.DataLoader(dataset=whole_dataset, batch_size=128, drop_last=False, shuffle=False, **data_loader_kwargs)


    # [Models]: Construct networks.
    # [Optimizer]: optimizer
    if do_separate:
        model1 = MINE(obs, hidden_size=3000).to(device)
        model2 = MINE(obs, hidden_size=3000).to(device)
        opt = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=lr)
    else:
        model = MINE(obs, hidden_size=3000).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    # [LR Scheduler] learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=lr_decay)
    # Use OnPlateau scheduler to prevent overfitting
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=10,
                                                              threshold=1e-4, threshold_mode='rel')  # maximize the MI

    # [Logs]: prepare logs
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

    # [Resume]: resume from existing pth.
    # if resume_from is not None:
    #     assert os.path.exists(resume_from) and os.path.isdir(resume_from)
    #     from glob import glob
    #     resume_pth = sorted(glob(os.path.join(resume_from, '*.pth')))[-1]
    #     cur_iter = resume(dict(model=model, opt=opt, lr_sch=lr_scheduler), resume_pth)

    # [Training]
    cur_iter = 0
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
        if do_separate:
            model1.train().requires_grad_(True)
            model2.train().requires_grad_(True)
        else:
            model.train().requires_grad_(True)

    while do_train:

        # Fetch data
        fg, bg, mask = next(trainset_4train_iterator)
        fg = fg.to(device)
        bg = bg.to(device)
        m = mask.to(device).unsqueeze(1)

        # Forward and backward
        opt.zero_grad()

        x_list, y_list = get_ILS_form_data(fg, bg, m, ILS_form)
        exist_abnormal = False
        if not do_separate:
            assert len(x_list) == len(y_list) == 1
            x = x_list[0]
            y = y_list[0]
            loss = model.learning_loss(x, y)
            if loss == 0:
                exist_abnormal = True

        else:
            assert len(x_list) == len(y_list) == 2
            loss = 0
            for i in range(len(x_list)):
                x = x_list[i]
                y = y_list[i]
                if i == 0:
                    loss_i = model1.learning_loss(x, y)
                    lossA = loss_i
                elif i == 1:
                    loss_i = model2.learning_loss(x, y)
                    lossB = loss_i

                if loss_i == 0:
                    exist_abnormal = True
                    break

                loss = loss + loss_i

            if not exist_abnormal:
                training_stats.report('Loss/loss1', lossA)
                training_stats.report('Loss/loss2', lossB)

        if exist_abnormal:
            print('Abnormal exists, skip optimization on this training batch!')
            continue

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
        # fields += [f"lr {training_stats.report0('Progress/lr', lr_scheduler.get_last_lr()[0]):<1.8f}"]
        fields += [f"lr {training_stats.report0('Progress/lr', opt.param_groups[0]['lr']):<1.8f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        print(' '.join(fields))

        # Learning rate scheduler
        if cur_tick % lr_step_ticks == 0 and isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
            lr_scheduler.step()

        # Save network snapshot.
        snapshot_data = None
        snapshot_pth = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            # snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            if do_separate:
                snapshot_data = {name: module.state_dict() for name, module in [('model1', model1), ('model2', model2), ('opt', opt), ('lr_sch', lr_scheduler)]}
            else:
                snapshot_data = {name: module.state_dict() for name, module in [('model', model), ('opt', opt), ('lr_sch', lr_scheduler)]}
            snapshot_data['cur_iter'] = cur_iter
            snapshot_pth = os.path.join(run_dir, f'network-snapshot-{cur_iter:08d}.pth')
            torch.save(snapshot_data, snapshot_pth)
            del snapshot_data  # conserve memory


        # Evaluate metrics.
        # MI train
        if (est_mi_trainset_ticks is not None) and (done or cur_tick % est_mi_trainset_ticks == 0):
            # Estimate MI using CLUB
            if do_separate:
                est_mi_train, est_mi_train2, mi_values_train, mi_values_train2 = compute_mi(model1, model2, device, trainset_4eval_loader, ILS_form)
                est_mi_train = est_mi_train.item()
                est_mi_train2 = est_mi_train2.item()
                result_dict = {
                    'MI_train1': est_mi_train,
                    'MI_train2': est_mi_train2,
                }
            else:
                est_mi_train, mi_values_train = compute_mi(model, None, device, trainset_4eval_loader, ILS_form)
                est_mi_train = est_mi_train.item()
                result_dict = {
                    'MI_train': est_mi_train,
                }

            report_metrics(result_dict, run_dir=run_dir, cur_iter=cur_iter, split='eval-MI-train')
            stats_metrics.update(result_dict)

            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if do_separate:
                    lr_scheduler.step(est_mi_train + est_mi_train2)
                else:
                    lr_scheduler.step(est_mi_train)

        # MI val
        if (est_mi_valset_ticks is not None) and (done or cur_tick % est_mi_valset_ticks == 0):
            # Estimate MI using CLUB
            if do_separate:
                est_mi_val, est_mi_val2, mi_values_val, mi_values_val2 = compute_mi(model1, model2, device, valset_loader, ILS_form)
                est_mi_val = est_mi_val.item()
                est_mi_val2 = est_mi_val2.item()
                result_dict = {
                    'MI_val1': est_mi_val,
                    'MI_val2': est_mi_val2,
                }
            else:
                est_mi_val, mi_values_val = compute_mi(model, None, device, valset_loader, ILS_form)
                est_mi_val = est_mi_val.item()
                result_dict = {
                    'MI_val': est_mi_val,
                }
            report_metrics(result_dict, run_dir=run_dir, cur_iter=cur_iter, split='eval-MI-val')
            stats_metrics.update(result_dict)

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

    # Evaluate the final outputs
    if do_separate:

        # Estimate MI using MINE
        est_mi_train, est_mi_train2, mi_values_train, mi_values_train2 = compute_mi(model1, model2, device, trainset_4eval_loader, ILS_form)
        est_mi_val, est_mi_val2, mi_values_val, mi_values_val2 = compute_mi(model1, model2, device, valset_loader, ILS_form)
        est_mi_whole, est_mi_whole2, mi_values_whole, mi_values_whole2 = compute_mi(model1, model2, device, wholeset_loader, ILS_form)

        result_dict = {
            'MI_train1': est_mi_train.item(),
            'MI_train2': est_mi_train2.item(),
            'MI_val1': est_mi_val.item(),
            'MI_val2': est_mi_val2.item(),
            'MI_whole1': est_mi_whole.item(),
            'MI_whole2': est_mi_whole2.item(),
            'final_lr': f"{opt.param_groups[0]['lr']:<1.8f}",
        }

    else:

        # Estimate MI using MINE
        est_mi_train, mi_values_train = compute_mi(model, None, device, trainset_4eval_loader, ILS_form)
        est_mi_val, mi_values_val = compute_mi(model, None, device, valset_loader, ILS_form)
        est_mi_whole, mi_values_whole = compute_mi(model, None, device, wholeset_loader, ILS_form)

        result_dict = {
            'MI_train': est_mi_train.item(),
            'MI_val': est_mi_val.item(),
            'MI_whole': est_mi_whole.item(),
            'final_lr': f"{opt.param_groups[0]['lr']:<1.8f}",
        }


    report_metrics(result_dict, run_dir=run_dir, cur_iter=cur_iter, split='eval-results')
    stats_metrics.update(result_dict)
    return result_dict


def get_ILS_form_data(fg, bg, m, ILS_form_config):
    """
    Convert fg and bg to ILS form.
    :param fg:
    :param bg:
    :param m:
    :param ILS_form_config:
    :return: x_list, y_list is the condition.
    """

    # [two region]: also need two estimators
    do_separate     = ILS_form_config['do_separate']
    sepmode     = ILS_form_config['sepmode']  # vis2invis, invis2vis. default:vis2invis.

    # [single region]
    # config for do_mf mode.
    do_mf           = ILS_form_config['do_mf']  # q(m*f|b)
    do_inv          = ILS_form_config['do_inv']  # q((1-m)*b|f)
    mask_condition  = ILS_form_config['mask_condition']  # if True: q(s*x|y*(1-s))
    # config for layer to layer, will be used if do_separate and do_mf are False.
    layer2layer     = ILS_form_config['layer2layer']  # fg2bg, bg2fg

    # get x and y list. y is the condition. q(x|y).
    if do_separate:  # 2 region
        fg_vis = fg * m
        bg_invis = bg * m
        fg_invis = fg * (1 - m)
        bg_vis = bg * (1 - m)
        if sepmode == 'invis2vis':  # use invisible region to predict visible region
            x_list = [fg_vis, bg_vis]
            y_list = [bg_invis, fg_invis]
        elif sepmode == 'vis2invis':  # inverse
            x_list = [bg_invis, fg_invis]
            y_list = [fg_vis, bg_vis]
        else:
            raise NotImplementedError('sepmode', sepmode)

    else:
        if do_mf:
            print('do_mf mode should be deprecated.')
            raise Exception()
            # the code below for do_mf mode actually can work, but we don't use it.
            # y is the condition.
            if do_inv:
                x = (1 - m) * bg
                y = m * fg if mask_condition else fg
            else:
                x = m * fg
                y = (1 - m) * bg if mask_condition else bg

        else:
            if layer2layer == 'fg2bg':  # q(b|f)
                x_list = [bg]
                y_list = [fg]
            elif layer2layer == 'bg2fg':  # q(f|b)
                x_list = [fg]
                y_list = [bg]
            else:
                raise NotImplementedError()

    return x_list, y_list


def compute_mi(net, net2, device, dataloader, ILS_form):
    """
    Compute the CLUB upper bound of mutual information
    :param net:
    :param device:
    :param dataloader:
    :return:
    """

    do_separate = ILS_form['do_separate']
    if not do_separate:
        assert net2 is None

    net.eval().requires_grad_(False)
    if do_separate:
        net2.eval().requires_grad_(False)

    mi_est_values = []
    mi_est_values2 = []
    with torch.no_grad():
        for batch_idx, (fg, bg, mask) in enumerate(dataloader):
            sample_size = fg.shape[0]
            random_index = torch.randperm(sample_size).long()

            fg = fg.to(device)
            bg = bg.to(device)
            m = mask.to(device).unsqueeze(1)

            x_list, y_list = get_ILS_form_data(fg, bg, m, ILS_form)

            if do_separate:
                assert len(x_list) == len(y_list) == 2
                for i in range(len(x_list)):
                    x = x_list[i]
                    y = y_list[i]
                    if i == 0:
                        mi = net(x, y)
                        mi_est_values.append(mi)
                    elif i == 1:
                        mi = net2(x, y)
                        mi_est_values2.append(mi)

            else:
                assert len(x_list) == len(y_list) == 1
                x = x_list[0]
                y = y_list[0]
                mi = net(x, y)
                mi_est_values.append(mi)

    net.train().requires_grad_(True)
    mi_lower_bound = torch.mean(torch.tensor(mi_est_values))
    if do_separate:
        net2.train().requires_grad_(True)
        mi_lower_bound2 = torch.mean(torch.tensor(mi_est_values2))
        return mi_lower_bound, mi_lower_bound2, mi_est_values, mi_est_values2

    return mi_lower_bound, mi_est_values


#----------------------------------------------------------------------------
def eval_MI(
    synthetic_path,
):
    """

    :return:
    """
    # [Assertion]
    if synthetic_path is None:
        raise Exception()

    # [Train Neural Estimator]
    # Pick output directory
    outdir = os.path.join(synthetic_path, 'MI-estimation-MINE')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}')
    assert not os.path.exists(run_dir)
    # Create output directory.
    print('Creating output directory...')
    os.makedirs(run_dir, exist_ok=True)


    # [Separate region]
    run_dir1 = os.path.join(run_dir, 'separate')
    os.makedirs(run_dir1, exist_ok=True)
    dnnlib.util.Logger(file_name=os.path.join(run_dir1, 'log.txt'), file_mode='a', should_flush=True)
    results = estimate_mi(
        run_dir1,
        synthetic_path,
        estimator='MINE',
        do_separate=True,
        sepmode='invis2vis',  # same as vis2invis when using MINE
        do_mf=False,
        do_inv=False,
        mask_condition=False,
    )
    final_lr_separate = results['final_lr']
    # print('\nFinal mi estimation results:\n', results)
    # results['run_id'] = f'MI-estimation-{cur_run_id:05d}'
    # with open(os.path.join(run_dir1, 'MI_estimation_results.json'), 'wt') as f:
    #     json.dump(results, f, indent=2)

    # compute average MI of last 20 evaluations on train
    train_MI_record = os.path.join(run_dir1, 'metric-eval-MI-train.jsonl')
    with open(train_MI_record, 'rt') as f:
        metrics = [json.loads(line) for line in f]
    MI_train1_list = []
    MI_train2_list = []
    for metric in metrics[-20:]:  # last 20
        MI_train1_list.append(metric['MI_train1'])
        MI_train2_list.append(metric['MI_train2'])
    ave_MI_train1 = np.mean(MI_train1_list)
    ave_MI_train2 = np.mean(MI_train2_list)
    results = {
        'ave_MI_train1': ave_MI_train1,
        'ave_MI_train2': ave_MI_train2,
    }
    print('\nFinal mi estimation results, '
          'average MI of last 20 evaluations on train:\n', results)
    with open(os.path.join(run_dir1, 'MI_estimation_results.json'), 'wt') as f:
        json.dump(results, f, indent=2)



    # [Single Region: Whole Layer]
    run_dir2 = os.path.join(run_dir, 'single')
    os.makedirs(run_dir2, exist_ok=True)
    dnnlib.util.Logger(file_name=os.path.join(run_dir2, 'log.txt'), file_mode='a', should_flush=True)
    results = estimate_mi(
        run_dir2,
        synthetic_path,
        estimator='MINE',
        do_separate=False,
        sepmode='invis2vis',
        do_mf=False,
        do_inv=False,
        mask_condition=False,
    )
    final_lr_single = results['final_lr']
    # print('\nFinal mi estimation results:\n', results)
    # results['run_id'] = f'MI-estimation-{cur_run_id:05d}'
    # with open(os.path.join(run_dir2, 'MI_estimation_results.json'), 'wt') as f:
    #     json.dump(results, f, indent=2)

    # compute average MI of last 20 evaluations on train
    train_MI_record = os.path.join(run_dir2, 'metric-eval-MI-train.jsonl')
    with open(train_MI_record, 'rt') as f:
        metrics = [json.loads(line) for line in f]
    MI_train_list = []
    for metric in metrics[-20:]:  # last 20
        MI_train_list.append(metric['MI_train'])
    ave_MI_train = np.mean(MI_train_list)
    results = {
        'ave_MI_train': ave_MI_train,
    }
    print('\nFinal mi estimation results, '
          'average MI of last 20 evaluations on train:\n', results)
    with open(os.path.join(run_dir2, 'MI_estimation_results.json'), 'wt') as f:
        json.dump(results, f, indent=2)


    print('\n MI estimation is done.\n')
    print('[ Separate ]:', ave_MI_train1, ave_MI_train2, '==> Final lr:', final_lr_separate)
    print('[ Single ]:', ave_MI_train, '==> Final lr:', final_lr_single)


#----------------------------------------------------------------------------
def report_metrics(result_dict, run_dir=None, cur_iter=None, split='val'):
    jsonl_line = json.dumps(dict(result_dict, cur_iter=cur_iter, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{split}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')


#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--path', help='Network pickle filename', required=True)
def call_eval_MI(
        ctx: click.Context,
        path: str,
):
    print("\n=====> Start MI Evaluation <=====\n")
    eval_MI(
        synthetic_path=path,
    )


#----------------------------------------------------------------------------

if __name__ == "__main__":
    call_eval_MI()


# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import copy
import json
import pickle
import psutil
import functools
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import dnnlib
import torchvision
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from metrics.validate_lyr_score import validate_lyr_score

import legacy
from metrics import metric_main

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def save_layer_grid(layer_list, fname, drange, grid_size):
    # remove the empty layer
    layer_list = [layer for layer in layer_list if layer]

    for layer in layer_list:
        for name, img in layer.items():
            lo, hi = drange if img.shape[1] == 3 else (0, 1)
            img = np.asarray(img, dtype=np.float32)
            img = (img - lo) * (255 / (hi - lo))
            if img.shape[1] == 1:
                img = np.tile(img, (1, 3, 1, 1))
            img = np.rint(img).clip(0, 255).astype(np.uint8)
            layer[name] = img

    def pre_reshape(gh, gw, x):
        x = x[:gh * gw]
        _N, C, H, W = x.shape
        x = x.reshape(gh, gw, C, H, W)
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape(gh * H, gw, 1, 1, W, C)
        return x

    gw, gh = grid_size
    gw = 2
    ordered = ["bg", "fg", "mask", "fg*mask", "img"]
    img = [np.concatenate([pre_reshape(gh, gw, layer[k]) for k in ordered], axis=2) for layer in layer_list]
    img = np.concatenate(img, axis=3)
    img = img.reshape(*img.shape[:1], -1, *img.shape[-1:])

    assert img.shape[-1] == 3
    PIL.Image.fromarray(img, 'RGB').save(fname)

def save_grad_layer_grid(layer_list, fname, drange, grid_size):
    # remove the empty layer
    layer_list = [layer for layer in layer_list if layer]

    for layer in layer_list:
        for name, img in layer.items():
            if name == "img_grad" or name == 'mask_grad':
                # drop the outliers, get the 5%-th as lo and 95%-th as hi
                lo, hi = np.percentile(img.numpy(), 5.), np.percentile(img.numpy(), 95.)

            else:
                # Note: other images in layer are already postprocessed by save_layer_grid !!!
                continue
                # lo, hi = drange if img.shape[1] == 3 else (0, 1)

            img = np.asarray(img, dtype=np.float32)
            img = (img - lo) * (255 / (hi - lo))
            if img.shape[1] == 1:
                img = np.tile(img, (1, 3, 1, 1))
            img = np.rint(img).clip(0, 255).astype(np.uint8)
            layer[name] = img

    def pre_reshape(gh, gw, x):
        x = x[:gh * gw]
        _N, C, H, W = x.shape
        x = x.reshape(gh, gw, C, H, W)
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape(gh * H, gw, 1, 1, W, C)
        return x

    gw, gh = grid_size
    gw = 2
    ordered = ["bg", "fg", "mask", "mask_grad", "fg*mask", "img_grad", "img"]
    img = [np.concatenate([pre_reshape(gh, gw, layer[k]) for k in ordered], axis=2) for layer in layer_list]
    img = np.concatenate(img, axis=3)
    img = img.reshape(*img.shape[:1], -1, *img.shape[-1:])

    assert img.shape[-1] == 3
    PIL.Image.fromarray(img, 'RGB').save(fname)


#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    Autoreg_kwargs          = {},       # Options for autoregressive models for mi estimation.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    if rank == 0:
        print('\ndebug\n')


    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Check the cache of FID and KID evaluation
    cache_path = os.path.join(os.environ['HOME'], '.cache/dnnlib/gan-metrics')
    if os.path.exists(cache_path) and rank == 0:
        cache_files = os.listdir(cache_path)
        dataset_name = training_set.name
        exist = False
        for cache in cache_files:
            if dataset_name in cache:
                exist = True
        if exist:
            print('\n####################################################################'
                  '\n======> Warning! The inception cache of dataset is existed!!'
                  '\n======> If the preprocess of dataset is not modified, ignore'
                  '\n======> the warning, or the cache must be deleted manually!'
                  '\n####################################################################')
            print('\nExisting caches:\n', cache_files, '\n')

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    if (not loss_kwargs['do_anti_leaking']) or \
            loss_kwargs['ILS_form_config']['estimator'] == 'l1_distance' or \
            loss_kwargs['ILS_form_config']['estimator'] == 'l1_distance_allowgrad' or \
            loss_kwargs['ILS_form_config']['distribution'] == 'predetermined_gaussian' or \
            loss_kwargs['ILS_form_config']['distribution'] == 'predetermined_laplace':
        Autoreg = None
        if rank == 0 and not loss_kwargs['do_anti_leaking']:
            print('\nNo ILS\n')
        if rank == 0 and (loss_kwargs['ILS_form_config']['estimator'] == 'l1_distance' or
                          loss_kwargs['ILS_form_config']['estimator'] == 'l1_distance_allowgrad'):
            if loss_kwargs['ILS_form_config']['estimator'] == 'l1_distance_allowgrad':
                print('\nMaximize L1 Distance between Fg and Bg, allow visible region grad backward\n')
            else:
                print('\nMaximize L1 Distance between Fg and Bg, stop grad in visible region\n')

        if rank == 0 and loss_kwargs['ILS_form_config']['distribution'] == 'predetermined_laplace' and loss_kwargs['ILS_form_config']['estimator'] == 'CLUB':
            print('\nDoing ILS with predetermined laplace distribution CLUB\n')
    else:
        if rank == 0:
            print('Auto regression model is applied to CLUB MI estimator now!!!! \n')
        Autoreg = dnnlib.util.construct_class_by_name(**Autoreg_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module

    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('Autoreg', Autoreg)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        layer = misc.print_module_summary(G, [z, c], in_kwargs=dict(return_layers=True))
        feature, _ = misc.print_module_summary(D, [layer["img"], c], in_kwargs=dict(get_feature=True))
        if Autoreg is not None:
            if Autoreg.num_autoreg == 1:
                pred_prob_params = misc.print_module_summary(Autoreg, [[layer['bg']]], in_kwargs=dict(condition_input_list=[layer['fg']]))
            elif Autoreg.num_autoreg == 2:
                pred_prob_params = misc.print_module_summary(Autoreg, [[layer['bg'], layer['bg']]], in_kwargs=dict(condition_input_list=[layer['fg'], layer['fg']]))
            else:
                raise NotImplementedError()
            del pred_prob_params
        del z, c, layer, feature

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [
        ('G_mapping_bg', G.mapping_bg), ('G_synthesis_bg', G.synthesis_bg),
        ('G_mapping_fg', G.mapping_fg), ('G_synthesis_fg', G.synthesis_fg),
        ('Tx', G.Tx), ('D', D), (None, G_ema), ('augment_pipe', augment_pipe),
        ('Autoreg', Autoreg)
    ]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            if name == 'Autoreg':
                module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    torch.cuda.empty_cache()
    loss_kwargs["anti_zero_mask"] = "mi" if G.c_dim > 0 else "msize" # [NEW] detect to use msize or mi to prevent zero mask
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss
    # If lazy regularization is enabled, the phases include Gmain, Greg, Dmain, Dreg
    phases = []
    for name, module, opt_kwargs, reg_interval, extra_module in [
        ('G', G, G_opt_kwargs, G_reg_interval, None),
        ('D', D, D_opt_kwargs, D_reg_interval, Autoreg),  # let the Autoreg module be optimized with D together
    ]:
        if extra_module is not None:
            params = list(module.parameters()) + list(extra_module.parameters())
        else:
            params = module.parameters()

        if reg_interval is None: # lazy regularization is disabled
            opt = dnnlib.util.construct_class_by_name(params=params, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1, extra_module=extra_module)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(params, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1, extra_module=extra_module)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval, extra_module=extra_module)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    grid_eps = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        # expt_batch_gpu = batch_gpu // 2 if G.Tw is not None else batch_gpu
        expt_batch_gpu = batch_gpu
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(expt_batch_gpu)
        if G.c_dim > 0:
            grid_c = torch.nn.functional.one_hot(torch.randint(G.c_dim, [labels.shape[0],], device=device), G.c_dim).split(expt_batch_gpu)
        else:
            grid_c = torch.from_numpy(labels).to(device).split(expt_batch_gpu)
        grid_eps = 2. * torch.rand(labels.shape[0], G.eps_dim, device=device) - 1. # Uniform[-1, 1]
        grid_eps = grid_eps.split(expt_batch_gpu)
        # depracate the previous
        # images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        # save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
        # [NEW] generate, collect, and save layers
        layer, t_layer, e_layer = None, None, None
        for z, c, eps in zip(grid_z, grid_c, grid_eps):
            _layer, _t_layer, _e_layer = G_ema.generate_layer(z=z, c=c, eps=eps, noise_mode='const')
            if layer is None and t_layer is None and e_layer is None:
                layer = {k: v.cpu() for k, v in _layer.items()}
                t_layer = {k: v.cpu() for k, v in _t_layer.items()}
                e_layer = {k: v.cpu() for k, v in _e_layer.items()}
            else:
                layer = {k: torch.cat([layer[k], _layer[k].cpu()]) for k in layer.keys()}
                t_layer = {k: torch.cat([t_layer[k], _t_layer[k].cpu()]) for k in t_layer.keys()}
                e_layer = {k: torch.cat([e_layer[k], _e_layer[k].cpu()]) for k in e_layer.keys()}
        save_image_grid(layer["img"], os.path.join(run_dir, 'fakes_init.png'), drange=[-1, 1], grid_size=grid_size)
        save_image_grid(layer["mask"], os.path.join(run_dir, 'masks_init.png'), drange=[0, 1], grid_size=grid_size)
        save_layer_grid([layer, t_layer, e_layer], os.path.join(run_dir, 'layers_init.png'), drange=[-1, 1], grid_size=grid_size)
    torch.cuda.empty_cache()

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0

    do_anti_leaking = False

    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            # deprecate the previous
            # all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            # all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            # all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
            # [NEW]
            if G.c_dim > 0:
                all_gen_c = torch.nn.functional.one_hot(torch.randint(G.c_dim, [len(phases) * batch_size,], device=device), G.c_dim)
                all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
            else:
                all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
                all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
                all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
            # [NEW] for transformation
            all_trs_eps = 2 * torch.rand([len(phases) * batch_size, G.eps_dim], device=device) - 1 # Uniform[-1, 1]
            all_trs_eps = [phase_trs_eps.split(batch_gpu) for phase_trs_eps in all_trs_eps.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c, phase_trs_eps in zip(phases, all_gen_z, all_gen_c, all_trs_eps):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)

            phase.module.requires_grad_(True)
            if phase.extra_module is not None:
                phase.extra_module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c, trs_eps) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c, phase_trs_eps)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c,
                                          trs_eps=trs_eps, sync=sync, gain=gain, cur_kimg=(cur_nimg//1000), do_anti_leaking=do_anti_leaking)

            # Update weights.
            phase.module.requires_grad_(False)
            if phase.extra_module is not None:
                phase.extra_module.requires_grad_(False)

            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

                if phase.extra_module is not None:
                    for param in phase.extra_module.parameters():
                        if param.grad is not None:
                            misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

                phase.opt.step()

            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1
        loss.cur_nimg = cur_nimg # Update the cur_nimg tracking in loss (Quick update)

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            # deprecate the previous
            # images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            # save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            # [NEW] generate, collect, and save layers
            layer, t_layer, e_layer, img_grad_viz, mask_grad_viz = None, None, None, None, None
            for z, c, eps in zip(grid_z, grid_c, grid_eps):
                _layer, _t_layer, _e_layer = G_ema.generate_layer(z=z, c=c, eps=eps, noise_mode='const')

                # Visualize the gradients
                G.requires_grad_(True)
                img_grad, mask_grad = loss.get_gradients_of_gen_img(gen_z=z, gen_c=c, trs_eps=eps)
                G.requires_grad_(False)
                G.zero_grad(set_to_none=True)
                # convert grad to heatmap or 0-255.
                # grayscale to heatmap: https://stackoverflow.com/questions/59478962/how-to-convert-a-grayscale-image-to-heatmap-image-with-python-opencv
                # we do the normalization in save_grad_layer_grid() function
                # note: if the do_Gmain in training is changed, the gradient we get
                # may not correspond to the gradient in training
                abs_grad = True
                if abs_grad:
                    img_grad = torch.abs(img_grad)
                    mask_grad = torch.abs(mask_grad)
                else:
                    raise NotImplementedError

                if layer is None and t_layer is None and e_layer is None and img_grad_viz is None and mask_grad_viz is None:
                    layer = {k: v.cpu() for k, v in _layer.items()}
                    t_layer = {k: v.cpu() for k, v in _t_layer.items()}
                    e_layer = {k: v.cpu() for k, v in _e_layer.items()}
                    img_grad_viz = img_grad.cpu()
                    mask_grad_viz = mask_grad.cpu()
                else:
                    layer = {k: torch.cat([layer[k], _layer[k].cpu()]) for k in layer.keys()}
                    t_layer = {k: torch.cat([t_layer[k], _t_layer[k].cpu()]) for k in t_layer.keys()}
                    e_layer = {k: torch.cat([e_layer[k], _e_layer[k].cpu()]) for k in e_layer.keys()}
                    img_grad_viz = torch.cat([img_grad_viz, img_grad.cpu()])
                    mask_grad_viz = torch.cat([mask_grad_viz, mask_grad.cpu()])

            save_image_grid(layer["img"], os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1, 1], grid_size=grid_size)
            save_image_grid(layer["mask"], os.path.join(run_dir, f'masks{cur_nimg//1000:06d}.png'), drange=[0, 1], grid_size=grid_size)
            save_layer_grid([layer, t_layer, e_layer], os.path.join(run_dir, f'layers{cur_nimg//1000:06d}.png'), drange=[-1, 1],
                            grid_size=grid_size)
            layer["img_grad"] = img_grad_viz
            layer["mask_grad"] = mask_grad_viz
            save_grad_layer_grid([layer], os.path.join(run_dir, f'grads{cur_nimg//1000:06d}.png'), drange=[-1, 1],
                                 grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')

            # Individually save the Autoreg with torch.save,
            # because Autoreg dose not support deepcopy
            if num_gpus > 1 and Autoreg is not None:
                misc.check_ddp_consistency(Autoreg, ignore_regex=r'.*\.num_batches_tracked')

            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
                if Autoreg is not None:
                    torch.save(Autoreg, os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}-Autoreg.pt'))

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl, sigma=loss.sigma)
                stats_metrics.update(result_dict.results)

            # [Validate lyr score]
            anti_leaking_by_score = True
            if anti_leaking_by_score:
                fg, bg, img = validate_lyr_score(snapshot_data['G_ema'], snapshot_data['D'], device=device, mode='mean')
                do_anti_leaking = True if bg > img else False
                del fg, bg, img

        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        # loss.cur_nimg = cur_nimg # TODO: Update the cur_nimg tracking in loss per tick (Slow update)
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

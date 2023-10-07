# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from collections import OrderedDict
import numpy as np
import re
import math
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from pixelcnnpp.utils import get_sample_log_prob, discretized_mix_logistic_loss
import torchvision
import json


#----------------------------------------------------------------------------
# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3, padding=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) / \
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, padding=padding,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()


#----------------------------------------------------------------------------

class ILSLoss(Loss):
    def __init__(self, device,
                 G_mapping_bg, G_synthesis_bg, G_mapping_fg, G_synthesis_fg,
                 Tx, D, augment_pipe=None, augment_pipe_bg=None, augment_pipe_color=None, Autoreg=None,
                 anti_zero_mask="mi", updateE='G', min_msize=0.25, msize_weight=2.0, bin_weight=2.0, mi_weight=1.0, t1kimg=2000,
                 mixing='orig', style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
                 do_perturb=True,
                 do_anti_leaking=False, ILS_weight_strategy={}, ILS_form_config={},
                 # ILS_sigma=1.0, initial_anti=True, initial_anti_sigma=0.2, strategy='all_time',
                 ):
        super().__init__()
        self.device = device
        self.G_mapping_bg = G_mapping_bg
        self.G_synthesis_bg = G_synthesis_bg
        self.G_mapping_fg = G_mapping_fg
        self.G_synthesis_fg = G_synthesis_fg
        self.D = D
        self.Tx = Tx
        self.Autoreg = Autoreg

        self.augment_pipe = augment_pipe
        self.augment_pipe_bg = augment_pipe_bg
        self.augment_pipe_color = augment_pipe_color

        self.mixing = mixing
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        # build perceptual loss if equivariance regularization is enabled
        self.cur_nimg = 0
        self.t1kimg = t1kimg

        # Use mutual information loss or mask size loss to prevent zero mask
        self.updateE = updateE
        self.anti_zero_mask = anti_zero_mask
        self.min_msize = min_msize

        # loss weights
        self.sigma = 0
        self.weights = {
            "msize": msize_weight if self.anti_zero_mask == "msize" else 0.,
            "bin": bin_weight,      # mask binarization
            "mi": mi_weight,        # mutual information maximization
        }


        # [Anti leaking]
        self.do_perturb = do_perturb
        if not do_perturb:
            print('\n####################################################'
                  '\n======> Warning! Perturbation is disabled!'
                  '\n####################################################')
        self.do_anti_leaking = do_anti_leaking
        self.ILS_weight_strategy = ILS_weight_strategy
        self.ILS_form_config = ILS_form_config
        assert ILS_weight_strategy['strategy'] in ['all_time', 'cool_down', 'cos']

        self.ILS_config = {
            'do_perturb': do_perturb,
            'do_anti_leaking': do_anti_leaking,
            'ILS_weight_strategy': ILS_weight_strategy,
            'ILS_form_config': ILS_form_config,
        }
        print('\nILS config\n', json.dumps(self.ILS_config, indent=2), '\n\n')

        if ILS_form_config['neg_weight'] != 1:
            print("\n===> Warning: ILS negative term's weight is not 1!\n")


    def get_w_avg(self, layer_name):
        module = getattr(self, f'G_mapping_{layer_name}')
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        return module.w_avg

    def run_G(self, z, c, sync):
        # produce w
        with misc.ddp_sync(self.G_mapping_bg, sync):
            ws_bg = self.G_mapping_bg(z, None)
        with misc.ddp_sync(self.G_mapping_fg, sync):
            ws_fg = self.G_mapping_fg(z, c)

        # mixing regularization
        with torch.autograd.profiler.record_function('mixing'):
            if self.mixing == 'orig':
                # we use separate cutoff to handle the possibility that num_ws_bg != num_ws_fg
                another_z = torch.randn_like(z)
                cutoff_bg = torch.empty([], dtype=torch.int64, device=z.device).random_(1, ws_bg.shape[1])
                cutoff_bg = torch.where(torch.rand([], device=z.device) < self.style_mixing_prob, cutoff_bg,
                                        torch.full_like(cutoff_bg, ws_bg.shape[1]))
                with misc.ddp_sync(self.G_mapping_bg, sync):
                    ws_bg[:, cutoff_bg:] = self.G_mapping_bg(another_z, None, skip_w_avg_update=True)[:, cutoff_bg:]
                cutoff_fg = torch.empty([], dtype=torch.int64, device=z.device).random_(1, ws_fg.shape[1])
                cutoff_fg = torch.where(torch.rand([], device=z.device) < self.style_mixing_prob, cutoff_fg,
                                        torch.full_like(cutoff_fg, ws_fg.shape[1]))
                with misc.ddp_sync(self.G_mapping_fg, sync):
                    ws_fg[:, cutoff_fg:] = self.G_mapping_fg(another_z, c, skip_w_avg_update=True)[:, cutoff_fg:]
            elif self.mixing == 'style':
                another_z = torch.randn_like(z)
                n_style = ws_bg.shape[1] + ws_fg.shape[1]
                cutoff = torch.empty([], dtype=torch.int64, device=z.device).random_(1, n_style)
                cutoff = torch.where(torch.rand([], device=z.device) < self.style_mixing_prob, cutoff,
                                     torch.full_like(cutoff, n_style))
                # we need to index original tensor to replace style, otherwise it might cause sync issues
                with misc.ddp_sync(self.G_mapping_bg, sync):
                    # the following is safe. if cut_off >= ws_bg.shape[1], it will return 0 dim tensor
                    ws_bg[:, cutoff:] = self.G_mapping_bg(another_z, None, skip_w_avg_update=True)[:, cutoff:]
                with misc.ddp_sync(self.G_mapping_fg, sync):
                    cutoff_fg = max(cutoff - ws_bg.shape[1], 0)
                    ws_fg[:, cutoff_fg:] = self.G_mapping_fg(another_z, c, skip_w_avg_update=True)[:, cutoff_fg:]
            elif self.mixing == 'layer':
                if torch.rand([], device=z.device) < self.style_mixing_prob:
                    # replace bg which is equivalent to replacing fg
                    another_z = torch.randn_like(z)
                    with misc.ddp_sync(self.G_mapping_bg, sync):
                        ws_bg[:] = self.G_mapping_bg(another_z, None, skip_w_avg_update=True)[:]
            else:
                raise NotImplementedError()

        # bg
        with misc.ddp_sync(self.G_synthesis_bg, sync):
            bg_rgb = self.G_synthesis_bg(ws_bg)

        # fg
        with misc.ddp_sync(self.G_synthesis_fg, sync):
            fg_rgba = self.G_synthesis_fg(ws_fg)
            fg_rgb = fg_rgba[:, :3]
            fg_a = fg_rgba[:, 3:].sigmoid()
            raw_fg_a = fg_rgba[:, 3:]

        layer = {
            "bg": bg_rgb,
            "fg": fg_rgb,
            "mask": fg_a,
            # 'raw_mask': raw_fg_a,
            "fg*mask": fg_a * fg_rgb,
            "img": (1. - fg_a) * bg_rgb + fg_a * fg_rgb
        }
        ws = {
            "bg": ws_bg,
            "fg": ws_fg
        }
        return layer, ws

    def run_Gw(self, ws, sync):
        # bg
        with misc.ddp_sync(self.G_synthesis_bg, sync):
            bg_rgb = self.G_synthesis_bg(ws['bg'])

        # fg
        with misc.ddp_sync(self.G_synthesis_fg, sync):
            fg_rgba = self.G_synthesis_fg(ws['fg'])
            fg_rgb = fg_rgba[:, :3]
            fg_a = fg_rgba[:, 3:].sigmoid()

        layer = {
            "bg": bg_rgb,
            "fg": fg_rgb,
            "mask": fg_a,
            "fg*mask": fg_a * fg_rgb,
            "img": (1. - fg_a) * bg_rgb + fg_a * fg_rgb
        }
        return layer

    def run_Tx(self, layer, eps):
        return self.Tx(layer, eps)

    def run_Tw(self, ws, eps, sync):
        with misc.ddp_sync(self.Tw, sync):
            trs_ws = self.Tw(ws, eps)
        return trs_ws

    def run_D(self, img, c, get_feature, sync, run_E=False, detachE=False):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            if run_E:
                d_out = self.D(img, c, get_feature=get_feature, run_E=run_E, detachE=detachE)
            else:
                d_out = self.D(img, c, get_feature=get_feature)
        return d_out

    def run_D_blitgeom_aug(self, img, c, get_feature, sync):
        # Note, the input img should be color perturbed
        if self.augment_pipe_bg is None:
            raise AssertionError()
        img = self.augment_pipe_bg(img)

        with misc.ddp_sync(self.D, sync):
            d_out = self.D(img, c, get_feature=get_feature)
        return d_out


    def get_logits_from_D(self, img, c, get_feature, sync):
        # compare to run_D, this function won't use the augment pipe
        with misc.ddp_sync(self.D, sync):
            d_out = self.D(img, c, get_feature=get_feature)
        return d_out

    def get_l1_norm(self, x, y, do_mean=False):
        # x and y: N,C,H,W
        # return: N, 1
        assert x.shape[0] == y.shape[0]
        assert len(x.shape) == len(y.shape) == 4
        N = x.shape[0]
        x = x.reshape(N, -1)
        y = y.reshape(N, -1)
        if do_mean:
            norm = (x - y).abs().mean(dim=[1])  # use mean() to down grade the scale of the loss.
        else:
            norm = (x - y).abs().sum(dim=[1])
        return norm

    def get_L2_SE(self, x, y, do_mean=False):
        # x and y: N,C,H,W
        # return: N, 1
        assert x.shape[0] == y.shape[0]
        assert len(x.shape) == len(y.shape) == 4
        N = x.shape[0]
        x = x.reshape(N, -1)
        y = y.reshape(N, -1)
        if do_mean:
            SE = ((x - y) ** 2).mean(dim=[1])  # use mean() to down grade the scale of the loss.
        else:
            SE = ((x - y) ** 2).sum(dim=[1])
        return SE

    def compute_club_mi(self, x, y, distribution, scale, neg_weight=1):
        """
        Compute the MI(x, y)
        :param x:
        :param y: the condition variable
        :param distribution:
        :param scale:
        :param neg_weight: used in predetermined_laplace
        :return:
        """
        do_mean = (scale == 'do_mean')
        assert len(x) == len(y)
        upper_bound_list = []

        if distribution == 'predetermined_laplace':
            for i in range(len(x)):
                sample_size = x[i].shape[0]
                random_index = torch.randperm(sample_size).long()
                positive = self.get_l1_norm(x[i], y[i], do_mean)
                negative = self.get_l1_norm(x[i][random_index], y[i], do_mean)
                upper_bound = - positive.mean() + negative.mean() * neg_weight
                upper_bound_list.append(upper_bound)

        elif distribution == 'predetermined_gaussian':
            for i in range(len(x)):
                sample_size = x[i].shape[0]
                random_index = torch.randperm(sample_size).long()
                positive = self.get_L2_SE(x[i], y[i], do_mean)
                negative = self.get_L2_SE(x[i][random_index], y[i], do_mean)
                upper_bound = - positive.mean() + negative.mean() * neg_weight
                upper_bound = 0.5 * upper_bound  # add coefficient
                upper_bound_list.append(upper_bound)

        elif distribution == 'neural_laplace':
            # we can not visit the attribute of module when using DDP
            # assert len(x) == self.Autoreg.num_autoreg
            pos_x_list = x
            pos_y_list = y  # conditional variable
            neg_x_list = []
            for i in range(len(x)):
                sample_size = x[i].shape[0]
                random_index = torch.randperm(sample_size).long()
                neg_x_list.append(pos_x_list[i][random_index])

            # pos_x_list actually won't be used in neural gaussian. We use it just for coding convenience.
            pred_x_param_list = self.Autoreg(pos_x_list, condition_input_list=pos_y_list)

            resolution = pred_x_param_list[0][0].shape[2]  # autoreg use small resolution for saving computation resources
            for i in range(len(x)):
                mu, logb = pred_x_param_list[i]
                # Reshape to vectors.
                sample_size = mu.shape[0]
                mu = mu.reshape(sample_size, -1)
                logb = logb.reshape(sample_size, -1)

                pos_x = F.interpolate(pos_x_list[i], size=resolution)
                neg_x = F.interpolate(neg_x_list[i], size=resolution)
                pos_x = pos_x.reshape(sample_size, -1)
                neg_x = neg_x.reshape(sample_size, -1)

                positive = - torch.abs(pos_x - mu) / logb.exp()
                negative = - torch.abs(neg_x - mu) / logb.exp()

                upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

                # club-sample upper bound
                upper_bound_list.append(upper_bound / 2)

        elif distribution == 'neural_gaussian':
            # we can not visit the attribute of module when using DDP
            # assert len(x) == self.Autoreg.num_autoreg
            pos_x_list = x
            pos_y_list = y  # conditional variable
            neg_x_list = []
            for i in range(len(x)):
                sample_size = x[i].shape[0]
                random_index = torch.randperm(sample_size).long()
                neg_x_list.append(pos_x_list[i][random_index])

            # pos_x_list actually won't be used in neural gaussian. We use it just for coding convenience.
            pred_x_param_list = self.Autoreg(pos_x_list, condition_input_list=pos_y_list)

            resolution = pred_x_param_list[0][0].shape[2]  # autoreg use small resolution for saving computation resources
            for i in range(len(x)):
                mu, logvar = pred_x_param_list[i]
                # Reshape to vectors.
                sample_size = mu.shape[0]
                mu = mu.reshape(sample_size, -1)
                logvar = logvar.reshape(sample_size, -1)

                pos_x = F.interpolate(pos_x_list[i], size=resolution)
                neg_x = F.interpolate(neg_x_list[i], size=resolution)
                pos_x = pos_x.reshape(sample_size, -1)
                neg_x = neg_x.reshape(sample_size, -1)

                positive = - (mu - pos_x)**2 / logvar.exp()
                negative = - (mu - neg_x)**2 / logvar.exp()
                upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

                # club-sample upper bound
                upper_bound_list.append(upper_bound / 2)

        elif distribution == 'pixelcnnpp':
            # we can not visit the attribute of module when using DDP
            # assert len(x) == self.Autoreg.num_autoreg
            pos_x_list = x
            pos_y_list = y
            neg_x_list = []
            for i in range(len(x)):
                sample_size = x[i].shape[0]
                random_index = torch.randperm(sample_size).long()
                neg_x_list.append(pos_x_list[i][random_index])

            # positive log prob
            pos_pred_x_param_list = self.Autoreg(pos_x_list, condition_input_list=pos_y_list, test=True)
            # negative log prob
            neg_pred_x_param_list = self.Autoreg(neg_x_list, condition_input_list=pos_y_list, test=True)
            pos_sample_log_prob_list = []
            neg_sample_log_prob_list = []
            resolution = pos_pred_x_param_list[0].shape[2]  # autoreg use small resolution for saving computation resources
            for i in range(len(x)):

                pos_sample_log_prob = get_sample_log_prob(
                    F.interpolate(pos_x_list[i], size=resolution), pos_pred_x_param_list[i])
                pos_sample_log_prob_list.append(pos_sample_log_prob)

                neg_sample_log_prob = get_sample_log_prob(
                    F.interpolate(neg_x_list[i], size=resolution), neg_pred_x_param_list[i])
                neg_sample_log_prob_list.append(neg_sample_log_prob)

                # club-sample upper bound
                upper_bound = (pos_sample_log_prob - neg_sample_log_prob).mean()  # upper_bound of mi
                upper_bound_list.append(upper_bound)

        return upper_bound_list


    def estimate_MI(self, ref_lyr, sigma, ILS_form_config):
        # Example config of ILS.
        example_config = {
            'Example': {
                'mode': 'anti_leakage',  # ['anti_leakage', 'segmentation']. If segmentation, it will replace the perturbation, and only use ILS to generate the mask.
                'estimator': 'club',  # ['club', 'l1out']
                'distribution': 'predetermined_laplace',  # ['predetermined_laplace', 'predetermined_gaussian', 'neural_laplace', 'neural_gaussian']
                'scale': 'do_mean',  # ['do_sum', 'do_mean']
                'mf_conditioned_b': dict(  # min MI(m*f|b) if do_mf is True
                    do_mf=False,
                    opt_mask=False,
                ),
                'separate_mi': {
                    'do_separate': False,  # MI(f,b) or (MI(m*f,m*b) + MI((1-m)*f,(1-m)*b)
                    'stop_grad': False,  # If do separate, whether stop the gradient of m*f and (1-m)*b?
                    'opt_mask': False,  # If do separate, optimize the mask?
                },
            },
        }

        mode = ILS_form_config['mode']
        estimator = ILS_form_config['estimator']
        distribution = ILS_form_config['distribution']
        scale = ILS_form_config['scale']
        mf_conditioned_b = ILS_form_config['mf_conditioned_b']
        separate_mi = ILS_form_config['separate_mi']
        neg_weight = ILS_form_config['neg_weight']

        # [Assertion]
        assert mode in ['anti_leakage', 'segmentation']
        assert estimator in ['club', 'l1out', 'l1_distance', 'l1_distance_allowgrad']
        assert distribution in [
            'pixelcnnpp',
            'predetermined_laplace',
            'predetermined_gaussian',
            'neural_laplace',
            'neural_gaussian']
        assert scale in ['do_sum', 'do_mean']
        assert isinstance(separate_mi, dict)

        # [Preparation]
        detached_mask = ref_lyr['mask'].detach()
        fg = ref_lyr['fg']
        detached_fg = fg.detach()
        bg = ref_lyr['bg']
        detached_bg = bg.detach()
        valid_bg = (bg > -1) * (bg < 1)
        valid_fg = (fg > -1) * (fg < 1)
        # Disable the optimization on the pixel value out of [-1,1]
        fg = fg * valid_fg + detached_fg * (~valid_fg)
        bg = bg * valid_bg + detached_bg * (~valid_bg)

        # [Anti Leakage]
        if mode == 'anti_leakage':
            # [Max L1 distance]
            if estimator == 'l1_distance':
                diff_regulate_bg = ((bg - detached_fg).abs() * detached_mask * valid_bg).mean()
                diff_regulate_fg = ((fg - detached_bg).abs() * (1 - detached_mask) * valid_fg).mean()
                upper_bound = - (diff_regulate_bg + diff_regulate_fg)  # Maximize the difference

            elif estimator == 'l1_distance_allowgrad':
                mask = ref_lyr['mask']
                diff_regulate_bg = ((bg - fg).abs() * mask * valid_bg * valid_fg).mean()
                diff_regulate_fg = ((fg - bg).abs() * (1 - mask) * valid_fg * valid_bg).mean()
                upper_bound = - (diff_regulate_bg + diff_regulate_fg)  # Maximize the difference

            # [CLUB]
            elif estimator == 'club':
                # [Separate MI]
                if separate_mi['do_separate']:

                    if separate_mi['opt_mask']:
                        mask = ref_lyr['mask']
                    else:
                        mask = detached_mask

                    # y1 = fg * mask  #
                    # x1 = bg * mask  #
                    # y2 = fg * (1 - mask)  #
                    # x2 = bg * (1 - mask)  #
                    # if separate_mi['stop_grad']:
                    #     y1 = y1.detach()
                    #     x2 = x2.detach()
                    # x_list = [x1, x2]
                    # y_list = [y1, y2]

                    fg_vis = fg * mask  # condition
                    bg_invis = bg * mask  #
                    fg_invis = fg * (1 - mask)  #
                    bg_vis = bg * (1 - mask)  # condition
                    if separate_mi['stop_grad']:
                        fg_vis = fg_vis.detach()
                        bg_vis = bg_vis.detach()
                    x_list = [bg_invis, fg_invis]
                    y_list = [fg_vis, bg_vis]

                    upper_bound_list = self.compute_club_mi(x_list, y_list, distribution=distribution, scale=scale, neg_weight=neg_weight)
                    upper_bound = 0
                    for up_b in upper_bound_list:
                        upper_bound = upper_bound + up_b


                # [Single MI]
                else:  # Without separation

                    if mf_conditioned_b['do_mf']:
                        # if mf_conditioned_b['opt_mask']:
                        #     mask = ref_lyr['mask']
                        # else:
                        #     mask = detached_mask
                        mask = ref_lyr['mask'] if mf_conditioned_b['opt_mask'] else detached_mask
                        fg = fg if mf_conditioned_b['opt_fg'] else detached_fg
                        bg = bg if mf_conditioned_b['opt_bg'] else detached_bg

                        if mf_conditioned_b['do_inv']:
                            x = [(1 - mask) * bg]
                            y = [fg * mask] if mf_conditioned_b['mask_condition'] else [fg]

                        else:
                            x = [mask * fg]
                            y = [bg * (1 - mask)] if mf_conditioned_b['mask_condition'] else [bg]
                        upper_bound_list = self.compute_club_mi(x, y, distribution=distribution, scale=scale)

                    else:
                        upper_bound_list = self.compute_club_mi([bg], [fg], distribution=distribution, scale=scale)
                    upper_bound = 0
                    for up_b in upper_bound_list:
                        upper_bound = upper_bound + up_b

            # [L1Out]
            elif estimator == 'l1out':
                raise NotImplementedError()

            else:
                raise Exception()

        # [Segmentation]
        elif mode == 'segmentation':
            raise NotImplementedError()


        mi = upper_bound * sigma

        assert torch.isnan(mi).sum() == 0
        return mi

    def get_autoreg_loss(self, ref_lyr):
        """
        Compute the NLL loss for training the autoregressive model pixelcnn++
        :param x:
        :param y: condtion variable
        :return:
        """
        distribution = self.ILS_form_config['distribution']
        mf_conditioned_b = self.ILS_form_config['mf_conditioned_b']
        fg = ref_lyr['fg']
        bg = ref_lyr['bg']
        mask = ref_lyr['mask']
        if self.ILS_form_config['separate_mi']['do_separate']:
            # y1 = fg * mask
            # x1 = bg * mask
            # y2 = fg * (1 - mask)
            # x2 = bg * (1 - mask)
            # x_list = [x1, x2]
            # y_list = [y1, y2]

            fg_vis = fg * mask  # condition
            bg_invis = bg * mask  #
            fg_invis = fg * (1 - mask)  #
            bg_vis = bg * (1 - mask)  # condition
            x_list = [bg_invis, fg_invis]
            y_list = [fg_vis, bg_vis]
        else:
            if self.ILS_form_config['mf_conditioned_b']['do_mf']:
                if mf_conditioned_b['do_inv']:
                    x_list = [(1 - mask) * bg]
                    y_list = [fg * mask] if mf_conditioned_b['mask_condition'] else [fg]

                else:
                    x_list = [mask * fg]
                    y_list = [bg * (1 - mask)] if mf_conditioned_b['mask_condition'] else [bg]

            else:
                x_list = [bg]
                y_list = [fg]

        pred_x_param_list = self.Autoreg(x_list, condition_input_list=y_list)  # fg condition
        loss = 0
        if distribution == 'pixelcnnpp':
            resolution = pred_x_param_list[0].shape[2]  # autoreg use small resolution for saving computation resources
            for i in range(len(x_list)):
                loss = loss + discretized_mix_logistic_loss(
                    F.interpolate(x_list[i], size=resolution), pred_x_param_list[i])

        elif distribution == 'neural_gaussian':
            resolution = pred_x_param_list[0][0].shape[2]
            for i in range(len(x_list)):
                mu, logvar = pred_x_param_list[i]
                sample_size = mu.shape[0]

                # Reshape to vectors.
                mu = mu.reshape(sample_size, -1)
                logvar = logvar.reshape(sample_size, -1)
                x_samples = F.interpolate(x_list[i], size=resolution)
                x_samples = x_samples.reshape(sample_size, -1)
                # NLL, unnormalized
                # loss = loss - (-(mu - x_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
                # NLL, normalized
                loss = loss - (-(mu - x_samples)**2/logvar.exp() - logvar - 0.5*torch.log(2*torch.tensor(math.pi))).sum(dim=1).mean(dim=0)

        elif distribution == 'neural_laplace':
            resolution = pred_x_param_list[0][0].shape[2]
            for i in range(len(x_list)):
                mu, logb = pred_x_param_list[i]
                sample_size = mu.shape[0]
                # Reshape to vectors.
                mu = mu.reshape(sample_size, -1)
                logb = logb.reshape(sample_size, -1)
                x_samples = F.interpolate(x_list[i], size=resolution)
                x_samples = x_samples.reshape(sample_size, -1)
                # NLL,
                loss = loss - (- logb - torch.abs(x_samples - mu) / logb.exp() ).sum(dim=1).mean(dim=0)
        else:
            raise NotImplementedError()

        return loss


    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, trs_eps, sync, gain, cur_kimg=None, do_anti_leaking=False):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        # do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0 and self.weights['equi'] == 0)
        do_Gpl   = False
        # do_Geq   = (phase in ['Greg', 'Gboth']) and (self.weights['equi'] != 0)
        do_Geq   = False
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        leaking_mark = False

        # Gmain: Maximize logits for generated images.
        #  [NEW] Minimize binarization loss, maximize mutual information
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):

                do_perturb = self.do_perturb  # default: True

                gen_lyr, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                if do_perturb:
                    trs_lyr = self.run_Tx(gen_lyr, trs_eps)
                else:
                    trs_lyr = gen_lyr

                # [Augment BG]
                bg_augment = False
                if bg_augment:
                    bg = trs_lyr['bg']
                    fg = trs_lyr['fg']
                    mask = trs_lyr['mask']
                    # augment bg, make it easier to expose the inconsistency of fg and bg,
                    # also can prevent leaking problem in our assumption
                    bg = self.augment_pipe(bg)
                    img = (1 - mask) * bg + fg * mask
                else:
                    img = trs_lyr["img"]


                gen_logits = self.run_D(img, gen_c, get_feature=False, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)


                # [Anti leaking]
                # do_anti_leaking = do_anti_leaking  # do anti leaking only control the ver1.0 anti-leaking loss
                do_anti_leaking = self.do_anti_leaking
                ILS_sigma = self.ILS_weight_strategy['ILS_sigma']
                strategy = self.ILS_weight_strategy['strategy']  # ['all_time', 'cool_down', 'cos']
                initial_anti = self.ILS_weight_strategy['initial_anti']
                initial_anti_sigma = self.ILS_weight_strategy['initial_anti_sigma']

                assert strategy in ['all_time', 'cool_down', 'cos']
                if strategy == 'all_time':
                    cosine_annealing = False
                    pulse = False
                elif strategy == 'cool_down':
                    cosine_annealing = False
                    pulse = True
                elif strategy == 'cos':
                    cosine_annealing = True
                    pulse = False


                start = 400

                if (cur_kimg <= start) and initial_anti and do_anti_leaking:
                    # Use small weight anti leaking to help initialization
                    sigma = initial_anti_sigma
                    mi = self.estimate_MI(ref_lyr=gen_lyr, sigma=sigma, ILS_form_config=self.ILS_form_config)
                    self.sigma = sigma


                elif (cur_kimg > start) and do_anti_leaking:
                    if cosine_annealing:
                        eta_min = 0  # equals to min sigma
                        eta_max = ILS_sigma  # equals to max sigma
                        T_max = 400
                        sigma = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * (cur_kimg / T_max)))

                    elif pulse:
                        sigma_strategy = [ILS_sigma, 0, 0, 0]
                        idx = int((cur_kimg - start - 1) / 200) % len(sigma_strategy)  # minus 1: to optimize 401~600 not 400~599
                        sigma = sigma_strategy[idx]

                        if sigma == 0 and leaking_mark is True:  # Apply epoch heuristic
                            sigma = ILS_sigma

                    else:
                        sigma = ILS_sigma

                    mi = self.estimate_MI(ref_lyr=gen_lyr, sigma=sigma, ILS_form_config=self.ILS_form_config)
                    self.sigma = sigma  # Assign sigma value to loss instance, it'll be used in the training_loop to mark whether the ckpt using anti-leaking
                else:
                    mi = 0
                    self.sigma = 0
                training_stats.report('Loss/G/anti_leaking', mi)


                # [Img-fg, Img-bg pairs divergence]
                do_divergence = False
                if do_divergence:
                    diver_weight = 5.0
                    ref_lyr = gen_lyr
                    # fg_score = self.get_logits_from_D(ref_lyr['fg'], gen_c, get_feature=False, sync=False)
                    bg_score = self.get_logits_from_D(ref_lyr['bg'], gen_c, get_feature=False, sync=False)
                    img_score = self.get_logits_from_D(ref_lyr['img'], gen_c, get_feature=False, sync=False)

                    # img_fg_gap = img_score - fg_score
                    img_bg_gap = img_score - bg_score

                    # diver = (img_fg_gap + img_bg_gap) * diver_weight
                    diver = img_bg_gap * diver_weight  # It will also bring the anti leaking effect
                else:
                    diver = 0
                training_stats.report('Loss/G/divergence', diver)

                # binarization loss
                loss_Gbin = torch.min(1 - trs_lyr["mask"], trs_lyr["mask"]).mean(dim=[2, 3])
                training_stats.report('Loss/G/loss_bin', loss_Gbin)

                # [Anti zero mask]
                # mask size loss
                if do_perturb:
                    loss_Ganti0 = torch.nn.functional.relu(self.min_msize - trs_lyr["mask"].mean(dim=(2, 3)))
                else:
                    loss_Ganti0 = torch.nn.functional.relu(self.min_msize - trs_lyr["mask"].mean(dim=(2, 3))) + \
                                  torch.nn.functional.relu(self.min_msize - (1 - trs_lyr["mask"].mean(dim=(2, 3))))
                training_stats.report('Loss/G/loss_msize', loss_Ganti0)

                # merge
                loss_Gtotal = loss_Gmain + loss_Gbin.mul(self.weights["bin"]) + loss_Ganti0.mul(self.weights[self.anti_zero_mask])
                loss_Gtotal = loss_Gtotal + mi # minimize mi

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gtotal.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        # In experiments, if Gpl is enabled, the value of pl_penalty remains at a very low level.
        # For PerturbGAN-Re, when Gpl is enabled, the layers fail to emerge; when Gpl is disabled,
        # the layers succeed in emerging.
        if do_Gpl:
            raise NotImplementedError()
            # with torch.autograd.profiler.record_function('Gpl_forward'):
            #     batch_size = gen_z.shape[0] // self.pl_batch_shrink
            #     gen_lyr, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
            #
            #     if self.do_up_down_sample:
            #         bg_rgb = gen_lyr['bg']
            #         fg_rgb = gen_lyr['fg']
            #         fg_a = gen_lyr['mask']
            #         fg_rgb = F.interpolate(fg_rgb, scale_factor=0.5, mode='bilinear')
            #         fg_a = F.interpolate(fg_a, scale_factor=0.5, mode='bilinear')
            #         gen_lyr = {
            #             "bg": bg_rgb,
            #             "fg": fg_rgb,
            #             "mask": fg_a,
            #             "fg*mask": fg_a * fg_rgb,
            #             "img": (1. - fg_a) * bg_rgb + fg_a * fg_rgb
            #         }
            #
            #     # randomly mix bg
            #     if self.mix_bg_ratio > 0 and bool(torch.rand([1]) < self.mix_bg_ratio) and self.G_mapping_fg.training:
            #         gen_lyr = self.mix_bg(gen_lyr)
            #
            #     # randomly do horizontal flip on bg
            #     if self.h_flip_bg_ratio > 0 and bool(torch.rand([1]) < self.h_flip_bg_ratio) and self.G_mapping_fg.training:
            #         gen_lyr = self.h_flip_bg(gen_lyr)
            #
            #     gen_img = gen_lyr["img"]
            #     pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            #     with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
            #         pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws["fg"], gen_ws["bg"]], create_graph=True, only_inputs=True)
            #     pl_lengths = torch.cat(pl_grads, dim=1).square().sum(2).mean(1).sqrt()
            #     pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            #     self.pl_mean.copy_(pl_mean.detach())
            #     pl_penalty = (pl_lengths - pl_mean).square()
            #     training_stats.report('Loss/pl_penalty', pl_penalty)
            #     loss_Gpl = pl_penalty * self.pl_weight
            #     training_stats.report('Loss/G/reg', loss_Gpl)
            # with torch.autograd.profiler.record_function('Gpl_backward'):
            #     (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()


        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_lyr, _gen_ws = self.run_G(gen_z, gen_c, sync=False)

                # trs_lyr = self.run_Tx(gen_lyr, trs_eps)

                if self.do_perturb:
                    trs_lyr = self.run_Tx(gen_lyr, trs_eps)
                else:
                    trs_lyr = gen_lyr

                gen_logits = self.run_D(trs_lyr["img"], gen_c, get_feature=False, sync=False)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                loss_Dtotal = loss_Dgen

            with torch.autograd.profiler.record_function('Dgen_backward'):
                with torch.autograd.set_detect_anomaly(True):
                    loss_Dtotal.mean().mul(gain).backward()

            if self.Autoreg is not None and \
                    (self.ILS_form_config['distribution'] == 'pixelcnnpp' or self.ILS_form_config['distribution'] == 'neural_gaussian' or self.ILS_form_config['distribution'] == 'neural_laplace') \
                    and self.do_anti_leaking:
                # we optimize the autoregressive model in the Dmain phase
                with torch.autograd.profiler.record_function('Autoreg_forward'):

                    NLL_loss = self.get_autoreg_loss(gen_lyr)  # set fg as condition
                    # because autoregressive model is separated from D, we can ignore the loss scale between D and autoreg.
                    # NLL_loss = NLL_loss / (gen_lyr['bg'].shape[0] * 16 * 16 * 3 * np.log(2.))
                    NLL_loss_weight = 0.1
                    NLL_loss = NLL_loss * NLL_loss_weight
                    training_stats.report('Loss/Autoreg/NLL', NLL_loss)

                with torch.autograd.profiler.record_function('Autoreg_backward'):
                    NLL_loss.mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                # real_logits = self.run_D(real_img_tmp, real_c, get_feature=False, sync=sync)
                real_logits = self.run_D(real_img_tmp, real_c, get_feature=False, sync=sync)

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

    def get_gradients_of_gen_img(self, gen_z, gen_c, trs_eps):
        gain = 1.
        sync = False
        do_Gmain = True

        # Gmain: Maximize logits for generated images.
        #  [NEW] Minimize binarization loss, maximize mutual information
        if do_Gmain:
            gen_lyr, _gen_ws = self.run_G(gen_z, gen_c, sync=sync) # May get synced by Gpl.
            trs_lyr = self.run_Tx(gen_lyr, trs_eps)

            gen_logits = self.run_D(trs_lyr["img"], gen_c, get_feature=False, sync=False)
            loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))

            # binarization loss
            loss_Gbin = torch.min(1 - trs_lyr["mask"], trs_lyr["mask"]).mean(dim=[2, 3])

            # anti zero mask
            # mask size loss
            loss_Ganti0 = torch.nn.functional.relu(self.min_msize - trs_lyr["mask"].mean(dim=(2, 3)))

            # merge
            loss_Gtotal = loss_Gmain + loss_Gbin.mul(self.weights["bin"]) + loss_Ganti0.mul(self.weights[self.anti_zero_mask])

            trs_lyr['img'].retain_grad()
            trs_lyr['mask'].retain_grad()
            loss_Gtotal.mean().mul(gain).backward()

            # return the gradients of gen_img and zero out the gradients out of the loss
            img_grad = trs_lyr['img'].grad.clone().detach()
            mask_grad = trs_lyr['mask'].grad.clone().detach()
            return img_grad, mask_grad

    def rampup(self, value):
        mod = self.cur_nimg / 1e3 / self.t1kimg
        return min(1., mod) * value

#----------------------------------------------------------------------------

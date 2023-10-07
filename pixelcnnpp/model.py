import pdb
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pixelcnnpp.layers import *
from pixelcnnpp.utils import *
import numpy as np
import torchvision.models as models


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, conditional=False, h_dim=None):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0, conditional=conditional, h_dim=h_dim)
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=1, conditional=conditional, h_dim=h_dim)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, hs=None):
        u_list, ul_list = [], []
        
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, h=hs)
            ul = self.ul_stream[i](ul, a=u, h=hs)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, conditional=False, h_dim=None):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=1, conditional=conditional, h_dim=h_dim)
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=2, conditional=conditional, h_dim=h_dim)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list, hs=None):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop(), h=hs)
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1), h=hs)
        
        return u, ul
         

class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, 
                    resnet_nonlinearity='concat_elu', input_channels=3, conditional=False, h_dim=None):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' : 
            # self.resnet_nonlinearity = lambda x : concat_elu(x)
            self.resnet_nonlinearity = concat_elu
        else : 
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters, 
                                                self.resnet_nonlinearity, conditional=conditional, h_dim=h_dim) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters, 
                                                self.resnet_nonlinearity, conditional=conditional, h_dim=h_dim) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters, 
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters, 
                                                    nr_filters, stride=(2,2)) for _ in range(2)])
        
        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters, 
                                                    stride=(2,2)) for _ in range(2)])
        
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters, 
                                                    nr_filters, stride=(2,2)) for _ in range(2)])
        
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3), 
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters, 
                                            filter_size=(1,3), shift_output_down=True), 
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters, 
                                            filter_size=(2,1), shift_output_right=True)])
    
        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None


    def forward(self, x, sample=False, hs=None):

        if x.is_cuda:
            device = x.device

        # similar as done in the tf repo :  
        if self.init_padding is None and not sample: 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.to(device) if x.is_cuda else padding
        
        if sample : 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.to(device) if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1], hs=hs)
            u_list  += u_out
            ul_list += ul_out

            if i != 2: 
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()
        
        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list, hs=hs)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out


class PixelCNN_extractor(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3, conditional=False, h_dim=None):
        super(PixelCNN_extractor, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            # self.resnet_nonlinearity = lambda x : concat_elu(x)
            self.resnet_nonlinearity = concat_elu
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                             self.resnet_nonlinearity, conditional=conditional, h_dim=h_dim) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                           self.resnet_nonlinearity, conditional=conditional, h_dim=h_dim) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                                     stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                                           nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                                     stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                                           nr_filters, stride=(2,2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                                          shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                                          filter_size=(1,3), shift_output_down=True),
                                      down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                                                filter_size=(2,1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

        # feature extractor
        self.extractor = models.resnet18(pretrained=False)
        self.to_h = nn.Linear(1000, h_dim)


    def forward(self, x, sample=False, condition_input=None, test=False):

        # Extract hidden feature of condition input to get hs
        if condition_input is not None:
            condition_feat = self.extractor(condition_input)
            hs = self.to_h(condition_feat)
            hs = nn.functional.relu(hs)
        else:
            hs = None

        if x.is_cuda:
            device = x.device

        # similar as done in the tf repo :
        if self.init_padding is None and not (sample or test):
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.to(device) if x.is_cuda else padding

        if (sample or test):
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.to(device) if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if (sample or test) else torch.cat((x, self.init_padding), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1], hs=hs)
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list, hs=hs)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out

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

#----------------------------------------------------------------------------
class NeuralGaussian(torch.nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, input_channels=3, output_channels=3, bilinear=True):
        super(NeuralGaussian, self).__init__()
        # self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
        #                           nn.ReLU(),
        #                           nn.Linear(hidden_size//2, y_dim))
        #
        # self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
        #                               nn.ReLU(),
        #                               nn.Linear(hidden_size//2, y_dim),
        #                               nn.Tanh())

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bilinear = bilinear

        self.inc = UNetBlock(self.input_channels, 64)

        self.down1 = UNetBlock(64, 128, down=True)
        self.down2 = UNetBlock(128, 256, down=True)
        self.down3 = UNetBlock(256, 256, down=True)
        self.down4 = UNetBlock(256, 256, down=True)

        self.up1 = UNetBlock(512, 256, up=True, bilinear=bilinear)
        self.up2 = UNetBlock(512, 128, up=True, bilinear=bilinear)
        self.up3 = UNetBlock(256, 64, up=True, bilinear=bilinear)
        self.up4 = UNetBlock(128, 64, up=True, bilinear=bilinear)

        self.out_p_mu = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, self.output_channels, kernel_size=1))

        self.out_p_logvar = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, self.output_channels, kernel_size=1),
            torch.nn.Tanh(),
        )


    def get_mu_logvar(self, x_samples):
        x1 = self.inc(x_samples)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        mu = self.out_p_mu(x)
        logvar = self.out_p_logvar(x)
        return mu, logvar


    # def loglikeli(self, x_samples, condition_input):
    #     y_samples = x_samples
    #     x_samples = condition_input
    #
    #     mu, logvar = self.get_mu_logvar(x_samples)
    #     # Reshape to vectors
    #     N = x_samples.shape[0]
    #     mu = mu.reshape(N, -1)
    #     logvar = logvar.reshape(N, -1)
    #     y_samples = y_samples.reshape(N, -1)
    #     return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    #
    # def get_mi(self, x_samples, condition_input):
    #     y_samples = x_samples
    #     x_samples = condition_input
    #
    #     mu, logvar = self.get_mu_logvar(x_samples)
    #
    #     sample_size = x_samples.shape[0]
    #     random_index = torch.randperm(sample_size).long()
    #
    #     # Reshape to vectors.
    #     mu = mu.reshape(sample_size, -1)
    #     logvar = logvar.reshape(sample_size, -1)
    #     y_samples = y_samples.reshape(sample_size, -1)
    #
    #     positive = - (mu - y_samples)**2 / logvar.exp()
    #     negative = - (mu - y_samples[random_index])**2 / logvar.exp()
    #     upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
    #     return upper_bound/2.
    #
    # def learning_loss(self, x_samples, condition_input):
    #     return - self.loglikeli(x_samples, condition_input)

    def forward(self, condition_input):
        mu, logvar = self.get_mu_logvar(condition_input)
        return (mu, logvar)


#----------------------------------------------------------------------------
class NeuralLaplace(torch.nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, input_channels=3, output_channels=3, bilinear=True):
        super(NeuralLaplace, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bilinear = bilinear

        self.inc = UNetBlock(self.input_channels, 64)

        self.down1 = UNetBlock(64, 128, down=True)
        self.down2 = UNetBlock(128, 256, down=True)
        self.down3 = UNetBlock(256, 256, down=True)
        self.down4 = UNetBlock(256, 256, down=True)

        self.up1 = UNetBlock(512, 256, up=True, bilinear=bilinear)
        self.up2 = UNetBlock(512, 128, up=True, bilinear=bilinear)
        self.up3 = UNetBlock(256, 64, up=True, bilinear=bilinear)
        self.up4 = UNetBlock(128, 64, up=True, bilinear=bilinear)

        self.out_p_mu = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, self.output_channels, kernel_size=1))

        self.out_p_logb = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, self.output_channels, kernel_size=1),
            torch.nn.Tanh(),
        )


    def get_mu_logb(self, x_samples):
        x1 = self.inc(x_samples)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        mu = self.out_p_mu(x)
        logb = self.out_p_logb(x)
        return mu, logb


    def forward(self, condition_input):
        mu, logb = self.get_mu_logb(condition_input)
        return (mu, logb)

#----------------------------------------------------------------------------
class MINE_resnet(nn.Module):
    def __init__(self, hidden_size=1000):
        super(MINE, self).__init__()
        self.x_feat_extractor = models.resnet18(pretrained=True)  # output 1000-d
        self.y_feat_extractor = models.resnet18(pretrained=True)
        self.T_func = nn.Sequential(nn.Linear(1000+1000, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, C, H, W]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        x_feat = self.x_feat_extractor(x_samples)  # [sample_size, hidden_size]
        y_feat = self.y_feat_extractor(y_samples)
        y_feat_shuffle = self.y_feat_extractor(y_shuffle)

        assert x_feat.dim() == 2

        T0 = self.T_func(torch.cat([x_feat,y_feat], dim = -1))
        T1 = self.T_func(torch.cat([x_feat,y_feat_shuffle], dim = -1))

        eps = 1e-7
        lower_bound = T0.mean() - torch.log(T1.exp().mean() + eps)

        if torch.isnan(lower_bound).sum() > 0:
            print('x_feat_nan_sum:', torch.isnan(x_feat).sum())
            print('y_feat_nan_sum:', torch.isnan(y_feat).sum())
            print('T0:', T0)
            print('T1:', T1)
            sys.exit(1)

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

#----------------------------------------------------------------------------
class MINE(nn.Module):
    def __init__(self, obs, hidden_size):
        super(MINE, self).__init__()
        x_dim = y_dim = obs[0] * obs[1] * obs[2]
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        sample_size = y_samples.shape[0]
        x_samples = x_samples.reshape(sample_size, -1)
        y_samples = y_samples.reshape(sample_size, -1)

        # shuffle and concatenate
        random_index = torch.randint(sample_size, (sample_size,)).long()
        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        if lower_bound.isnan().sum() > 0 or lower_bound.isinf().sum() > 0:
            print('\nDetected abnormal lowerbound value:', lower_bound, 'set lowerbound to 0!\n')
            lower_bound = 0

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class ModelBag(nn.Module):
    def __init__(self, num_autoreg=1, resolution=16, distribution='pixelcnnpp', nr_resnet=3, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3, conditional=True, h_dim=200):
        super(ModelBag, self).__init__()
        self.num_autoreg = num_autoreg
        self.resolution = resolution
        self.distribution = distribution
        self.nr_resnet = nr_resnet
        self.nr_filters = nr_filters
        self.nr_logistic_mix = nr_logistic_mix
        self.resnet_nonlinearity = resnet_nonlinearity
        self.input_channels = input_channels
        self.conditional = conditional
        self.h_dim = h_dim

        # Construct Autoregressive models accroding to num_autoreg
        for i in range(num_autoreg):
            if distribution == 'pixelcnnpp':
                model = PixelCNN_extractor(nr_resnet=nr_resnet, nr_filters=nr_filters, resnet_nonlinearity=resnet_nonlinearity,
                                       input_channels=input_channels, nr_logistic_mix=nr_logistic_mix, conditional=conditional, h_dim=h_dim)
            elif distribution == 'neural_gaussian':
                model = NeuralGaussian(input_channels=3, output_channels=3, bilinear=True)
            elif distribution == 'neural_laplace':
                model = NeuralLaplace(input_channels=3, output_channels=3, bilinear=True)
            else:
                raise NotImplementedError()
            setattr(self, f'autoreg{i}', model)

    def forward(self, x_list, condition_input_list, sample=False, test=False):
        assert len(x_list) == len(condition_input_list) == self.num_autoreg
        pred_x_param_list = []
        for i in range(self.num_autoreg):
            x = F.interpolate(x_list[i], size=self.resolution)
            condition_input = F.interpolate(condition_input_list[i], size=self.resolution)
            autoreg = getattr(self, f'autoreg{i}')
            if self.distribution == 'pixelcnnpp':
                pred_x_param = autoreg(x, condition_input=condition_input, sample=sample, test=test)
            elif self.distribution == 'neural_gaussian':
                pred_x_param = autoreg(condition_input=condition_input)
            elif self.distribution == 'neural_laplace':
                pred_x_param = autoreg(condition_input=condition_input)
            else:
                raise NotImplementedError()
            pred_x_param_list.append(pred_x_param)
        # return the param list according to the number of Autoreg modules
        return pred_x_param_list



if __name__ == '__main__':
    pass
    # ''' testing loss with tf version '''
    # np.random.seed(1)
    # xx_t = (np.random.rand(15, 32, 32, 100) * 3).astype('float32')
    # yy_t  = np.random.uniform(-1, 1, size=(15, 32, 32, 3)).astype('float32')
    # x_t = Variable(torch.from_numpy(xx_t)).cuda()
    # y_t = Variable(torch.from_numpy(yy_t)).cuda()
    # loss = discretized_mix_logistic_loss(y_t, x_t)
    #
    # ''' testing model and deconv dimensions '''
    # x = torch.cuda.FloatTensor(32, 3, 32, 32).uniform_(-1., 1.)
    # xv = Variable(x).cpu()
    # ds = down_shifted_deconv2d(3, 40, stride=(2,2))
    # x_v = Variable(x)
    #
    # ''' testing loss compatibility '''
    # model = PixelCNN(nr_resnet=3, nr_filters=100, input_channels=x.size(1))
    # model = model.cuda()
    # out = model(x_v)
    # loss = discretized_mix_logistic_loss(x_v, out)
    # print('loss : %s' % loss.data[0])

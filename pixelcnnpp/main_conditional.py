import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from pixelcnnpp.utils import *
from pixelcnnpp.model import *
from PIL import Image
import numpy as np
import re


def sample(model, device, label):
    model.eval().requires_grad_(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.to(device)

    label = torch.full([sample_batch_size], label)
    hs = torch.nn.functional.one_hot(label, h_dim).to(device, dtype=input.dtype)


    with torch.no_grad():
        for i in range(obs[1]):
            for j in range(obs[2]):
                out = model(data, sample=True, hs=hs)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample.data[:, :, i, j]

    model.train().requires_grad_(True)
    return data


parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--outdir', type=str, default='output',
                    help='Location for outputs')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=5,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.lr, args.nr_resnet, args.nr_filters)

# Prepare output directory
outdir = args.outdir
if not os.path.exists(outdir):
    os.makedirs(outdir)
if os.path.isdir(outdir):
    prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
cur_run_id = max(prev_run_ids, default=-1) + 1
run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{model_name}')
print('running directory:', run_dir)

# assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)

writer = SummaryWriter(log_dir=run_dir)

sample_batch_size = 4
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

if 'mnist' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

    h_dim = 10

elif 'cifar' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    h_dim = 10

else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix, conditional=True, h_dim=h_dim)

device = torch.device('cuda')
model = model.to(device).train().requires_grad_(True)

if args.load_params:
    load_part_of_model(model, args.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

print('starting training')
writes = 0
for epoch in range(args.max_epochs):

    torch.cuda.synchronize()
    train_loss = 0.
    epoch_start_time = time.time()
    time_ = time.time()
    model.train().requires_grad_(True)
    for batch_idx, (input, label) in enumerate(train_loader):

        # input = input.cuda(async=True)
        # input = Variable(input)

        input = input.to(device)
        hs = torch.nn.functional.one_hot(label, h_dim).to(device, dtype=input.dtype)

        output = model(input, hs=hs)
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_loss += loss.data[0]
        train_loss += loss.item()

        if (batch_idx +1) % args.print_every == 0 : 
            deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
            writer.add_scalar('train/bpd', (train_loss / deno), writes)
            print('loss : {:.4f}, time : {:.4f}'.format(
                (train_loss / deno), 
                (time.time() - time_)))
            train_loss = 0.
            writes += 1
            time_ = time.time()

        # # Debug
        # break
            

    # decrease learning rate
    scheduler.step()
    torch.cuda.synchronize()
    epoch_end_time = time.time()
    print('Epoch training time: {:.4f}'.format(epoch_end_time - epoch_start_time))

    # Evaluation
    print('Evaluating...')
    eval_start_time = time.time()
    model.eval().requires_grad_(False)
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(test_loader):

        # input = input.cuda(async=True)
        # input_var = Variable(input)
        input_var = input.to(device)

        output = model(input_var)
        loss = loss_op(input_var, output)
        test_loss += loss.item()
        del loss, output
    model.train().requires_grad_(True)

    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    writer.add_scalar('test/bpd', (test_loss / deno), writes)
    print('test loss : %s' % (test_loss / deno))
    print('Current epoch number:', epoch)
    eval_end_time = time.time()
    print('Testing time: {:.4f}'.format(eval_end_time - eval_start_time))
    
    if (epoch + 1) % args.save_interval == 0:
        # model_dir = os.path.join(run_dir, 'models')
        # image_dir = os.path.join(run_dir, 'images')
        # if not os.path.exists(model_dir):
        #     os.mkdir(model_dir)
        # if not os.path.exists(image_dir):
        #     os.mkdir((image_dir))

        torch.save(model.state_dict(), os.path.join(run_dir, f'network-pkl-{epoch:05d}.pth'))
        print('sampling...')
        for gen_label in range(h_dim):
            sample_t = sample(model, device, label=gen_label)  # conditioned by gen_label
            sample_t = rescaling_inv(sample_t)
            utils.save_image(sample_t, os.path.join(run_dir, f'synthetic-samples-{epoch:05d}-label-{gen_label}.png'),
                    nrow=2, padding=0)

writer.close()

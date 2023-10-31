from __future__ import print_function, division
import json
import sys
import argparse
import os
import cv2
import time
import torch
import torchvision
from torch.cuda.amp import GradScaler

from rhwf import RHWF
from utils import *
from utils_train import *

from evaluate import validate_process

from wandb_logger import WandbLogger

def train(args):
    # device
    device = torch.device('cuda:'+ str(args.gpuid[1]))
    print(f"device: {device}")

    # model
    model = RHWF(args)
    model.cuda()
    model.train()
    print(f"Parameter Count: {count_parameters(model)}")
    optimizer, scheduler = fetch_optimizer(args, model)

    # restore
    if args.restore_ckpt is not None:
        save_model = torch.load(args.restore_ckpt)
        model.load_state_dict(save_model['net'])

        optimizer.load_state_dict(save_model['optimizer'])
        scheduler.load_state_dict(save_model['scheduler'])

    # dataset
    if args.dataset=='ggearth':
        import dataset as datasets
    else:
        import datasets_4cor_img as datasets
        
    train_loader = datasets.fetch_dataloader(args, split="train")
    scaler = GradScaler(enabled=args.mixed_precision)
    # logger = Logger(model, scheduler, args)

    # log
    if args.wandb:
        wandb_config = dict(project="cvgl", entity='jayhi-park', name=args.name)
        wandb_logger = WandbLogger(wandb_config, args)
        wandb_logger.before_run()
    else:
        wandb_logger = WandbLogger(None)

    # epoch
    total_steps = 0
    best_results = {'val/mace': 100}

    while total_steps <= args.num_steps:
        for i_batch, data_blob in enumerate(train_loader):
            tic = time.time()
            image1, image2, image2w, flow,  H  = [x.cuda() for x in data_blob]
            image2_w = warp(image2, flow)

            # if i_batch==0:
            #     if not os.path.exists('watch'):
            #         os.makedirs('watch')
            #     save_img(torchvision.utils.make_grid(image1, nrow=16, padding = 16, pad_value=255), './watch/' + 'train_img1.bmp')
            #     save_img(torchvision.utils.make_grid(image2, nrow=16, padding = 16, pad_value=255), './watch/' + 'train_img2.bmp')
            #     save_img(torchvision.utils.make_grid(image2_w, nrow=16, padding = 16, pad_value=255), './watch/' + 'train_img2w.bmp')

            optimizer.zero_grad()

            # predict
            four_pred = model(image1, image2, iters_lev0=args.iters_lev0, iters_lev1=args.iters_lev1)

            # loss
            loss, metrics = sequence_loss(four_pred, flow, H, args.gamma, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            toc = time.time()

            # metrics['time'] = toc - tic
            # logger.push(metrics)

            # log
            if total_steps % args.train_freq == 0:
                wandb_logger.log_dict({'training/loss': loss})
                time_left_sec = int(((args.num_steps - (total_steps+1)) * (toc - tic)))
                time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60,
                                                               time_left_sec % 3600 % 60)
                print(f"training steps: {total_steps}, loss: {loss: .3f}, left_hms: {time_left_hms}")

                wandb_logger.log_dict({'training/lr': scheduler.get_lr()[0]})

                for key, value in metrics.items():
                    wandb_logger.log_dict(({f'training/{key}': value}))

            # Validate
            if total_steps % args.val_freq == args.val_freq - 1:
                # validate(model, args, logger)
                # plot_train(logger, args)
                # plot_val(logger, args)
                results = validate_process(model, args)
                wandb_logger.log_dict(results)

                if results['val/mace'] < best_results['val/mace']:
                    best_results = results
                    # PATH = args.output + f'/{total_steps+1}_{args.name}.pth'
                    PATH = args.output + f'/best.pth'
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }
                    torch.save(checkpoint, PATH)

            # add step
            total_steps += 1

    PATH = args.output + f'/last.pth'
    torch.save(model.state_dict(), PATH)

    return PATH


def validate(model, args, logger):
    results = {}
    # Evaluate results
    results.update(validate_process(model, args))
    # Record results in logger
    for key in results.keys():
        if key not in logger.val_results_dict.keys():
            logger.val_results_dict[key] = []
        logger.val_results_dict[key].append(results[key])
    logger.val_steps_list.append(logger.total_steps)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='RHWF', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    
    parser.add_argument('--gpuid', type=int, nargs='+', default = [0, 1])
    parser.add_argument('--output', type=str, default='results/ggearth_6_6', help='output directory to save checkpoints and plots')
    # parser.add_argument('--logname', type=str, default='ggearth_6_6.log', help='printing frequency')
    parser.add_argument('--dataset', type=str, default='ggearth', help='dataset')

    parser.add_argument('--lev0', default=True, action='store_true', help='warp no')
    parser.add_argument('--lev1', default=True, action='store_true', help='warp once')
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=6)

    parser.add_argument('--train_freq', type=int, default=20, help='train frequency')
    parser.add_argument('--val_freq', type=int, default=250, help='validation frequency')
    # parser.add_argument('--print_freq', type=int, default=100, help='printing frequency')

    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    
    parser.add_argument('--model_name', default='', help='specify model name')
    parser.add_argument('--resume', default=False, action='store_true', help='resume_training')

    # wandb
    parser.add_argument('--wandb', action='store_true', default=False)

    args = parser.parse_args()

    setup_seed(1024)

    # sys.stdout = Logger_(args.logname, sys.stdout)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    train(args)

import argparse
import sys
import os

import torch
import torch.distributed as dist

if str(os.getcwd()) not in sys.path:
    sys.path.append(str(os.getcwd()))

from src.utils.general import LOGGER
from src.core.trainer import Trainer
from src.utils.envs import get_envs, select_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_afn_root', default='/media/manu/kingstoo/AffectNet', type=str, help='data root dir')
    parser.add_argument('--batch_size', default=64, type=int, help='total batch size for all GPUs')
    parser.add_argument('--img_size', default=260, type=int, help='train, val image size (pixels)')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--epoch_ft_a', default=3, type=int, help='step one fine tune epoch')
    parser.add_argument('--epoch_ft_b', default=6, type=int, help='step two fine tune epoch')
    parser.add_argument('--lr_ft_a', default=1e-3, type=float, help='step one learning rate')
    parser.add_argument('--lr_ft_b', default=1e-4, type=float, help='step two learning rate')
    parser.add_argument('--path_resume', default='', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    args.local_rank, args.rank, args.world_size = get_envs()
    args.device_pick = select_device(args.device)
    args.local_rank, args.rank, args.world_size = get_envs()
    LOGGER.info(f'training args are: {args}\n')

    if args.local_rank != -1:  # if DDP mode
        torch.cuda.set_device(args.local_rank)
        args.device_pick = torch.device('cuda', args.local_rank)
        LOGGER.info('Initializing process group... ')
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", init_method=args.dist_url,
                                rank=args.local_rank, world_size=args.world_size)

    trainer = Trainer(args)
    trainer.train(n_epochs=args.epoch_ft_a, learningrate=args.lr_ft_a, dft=False)
    trainer.train(n_epochs=args.epoch_ft_b, learningrate=args.lr_ft_b, dft=True)


if __name__ == '__main__':
    main()

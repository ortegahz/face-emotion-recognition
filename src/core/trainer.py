from copy import deepcopy
from tqdm import tqdm
import numpy as np
import shutil
import copy
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import timm

from src.core.loss import cross_entropy_with_label_smoothing
from src.affectnet.robust_optimization import RobustOptimizer
from src.utils.torch_utils import torch_distributed_zero_first
from src.utils.general import LOGGER


class Trainer:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.device = args.device_pick
        self.tf_mean = [0.485, 0.456, 0.406]
        self.tf_std = [0.229, 0.224, 0.225]
        self.lr = 3e-5
        self.robust = True
        self.path_ckpt = 'models/pretrained_faces/state_vggface2_enet0_new.pt'
        self.path_save = 'run/fer.pt'
        self.main_process = args.rank in [-1, 0]

        self.dir_train = os.path.join(args.dir_afn_root, 'full_res', 'train')
        self.dir_val = os.path.join(args.dir_afn_root, 'full_res', 'val')

        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tf_mean,
                                     std=self.tf_std)
            ]
        )

        self.test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tf_mean,
                                     std=self.tf_std)
            ]
        )

        self.criterion = cross_entropy_with_label_smoothing

        self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
        self.model.classifier = torch.nn.Identity()
        self.model.load_state_dict(torch.load(self.path_ckpt, map_location=self.device))
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=8))  # TODO
        self.model = self.model.cuda(self.device)
        self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model, self.is_parallel = self.parallel_model(args, self.model, self.device)

        with torch_distributed_zero_first(args.rank):
            kwargs = {} if self.device == 'cpu' else {'num_workers': 8, 'pin_memory': False}
            self.train_dataset = datasets.ImageFolder(root=self.dir_train, transform=self.train_transforms)
            if not self.is_parallel:
                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                                shuffle=True,
                                                                **kwargs)

            else:
                self.train_sampler = DistributedSampler(self.train_dataset)
                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                                shuffle=False, sampler=self.train_sampler,
                                                                **kwargs)
            self.test_loader = None
            if args.rank in [-1, 0]:
                self.test_dataset = datasets.ImageFolder(root=self.dir_val, transform=self.test_transforms)
                self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                               shuffle=False,
                                                               **kwargs)

        (unique, counts) = np.unique(self.train_dataset.targets, return_counts=True)
        cw = 1 / counts
        cw /= cw.min()
        self.class_weights = {i: cwi for i, cwi in zip(unique, cw)}
        self.class_weights = torch.FloatTensor(list(self.class_weights.values())).cuda(self.device)
        self.num_classes = len(self.train_dataset.classes)

        if self.robust:
            self.optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, self.model.parameters()), optim.Adam,
                                             lr=self.lr)
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

    @staticmethod
    def parallel_model(args, model, device):
        is_parallel = False
        # If DP mode
        dp_mode = device.type != 'cpu' and args.rank == -1
        if dp_mode and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use DDP instead.\n')
            model = torch.nn.DataParallel(model)
            is_parallel = True

        # If DDP mode
        ddp_mode = device.type != 'cpu' and args.rank != -1
        if ddp_mode:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
            is_parallel = True

        return model, is_parallel

    def de_parallel(self):
        '''De-parallelize a model. Return single-GPU model if model's type is DP or DDP.'''
        return self.model.module if self.is_parallel else self.model

    def save_model(self, best_acc, best_model):
        self.model.load_state_dict(best_model)
        LOGGER.info(f"Best acc:{best_acc}")
        self.model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                val_output = self.model(data)
                val_loss = self.criterion(val_output, label, self.class_weights)

                acc = (val_output.argmax(dim=1) == label).float().sum()
                epoch_val_accuracy += acc
                epoch_val_loss += val_loss
        epoch_val_accuracy /= len(self.test_dataset)
        epoch_val_loss /= len(self.test_dataset)
        LOGGER.info(
            f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        torch.save(deepcopy(self.de_parallel()), self.path_save)

    def eval_model(self, best_acc, best_model, epoch, epoch_loss, epoch_accuracy):
        self.model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                val_output = self.model(data)
                val_loss = self.criterion(val_output, label, self.class_weights)

                acc = (val_output.argmax(dim=1) == label).float().sum()
                epoch_val_accuracy += acc
                epoch_val_loss += val_loss
        epoch_val_accuracy /= len(self.test_dataset)
        epoch_val_loss /= len(self.test_dataset)
        LOGGER.info(
            f"Epoch : {epoch} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
        # self.pbar.set_description(
        #     f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
        if best_acc < epoch_val_accuracy:
            best_acc = epoch_val_accuracy
            best_model = copy.deepcopy(self.model.state_dict())
        # scheduler.step()
        return best_acc, best_model

    def set_parameter_requires_grad(self, requires_grad):
        for param in self.model.module.parameters():
            param.requires_grad = requires_grad

    def set_parameter_requires_grad_cls(self, requires_grad):
        if self.is_parallel:
            for param in self.model.module.classifier.parameters():
                param.requires_grad = requires_grad
        else:
            for param in self.model.classifier.parameters():
                param.requires_grad = requires_grad

    def train_in_step(self, data, label, epoch_accuracy, epoch_loss):
        data = data.to(self.device)
        label = label.to(self.device)

        output = self.model(data)
        loss = self.criterion(output, label, self.class_weights)

        if self.robust:
            # optimizer.zero_grad()
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            output = self.model(data)
            loss = self.criterion(output, label, self.class_weights)
            loss.backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        acc = (output.argmax(dim=1) == label).float().sum()
        epoch_accuracy += acc
        epoch_loss += loss
        return epoch_accuracy, epoch_loss

    def train(self, n_epochs=40, learningrate=3e-5, dft=False):
        if dft:
            self.set_parameter_requires_grad(requires_grad=True)
        else:
            self.set_parameter_requires_grad(requires_grad=False)
            self.set_parameter_requires_grad_cls(requires_grad=True)

        self.optimizer.lr = learningrate

        best_acc = 0
        best_model = None
        for epoch in range(n_epochs):
            if self.is_parallel:
                self.train_sampler.set_epoch(epoch)
            epoch_loss = 0
            epoch_accuracy = 0
            self.model.train()
            self.pbar = enumerate(self.train_loader)
            if self.main_process:
                self.pbar = tqdm(self.pbar, total=len(self.train_loader),
                                 ncols=min(200, shutil.get_terminal_size().columns),
                                 desc=f'Epoch {epoch}/{n_epochs - 1}')
            for _, (data, label) in self.pbar:
                epoch_accuracy, epoch_loss = self.train_in_step(data, label, epoch_accuracy, epoch_loss)

            epoch_accuracy /= len(self.train_dataset)
            epoch_loss /= len(self.train_dataset)

            if self.main_process:
                best_acc, best_model = self.eval_model(best_acc, best_model, epoch, epoch_loss, epoch_accuracy)

        if best_model is not None and self.main_process:
            self.save_model(best_acc, best_model)

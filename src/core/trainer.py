from tqdm import tqdm
import numpy as np
import copy
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader

import timm

from core.loss import cross_entropy_with_label_smoothing
from affectnet.robust_optimization import RobustOptimizer
from utils.general import LOGGER


class Trainer:
    def __init__(self, args):
        self.args = args
        self.batch_size = 64
        self.lr = 3e-5
        self.gamma = 0.7
        self.img_size = 260
        self.tf_mean = [0.485, 0.456, 0.406]
        self.tf_std = [0.229, 0.224, 0.225]
        self.robust = True
        self.path_ckpt = '../models/pretrained_faces/state_vggface2_enet0_new.pt'
        self.path_save = '../run/fer.pt'

        self.dir_train = os.path.join(args.dir_afn_root, 'full_res', 'train')
        self.dir_val = os.path.join(args.dir_afn_root, 'full_res', 'val')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda:7' if self.use_cuda else 'cpu'

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

        kwargs = {'num_workers': 4, 'pin_memory': True} if self.use_cuda else {}
        self.train_dataset = datasets.ImageFolder(root=self.dir_train, transform=self.train_transforms)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                                        **kwargs)
        self.test_dataset = datasets.ImageFolder(root=self.dir_val, transform=self.test_transforms)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                                       **kwargs)

        (unique, counts) = np.unique(self.train_dataset.targets, return_counts=True)
        cw = 1 / counts
        cw /= cw.min()
        self.class_weights = {i: cwi for i, cwi in zip(unique, cw)}
        self.class_weights = torch.FloatTensor(list(self.class_weights.values())).cuda(self.device)
        self.num_classes = len(self.train_dataset.classes)

        self.criterion = cross_entropy_with_label_smoothing

        self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
        self.model.classifier = torch.nn.Identity()
        self.model.load_state_dict(torch.load(self.path_ckpt))

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=self.num_classes))  # 1792 #1280 #1536
        self.model = self.model.to(self.device)

        if self.robust:
            self.optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, self.model.parameters()), optim.Adam,
                                             lr=self.lr)
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

    def set_parameter_requires_grad(self, requires_grad):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def set_parameter_requires_grad_cls(self, requires_grad):
        for param in self.model.classifier.parameters():
            param.requires_grad = requires_grad

    def train(self, n_epochs=40, learningrate=3e-5, dft=False):
        if dft:
            self.set_parameter_requires_grad(requires_grad=True)
        else:
            self.set_parameter_requires_grad(requires_grad=False)
            self.set_parameter_requires_grad_cls(requires_grad=True)

        self.optimizer.lr = learningrate

        best_acc = 0
        best_model = None
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0
            epoch_accuracy = 0
            self.model.train()
            for data, label in tqdm(self.train_loader):
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
            epoch_accuracy /= len(self.train_dataset)
            epoch_loss /= len(self.train_dataset)

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
                f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )
            if best_acc < epoch_val_accuracy:
                best_acc = epoch_val_accuracy
                best_model = copy.deepcopy(self.model.state_dict())
            # scheduler.step()

        if best_model is not None:
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
            torch.save(self.model, self.path_save)
        else:
            LOGGER.info(f"No best model Best acc:{best_acc}")

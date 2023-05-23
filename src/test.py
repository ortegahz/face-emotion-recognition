import os
import sys
import argparse
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

import torch
from torchvision import datasets, transforms

if str(os.getcwd()) not in sys.path:
    sys.path.append(str(os.getcwd()))


def plt_conf_matrix(y_true, y_pred, labels, ic, save_name):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_confusion_matrix(ic, y_pred, y_true, display_labels=labels, cmap=plt.cm.Blues, ax=ax)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'run/conf_matrix_{save_name}.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ckpt', default='/tmp/pycharm_project_359/run/fer.pt', type=str)
    parser.add_argument('--dir_afn_root', default='/media/sdb/data/AffectNet', type=str)
    parser.add_argument('--img_size', default=260, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info(args)

    device = args.device

    model = torch.load(args.path_ckpt, map_location=device)
    model = model.eval()

    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    dir_train = os.path.join(args.dir_afn_root, 'full_res', 'train')
    dir_val = os.path.join(args.dir_afn_root, 'full_res', 'val')

    train_dataset = datasets.ImageFolder(root=dir_train, transform=train_transforms)
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    y_val, y_scores_val = [], []
    model.eval()
    for class_name in tqdm(os.listdir(dir_val)):
        if class_name in class_to_idx:
            class_dir = os.path.join(dir_val, class_name)
            y = class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                filepath = os.path.join(class_dir, img_name)
                img = Image.open(filepath)
                img_tensor = test_transforms(img)
                img_tensor.unsqueeze_(0)
                scores = model(img_tensor.to(device))
                scores = scores[0].data.cpu().numpy()
                y_scores_val.append(scores)
                y_val.append(y)

    y_scores_val = np.array(y_scores_val)
    y_val = np.array(y_val)

    y_pred = np.argmax(y_scores_val, axis=1)
    acc = 100.0 * (y_val == y_pred).sum() / len(y_val)
    logging.info(f'mean acc for eight classes -> {acc}')
    y_train = np.array(train_dataset.targets)
    for i in range(y_scores_val.shape[1]):
        _val_acc = (y_pred[y_val == i] == i).sum() / (y_val == i).sum()
        logging.info('%s %d/%d acc: %f' % (idx_to_class[i], (y_train == i).sum(), (y_val == i).sum(), 100 * _val_acc))

    # minus Contempt
    idx_contempt = class_to_idx['Contempt']
    y_scores_val_filtered = y_scores_val[:, [i != idx_contempt for i in idx_to_class]]
    y_pred_filtered = np.argmax(y_scores_val_filtered, axis=1)
    other_indices = y_val != idx_contempt
    y_val_new = np.array([y if y < idx_contempt else y - 1 for y in y_val if y != idx_contempt])
    acc = 100.0 * np.mean(y_val_new == y_pred_filtered[other_indices])
    logging.info(f'mean acc for minus-contempt classes -> {acc}')

    labels = list(class_to_idx.keys())
    ic = type('IdentityClassifier', (), {"predict": lambda i: i, "_estimator_type": "classifier"})
    plt_conf_matrix(y_val, y_pred, labels, ic, 'eight')
    labels_7 = [idx_to_class[i] for i in idx_to_class if i != idx_contempt]  # list(class_to_idx.keys())
    plt_conf_matrix(y_val_new, y_pred_filtered[other_indices], labels_7, ic, 'seven')


if __name__ == '__main__':
    main()

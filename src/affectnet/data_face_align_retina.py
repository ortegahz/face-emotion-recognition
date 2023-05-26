import argparse
import logging
import sys
import cv2
import glob
import os

from tqdm import tqdm
import numpy as np

if str(os.getcwd()) not in sys.path:
    sys.path.append(str(os.getcwd()))
from src.utils.general import set_logging

sys.path.append('/tmp/pycharm_project_278')
from retinaface import RetinaFace

sys.path.append('/tmp/pycharm_project_278/common')
import face_preprocess


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def init_dirs(args):
    if args.dir_out_reset:
        os.system(f'rm {args.dir_root_out} -rvf')
    os.makedirs(args.dir_root_out, exist_ok=True)


def init_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root_in', default='/media/sdb/data/AffectNet/full_res', type=str)
    parser.add_argument('--dir_root_out', default='/media/sdb/data/AffectNet/full_res_align', type=str)
    parser.add_argument('--path_model_fd', default='/media/manu-pc/tmp/r50/R50', type=str)
    parser.add_argument('--dir_out_reset', default=True, action='store_true')
    parser.add_argument('--img_sfx', nargs='+', type=str, default=['.jpg', '.png', '.bmp'])
    parser.add_argument('--img_size', default=640, type=int)
    return parser.parse_args()


def main():
    set_logging(name='dfar')
    args = parse_args()
    logging.info(args)
    init_dirs(args)

    detector = RetinaFace(args.path_model_fd, 0, 3, 'net3')
    for subset in ['train', 'val']:
        dir_root_out_subset = os.path.join(args.dir_root_out, subset)
        dir_root_in_subset = os.path.join(args.dir_root_in, subset)
        os.makedirs(dir_root_out_subset, exist_ok=True)
        dirs_id = glob.glob(os.path.join(dir_root_in_subset, '*'))

        for i, dir_id in enumerate(dirs_id):
            basename_dir = os.path.basename(dir_id)
            dir_id_out = os.path.join(dir_root_out_subset, basename_dir)
            os.makedirs(dir_id_out, exist_ok=True)
            paths_img = glob.glob(os.path.join(dir_id, '*'))
            for path_img in tqdm(paths_img):
                logging.info(f'{i} / {len(dirs_id) - 1}')
                basename = os.path.basename(path_img)
                sfx = os.path.splitext(path_img)[-1]
                if sfx not in args.img_sfx:
                    continue
                name, _ = os.path.splitext(basename)
                img = cv2.imread(path_img)
                img_rs, ratio, _ = letterbox(img, new_shape=(args.img_size, args.img_size))
                faces, landmarks = detector.detect(img_rs, 0.8, scales=[1.0], do_flip=True)
                faces /= ratio
                landmarks /= ratio
                if len(faces) != 1:
                    continue
                bbox = faces
                points = np.squeeze(landmarks).transpose().reshape(1, 10)
                bbox = bbox[0, 0:4]
                points = points[0, :].reshape((2, 5)).T
                img_aligned = face_preprocess.preprocess(img, bbox, points, image_size='112,112')
                out_path = os.path.join(dir_id_out, name + '.jpg')
                cv2.imwrite(out_path, img_aligned)


if __name__ == '__main__':
    main()

# -*-coding:utf-8-*-

import argparse
import logging
import copy
import sys
import cv2
import os
import torch
import numpy as np
from multiprocessing import Process, Queue

sys.path.append('/media/manu/kingstop/workspace/demo')
from retinaface import RetinaFace
from process import process_decoder

sys.path.append('/media/manu/kingstop/workspace/demo/common')
import face_preprocess

if str(os.getcwd()) not in sys.path:
    sys.path.append(str(os.getcwd()))
from src.utils.general import set_logging


def plot_fd_results(frame, faces, landmarks):
    for i in range(faces.shape[0]):
        box = faces[i].astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        if landmarks is not None:
            landmark5 = landmarks[i].astype(int)
            for l in range(landmark5.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), 1, color, 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ckpt_fer', default='/home/manu/tmp/run/fer.pt', type=str)
    parser.add_argument('--path_ckpt_fd', default='/home/manu/tmp/mobilenet_v1_0_25/retina', type=str)
    # parser.add_argument('--path_video', default='/media/manu/samsung/videos/at2021/mp4/Video1.mp4', type=str)
    parser.add_argument('--path_video', default='rtsp://192.168.1.185:554/ch0_1', type=str)
    parser.add_argument('--name_window', default='result', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--th_fer', default=0.0, type=float)
    return parser.parse_args()


def main():
    CLASS_NAMES = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    args = parse_args()
    set_logging(0, name='demo')
    logging.info(f'demo args are: {args}\n')

    model_fer = torch.load(args.path_ckpt_fer, map_location=args.device)
    model_fer = model_fer.eval()

    model_fd = RetinaFace(args.path_ckpt_fd, 0, 1, 'net3')
    img = cv2.imread('/media/manu/samsung/pics/material3000_1920x1080.jpg')
    # model_fd warm up
    for _ in range(10):
        _, _ = model_fd.detect(img, 0.3, scales=[1.0], do_flip=True)

    q_decoder = Queue()
    p_decoder = Process(target=process_decoder, args=(q_decoder, args.path_video), daemon=True)
    p_decoder.start()

    b_init = False
    while True:
        try:
            item_frame = q_decoder.get(timeout=10)
        except:
            break
        frame_org = item_frame[0]
        frame = copy.deepcopy(frame_org)
        h, w, c = frame.shape
        if not b_init:
            cv2.namedWindow(args.name_window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(args.name_window, w, h)
            b_init = True
        scales = [0.5] if w > 1920 else [1.0]
        faces, landmarks = model_fd.detect(frame, 0.3, scales=scales, do_flip=False)
        if faces is not None:
            plot_fd_results(frame, faces, landmarks)

            for bbox, kps in zip(faces, landmarks):
                kps = kps.transpose().reshape(1, 10)
                bbox = bbox[0:4]
                kps = kps[0, :].reshape((2, 5)).T
                # img_aligned = cv2.resize(frame_org, (112, 112))
                img_aligned = face_preprocess.preprocess(frame_org, bbox, kps, image_size='112,112')
                img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
                img_aligned = np.transpose(img_aligned, (2, 0, 1))
                img_aligned = torch.from_numpy(img_aligned).unsqueeze(0).float()
                img_aligned.div_(255).sub_(0.5).div_(0.5)
                img_aligned = img_aligned.cuda(args.device)
                scores = model_fer(img_aligned).cpu().detach().numpy().astype('float')
                scores = np.squeeze(scores)
                # [-0.48006934, 0.6168594, 0.6063671, -0.1295718, 1.2593273, -0.70998496, -0.78540003, -0.1893796]
                scores = np.exp(scores) / np.sum(np.exp(scores))
                idx_pred = np.argmax(scores)
                name_pred = CLASS_NAMES[idx_pred]
                score_pred = scores[idx_pred]
                info = name_pred + ' ' + '%f' % score_pred
                pt = (int(bbox[0]), int(bbox[1]))
                if score_pred > args.th_fer:
                    cv2.putText(frame, info, pt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow(args.name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()

import argparse

from utils.general import LOGGER
from core.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_afn_root', default='/media/manu/kingstoo/AffectNet', type=str)
    parser.add_argument('--label_names', nargs='*',
                        default=['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'],
                        type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    LOGGER.info(args)
    trainer = Trainer(args)
    trainer.train(3, 0.001, False)
    trainer.train(6, 1e-4, True)


if __name__ == '__main__':
    main()

from tqdm import tqdm
import pandas as pd
import argparse
import shutil
import os


def new_dirs(args):
    if os.path.exists(args.dir_out_root):
        shutil.rmtree(args.dir_out_root)
    os.makedirs(args.dir_out_root)
    for subset in args.subsets:
        dir_out_subset = os.path.join(args.dir_out_root, f'{subset}')
        os.makedirs(dir_out_subset)
        for label_name in args.label_names:
            dir_out_subset_label = os.path.join(dir_out_subset, f'{label_name}')
            os.makedirs(dir_out_subset_label)


def data_gen(args):
    for subset in args.subsets:
        filename = os.path.join(args.dir_in_root, 'validation.csv') if subset == 'val' else os.path.join(
            args.dir_in_root, 'training.csv')
        img_affect_data_dir = os.path.join(args.dir_in_root, 'Manually_Annotated', 'Manually_Annotated_Images')
        affect_df = pd.read_csv(filename)
        affect_vals = [d for _, d in affect_df.iterrows()]
        for d in tqdm(affect_vals):
            if d.expression >= len(args.label_names) or d.face_width < 0:
                continue
            src_file_path = os.path.join(img_affect_data_dir, d.subDirectory_filePath)
            dst_file_path = os.path.join(args.dir_out_root, subset, args.label_names[d.expression],
                                         os.path.basename(d.subDirectory_filePath))
            shutil.copyfile(src_file_path, dst_file_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in_root', default='/media/manu/kingstoo/AffectNet', type=str)
    parser.add_argument('--dir_out_root', default='/media/manu/kingstoo/AffectNet/full_res', type=str)
    parser.add_argument('--subsets', nargs='*', default=['val', 'train'], type=str)
    parser.add_argument('--label_names', nargs='*',
                        default=['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'],
                        type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    new_dirs(args)
    data_gen(args)


if __name__ == '__main__':
    main()

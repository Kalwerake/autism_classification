import pathlib
import pandas as pd
from feature_extraction.dfc import TwinSurrDFC
import argparse

def main(df_path, data_dir, save_dir, window_length=70):
    df = pd.read_csv(df_path)
    twin_surr_dfc = TwinSurrDFC(df=df, roi_folder=data_dir, save_folder=save_dir)
    twin_surr_dfc.pickle_jar(window_length=window_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='AugDFC',
        description='calculates and stores dfc in .npy format')
    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to data directory', type=pathlib.Path)
    parser.add_argument('--save', help='path to save directory', type=pathlib.Path)
    parser.add_argument('--window', help='sliding window length, default 70', type=int, required=False)

    args = parser.parse_args()

    main(df_path=args.df, data_dir=args.data, save_dir=args.save, window_length=args.window)

import os
import pandas as pd
from feature_extraction.dfc import DFC
import argparse
import pathlib


def main(df_path, data, extension, window, save):
    try:
        os.mkdir(save)
    except FileExistsError:
        pass

    df = pd.read_csv(df_path)

    dfc_construct = DFC(df, data, extension, save)  # DFC class instance initialise

    dfc_construct.get_dfc(window)  # calculate and store dfc matrices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='CalculateDFC',
        description='get all DFC matrices .npy')

    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to time series data directory', type=pathlib.Path)
    parser.add_argument('--extension', help='file extension for time series', type=str)
    parser.add_argument('--window', help='sliding window length', type=int)
    parser.add_argument('--save', help='save directory path', type=pathlib.Path)

    args = parser.parse_args()
    main(df_path=args.df, data=args.data, extension=args.extension, window=args.window, save=args.save)

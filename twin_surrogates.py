import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import os
import argparse
from pyunicorn.timeseries.surrogates import Surrogates
import pathlib

"""
Synthesis twin surrogate data based on time series data in the rois_cc200 directory. 
a 100 twin surrogate multivariate systems are synthesised
Synthesised data is saved in new subdirectory as {subdirectory}/{FILE_ID}_surr_{:03d}.npy' where d goes from 1-100
"""

def main(df_path,data_dir,save_dir):
    """
    df_path: (path) path to phenotype descripton file `~/phenotype_files/pheno_nn.csv`
    data_dir: (path) directory containing original roi time series data
    save_dir: (path) directory to save augmented data
    """
    df = pd.read_csv(df_path)
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    # get all unique identifiers in FILE_ID
    file_id = df.FILE_ID
    # attach suffix to FILE_ID to get time series data file name
    roi_files = [idx + '_rois_cc200.1D' for idx in file_id]
    # get paths to all time series files in rois_cc200
    roi_paths = [os.path.join(data_dir, file) for file in roi_files]

    # all the original time series data will be stored in the same directory as synthesised data
    # to distinguish they will be stored as {FILE_ID}_real.npy
    for i, path in enumerate(roi_paths):
        # make save path for original data
        real_save_path = os.path.join(save_dir, file_id[i] + '_real.npy')
        # load original data
        data = pd.read_csv(path, sep='\t', lineterminator='\n')
        # turn dataframe int0 a numpy array
        data = data.to_numpy()
        # save data to save path
        np.save(real_save_path, data)
        # normalise the data after transposing, transposing is neccasry since ths SUrrogates() module takes a array in the
        # shape NxT (where N is the number of variable, and T the temporal samples)
        data = normalize(data).T
        # Inialise Surrogate module with original data
        t = Surrogates(data)
        # iterating 100 times produces 100 unique synthesized multi-variate time series systems
        for j in range(100):
            # new file name for synthesied data under desired format
            save_name = '{}_surr_{:03d}.npy'.format(file_id[i], j+1)
            # path for saving defined file name
            surr_save = os.path.join(save_dir, save_name)
            # produce a twin surrogate, output is in for NxT
            s = t.twin_surrogates(original_data=data, dimension=3, delay=2, threshold=0.02)
            # output is in then transposed to get array of shapt TxN
            s = s.T
            #save data
            np.save(surr_save, s)
        # clear cache before starting on next sample
        t.clear_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='TwinSurrogates',
        description='Uses the twin surrogate method to synthesise multivariate time series data')
    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to data directory', type=pathlib.Path)
    parser.add_argument('--save', help='path to save directory', type=pathlib.Path)

    args = parser.parse_args()

    main(df_path=args.df, data_dir=args.data, save_dir=args.save)



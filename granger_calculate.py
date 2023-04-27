import pathlib
# import granger causality package
from feature_extraction.granger import gci, large_scale_gci
import os
import pandas as pd
import numpy as np
import argparse


def main(df_path, roi_dir, gci_dir, extension, large_scale=True, vectorise=False):
    """
    df_path: (path) path to phenotype description file (phenotype_file/pheno_nn.csv)
    roi_dir: (path) directory containing time series data
    gci_dir: (path) directory to save gci matrix
    extension: (str) extension
    large_scale: (bool) large scale granger causality (TRUE) or traditional granger causalaity (FALSE)
    vectorise: (bool) stack and vectorise data or not
    """
    # load description df
    df = pd.read_csv(df_path)
    # get file ids
    subjects = df.FILE_ID
    # get names of all time series files
    roi_files = [sub + extension for sub in subjects]
    # get paths to time series files
    roi_paths = [os.path.join(roi_dir, file) for file in roi_files]
    # if directory for gci storage does not exist create
    try:
        os.mkdir(gci_dir)
    except FileExistsError:
        pass

    all_gci = []
    for i, path in enumerate(roi_paths):
        # load timeseries data
        data = pd.read_csv(path, sep='\t', lineterminator='\n')
        # if large scale granger causality calculation
        if large_scale:
            # save granger causlity matrix {gci_dir}/{file_id}_ls_gci.npy
            matrix_name = subjects[i] + '_ls_gci.npy'
            matrix_path = os.path.join(gci_dir, matrix_name)
            # calculate matrix using large_scale_gci() function
            gci_matrix = large_scale_gci(data)
        # if traditional granger causality calculation
        else:
            # save granger causality matrix {gci_dir}/{file_id}_gci.np
            matrix_name = subjects[i] + '_gci.npy'
            matrix_path = os.path.join(gci_dir, matrix_name)
            gci_matrix = gci(data)
        # if vectorised data is to be saved
        if vectorise:
            all_gci.append(gci_matrix.flatten())
        # if square matrix is to be saved
        elif not vectorise:
            np.save(matrix_path, gci_matrix)

    if vectorise:
        # turn all_gci into 2d array of shape (sample_number, roi_number * roi_number)
        vectorised = np.stack(all_gci)
        # the binary classes, binazrise DX_GROUP column
        binary_target = [1 if target == 1 else 0 for target in df.DX_GROUP]
        # get multi-class labels
        multi_target = df.DSM_IV_TR
        # concatenate along columns
        vecs = np.c_[vectorised, binary_target, multi_target]
        # turn into pd data frame
        out_df = pd.DataFrame(vecs)
        # get indexes of columns containing diagonal values
        diagonals = [i for i in range(out_df.shape[1]) if
                     len(out_df.iloc[:, i].unique()) == 1]
        # drop columns containing diagonal data
        out_df.drop(diagonals, axis=1, inplace=True)
        # number of features
        feature_number = vectorised.shape[1] - len(diagonals)
        # make column names out of feature names
        col_names = [f'{i}' for i in range(feature_number)]
        # add name of target columns to column name list
        col_names.extend(['DX_GROUP', 'DSM_IV_TR'])
        # assign names to columns dataframe
        out_df.columns = col_names
        # ensure target columns are integars
        out_df = out_df.astype({'DX_GROUP': 'int32', 'DSM_IV_TR': 'int32'})
        # file name if large scale granger causlity calculation is used
        if large_scale:
            out_name ='gci_ls_cc200_vectorised.csv.gz'
        elif not large_scale:
            out_name = 'gci_cc200_vectorised.csv.gz'
        # save vectorised file in 'vectorised/{out_name}'
        save_path = os.path.join('vectorised', out_name)
        out_df.to_csv(save_path, index=False, sep='\t', compression='gzip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LargeScaleGranger',
        description='Calculate and stores granger causality measures')

    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to time series data directory', type=pathlib.Path)
    parser.add_argument('--save', help='save directory path', type=pathlib.Path)
    parser.add_argument('--suffix', help='file name suffix after id', type=str)
    parser.add_argument('--large', help='calculate large scale index or not', action='store_true')
    parser.add_argument('--no-large', dest='large', action='store_false')
    parser.add_argument('--vectorise', help='vectorise or not', action='store_true', required=False)
    parser.add_argument('--no-vectorise', dest='vectorise', action='store_false', required=False)
    args = parser.parse_args()

    main(df_path=args.df, roi_dir=args.data, gci_dir=args.save, extension=args.suffix, large_scale=args.large,
         vectorise=args.vectorise)

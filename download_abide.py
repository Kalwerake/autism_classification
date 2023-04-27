import os
import wget
from nilearn.datasets import fetch_abide_pcp
import argparse
import shutil
"""
Download all phenotype description files and derivative data if required.
"""


def main(phenotype_files = True, atlas_labels = True,  derivative = 0):
    '''
    Download phenotype files
    If derivative is given then download derivative
    phenotype_files: (bool) if phenotype files need to be downloaded True
    derivative: (str) options: func_preproc,rois_aal, rois_cc200, rois_cc400,
                            rois_dosenbach160, rois_ez, rois_ho, rois_tt
    '''

    # directory where all phenotype description files are stored
    if phenotype_files:
        dir_name = 'phenotype_files'
        try:
            os.mkdir(dir_name)
        except FileExistsError:
            pass

        # URL for downloading phenotype description file
        pheno_url = 'http://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE/Phenotypic_V1_0b_preprocessed.csv'

        # download file to `phenotype_files/`
        wget.download(pheno_url, out=dir_name)

        print(f'phentype description file saved in {dir_name}')

    if atlas_labels:

        # store all atlas labels in directory `atlas`
        atlas_dir_name = 'atlas'
        try:
            os.mkdir(atlas_dir_name)
        except FileExistsError:
            pass

        # all atlases used in study
        atlas_names = ['aal', 'CC200_ROI', 'CC400_ROI', 'dos160', 'ez', 'tt', 'ho']

        # download urls for atlas label files
        url_all = [f'http://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE/Resources/{atlas}_labels.csv' for atlas in atlas_names]

        for url in url_all:
            wget.download(url, out=atlas_dir_name)

        print(f'atlas_label files saved in {atlas_dir_name}')

    # if derivative must be downloaded
    if derivative != 0:
        try:
            # make subdirectory with derivative name
            os.mkdir(derivative)
        except FileExistsError:
            pass
        # download derivative to folder
        fetch_abide_pcp(data_dir=derivative, pipeline='cpac', band_pass_filtering=True,
                        global_signal_regression=True, derivatives=derivative)

        # the fetch_abide_pcp() function creates a subdirectory tree within derivative and saves within
        data_path = os.path.join(derivative, 'ABIDE_pcp/cpac/filt_global')

        # move all files to derivative/ and remove tree
        for filename in os.listdir(data_path):
            shutil.move(os.path.join(data_path, filename), os.path.join(derivative, filename))

        shutil.rmtree(os.path.join(derivative, 'ABIDE_pcp'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Download',
        description='Download phenotypic description files and atlas label files')
    parser.add_argument('--phenotype', help='download phenotype', action='store_true', required=False)
    parser.add_argument('--no-phenotype', dest='phenotype', action='store_false', required=False)
    parser.add_argument('--atlas', help='download atlas labels', action='store_true', required=False)
    parser.add_argument('--no-atlas', dest='atlas', action='store_false', required=False)
    parser.add_argument('--derivative', help='derivative to download (optional)', type=str, required=False)

    args = parser.parse_args()
    main(phenotype_files=args.phenotype, atlas_labels=args.atlas, derivative=args.derivative)

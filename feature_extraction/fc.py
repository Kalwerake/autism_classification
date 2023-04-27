import os
import pandas as pd
import numpy as np
import itertools
import sys

class FetchROI:
    """
    load ROI time series data
    """

    def __init__(self, roi_path):
        """
        iniatialise object
        :param roi_path: full file path to roi timeseries data
        """
        self.roi_dir = roi_path

    def fetch_roi_avg_ts(self, roi_file):
        """
        :param roi_file: specific time series  data file name
        :return:
        """
        roi_path = os.path.join(self.roi_dir, roi_file)  # path to time series data
        output = pd.read_csv(roi_path, sep='\t', lineterminator='\n')  # load tap seperated file as pandas dataframe
        return output


class FC:
    """
    Calculating and storing  functional correlation data in .pkl format.

    Input folder should contain extracted time series data based on CC200 atlas.

    Data must be stored in BIDS format, with file extension `.1D`, tab seperated '\t' and line terminator \n

    needs:
    import os
    import pickle
    import pandas as pd """

    def __init__(self, df, save_dir, data_source='local', roi_folder=0, extension=0):
        """
        description_df: (pandas DataFrame)
            pandas dataframe containing phenotypic data,
            and extracted time series data file names under column 'CC200'.
        roi_folder: (path)
                    input folder path containing all time series data as subdirectory in main directory.
        pickle_folder:(path)
                     subdirectory folder name for storing pickle files containing dynamic correlation data.

        """
        self.df = df
        # Access roi data file names
        # subdirectory containing  roi data
        self.roi_folder = roi_folder
        self.roi_ids = df.FILE_ID

        # if roi time series data is stored locally
        if data_source == 'local':
            self.roi_files = [idx + extension for idx in self.roi_ids]
            # all file paths for accessing ROI time series data

            self.roi_paths_all = [os.path.join(roi_folder, file) for file in self.roi_files]

        # if roi data hasnt been downloaded
        elif data_source == 'aws':
            # all url paths
            self.roi_paths_all = [f'http://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE/Outputs/cpac/filt_global/' \
                                  f'{roi_folder}/{file_id}_{roi_folder}.1D' for file_id in self.roi_ids]

        self.save_dir = save_dir

        self.fc_file_all = [idx + '_fc.npy' for idx in self.roi_ids]  # paths to all .pkl files

        self.save_paths_all = [os.path.join(self.save_dir, file) for file in self.fc_file_all]

    @staticmethod
    def fc_calculate(roi_path, triangle=False):
        """
        calculate and return FC matrix, if triangle =True only the upper triangle of matrix is returned
        roi_path: path to roi url or local path
        triangle: only extract triangle bool (False)
        """
        # load time series
        ts_df = pd.read_csv(roi_path, sep='\t', lineterminator='\n')
        # check for missing data in time series, if time series is absent all values are at 0
        missing = []
        for i in range(ts_df.shape[1]):
            # if the number unique values in a roi is 1 then the data is missing append to missing
            if len(ts_df.iloc[:, i].unique()) == 1:
                missing.append(i)
        # if file contains missing roi time series then raise error
        if len(missing) != 0:
            raise ValueError('This subject contains missing values')

        else:
            fc = ts_df.corr(method='pearson')

        # if triangle equal true return only upper triangle of matrix and shape of matrix ie roi number
        if triangle:
            fc_np = fc.to_numpy()
            return fc_np.shape[1], upper_tri_indexing(fc_np)
        else:
            return fc.to_numpy()

    def fc(self):
        """
        Calculate functional connectivity via Pearson correlation,
        if save = True then each matrix is saved
        else return a 3d numpy array with all matrices
        """
        # initialise missing data list
        missing_data = []
        for i, roi_path in enumerate(self.roi_paths_all):
            try:
                # try calculate fc matrix and save
                fc = self.fc_calculate(roi_path)
                np.save(self.save_paths_all[i], fc)
            # if error raised due to missing roi data, append file_id to missing and skip to next
            except ValueError:
                missing_data.append(self.roi_ids[i])
                pass

        # if files with missing data was detected save file ids to txt
        if len(missing_data) > 0:
            missing_name = f'{self.roi_folder}_missing.txt'

            with open(os.path.join(self.save_dir, missing_name), mode='wt', encoding='utf-8') as miss_file:
                miss_file.write('\n'.join(missing_data))

    def vectorise(self, save=True):
        """
        Vectorise and save all fc matrices
        """


        # calculate all fc_matrices and store in 3d array
        all_fc = []
        # initialise missing_data for missing file_id storage
        missing_data = []
        non_missing = []
        for i, roi_path in enumerate(self.roi_paths_all):
            try:
                # try to calculate fc take only the upper triangle of matrix
                roi_number, fc = self.fc_calculate(roi_path, triangle=True)
                # append vectorised fc upper triangle to all_fc
                all_fc.append(fc)
                # append index to non_missing
                non_missing.append(i)
            # if ValueError is met then there is a null value in one of the rois of dataset, append to missing_data
            except ValueError:
                missing_data.append(self.roi_ids[i])
                pass
        # if any missing value rois found in subject
        if len(missing_data) > 0:
            # sub directory to contain missing data lists
            try:
                os.mkdir('missing_lists')
            except FileExistsError:
                pass
            # name of missing list file
            missing_name = f'{self.roi_folder}_missing.txt'
            # write missing file id list to txt file
            with open(os.path.join(self.save_dir, missing_name), mode='wt', encoding='utf-8') as miss_file:
                miss_file.write('\n'.join(missing_data))

        # binary targets for all non-missing subjects
        b_t = self.df.loc[non_missing, 'DX_GROUP']
        binary_target = [1 if target == 1 else 0 for target in b_t]
        # multiclass targets for all non-missing subjects
        multi_target = self.df.loc[non_missing, 'DSM_IV_TR']
        # turn all_fc list to np array
        vectorised = np.stack(all_fc)
        # concatenate vectorised matrices with target data
        vecs = np.c_[vectorised, binary_target, multi_target]
        # turn to pd dataframe
        out_df = pd.DataFrame(vecs)

        # get col names by making a non repeating combination of all roi numbers,
        # if atlas cc400 then numbers need to based of CC400_ROI_labels.csv data
        if 'cc400' in self.roi_folder:
            atlas_labels = pd.read_csv('atlas/CC400_ROI_labels.csv').iloc[:, 0]
            col_names = [f"#{i[0]}-#{i[1]}" for i in list(itertools.combinations(atlas_labels.values, 2))]
        else:
            col_names = [f"#{i[0] + 1}-#{i[1] + 1}" for i in list(itertools.combinations(range(roi_number), 2))]

        col_names.extend(['DX_GROUP', 'DSM_IV_TR'])
        out_df.columns = col_names

        out_df = out_df.astype({'DX_GROUP': 'int32', 'DSM_IV_TR': 'int32'})
        # if save then save in standardised format
        if save:

            out_name = f'fc_{self.roi_folder.replace("_rois_", "")}_vectorised.csv.gz'
            save_path = os.path.join(self.save_dir, out_name)
            out_df.to_csv(save_path, index=False, sep='\t', compression='gzip')
        elif not save:
            sys.stdout.write(f"Data stored in {vectorised}")
            return out_df


class TwinSurrFC:
    """
    Calculating and storing  functional correlation data in .pkl format.

    Input folder should contain extracted time series data based on CC200 atlas.

    Data must be stored in BIDS format, with file extension `.1D`, tab seperated '\t' and line terminator \n

    needs:
    import os
    import pickle
    import pandas as pd """

    def __init__(self, df, idx_column, roi_folder, save_dir):
        """
        description_df: (pd.DataFrame) train_surr.csv or test_surr.csv

        roi_folder: (path)
                    input folder path containing all time series data as subdirectory in main directory.
        pickle_folder:(path)
                     subdirectory folder name for storing pickle files containing dynamic correlation data.

        """
        self.df = df
        # Access roi data file names
        # subdirectory containing  roi data
        self.roi_folder = roi_folder
        self.save_dir = save_dir

        self.roi_ids = df.loc[:, idx_column]

        self.roi_files = [idx.replace('_dfc.npy', '.npy') for idx in self.roi_ids]
        # all file paths for accessing ROI time series data
        self.roi_paths_all = [os.path.join(self.roi_folder, file) for file in self.roi_files]

        self.fc_file_all = [idx.replace('_dfc.npy', '_fc.npy') for idx in self.roi_ids]  # save to paths

        self.save_paths_all = [os.path.join(self.save_dir, file) for file in self.fc_file_all]

        self.roi_number = 0

    def fc(self):
        """
        call pickle_jar() method for extraction of DFC data and storage in .pkl format, no arguments needed.
        """

        for i, roi_path in enumerate(self.roi_paths_all):  # take index and value of list roi_paths_all containing
            # path names for all time series data
            ts_df = np.load(roi_path)
            fc = np.corrcoef(ts_df, rowvar=False)
            self.roi_number = fc.shape[0]
            np.save(self.save_paths_all[i], fc)


def upper_tri_indexing(cov_matrix):
    matrix_length = cov_matrix.shape[0]
    r, c = np.triu_indices(matrix_length, 1)
    return cov_matrix[r, c]

import os
import numpy as np

from sklearn.preprocessing import normalize
from numpy.linalg import eigh, pinv
from statsmodels.tsa.api import VAR


def gci(df):
    """
    :param df: time series file for one subject
    :return:granger causality index matrix

    Traditional granger causality index calculation without using the large scale granger causality algorithm.
    No feature reduction via PCA
    """
    _, rois = df.shape
    X = df.to_numpy()  #
    Xn = normalize(X)  # normalize X

    mvar = VAR(Xn)
    results = mvar.fit(2)  # fit model with maxlag of 2

    E_hat = results.resid

    GCI = np.zeros((rois, rois))
    for i in range(rois):
        GCI[i, i] = 1

        X_iMinus = np.delete(Xn, i, 1)  # remove ith column from X
        mvar_minus = VAR(X_iMinus)  # initialise new model with X_iminus
        results_minus = mvar_minus.fit(2)  # fit model with maxlag of 2

        E_m = results_minus.resid  # get error matrix of predictions without ith feature
        E_minus = np.insert(E_m, i, 0, axis=1)  # add a dummy column at removed column

        for j in range(rois):
            if j != i:
                gci = np.log(np.var(E_minus[:, j]) / np.var(E_hat[:, j]))

                GCI[j, i] = max(gci, 0)  # from i to j at [j,i] insert calculate gci
            else:
                continue

    return GCI


def large_scale_gci(df, is_pd=True):
    if is_pd:
        X = df.to_numpy()  #
        Xn = normalize(X)  # normalize X
    else:
        Xn = normalize(df)
    roi_number = Xn.shape[-1]

    cov = np.cov(Xn, rowvar=False)  # construct covariance matrix of features, state that feature data is not in row
    eigval, eigvec = eigh(cov)  # eigenvalue decomposition, eval(eigenvalues), eigvec(eigenvectors) is a matrix of
    # eigen vectors
    idx = eigval.argsort()[::-1]  # get indices of sorted eigenvalues and reverse list to get descending order
    eigval = eigval[idx]  # eigen values sorted in descending order
    W = eigvec[:, idx]  # eigen vectors sorted with respect to eigenvalues giving projection matrix W
    W_c = W[:, :35]  # choose the first 35 eigen vectors
    Z = np.dot(Xn, W_c)  # project data on to 35 dimensional space

    mvar = VAR(Z)
    results = mvar.fit(2)  # fit model with maxlag of 2

    z = results.fittedvalues  # model predictions has only 194 rows due to the lag of 2, I will add first two rows from X_ld
    z_hat = np.concatenate((Z[:2, :], z), axis=0)  # concatenate first two rows of X_ld to z
    W_plus = pinv(W_c)  # get pseudo inverse of projection matrix
    E_hat = Xn - np.dot(z_hat, W_plus)

    lsGCI = np.zeros((roi_number, roi_number))
    for i in range(roi_number):
        lsGCI[i, i] = 1
        X_iMinus = np.delete(Xn, i, 1)  # remove ith column from X remove feature from HD space
        W_iMinus = np.delete(W_c, i, 0)  # Remove ith row from projection matrix W
        Z_minus = np.dot(X_iMinus, W_iMinus)  # project matrix without column i onto 35d space
        mvar_minus = VAR(Z_minus)  # initialise new model with Z_minus
        results = mvar_minus.fit(2)  # fit model with maxlag of 2
        z_minus_pred = results.fittedvalues  # model predictions has only 194 rows due to the lag of 2, I will add first two rows from Z_minus
        z_m_hat = np.concatenate((Z_minus[:2, :], z_minus_pred), axis=0)  # concatenate first two rows of X_ld to z

        W_m_plus = pinv(W_iMinus)  # get pseudo inverse of W_iMinus
        E_m = X_iMinus - np.dot(z_m_hat, W_m_plus)  # get error matrix of predictions without ith feature
        E_minus = np.insert(E_m, i, 0, axis=1)  # add a dummy column at removed column

        for j in range(roi_number):
            if j != i:

                GCI = np.log(np.var(E_minus[:, j]) / np.var(E_hat[:, j]))
                lsGCI[i, j] = max(GCI, 0)  # from i to j
            else:
                continue

    return lsGCI


class MatrixMean:
    def __init__(self, df, data_dir, extension, binary=True):
        """
                    :param data_dir:
                    :param df:
                    :param extension: file extension and unique identifier after file id
                    :return:
                    """
        self.df = df
        self.data_dir = data_dir
        self.ext = extension

        self.groups = {}
        if binary:
            self.column_name = 'DX_GROUP'
            self.groups['asd'] = 1
            self.groups['control'] = 2
        else:
            self.column_name = 'DSM_IV_TR'
            self.groups['mc_0'] = 0
            self.groups['mc_1'] = 1
            self.groups['mc_2'] = 2
            self.groups['mc_3'] = 3
            self.groups['mc_4'] = 4

    def mean_gci(self):

        for name, label in self.groups.items():  # parse **groups obtain names, labels

            file_ids = self.df[self.df[self.column_name] == label].FILE_ID  # get unique ids for all subject in class

            files = [idx + self.ext for idx in file_ids]  # get file names for metrices
            gci_paths = [os.path.join(self.data_dir,f) for f in files]

            all_matrices = [np.load(gci_paths[i]) for i in range(len(files))]  # stack each matrix into 3d list
            mean = np.mean(np.array(all_matrices),
                           axis=0)  # make 3d matrix into 3d numpy array and find elementwise mean
            save_path = os.path.join(self.data_dir, f'{name}_mean.npy')  # path for saving mean matrix

            np.save(save_path, mean)

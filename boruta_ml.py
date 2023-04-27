import os

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.svm import SVC

import xgboost as xgb
import argparse
import pathlib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""
Feature selection via boruta algorithm and cross validation on three ml models, for each atlas

"""


def get_spec_sense(true_y, preds):
    """
    Calculate specificity and sensitivity
    true_y: True classes
    preds: predicted classes
    """
    # Use sklearn confusion matrixs to get true positive, etc
    tn, fp, fn, tp = confusion_matrix(true_y, preds).ravel()
    # calculate specificity and sensitivity
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    # output specificity and sensitivity
    return specificity, sensitivity


def get_metrics_cv(model, cv, x, y):
    """
    Fit model and out metrics as a pandas dataframe
    model: (sklearn module) Un-initialised sklearn model class
    cv: (sklearn module) cross validation module class
    x: features
    y: target
    """
    train_accuracy = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    specificity = []
    sensitivity = []
    cross_val = cv

    # for cross validation split
    for train, test in cross_val.split(x, y):

        model = model
        # get train and test data
        X_train, X_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        # fit model and get predicitons
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # get all metrics
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f = f1_score(y_test, predictions)
        spec, sense = get_spec_sense(y_test, predictions)
        # get confusion matrix

        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1.append(f)
        specificity.append(spec)
        sensitivity.append(sense)
    # make dictionary with keys and values
    metric_dict = {'train_accuracy': train_accuracy, 'test_accuracy': accuracy,
                   'precision': precision, 'recall': recall, 'f1_score': f1,
                   'specificity': specificity, 'sensitivity': sensitivity}
    # output a dataframe with all metrics
    return pd.DataFrame(metric_dict)


def main(df_path, atlas, filtered=False):
    """
    df_path: path to vectorised data set
    atlas: name of atlas used
    filtered: (bool) if True then the data has been filtered by the Boruta algorithm
    """

    fc_df = pd.read_csv(df_path, compression='gzip', header=0, sep='\t')

    # if data is not filtered via boruta then apply boruta algorithm to vectorised data
    if filtered is False:
        # take all features
        X = fc_df.iloc[:, :-2].to_numpy()
        # take target column
        y = fc_df.loc[:, 'DX_GROUP']
        # apply seed algorithm max depth of 5
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

        # define Boruta feature selection method
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)

        # find all relevant features
        feat_selector.fit(X, y)

        # call transform() on X to filter it down to selected features
        X_filtered = feat_selector.transform(X)

        # 1D boolean mask to be applied to feature list gives list of selected features
        support = feat_selector.support_

        selected = fc_df.columns[:-2][support]  # selected features
        selected = np.r_[selected, ['DX_GROUP']]  # add target column names to list

        filtered_df = fc_df[selected]  # filter out selected columns and store

        # save filtered dataset in a standardise format
        filter_save = f'vectorised/{atlas}_filtered.csv'
        filtered_df.to_csv(filter_save, index=False)  # store filtered data as csv
        print(f"{atlas} filtered")

    # if inputting a filtered dataframe with selected features
    elif filtered:
        # read dataset
        df = pd.read_csv(df_path)
        # Get features
        X_filtered = df.drop(columns='DX_GROUP').to_numpy()
        # get target
        y = df.loc[:, 'DX_GROUP']

    # if model evaluation folder does not exist, create
    try:
        os.mkdir('model_evaluation')
    except FileExistsError:
        pass
    # a subdirectory is needed to store evaluation metrics for each atlas based dataset
    try:
        atlas_store = f'model_evaluation/{atlas}'
        os.mkdir(atlas_store)
    except FileExistsError:
        pass
    # initialise models
    svm = SVC(kernel='rbf')
    boost = xgb.XGBClassifier()
    lr = LogisticRegression()
    # initialise the k-fold cross validation, 80/20 split
    cv = StratifiedShuffleSplit(n_splits=10, random_state=0, test_size=0.2)
    # fit models with filtered features and get all metrics
    svm_metrics, svm_confusion = get_metrics_cv(svm, cv, X_filtered, y)
    boost_metrics, boost_confusion = get_metrics_cv(boost, cv, X_filtered, y)
    lr_metrics, lr_confusion = get_metrics_cv(lr, cv, X_filtered, y)
    # save all metrics as csv in 'model_evaluation/{atlas}/'
    svm_metrics.to_csv(os.path.join(atlas_store, 'svm_scores.csv'), index=False)
    boost_metrics.to_csv(os.path.join(atlas_store, 'xgb_scores.csv'), index=False)
    lr_metrics.to_csv(os.path.join(atlas_store, 'log_scores.csv'), index=False)

    np.save(os.path.join(atlas_store, 'mean_conf_matrix_svm.npy'), svm_confusion)
    np.save(os.path.join(atlas_store, 'mean_conf_matrix_xgb.npy'), boost_confusion)
    np.save(os.path.join(atlas_store, 'mean_conf_matrix_log.npy'), lr_confusion)

    print(f"{atlas} metrics saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='',
                                     description='cross validate multiple models on one atlas data')

    parser.add_argument('--atlas', help='atlas name', type=str)
    parser.add_argument('--df', help='path to data', type=pathlib.Path, required=False)
    parser.add_argument('--filtered', help='is data boruta filtered', action='store_true', required=False)
    parser.add_argument('--no-filtered', dest='filtered', action='store_false', required=False)
    args = parser.parse_args()

    main(atlas=args.atlas, df_path=args.df, filtered=args.filtered)

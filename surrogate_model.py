import torch
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
# import functions needed for model training from nn_functions package
from nn_functions.functions import train_one_epoch_binary, validate_one_epoch_binary, EarlyStopper
# import crnn model
from nn_functions.models import CRNN5
# import custom dataset
from nn_functions.custom_dataset import SurrDataset

from torch.utils.data import DataLoader
import torch.optim as optim

import torch.nn as nn
import argparse
import pathlib


def train_test_df(X, y):
    '''The original data annotation file  contain unique identifiers in column FILE_ID,
    in order to keep train and test data separate, twin surrogates obtained from original data in train and test must be
    kept separate. This script will preserve segregation, by adding suffixes to original unique identifiers and create
    train and test annotation files that point towards augmented data.
    '''
    # create names for all surrogate data files and pair with target class
    surr_files = [['{}_surr_{:03d}_dfc.npy'.format(X.values[i], j + 1), y[i]] for i in range(len(X)) for j in
                      range(100)]  # add suffix to ids indicating surrogate origin, add target to second column
    # create names for real data
    real_files = [[f'{X[i]}_real_dfc.npy', y[i]] for i in
                      range(len(X.values))]  # add suffix indicating real origin
    roi_files = [*surr_files, *real_files]  # concatenate two nested lists
    return pd.DataFrame(roi_files, columns=['FILE_ID', 'TARGET'])  # make pandas dataframe with column names


def remove_low(in_df, data_dir):
    """
    The shape of DFC arrays must remain constant at (42,200,200), to achieve this the temporal samples must be constant at
    116 time lengths. The surrogate method, synthesizes time series of less length compared to original.
    Therefore, any time series below 116 time points must be removed.

    in_df: (pandas DataFrame) train_df or test_df
    data_dir: (path) subdirectory containing augmented data
    """
    # get names for all augmented data files
    all_files = os.listdir(data_dir)
    # paths to all augmented files
    all_paths = [data_dir + file for file in all_files]

    ids = []
    # iterate over all file paths
    for i, f in enumerate(all_paths):
        # load time series data
        got = np.load(f)
        # if the length is less than 116 append the file name to ids
        if len(got) < 116:
            ids.append(all_files[i])
    # replace .py extension with '_dfc.npy' to get file_name of DFC array
    cut = [ts.replace('.npy', '_dfc.npy') for ts in ids]
    # remove rows in original df where FILE_ID is in list cut
    new_df = in_df[~in_df.FILE_ID.isin(cut)]
    # reset index
    new_df.reset_index(drop=True, inplace=True)
    #output altered df
    return new_df

def main(df_path, data_dir, batch_size, lr, epochs, workers, model_save = False, model_name = 0,
         early_stop=False, patience=0, min_delta=0):
    """

    :param df_path: path to annotation file (pheno_nn.csv)
    :param data_dir:
    :param batch_size:
    :param lr:
    :param epochs:
    :param workers:
    :param model_name: (str) name of model
    :param metric_dir: path for saving model_metric
    :param early_stop: (bool) if true early stop condition is enacted default = False
    :param min_delta: minimum
    :param patience:
    :return:
    """



    df = pd.read_csv(df_path)

    sub_ids = df.FILE_ID  # file ids unique identifiers
    targets = [1 if i == 1 else 0 for i in df.DX_GROUP]  # binarize target classes

    # split dataset to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(sub_ids, targets, test_size = 0.2, random_state = 42)
    # get train and test annotation files for surrogate data
    train_df = train_test_df(X_train, y_train)
    test_df = train_test_df(X_test, y_test)
    # remove all references to dfc arrays with less than 42 frames
    train_df = remove_low(train_df, 'rois_cc200')
    test_df = train_test_df(test_df, 'rois_cc200')

    # iniatlise Dataset class for train set
    train_data = SurrDataset(train_df, data_dir)
    # make dataloader class for minibatching
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=workers)

    test_data = SurrDataset(test_df, data_dir)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=workers)
    # set device as gpu or cpu if not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inialise model
    model = CRNN5()
    # define loss function binary cross entrophy loss
    loss_fn = nn.BCELoss()
    # define optimser algorithm as adam with weight decay parameter of 1e-2, lr can be defined in main function
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # send model to device
    model.to(device)

    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    # if early stop condition is true, iniatlise Early stopper condition, using defined parameters
    if early_stop:
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    else:
        pass

    # iterate over defined number of epochs
    for i in range(epochs):
        # set model to training mode
        model.train()
        # train for one epoch get train_loss and train accuracy
        train_loss, train_acc = train_one_epoch_binary(model, loss_fn, optimiser, train_dataloader, device)
        # append to loss and accuracy hostory
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        # set model to evaluation model
        model.eval()
        # get loss and acc for validation set
        test_loss, test_acc = validate_one_epoch_binary(model, loss_fn, test_dataloader, device)

        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        # if early_stopper counter reachers patience limit then it outputs true if True then stop training
        if early_stop and early_stopper.early_stop(test_loss):
            break
        # print the train and validation metrics every 5 epochs
        if (i + 1) % 10 == 0:
            print(
                f'Epoch {i + 1} train_loss: {round(train_loss, 2)}, accuracy: {round(train_acc, 2)}, test_loss:{round(test_loss, 2)}, test_acc: {round(test_acc, 2)} ')
    #save model parameters to model_save pathname
    if model_save:
        try:
            os.mkdir('network_params')
        except FileExistsError:
            pass


        torch.save(model.state_dict(), os.path.join('network_params', f'{model_name}.pth' ))

    # if subdirectory for metric storage does not exist create
    try:
        os.mkdir('model_evaluation')
    except FileExistsError:
        pass
    # make dictionary out of metrics
    metrics = {'train_acc_history': train_acc_history, 'train_loss_history': train_loss_history,
               'test_acc': test_acc_history, 'test_loss': test_loss_history}  # make dictionary of metrics

    metric_path = os.path.join('model_evaluation', f'{model_name}.pickle')
    # save dictionary in pickle format
    with open(metric_path, 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Train',
        description='train, save and evaluate model')

    parser.add_argument('--df', help='path to df csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to data directory', type=pathlib.Path)
    parser.add_argument('--batch', help='batch size', type=int)
    parser.add_argument('--lr', help='learning rate', type=float)
    parser.add_argument('--epochs', help='epoch number', type=int)
    parser.add_argument('--workers', help='num workers for data loader', type=int)
    parser.add_argument('--model_save', help='save model parameters or not', action='store_true', required=False)
    parser.add_argument('--no-model_save', dest='model_save', action='store_false', required=False)
    parser.add_argument('--model_name', help='name of model', type=str, required=False)
    parser.add_argument('--early_stop', help='enact early stopping?', action='store_true', required=False)
    parser.add_argument('--no-early_stop', dest='early_stop', action='store_false', required=False)
    parser.add_argument('--patience', help='patience', type=int, required=False)
    parser.add_argument('--delta', help='minimum change in validation loss', type=float, required=False)

    args = parser.parse_args()

    main(df_path=args.df, data_dir=args.data, batch_size=args.batch, lr=args.lr,
         epochs=args.epochs, workers=args.workers, model_save=args.model_save, model_name=args.model_name,
         early_stop=args.early_stop, patience=args.patience, min_delta=args.delta)

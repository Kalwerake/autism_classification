
import numpy as np
import os
from torch.utils.data import Dataset
import torch


class SurrDataset(Dataset):

    def __init__(self, df, dfc_dir):
        """
        :param df: pandas dataframe containing subject ids in column `FILE_ID` and target labels in column `TARGET`
        :param dfc_dir: path to subdirectory containing the pickle files
        """
        self.df_main = df
        self.dfc_dir = dfc_dir
        self.files = df.FILE_ID
        self.paths = [os.path.join(dfc_dir, f) for f in self.files]  # paths to pkl files

    def __getitem__(self, idx):
        """
        :param idx: index
        :return: 4D tensor with 2 spatial dimensions [Batch_size,1,200,200], and label
        """
        dfc_pickle = np.load(self.paths[idx])  # load pkl file as dictionary

        dfc_tensor = torch.tensor(dfc_pickle, dtype=torch.float32)  # convert arrays to torch tensor of type float32

        x_out = torch.unsqueeze(dfc_tensor, dim=0)  # add dimension for batch, shape [batch_size, height, width]

        label = self.df_main.loc[idx, 'TARGET']  # class label of idx as np array
        label_tensor = torch.tensor(label, dtype=torch.float32)
        label_out = torch.unsqueeze(label_tensor, dim=-1)  # label output as tensor of shape [batch size, 1]

        return x_out, label_out  # return label as tensor of shape [batch size, 1]

    def __len__(self):
        """
        :return: length of dataset
        """
        return len(self.df_main)

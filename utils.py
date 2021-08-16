# Utility functions required for preprocessing, training and testing
# AudioDataset class reused from https://github.com/dansuh17/segan-pytorch 

import os

import numpy as np
import torch
from torch.utils import data

from data_preprocess import serialized_test_folder, serialized_train_folder


from scipy import signal

# Pre-emphasis filter
def pre_emphasis(signal_batch, emph_coeff=0.95) -> np.array:
    """
    Pre-emphasis of higher frequencies given a batch of signal.
    Args:
        signal_batch(np.array): batch of signals, represented as numpy arrays
        emph_coeff(float): emphasis coefficient
    Returns:
        result: pre-emphasized signal batch
    """
    return signal.lfilter([1, -emph_coeff], [1], signal_batch)

# De-emphasis filter
def de_emphasis(signal_batch, emph_coeff=0.95) -> np.array:
    """
    De-emphasis operation given a batch of signal.
    Reverts the pre-emphasized signal.
    Args:
        signal_batch(np.array): batch of signals, represented as numpy arrays
        emph_coeff(float): emphasis coefficient
    Returns:
        result: de-emphasized signal batch
    """
    return signal.lfilter([1], [1, -emph_coeff], signal_batch)

class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_type):

        if data_type == 'train':
            data_path = serialized_train_folder
        else:
            data_path = serialized_test_folder
        if not os.path.exists(data_path):
            raise FileNotFoundError('The {} data folder does not exist!'.format(data_type))

        self.data_type = data_type
        self.file_names = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]

    def reference_batch(self, batch_size):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.

        Args:
            batch_size(int): batch size

        Returns:
            ref_batch: reference batch
        """
        ref_file_names = np.random.choice(self.file_names, batch_size)
        ref_batch = np.stack([np.load(f) for f in ref_file_names])

        ref_batch = pre_emphasis(ref_batch, emph_coeff=0.95)
        return torch.from_numpy(ref_batch).type(torch.FloatTensor)

    def __getitem__(self, idx):
        pair = np.load(self.file_names[idx], allow_pickle = True)
        pair = pre_emphasis(pair[np.newaxis, :, :], emph_coeff=0.95).reshape(2, -1)
        noisy = pair[1].reshape(1, -1)
        if self.data_type == 'train':
            clean = pair[0].reshape(1, -1)
            return torch.from_numpy(pair).type(torch.FloatTensor), torch.from_numpy(clean).type(
                torch.FloatTensor), torch.from_numpy(noisy).type(torch.FloatTensor)
        else:
            return os.path.basename(self.file_names[idx]), torch.from_numpy(noisy).type(torch.FloatTensor)

    def __len__(self):
        return len(self.file_names)

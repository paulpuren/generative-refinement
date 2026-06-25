'''
Data loader for the NSKT dataset.
'''

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import random

class NSKT(Dataset):
    def __init__(
            self, 
            patch_size = 256, 
            stride = 128,
            scratch_dir = './',
            train = True,
            is_T_fixed = True
        ):
        """
        Load and preprocess the dataset for training.
        
        Args:
            - patch_size: torch.Tensor, int
                    Size of the patches to extract.
            - stride: torch.Tensor, int 
                    Stride for extracting patches.
            - scratch_dir: str
                    Directory where the dataset files are located.
            - train: bool
                    If True, load training data; otherwise, load validation data.
        """
        
        super(NSKT, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.is_T_fixed = is_T_fixed
        
        if train:
            seed = '2150'
        else:
            seed = '3407'
        
        # get the data path
        self.paths = [
            os.path.join(scratch_dir, f'2000_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'4000_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'8000_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'16000_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'32000_2048_2048_seed_{seed}.h5')
        ]
        
        # Reynolds numbers for the datasets
        self.RE_list = [2000., 4000., 8000., 16000., 32000.]
        
        # Load initial dataset shape for determining patch boundaries
        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['w'].shape 

        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1 

    def open_hdf5(self):
        """
        Load 'w' and 'u' datasets for the training process.
        'w' -> vorticity
        'u' -> velocity field
        """
        self.datasets = [h5py.File(path, 'r')['w'] for path in self.paths]

    def __getitem__(self, time_index):
        # time_index is the index of the initial state in the dataset
        if not hasattr(self, 'datasets'):
            self.open_hdf5()
        
        # Randomly select a dataset and Reynolds number
        #dataset_id = np.random.randint(len(self.datasets))
        dataset_id = 3 #use just Re=16k

        # reynolds_number = self.RE_list[dataset_id]
        
        # specific embedding for different reynolds numbers
        # kind of normalization on reynolds number
        reynolds_number = self.RE_list[dataset_id] ** (1/4) / 14 
        reynolds_number = reynolds_number if np.random.uniform() < 0.9 else 0. 

        # Randomly choose between datasets for variation
        dataset = self.datasets[dataset_id]

        # Select a time index for intial state
        time_index = time_index // 17  # (should be less than 1497)

        # Randomly select a patch
        row_start = np.random.randint(0, self.max_row) * self.stride
        col_start = np.random.randint(0, self.max_col) * self.stride

        # define a random total pred steps
        if self.is_T_fixed:
            total_interp_steps = 10
        else:
            total_interp_steps = np.random.randint(5, 20)

        # define a random time index for the target within the range of predicted steps
        target_interp_step = np.random.randint(0, total_interp_steps) + 1
        
        # extract the input patch
        condition_start = torch.from_numpy(
            dataset[
                time_index, 
                row_start : (row_start + self.patch_size), 
                col_start : (col_start + self.patch_size)
            ]
        ).float().unsqueeze(0)
        condition_end = torch.from_numpy(
            dataset[
                (time_index + total_interp_steps + 1), 
                row_start : (row_start + self.patch_size), 
                col_start : (col_start + self.patch_size)
            ]
        ).float().unsqueeze(0)
        inputs = [condition_start, condition_end]
        
        # extract the target patch
        targets = torch.from_numpy(
            dataset[
                (time_index + target_interp_step), 
                row_start : (row_start + self.patch_size), 
                col_start : (col_start + self.patch_size)
            ]
        ).float().unsqueeze(0)

        cond_params = [
            torch.tensor(target_interp_step, dtype=torch.float32), 
            torch.tensor(reynolds_number),
            torch.tensor(total_interp_steps, dtype=torch.float32)
        ]
        
        return inputs, targets, cond_params

    def __len__(self):
        return 25000  # Adjust as needed for the dataset size
    
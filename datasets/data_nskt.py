'''
Dataloader for the NSKT dataset.
'''

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import random
import cv2

# define the Reynolds numbers for train and eval
RE_TRAIN_LIST = [2000, 4000, 8000, 16000, 32000]
RE_EVAL_LIST = [
    600, 1000, 2000, 4000, 8000, 
    12000, 16000, 24000, 32000, 36000
]

class NSKT(Dataset):
    def __init__(
            self, 
            patch_size = 256, 
            stride = 128,
            num_interp_steps = 10,
            re_id = 3,
            scratch_dir = './',
            flag = "train", 
            is_T_fixed = True
        ):
        """
        Load and preprocess the dataset for training and evaluation.
        
        Args:
        ---
        patch_size: torch.Tensor, int
            Size of the patches to extract.
        stride: torch.Tensor, int 
            Stride for extracting patches.
        num_interp_steps: torch.Tensor, int
            Number of interpolation steps
        re_id: torch.Tensor, int
            The Reynolds index for retrieving fluid datasets 
        scratch_dir: str
            Directory where the dataset files are located.
        flag: str
            If "train", load training data; otherwise, load evaluation data.
        """
        super(NSKT, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_interp_steps = num_interp_steps
        self.re_id = re_id
        self.is_T_fixed = is_T_fixed
        self.scratch_dir = scratch_dir
        self.flag = flag

        # different seeds for simulated train and eval data
        self.seed = '2150' if self.flag == "train" else '3407'
        self.paths = self.build_file_list()
        
        # Load initial dataset shape for determining patch boundaries
        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['w'].shape 
        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1 

    def build_file_list(self):
        self.re_list = RE_TRAIN_LIST if self.flag == "train" else RE_EVAL_LIST
        return [
            os.path.join(
                self.scratch_dir,
                f"{re}_2048_2048_seed_{self.seed}.h5"
            ) for re in self.re_list
        ]
    
    def open_hdf5(self):
        """
        Load 'w' and 'u' datasets for the training process.
        'w' -> vorticity
        'u' -> velocity field in the x direction
        """
        self.datasets = [h5py.File(path, 'r')['w'] for path in self.paths]

    def __getitem__(self, time_index):
        # time_index is the index of the initial state in the dataset
        if not hasattr(self, 'datasets'):
            self.open_hdf5()
   
        # Randomly select a dataset and Reynolds number
        dataset_id = np.random.randint(len(self.datasets))
        # dataset_id = 3 # previously using fixed Re=16k
        
        # specific embedding for different reynolds numbers
        # # kind of normalization on reynolds number
        # reynolds_number = self.RE_list[dataset_id] ** (1/4) / 14 
        # reynolds_number = reynolds_number if np.random.uniform() < 0.9 else 0. 
        reynolds_number = self.re_list[dataset_id] / 40000.0 # from Vini

        # Randomly choose between datasets for variation
        dataset = self.datasets[dataset_id]

        # Select a time index for intial state
        time_index = time_index // 17  # (should be less than 1497)

        # Randomly select a patch
        row_start = np.random.randint(0, self.max_row) * self.stride
        col_start = np.random.randint(0, self.max_col) * self.stride

        # define a random total pred steps
        if self.is_T_fixed:
            total_interp_steps = self.num_interp_steps
        else:
            total_interp_steps = np.random.randint(
                int(self.num_interp_steps // 4), 
                self.num_interp_steps
            ) # was random(5, 20)

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

        # --- RESIZE OPERATION ---
        # # resize
        # # start condition
        # select_region = self.patch_size * 2
        # condition_start = dataset[
        #     time_index, 
        #     row_start : (row_start + select_region), 
        #     col_start : (col_start + select_region)
        # ]
        # condition_start = cv2.resize(
        #     condition_start, 
        #     (self.patch_size, self.patch_size)
        # )
        # condition_start = torch.from_numpy(condition_start).float().unsqueeze(0)

        # # end condition
        # condition_end = dataset[
        #     (time_index + total_interp_steps + 1), 
        #     row_start : (row_start + select_region), 
        #     col_start : (col_start + select_region)
        # ]
        # condition_end = cv2.resize(
        #     condition_end, 
        #     (self.patch_size, self.patch_size)
        # )       
        # condition_end = torch.from_numpy(condition_end).float().unsqueeze(0)
        # --- END RESIZE OPERATION ---
        inputs = [condition_start, condition_end]

        
        # extract the target patch
        targets = torch.from_numpy(
            dataset[
                (time_index + target_interp_step), 
                row_start : (row_start + self.patch_size), 
                col_start : (col_start + self.patch_size)
            ]
        ).float().unsqueeze(0)
        # targets = dataset[
        #     (time_index + target_interp_step), 
        #     row_start : (row_start + select_region), 
        #     col_start : (col_start + select_region)
        # ]
        # targets = cv2.resize(
        #     targets, 
        #     (self.patch_size, self.patch_size)
        # )       
        # targets = torch.from_numpy(targets).float().unsqueeze(0)

        cond_params = [
            torch.tensor(target_interp_step, dtype=torch.float32), 
            torch.tensor(total_interp_steps, dtype=torch.float32),
            torch.tensor(reynolds_number, dtype=torch.float32)
        ]
        return inputs, targets, cond_params

    def __len__(self):
        return 25000  # Adjust as needed for the dataset size


class NSKT_eval(Dataset):
    def __init__(
            self,
            patch_size = 256,
            stride = 128,
            num_interp_steps = 1,
            re_id = -1,
            scratch_dir = './'
        ):
        """
        Load and preprocess the dataset for evaluation.
        
        Args:
        ---
        patch_size: torch.Tensor, int
            Size of the patches to extract.
        stride: torch.Tensor, int 
            Stride for extracting patches.
        num_interp_steps: torch.Tensor, int
            Number of interpolation steps
        re_id: torch.Tensor, int
            The Reynolds index for retrieving fluid datasets 
        scratch_dir: str
            Directory where the dataset files are located.
        """
        super(NSKT_eval, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_interp_steps = num_interp_steps
        self.re_id = re_id
        self.scratch_dir = scratch_dir

        # get the data path
        self.seed = '3407'
        self.re_list = RE_EVAL_LIST
        self.paths = self.build_file_list()

        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)

        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1 
        self.num_patches_per_image = (
            (self.data_shape[1] - self.patch_size) // self.stride + 1) * \
            ((self.data_shape[2] - self.patch_size) // self.stride + 1)
                                     
        print(f'Number of patches per snapshot: {self.num_patches_per_image}')

    def build_file_list(self):
        return [
            os.path.join(
                self.scratch_dir,
                f"{re}_2048_2048_seed_{self.seed}.h5"
            ) for re in self.re_list
        ]

    def open_hdf5(self):
        self.datasets = [h5py.File(path, 'r')['w'] for path in self.paths]

    def __getitem__(self, time_index):
        
        if not hasattr(self, 'datasets'):
            self.open_hdf5()

        # # specific embedding for different reynolds numbers
        # # kind of normalization on reynolds number
        # reynolds_number = RE_list[self.re_num_id] ** (1 / 4) / 14 
        # reynolds_number = reynolds_number if np.random.uniform() < 0.9 else 0. 
        reynolds_number = self.re_list[self.re_id] / 40000.0 # from Vini

        # Randomly choose between datasets for variation
        dataset = self.datasets[self.re_id]

        # Select a time index for intial state
        time_index = time_index // 17  # (should be less than 1497)

        # Randomly select a patch
        row_start = np.random.randint(0, self.max_row) * self.stride
        col_start = np.random.randint(0, self.max_col) * self.stride

        # extract the input patch
        select_region = self.patch_size * 2
        condition_start = dataset[
            time_index, 
            row_start : (row_start + select_region), 
            col_start : (col_start + select_region)
        ]
        condition_start = cv2.resize(
            condition_start, 
            (self.patch_size, self.patch_size)
        )
        condition_start = torch.from_numpy(condition_start).float().unsqueeze(0)

        # end condition
        condition_end = dataset[
            (time_index + self.num_interp_steps + 1), 
            row_start : (row_start + select_region), 
            col_start : (col_start + select_region)
        ]
        condition_end = cv2.resize(
            condition_end, 
            (self.patch_size, self.patch_size)
        )       
        condition_end = torch.from_numpy(condition_end).float().unsqueeze(0)
        inputs = [condition_start, condition_end]
    
        # create a list to hold target patches
        targets = []
        for i in range(1, (self.num_interp_steps + 1)):
            snapshot = dataset[
                time_index + i, 
                row_start : (row_start + select_region), 
                col_start : (col_start + select_region)
            ]
            snapshot = cv2.resize(
                snapshot, 
                (self.patch_size, self.patch_size)
            )       
            snapshot = torch.from_numpy(snapshot).float().unsqueeze(0)
            targets.append(snapshot)

        # extract physical parameters
        cond_params = [
            torch.tensor(self.num_interp_steps, dtype = torch.float32), 
            torch.tensor(reynolds_number)
        ]
        return inputs, targets, cond_params

    def __len__(self):
        # return 100
        return 2000
        # return 25000
        # return self.num_patches_per_image * 70
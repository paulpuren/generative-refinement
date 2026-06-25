"""
Dataloader for the NSKT dataset.
"""

import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


RE_TRAIN_LIST = [2000, 4000, 8000, 16000, 32000]
# select 12000, 24000, and 36000: 5, 7, 9 for eval OOD generalization
RE_EVAL_LIST = [600, 1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 36000]
TRAIN_VAL_SPLIT = 0.8
TRAIN_VAL_TOTAL_SAMPLES = 20000
EVAL_TOTAL_SAMPLES = 2000


class _NSKTBase(Dataset):
    def __init__(
            self, 
            patch_size, 
            stride, 
            scratch_dir, 
            re_list, 
            seed, 
            crop_size=None
        ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.crop_size = int(crop_size) if crop_size is not None else self.patch_size
        self.stride = int(stride)
        self.scratch_dir = scratch_dir
        self.re_list = list(re_list)
        self.seed = str(seed)
        self.paths = self._build_file_list()

        if self.crop_size < self.patch_size:
            raise ValueError(
                f"crop_size ({self.crop_size}) must be >= patch_size ({self.patch_size})."
            )

        self._h5_files = None
        self.datasets = None

        with h5py.File(self.paths[0], "r") as f:
            self.data_shape = f["w"].shape

        self._time_dim, self._height, self._width = self.data_shape
        self.max_row = (self._height - self.crop_size) // self.stride + 1
        self.max_col = (self._width - self.crop_size) // self.stride + 1
        self.num_patches_per_image = self.max_row * self.max_col

    def _build_file_list(self):
        return [
            os.path.join(
                self.scratch_dir, 
                f"{re}_2048_2048_seed_{self.seed}.h5"
            )
            for re in self.re_list
        ]

    def _ensure_open(self):
        """
        Load 'w' and 'u' datasets for the training process.
        'w' -> vorticity
        'u' -> velocity field in the x direction
        """
        if self.datasets is not None:
            return
        self._h5_files = [h5py.File(path, "r") for path in self.paths]
        self.datasets = [f["w"] for f in self._h5_files]

    def _pick_patch_origin(self):
        row_start = np.random.randint(self.max_row) * self.stride
        col_start = np.random.randint(self.max_col) * self.stride
        return row_start, col_start

    @staticmethod
    def _to_tensor(array_2d):
        return torch.from_numpy(array_2d).float().unsqueeze(0)

    def _maybe_downsample(self, tensor):
        if self.crop_size == self.patch_size:
            return tensor
        return F.interpolate(
            tensor.unsqueeze(0),
            size=(self.patch_size, self.patch_size),
            mode="area",
        ).squeeze(0)

    def _normalize_reynolds(self, re_value):
        return float(re_value) # / 40000.0 # from Vini

    def __del__(self):
        if self._h5_files is not None:
            for f in self._h5_files:
                try:
                    f.close()
                except Exception:
                    pass


class NSKT(_NSKTBase):
    def __init__(
        self,
        patch_size=256,
        stride=128,
        num_interp_steps=10,
        re_id=3,
        scratch_dir="./",
        flag="train",
        is_T_fixed=True,
        crop_size=None,
    ):
        if flag not in {"train", "valid", "test"}:
            raise ValueError("NSKT supports 'train', 'valid', and 'test' flags.")

        normalized_flag = flag
        if normalized_flag == "test":
            re_list = RE_EVAL_LIST
            seed = "3407"
        else:
            re_list = RE_TRAIN_LIST
            seed = "2150"
        super().__init__(
            patch_size, 
            stride, 
            scratch_dir, 
            re_list, 
            seed, 
            crop_size=crop_size
        )

        self.flag = normalized_flag
        self.num_interp_steps = int(num_interp_steps)
        self.re_id = int(re_id)
        self.is_T_fixed = bool(is_T_fixed)
        train_cutoff = int(TRAIN_VAL_TOTAL_SAMPLES * TRAIN_VAL_SPLIT)
        if self.flag == "train":
            self.index_offset = 0
            self.dataset_length = train_cutoff
        elif self.flag == "valid":
            self.index_offset = train_cutoff
            self.dataset_length = TRAIN_VAL_TOTAL_SAMPLES - train_cutoff
        else:
            self.index_offset = 0
            self.dataset_length = EVAL_TOTAL_SAMPLES
        if self.re_id < 0:
            self.dataset_length *= len(self.re_list)

    def _resolve_dataset_id_and_index(self, index):
        if 0 <= self.re_id < len(self.re_list):
            return self.re_id, index
        dataset_count = len(self.re_list)
        dataset_id = int(index) % dataset_count
        dataset_index = int(index) // dataset_count
        return dataset_id, dataset_index

    def _pick_total_interp_steps(self):
        if self.is_T_fixed:
            return self.num_interp_steps
        min_steps = max(1, self.num_interp_steps // 4)
        return np.random.randint(min_steps, self.num_interp_steps + 1)

    def __getitem__(self, index):
        self._ensure_open()
        dataset_id, dataset_index = self._resolve_dataset_id_and_index(index)
        dataset_index = int(dataset_index) + self.index_offset

        dataset = self.datasets[dataset_id]
        reynolds_number = self._normalize_reynolds(self.re_list[dataset_id])

        total_interp_steps = self._pick_total_interp_steps()
        target_interp_step = np.random.randint(1, total_interp_steps + 1)

        # Preserve historical sampling behavior while keeping index safe.
        base_time = dataset_index // 17
        max_base_time = self._time_dim - (total_interp_steps + 2)
        base_time = min(base_time, max_base_time)

        row_start, col_start = self._pick_patch_origin()
        row_end = row_start + self.crop_size
        col_end = col_start + self.crop_size

        condition_start = self._maybe_downsample(self._to_tensor(
            dataset[base_time, row_start:row_end, col_start:col_end]
        ))
        condition_end = self._maybe_downsample(self._to_tensor(
            dataset[
                base_time + total_interp_steps + 1, 
                row_start:row_end, 
                col_start:col_end
            ]
        ))
        target = self._maybe_downsample(self._to_tensor(
            dataset[
                base_time + target_interp_step, 
                row_start:row_end, 
                col_start:col_end
            ]
        ))

        inputs = [condition_start, condition_end]
        cond_params = [
            torch.tensor(target_interp_step, dtype=torch.float32),
            torch.tensor(total_interp_steps, dtype=torch.float32),
            torch.tensor(reynolds_number, dtype=torch.float32),
        ]
        return inputs, target, cond_params

    def __len__(self):
        return self.dataset_length


class NSKT_eval(_NSKTBase):
    def __init__(
        self,
        patch_size=256,
        stride=128,
        num_interp_steps=1,
        re_id=-1,
        scratch_dir="./",
        crop_size=None,
    ):
        super().__init__(
            patch_size,
            stride,
            scratch_dir,
            RE_EVAL_LIST,
            "3407",
            crop_size=crop_size,
        )
        self.num_interp_steps = int(num_interp_steps)
        self.re_id = int(re_id)
        self.dataset_length = EVAL_TOTAL_SAMPLES
        if self.re_id < 0:
            self.dataset_length *= len(self.re_list)

    def _resolve_dataset_id_and_index(self, index):
        if 0 <= self.re_id < len(self.re_list):
            return self.re_id, index
        dataset_count = len(self.re_list)
        dataset_id = int(index) % dataset_count
        dataset_index = int(index) // dataset_count
        return dataset_id, dataset_index

    def __getitem__(self, index):
        self._ensure_open()

        dataset_id, dataset_index = self._resolve_dataset_id_and_index(index)
        dataset = self.datasets[dataset_id]
        reynolds_number = self._normalize_reynolds(self.re_list[dataset_id])

        base_time = int(dataset_index) // 17
        max_base_time = self._time_dim - (self.num_interp_steps + 2)
        base_time = min(base_time, max_base_time)

        row_start, col_start = self._pick_patch_origin()
        row_end = row_start + self.crop_size
        col_end = col_start + self.crop_size

        condition_start = self._maybe_downsample(self._to_tensor(
            dataset[base_time, row_start:row_end, col_start:col_end]
        ))
        condition_end = self._maybe_downsample(self._to_tensor(
            dataset[
                base_time + self.num_interp_steps + 1, 
                row_start:row_end, 
                col_start:col_end
            ]
        ))
        inputs = [condition_start, condition_end]

        # Generate targets for all interpolation steps up to num_interp_steps.
        targets = []
        for step in range(1, self.num_interp_steps + 1):
            targets.append(
                self._maybe_downsample(self._to_tensor(
                    dataset[
                        base_time + step, 
                        row_start:row_end, 
                        col_start:col_end
                    ]
                ))
            )

        cond_params = [
            torch.tensor(self.num_interp_steps, dtype=torch.float32),
            torch.tensor(reynolds_number, dtype=torch.float32),
        ]
        return inputs, targets, cond_params

    def __len__(self):
        return self.dataset_length

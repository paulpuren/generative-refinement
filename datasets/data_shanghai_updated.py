"""
Dataloader for Shanghai Radar data

Link: https://github.com/DeminYu98/DiffCast/blob/main/datasets/dataset_shanghai.py
"""

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms 

PIXEL_SCALE = 90.0

COLOR_MAP = np.array([
    [0, 0, 0,0],
    [0, 236, 236, 255],
    [1, 160, 246, 255],
    [1, 0, 246, 255],
    [0, 239, 0, 255],
    [0, 200, 0, 255],
    [0, 144, 0, 255],
    [255, 255, 0, 255],
    [231, 192, 0, 255],
    [255, 144, 2, 255],
    [255, 0, 0, 255],
    [166, 0, 0, 255],
    [101, 0, 0, 255],
    [255, 0, 255, 255],
    [153, 85, 201, 255],
    [255, 255, 255, 255]
    ]) / 255

BOUNDS = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75, PIXEL_SCALE]
THRESHOLDS = [20, 30, 35, 40]

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255

class Shanghai(Dataset):
    def __init__(
            self, 
            data_path, 
            img_size, 
            type = 'train', 
            trans = None, 
            total_interp_steps = 20,
            seq_len = -1
        ):
        super().__init__()
        self.pixel_scale = PIXEL_SCALE
        self.data_path = data_path
        self.img_size = img_size
        self.total_interp_steps = total_interp_steps

        assert type in ['train', 'test', 'val']
        self.type = type if type!='val' else 'test'
        with h5py.File(data_path,'r') as f:
            self.all_len = int(f[self.type]['all_len'][()]) 
        if trans is not None:
            self.transform = trans
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    # transforms.ToTensor(),
                    # trans.Lambda(lambda x: x/255.0),
                    # transforms.Normalize(mean=[0.5], std=[0.5]),
                    # trans.RandomCrop(data_config["img_size"]),
                ]
            )
                    
    def __len__(self):
        return self.all_len

    def sample(self):
        index = np.random.randint(0, self.all_len)
        return self.__getitem__(index)
    
    def __getitem__(self, index):

        with h5py.File(self.data_path, 'r') as f:
            # numpy array: (25, 565, 784), dtype=uint8, range(0,70)
            # 25 is seq len
            imgs = f[self.type][str(index)][()]   
            frames = torch.from_numpy(imgs).float().squeeze() 
            frames = frames / 255.0
            frames = self.transform(frames)  # [25, 128, 128]   
        # frames = frames.unsqueeze(1) # (25,1,128,128)

        # --- MODIFIED BY PU REN ---
        # define a random total pred steps
        # total_interp_steps = np.random.randint(5, 20)
        total_interp_steps = np.random.randint(
            self.total_interp_steps // 4, 
            self.total_interp_steps
        )

        # define a random time index for the target within the range of predicted steps
        target_interp_step = np.random.randint(0, total_interp_steps) + 1

        # extract the input patch
        condition_start = frames[0].unsqueeze(0)
        condition_end = frames[total_interp_steps + 1].unsqueeze(0)
        inputs = [condition_start, condition_end]
        
        # extract the target patch
        targets = frames[target_interp_step].unsqueeze(0)

        cond_params = [
            torch.tensor(target_interp_step, dtype=torch.float32), 
            torch.tensor(total_interp_steps, dtype=torch.float32),
            torch.tensor(0.0,  dtype=torch.float32),
        ]
        
        return inputs, targets, cond_params
    

def gray2color(image, **kwargs):
    from matplotlib import colors

    # define the color mapping and boundaries
    cmap = colors.ListedColormap(COLOR_MAP)
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 将图像进行染色
    colored_image = cmap(norm(image))
    return colored_image


if __name__ == '__main__':
    data_path = "/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/"
    dataset = Shanghai(data_path + 'shanghai.h5', 128)
    sample1 = dataset.sample()
    sample2 = dataset.sample()

    print(len(dataset)) # 1545

    # print(sample1.shape, sample2.shape)
    # print(sample1.min(), sample1.max())
    # print(sample2.min(), sample2.max())
    # print(sample1[0,0].numpy())
    # import matplotlib.pyplot as plt

    # plt.subplot(1,2,1)
    # plt.imshow(sample1[10,0].numpy())
    # plt.subplot(1,2,2)
    # plt.imshow(sample2[10,0].numpy())
    # # plt.show()
    # plt.savefig('sample.png')

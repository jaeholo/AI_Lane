import numpy as np
import torch
import os
import shutil
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class VAEDataset1000ENZ1(Dataset):
    def __read_npy(self):
        npydata = np.load(self.__data_path)
        for i in range(npydata.shape[0]):
            data = npydata[i]
            sac = data[:598]
            img = data[598:]
            img_modified = np.where(img != 0, 1, img)
            sac_tensor = torch.tensor(sac).float().unsqueeze(dim=0)
            img_tensor = torch.tensor(img_modified).float().reshape(64, 64).unsqueeze(dim=0)
            self.__sac.append(sac_tensor)
            self.__img.append(img_tensor)

        # files = os.listdir(self.__data_path)
        # for file in tqdm(files):
        #     big_in_file = np.load(os.path.join(self.__data_path, file))
        #     for _, v in big_in_file.items():
        #         # E N Z
        #         # sac = np.vstack((v[0:1024],v[1024:2048],v[2048:3072]))
        #         sac = np.vstack((v[0:1024], v[1024:2048])) # n and z
        #         # sac = v[4096:8192] # z only
        #
        #         img = v[2048:]
        #
        #         img_modified = np.where(img != 0, 1, img)
        #         sac_tensor = torch.tensor(sac).float()#.unsqueeze(dim=0)
        #         img_tensor = torch.tensor(img_modified).float().reshape(64, 64).unsqueeze(dim=0)
        #         self.__sac.append(sac_tensor)
        #         self.__img.append(img_tensor)


    def __init__(self, path):
        self.__data_path = path         # base directory path of the data
        self.__sac = []
        self.__img = []
        self.__device = get_device()
        self.__read_npy()

    def __len__(self):
        return len(self.__sac)

    def __getitem__(self, index):
        return self.__sac[index].to(self.__device), self.__img[index].to(self.__device)


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


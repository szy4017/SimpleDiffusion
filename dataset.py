import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transform
import torch.nn.functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size=255):
        """
        Image loader base class
        
        Takes a path to folder of images. Uses every image in the folder as the dataset.
        All images are resized to [im_len, im_len] when an item is requested.
        """
        self.path = path
        self.files = os.listdir(self.path)
        self.len = len(self.files)
        self.img_size = img_size
        self.trf = transform.ToTensor()

    def __len__(self):
        """
        Returns number of images in dataset
        """
        return self.len
    
    def __getitem__(self, index):
        """
        Gets the image at index
        """
        image = Image.open(self.path + self.files[index])
        image = image.resize((self.img_size, self.img_size))
        image = image.convert('RGB')
        return self.trf(image)  * 2 - 1


class MMFiDataset(torch.utils.data.Dataset):
    def __init__(self, path, size=(64, 64, 40), range=((0.0, 6.4), (-3.2, 3.2), (-2.0, 2.0)), point_num_max=5):
        self.path = path
        self.size_x = size[0]
        self.size_y = size[1]
        self.size_z = size[2]
        self.range_x = range[0]
        self.range_y = range[1]
        self.range_z = range[2]
        self.scale_x = self.size_x / (self.range_x[1] - self.range_x[0])
        self.scale_y = self.size_y / (self.range_y[1] - self.range_y[0])
        self.scale_z = self.size_z / (self.range_z[1] - self.range_z[0])
        self.point_num_max = point_num_max

        data_list = sorted(os.listdir(self.path))
        self.all_file_list = []
        for dl in data_list:
            file = os.path.join(self.path, dl, 'lidar')
            file_list = sorted(os.listdir(file))
            for fl in file_list:
                self.all_file_list.append(os.path.join(self.path, dl, 'lidar', fl))
        self.len = len(self.all_file_list)

        self.trf = transform.ToTensor()

    def __len__(self):
        """
        Returns number of images in dataset
        """
        return self.len

    def convert_pc2vl(self, pc_arr):
        pc_arr_translated = pc_arr - (self.range_x[0], self.range_y[0], self.range_z[0])
        pc_arr_scaled = pc_arr_translated * [self.scale_x, self.scale_y, self.scale_z]
        pc_arr_scaled = pc_arr_scaled.astype(int)
        unique_indices, counts = np.unique(pc_arr_scaled, return_counts=True, axis=0)
        voxel_grid = np.zeros((self.size_x, self.size_y, self.size_z))
        voxel_indices = unique_indices
        voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = counts
        voxel_grid = voxel_grid / self.point_num_max
        voxel_grid = voxel_grid.astype(np.float32)
        return voxel_grid

    def __getitem__(self, index):
        """
        Gets the image at index
        """
        pointcloud = np.fromfile(self.all_file_list[index], dtype=np.float64)
        pointcloud = pointcloud.reshape((-1, 3))
        voxel = self.convert_pc2vl(pointcloud)
        voxel_t = self.trf(voxel) * 2 - 1
        # print(voxel_t)
        return voxel_t
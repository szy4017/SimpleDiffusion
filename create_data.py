'''
create voxel data from lidar point cloud data
'''
import os

import numpy as np
import os


mmfi_root = '/data/szy4017/data/mmfi/E01/S01'
data_list = sorted(os.listdir(mmfi_root))
print(data_list)
all_file_list = []
for dl in data_list:
    file = os.path.join(mmfi_root, dl, 'lidar')
    file_list = sorted(os.listdir(file))
    for fl in file_list:
        all_file_list.append(os.path.join(mmfi_root, dl, 'lidar', fl))
print(all_file_list)
print(len(all_file_list))

for pc_file in all_file_list:
    print(pc_file)
    pc_arr = np.fromfile(pc_file, dtype=np.float32)
    pc_arr = pc_arr.reshape((-1, 3))
    print(pc_arr)
    print(pc_arr.shape)

    # TODO: convert point cloud to voxel
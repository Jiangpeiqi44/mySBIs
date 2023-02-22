from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob
import os
import pandas as pd


def init_ff(phase, level='frame', n_frames=8):
    # raw or c23 or c40
    dataset_path = 'data/FF++/original_sequences/youtube/raw/frames/'

    image_list = []
    label_list = []

    # folder_list = sorted(glob(dataset_path+'*'))
    folder_list = [os.path.join(dataset_path, i)
                   for i in sorted(os.listdir(dataset_path))]
    # print(folder_list[:5])
    # print(len(folder_list))
    filelist = []
    list_dict = json.load(
        open(f'data/FF++/{phase}.json', 'r'))

    # print(list_dict)
    for i in list_dict:
        filelist += i
    # print(filelist)
    folder_list = [i for i in folder_list if os.path.basename(i)[
        :3] in filelist]
    # print(len(folder_list))
    if level == 'video':
        label_list = [0]*len(folder_list)
        return folder_list, label_list
    for i in range(len(folder_list)):
        # images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
        # images_temp = sorted(glob((folder_list[i]+'/*.png')))
        images_temp = [os.path.join(folder_list[i], k).replace('\\', '/')
                   for k in sorted(os.listdir(folder_list[i]))]
        if n_frames < len(images_temp):
            images_temp = [images_temp[round(i)] for i in np.linspace(
                0, len(images_temp)-1, n_frames)]
        image_list += images_temp
        label_list += [0]*len(images_temp)

    return image_list, label_list

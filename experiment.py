from random import random
from turtle import color
import torch
import torch.nn as nn
from active_segmentation import random_sampling,Active_sampling
from visualizations import visualise_masks
from utils import reset_DATA
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from AB_UNET_base_model import AB_UNET
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt


def experiment(sample_size=10,dropout_iteration=4,dropout=0,max_dropout=0.3,exp_path=r".\experiments\exp2"):
    print('BEGINNING RANDOM SAMPLING :')
    reset_DATA(r".\DATA")
    random_sampling(sample_size=5,dropout=dropout,max_dropout=max_dropout,label_split_ratio=0.025,test_split_ratio=0.3,exp_path=exp_path)
    reset_DATA(r".\DATA")
    acq_fn=['entropy','BALD','KL-Divergence','JS-divergence']
    for acq in [1,2,3,4]:
        print(f"BEGINNING {acq_fn[acq-1]} SAMPLING :")
        Active_sampling(sample_size=sample_size,acquistion_type=acq,dropout_iteration=dropout_iteration,dropout=dropout,max_dropout=max_dropout,exp_path=exp_path)
        reset_DATA(r".\DATA")


if __name__ == "__main__":
    experiment(sample_size=5,dropout_iteration=4,dropout=0,max_dropout=0.25,exp_path=r".\experiments\exp6")
    #arr1=np.load(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\experiments\exp3\dice_stats\entropy.npy')
    #arr2=np.load(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\experiments\exp3\dice_stats\JS-divergence.npy')
    #arr3=np.load(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\experiments\exp3\dice_stats\KL-Divergence.npy')
    #arr4=np.load(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\experiments\exp3\dice_stats\random.npy')
    #plt.plot(np.arange(1,16,1)/15,arr1,label='max entropy',color='y')
    #plt.plot(np.arange(1,16,1)/15,arr2,label='JS divergence',color='g')
    #plt.plot(np.arange(1,16,1)/15,arr3,label='KL divergence',color='b')
    #plt.plot(np.arange(1,16,1)/15,arr4,label='random',color='r')
    #plt.ylim(ymin=0.1)
    #plt.title("Active learning dice")
    #plt.xlabel("% of the dataset")
    #plt.ylabel("Dice metric")
    #plt.legend()
    #plt.show()


from itertools import dropwhile
from turtle import color
import torch
import torch.nn as nn
from utils import train_val_split,get_loaders_active,get_loaders,check_accuracy,create_score_dict,labeled_unlabeled_test_split,reset_DATA
from train import train_fn
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








def move_images(BASE_DIR,TRAIN_DIR,VAL_DIR,num):
        print(len(os.listdir(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images'))))

        images=os.listdir(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images'))
        for im in images[:num] : 
            im_path=os.path.join(BASE_DIR,'images',im)
            target_path=os.path.join(BASE_DIR,TRAIN_DIR,'labeled_images')
            shutil.copy(im_path, target_path)
            os.remove(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images',im))

            for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
                mask_path=os.path.join(os.path.join(BASE_DIR,'masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp"))
                target_mask_path=os.path.join(BASE_DIR,TRAIN_DIR,'labeled_masks',mask)
                shutil.copy(mask_path, target_mask_path)
                os.remove(os.path.join(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp")))

        print(len(os.listdir(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images'))))

def move_images_with_dict(BASE_DIR,TRAIN_DIR,VAL_DIR,score_dict,num):

        print(len(os.listdir(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images'))))

        dict_iterator=iter(score_dict)
        for i in range(num) : 
            im=next(dict_iterator)
            im_path=os.path.join(BASE_DIR,'images',im)
            target_path=os.path.join(BASE_DIR,TRAIN_DIR,'labeled_images')
            shutil.copy(im_path, target_path)
            os.remove(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images',im))

            for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
                mask_path=os.path.join(os.path.join(BASE_DIR,'masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp"))
                target_mask_path=os.path.join(BASE_DIR,TRAIN_DIR,'labeled_masks',mask)
                shutil.copy(mask_path, target_mask_path)
                os.remove(os.path.join(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp")))

        print(len(os.listdir(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images'))))




def save_model_dict(model,step):
    FILE = f"models/model_step_{step}.pth"
    torch.save(model, FILE)



def random_sampling(sample_size=10,dropout=0,max_dropout=0.3,label_split_ratio=0.05,test_split_ratio=0.3,exp_path=r""):

    
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #DEVICE="cpu"
    BATCH_SIZE = 1
    NUM_EPOCHS = 5
    NUM_WORKERS = 4
    PIN_MEMORY = False
    LOAD_MODEL = False
    LABELED_IMG_DIR = r".\DATA\Labeled_pool\labeled_images"
    LABELED_MASK_DIR = r".\DATA\Labeled_pool\labeled_masks"
    UNLABELED_IMG_DIR = r".\DATA\Unlabeled_pool\unlabeled_images"
    UNLABELED_MASK_DIR = r".\DATA\Unlabeled_pool\unlabeled_masks"
    TEST_IMG_DIR=r".\DATA\Test\test_images"
    TEST_MASK_DIR=r".\DATA\Test\test_masks"


#intial split

    BASE_DIR=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA"
    LABELED_DIR=r"Labeled_pool"
    UNLABELED_DIR=r"Unlabeled_pool"
    TEST_DIR=r"Test"
    labeled_unlabeled_test_split(BASE_DIR,LABELED_DIR,UNLABELED_DIR,TEST_DIR,label_split_ratio=label_split_ratio,test_split_ratio=test_split_ratio,shuffle=True)
    num_images=len(os.listdir(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Unlabeled_pool\unlabeled_images'))

    model = AB_UNET(in_channels=3, out_channels=4,dropout=dropout,max_dropout=max_dropout).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    #torch.cuda.empty_cache()
    train_transform = A.Compose(
        [   #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.3),
            #A.VerticalFlip(p=0.1),
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.augmentations.transforms.Equalize(mode="cv",p=1),
            ToTensorV2()
        ],
    )
    dice_array=[]
    for step in range(int(num_images/sample_size)):
        
        print(f'step number {step} : ')
        labeled_loader,_,test_loader= get_loaders_active(
            LABELED_IMG_DIR,
            LABELED_MASK_DIR,
            UNLABELED_IMG_DIR,
            UNLABELED_MASK_DIR,
            TEST_IMG_DIR,
            TEST_MASK_DIR,
            BATCH_SIZE,
            train_transform,
            train_transform,
            NUM_WORKERS,
        )
        
        for epoch in tqdm(range(NUM_EPOCHS)):
            #scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,verbose=True)
            train_fn(labeled_loader, model, optimizer, loss_fn, scaler,scheduler=None)
            if epoch==NUM_EPOCHS-1 : 
                acc,dice=check_accuracy(test_loader,model,BATCH_SIZE, device=DEVICE)
            #save_model_dict(model,step)

        dice_array.append(dice)
        #move 10 images at random to labeled pool
        move_images(BASE_DIR,LABELED_DIR,UNLABELED_DIR,sample_size)

    dice_stats=torch.tensor(dice_array).detach().cpu().numpy()
    path=os.path.join(exp_path,r"dice_stats",f"random.npy")
    np.save(path,np.array(dice_stats))


def Active_sampling(sample_size=10,acquistion_type=3,dropout_iteration=5,dropout=0,max_dropout=0.3,exp_path=r""):

    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #DEVICE="cpu"
    BATCH_SIZE = 1
    NUM_EPOCHS = 5
    NUM_WORKERS = 4
    PIN_MEMORY = False
    LOAD_MODEL = False
    LABELED_IMG_DIR = r".\DATA\Labeled_pool\labeled_images"
    LABELED_MASK_DIR = r".\DATA\Labeled_pool\labeled_masks"
    UNLABELED_IMG_DIR = r".\DATA\Unlabeled_pool\unlabeled_images"
    UNLABELED_MASK_DIR = r".\DATA\Unlabeled_pool\unlabeled_masks"
    TEST_IMG_DIR=r".\DATA\Test\test_images"
    TEST_MASK_DIR=r".\DATA\Test\test_masks"


#intial split

    BASE_DIR=r".\DATA"
    LABELED_DIR=r"Labeled_pool"
    UNLABELED_DIR=r"Unlabeled_pool"
    TEST_DIR=r"Test"
    labeled_unlabeled_test_split(BASE_DIR,LABELED_DIR,UNLABELED_DIR,TEST_DIR,label_split_ratio=0.025,test_split_ratio=0.3,shuffle=True)
    num_images=len(os.listdir(r'.\DATA\Unlabeled_pool\unlabeled_images'))
    
    model = AB_UNET(in_channels=3, out_channels=4,dropout=dropout,max_dropout=max_dropout).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    #torch.cuda.empty_cache()
    train_transform = A.Compose(
        [   #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.3),
            #A.VerticalFlip(p=0.1),
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.augmentations.transforms.Equalize(mode="cv",p=1),
            ToTensorV2()
        ],
    )
    dice_array=[]
    for step in range(int(num_images/sample_size)):

        print(f'step number {step} : ')
        #scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,verbose=True)
        labeled_loader,unlabeled_loader,test_loader= get_loaders_active(
            LABELED_IMG_DIR,
            LABELED_MASK_DIR,
            UNLABELED_IMG_DIR,
            UNLABELED_MASK_DIR,
            TEST_IMG_DIR,
            TEST_MASK_DIR,
            BATCH_SIZE,
            train_transform,
            train_transform,
            NUM_WORKERS,
        )
        
        for epoch in tqdm(range(NUM_EPOCHS)):
            train_fn(labeled_loader, model, optimizer, loss_fn, scaler,scheduler=None)
            if epoch==NUM_EPOCHS-1 : 
                acc,dice=check_accuracy(test_loader,model,BATCH_SIZE, device=DEVICE)
            #save_model_dict(model,step)
        dice_array.append(dice)
        score_dict=create_score_dict(model,unlabeled_loader,DEVICE,acquistion_type,dropout_iteration=dropout_iteration)
        move_images_with_dict(BASE_DIR,LABELED_DIR,UNLABELED_DIR,score_dict,sample_size)
        
    dice_stats=torch.tensor(dice_array).detach().cpu().numpy()
    acq_fn=['entropy','BALD','KL-Divergence','JS-divergence']
    path=os.path.join(exp_path,r"dice_stats",f"{acq_fn[acquistion_type-1]}.npy")
    np.save(path,np.array(dice_stats))



def Active_sampling_step(sample_size=4,acquistion_type=4,dropout_iteration=10,dropout=0,max_dropout=0.3):
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #DEVICE="cpu"
    BATCH_SIZE = 1
    NUM_EPOCHS = 30
    NUM_WORKERS = 2
    PIN_MEMORY = False
    LOAD_MODEL = False
    LABELED_IMG_DIR = r".\DATA_AO_preprocessed\Labeled_pool\labeled_images"
    LABELED_MASK_DIR = r".\DATA_AO_preprocessed\Labeled_pool\labeled_masks"
    UNLABELED_IMG_DIR = r".\DATA_AO_preprocessed\Unlabeled_pool\unlabeled_images"
    BASE_DIR=r".\DATA_AO_preprocessed"
    LABELED_DIR=r"Labeled_pool"
    UNLABELED_DIR=r"Unlabeled_pool"

    train_transform = A.Compose(
        [   #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.3),
            #A.VerticalFlip(p=0.1),
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.augmentations.transforms.Equalize(mode="cv",p=1),
            ToTensorV2()
        ],
    )
    
    model = AB_UNET(in_channels=3, out_channels=4,dropout=dropout,max_dropout=max_dropout).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    #scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,verbose=True)
    labeled_loader,unlabeled_loader= get_loaders(
        LABELED_IMG_DIR,
        LABELED_MASK_DIR,
        UNLABELED_IMG_DIR,
        '',
        BATCH_SIZE,
        train_transform,
        train_transform,
        NUM_WORKERS,
        )
        
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_fn(labeled_loader, model, optimizer, loss_fn, scaler,scheduler=None)
        if epoch==NUM_EPOCHS-1 : 
            acc,dice=check_accuracy(labeled_loader,model,BATCH_SIZE, device=DEVICE)
    

    score_dict=create_score_dict(model,unlabeled_loader,DEVICE,acquistion_type,dropout_iteration=dropout_iteration)
    dict_iterator=iter(score_dict)
    print(score_dict)
    for i in range(sample_size) : 
        im=next(dict_iterator)
        print(im)
        



if __name__ == "__main__":
    #reset_DATA(r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA")
    #random_sampling(sample_size=4,dropout=0,max_dropout=0.3)
    Active_sampling_step(sample_size=4,acquistion_type=1,dropout_iteration=10,dropout=0,max_dropout=0.3)

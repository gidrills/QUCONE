import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
torch.manual_seed(5)

from torch.utils.data import DataLoader, TensorDataset

def load_images(fileName):
    directory = fileName
    print(directory)
    images = np.zeros((0, 1, 64, 64))
    for filename in os.listdir(directory):
        if((filename)!='.ipynb_checkpoints'):
            #resize and convert to black and white the image
            img = Image.open(directory+filename).resize((64, 64)).convert('L')
            #img.show()
            #we reshape img one size up.The image lost 1 size because was converted to black and white
            imgarray = np.reshape(np.array(img), [1, 1, 64, 64])
            images = np.concatenate([images, imgarray], axis = 0)
    return images

def create_2_subsets(parasitized,uninfected,seed):
    my_parasitized_images=parasitized
    my_uninfected_images=uninfected
    if seed==0:
        my_seed=0
    else:
        my_seed=seed
        
    lb = np.concatenate([np.ones(my_parasitized_images.shape[0]), np.zeros(my_uninfected_images.shape[0])])
    cells = np.concatenate([my_parasitized_images, my_uninfected_images])
    
    return cells,lb

def data_to_torch(train_set,train_lb):
    train_set_torch  = torch.from_numpy(train_set).float()
    train_lb_torch  = torch.from_numpy(train_lb).float()
    
    return train_set_torch,train_lb_torch;

def create_dataloader_from_torchdata(train_set_torch, train_lb_torch,val_set_torch, val_lb_torch):
    #create a dataset that contains tensors for input features and labels
    dataset_train = TensorDataset(train_set_torch, train_lb_torch)
    dataset_val = TensorDataset(val_set_torch, val_lb_torch)
    
    #create a dataloader from dataset. A dataloader represents an iterable over a dataset
    #batchsize: how many samples per batch to load
    #shuffle: set to True to have the data reshuffled at every epoch
    dataloader_train = DataLoader(dataset_train, batch_size=30, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=30, shuffle=True)
    
    return dataloader_train,dataloader_val;
    
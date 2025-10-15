import os
import sys
import numpy as np
import scipy.io as sio 
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#code by black-y 2025.10.15 16:02
#Data process tools using in train/test/other flow

#=====Data loading tools=====#
def trans(array, image_size=0):
    """
    This function implements the conversion from a 1D signal to a 2D image, 
    following the arrangement order mentioned in the TEMDNet paper, 
    with attention paid to the non-conventional raster order.
    array: 1d tem signal data,
    image_size: size of the converted 2D image.
    """
    array = array.reshape((image_size, image_size))
    ite = int((image_size / 2) + 1)
    
    for num in range(1, ite):
        array[(num*2)-1] = array[(num*2)-1, ::-1]
        
    return array

def reverse_trans(array, image_size=0):
    ite = int((image_size / 2) + 1)
    flattened_array = []
    for num in range(image_size):
        if num % 2 == 0: 
            flattened_array.extend(array[num, :])
        else: 
            flattened_array.extend(array[num, ::-1])
            
    return np.array(flattened_array)

def get_next_batch(input_img, pointer, batch_size, IMG_SIZE=0):
    """
    Batch data acquisition function
    input_img: A list or array of input images
    pointer: The starting pointer of the current batch, pointing to the image data to be obtained.
    batch_size: The number of images contained in each batch.
    IMG_SIZE: The size of the image (default is 0, indicating that the actual value needs to be passed in).
    test: Whether it is in test mode, default is False.
    is_TEMDNet: Whether to use the TEMDNet network architecture, default is False.
    """
    img_batch = []
    imgs = input_img[pointer * batch_size : (pointer + 1) * batch_size]

    for img in imgs:      
        array = np.array(img)                  
        array = trans(array, IMG_SIZE)
        img_batch.append(array)
    img_batch = np.array(img_batch)
    img_batch = torch.tensor(img_batch, dtype=torch.float32).unsqueeze(1)
    
    return img_batch

def get_next_batch_1d(input_data, pointer, batch_size):
    signal_batch = []
    signals = input_data[pointer * batch_size : (pointer + 1) * batch_size]
    signal_batch = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
    
    return signal_batch

def get_sparse_batch(sparse_matrix, pointer, batch_size):
    """
    get batch data of the sparse matrix. 
    sparse_matrix (np.ndarray or torch.Tensor): A sparse matrix with a shape of (256, N), where each column represents a sample.
    pointer: The starting pointer for the current batch, indicating the column of the sparse matrix to be retrieved.
    batch_size: The number of samples included in each batch. 
    return: torch.Tensor: A batch of sparse matrices with a shape of (batch_size, 256).
    """
    if isinstance(sparse_matrix, np.ndarray):
        sparse_matrix = torch.tensor(sparse_matrix, dtype=torch.float32)
    sparse_batch = sparse_matrix[:, pointer * batch_size : (pointer + 1) * batch_size]
    sparse_batch = sparse_batch.T
    
    return sparse_batch

def create_save_folder(path):
    if os.path.exists(path): 
        print('{} exists!\n'.format(path))
    else: 
        os.makedirs(path)
        print('{} create successfully\n'.format(path))

def get_img(path,name):
    """get the mat matrix"""
    data=sio.loadmat(path)
    load_matrix = data[name]
    return load_matrix

def sig_proprocess(
                   MODEL_SAVE_PATH=None,
                   DATASET_PATH_INPUT=None,
                   DATASET_PATH_REAL=None,
                   TRAIN_SAVE_PATH=None,
                   ):
    """mat signal preprocess"""
    create_save_folder(MODEL_SAVE_PATH)
    create_save_folder(TRAIN_SAVE_PATH)
    input_data = get_img(DATASET_PATH_INPUT,'noise_train')
    real_data = get_img(DATASET_PATH_REAL,'clean_train')
    return input_data,real_data


#=====Loss functions=====#
def res_loss(input, output, real):
    """Res_MSE"""
    loss = torch.mean((output - (input - real)) ** 2)
    return loss

def res_loss_MAE(input, output, real):
    """Res_MAE"""
    loss = torch.mean(torch.abs(output - (input - real)))
    return loss

def MSE_loss(output, real):
    """MSE"""
    loss = torch.mean((output - real) ** 2)
    return loss

def MAE_loss(output,real):
    """MAE"""
    loss = torch.mean(torch.abs(output-real))
    return loss

def combined_loss(output, real, alpha=0.0):
    """
    Calculate the weighted MSE and MAE losses.
    alpha: Controls the weights of MSE and MAE losses. The value range is [0, 1].
    """
    mse_loss = torch.mean((output - real) ** 2)
    mae_loss = torch.mean(torch.abs(output - real))
    loss = alpha * mse_loss + (1 - alpha) * mae_loss
    return loss

def combined_loss_L2(output, real, model, alpha=0.0, l2_lambda=0.0):
    """
    MSE + MAE + L2
    """
    mse_loss = torch.mean((output - real) ** 2)
    mae_loss = torch.mean(torch.abs(output - real))
    loss = alpha * mse_loss + (1 - alpha) * mae_loss
    
    l2_reg = torch.tensor(0., device=output.device)
    for param in model.parameters():
        l2_reg += torch.norm(param, 2) ** 2

    loss = loss + l2_lambda * l2_reg
    return loss

def dual_consistency_loss(out1,out2,sparse1,sparse2):
    """
    Sparse code loss
    denoised signal loss
    """
    loss_shape = MAE_loss(sparse1,sparse2)
    loss_output = MSE_loss(out1,out2)
    return loss_output+loss_shape

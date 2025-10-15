import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm 
from model.DTEMDNet import DTEMDNet
from lib import *
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#code by black-y 2025.10.15 16:02

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#kaiming initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')#LeakyReLU

# Warmup
class GradualWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.total_epoch:
            return [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0) 
                    for base_lr in self.base_lrs]
        if self.after_scheduler:
            if not self.finished:
                self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                self.finished = True
            return self.after_scheduler.get_last_lr()
        return [base_lr * self.multiplier for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def train(input_data, real_data, MODEL_SAVE_PATH, TRAIN_RESULTLS_PATH,
          LEARNING_RATE_BASE, EPOCHS, SEED, LOAD_MODEL=False, 
          BATCH_SIZE=None, IMG_SIZE=None, 
          WEIGHT_DECAY=None, T_MAX=None, MULTIPLIER=None, WARMUP_EPOCHS=None,LOSS_ALPHA=None):
    """
    trains DTEMDNet on (input_data -> real_data) pairs with warmup + cosine lr
    saves best checkpoint by running loss and exports loss history to excel
    """
    
    global_steps = 0
    set_seed(SEED)
    clip_value = None #gradient clipping
    model = DTEMDNet().to(device)
    model.apply(init_weights) #weight initialize
    
    #mse reconstruction loss by default (replaceable via lib.*)
    criterion = MSE_loss 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_BASE, weight_decay=WEIGHT_DECAY)
    
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=None)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=MULTIPLIER, total_epoch=WARMUP_EPOCHS, after_scheduler=scheduler_cosine)
    
    best_loss = float('inf')
    loss_dict = {"train_batch_loss": []}

    if LOAD_MODEL:
        print("Loading checkpoints...")
        checkpoint_path = os.path.join(MODEL_SAVE_PATH, 'pretrained model path')
        if os.path.isfile(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded model from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

    print('Start Training....\n')

    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.RandomState(SEED)
        rand_list = rng.choice(range(len(input_data) // BATCH_SIZE), len(input_data) // BATCH_SIZE, replace=False)
        with tqdm(total=len(rand_list), desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", leave=False) as pbar:
            total_train_loss = 0
            for rand_point in rand_list:
                # fetch a mini-batch from both input and target sets
                inputs = get_next_batch(input_data, rand_point, BATCH_SIZE, IMG_SIZE).to(device)
                targets = get_next_batch(real_data, rand_point, BATCH_SIZE, IMG_SIZE).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                train_loss = criterion(outputs, targets) # model=model,alpha=LOSS_ALPHA)
                total_train_loss += train_loss.item()
                train_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                pbar.update(1)
                global_steps += 1
        
        #epoch-level metrics
        avg_train_loss = total_train_loss / len(rand_list)
        loss_dict["train_batch_loss"].append(avg_train_loss)

        scheduler_warmup.step()
        
        print(f"Epoch: {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Learning Rate: {optimizer.param_groups[0]['lr']}")
        
        # save best model by lowest average training loss
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_model_path = os.path.join(MODEL_SAVE_PATH, 'xxx.pth')
            torch.save(model.state_dict(), best_model_path)
    
    # export loss trajectory to excel for downstream analysis/plots
    loss_df = pd.DataFrame(loss_dict)
    excel_path = os.path.join(TRAIN_RESULTLS_PATH, 'loss.xlsx')
    loss_df.to_excel(excel_path, index=False)
    print(f"Losses saved to {excel_path}")
    print('Training Completed.')

def main():
    # set project-specific paths (replace with your actual directories)
    MODEL_SAVE_PATH = 'pretrained model path'
    TRAIN_SAVE_PATH = 'model save path after training'
    DATASET_PATH_INPUT = 'your noise signal path, xxx.mat'
    DATASET_PATH_REAL = 'your clean signal path, xxx.mat'

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    if not os.path.exists(TRAIN_SAVE_PATH):
        os.makedirs(TRAIN_SAVE_PATH)
    
    
    # your para settings
    LEARNING_RATE_BASE = None
    EPOCHS = None
    SEED = None
    LOAD_MODEL = False
    BATCH_SIZE = None
    IMG_SIZE = None
    WEIGHT_DECAY = None     #Weight decay for the optimizer
    T_MAX = None            #Cosine annealing period, number of epochs before the cycle resets to 0
    MULTIPLIER = None       #Warm-up multiplier for the learning rate
    WARMUP_EPOCHS = None    #Number of warm-up epochs for the learning rate
    LOSS_ALPHA = None       #Weight factor for the combined loss: pred sparse code loss (MAE) + pred denoised signal (MSE)
    
    # dump hyperparameters for experiment tracking
    hyperparameter_file_path = os.path.join(TRAIN_SAVE_PATH, 'hyperparameters_train.txt')
    with open(hyperparameter_file_path, 'w') as f:
        f.write("\nModel: ResNet6 \n")
        f.write("\nHyperparameters:\n")
        f.write(f"Learning Rate: {LEARNING_RATE_BASE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Random Seed: {SEED}\n")
        f.write(f"Load Pre-trained Model: {LOAD_MODEL}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Image Size: {IMG_SIZE}\n")
        f.write(f"Weight Decay: {WEIGHT_DECAY}\n")
        f.write(f"T_max: {T_MAX}\n")
        f.write(f"Multiplier: {MULTIPLIER}\n")
        f.write(f"Warmup Epochs: {WARMUP_EPOCHS}\n")
        f.write(f"LOSS_ALPHA: {LOSS_ALPHA}\n")

    # prepare dataset tensors/arrays needed for training (preprocessing)
    input_data, real_data = sig_proprocess(
        MODEL_SAVE_PATH=MODEL_SAVE_PATH,
        DATASET_PATH_INPUT=DATASET_PATH_INPUT,
        DATASET_PATH_REAL=DATASET_PATH_REAL,
        TRAIN_SAVE_PATH=TRAIN_SAVE_PATH
    )
    
    train(input_data, real_data, MODEL_SAVE_PATH, TRAIN_SAVE_PATH,
          LEARNING_RATE_BASE, EPOCHS, SEED, 
          LOAD_MODEL=LOAD_MODEL,
          BATCH_SIZE=BATCH_SIZE, 
          IMG_SIZE=IMG_SIZE, 
          #STDDEV=STDDEV, 
          WEIGHT_DECAY=WEIGHT_DECAY, 
          T_MAX=T_MAX, 
          MULTIPLIER=MULTIPLIER, 
          WARMUP_EPOCHS=WARMUP_EPOCHS,
          LOSS_ALPHA=LOSS_ALPHA,
          )

if __name__ == '__main__':
    main()

import os
import random
import torch
import numpy as np
import scipy.io as scio
import torch.nn as nn
import pandas as pd
from lib import *
from model.DTEMDNet import DTEMDNet

#code by black-y 2025.10.15 16:02

def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

#weak augmentation
def add_noise(inputs, noise_level=0.1):
    noise = torch.randn_like(inputs) * noise_level
    return inputs + noise

def compute_snr(clean_image, denoise_image):
    signal_power = np.sum(clean_image ** 2)
    noise_power = np.sum((clean_image - denoise_image) ** 2)
    return 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')

#One-oreder loss
def gradient_loss(student_output, dictionary_output):

    #Calculate the horizontal and vertical gradients of the model output
    grad_student_h = torch.abs(student_output[:, :, :, :-1] - student_output[:, :, :, 1:])
    grad_student_v = torch.abs(student_output[:, :, :-1, :] - student_output[:, :, 1:, :])
    
    #Calculate the horizontal and vertical gradients of the diction_reconstruction output
    grad_dict_h = torch.abs(dictionary_output[:, :, :, :-1] - dictionary_output[:, :, :, 1:])
    grad_dict_v = torch.abs(dictionary_output[:, :, :-1, :] - dictionary_output[:, :, 1:, :])
    
    #mse between model and diction_learning
    loss_h = MSE_loss(grad_student_h, grad_dict_h)
    loss_v = MSE_loss(grad_student_v, grad_dict_v)
    return loss_h + loss_v

#results save
def save_to_mat(Original_signals, Student_signals, SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)
    scio.savemat(os.path.join(SAVE_PATH, "Original_denoise_signals.mat"), {"Original_denoise_signals": np.array(Original_signals)})
    scio.savemat(os.path.join(SAVE_PATH, "DPTTA_denoise_signals.mat"), {"DPTTA_denoise_signals": np.array(Student_signals)})
    print(f"Signals saved to {SAVE_PATH}")

#Dictionary reconstruction flow
def process_student_sample(dictionary, Student_predicted_sparse_code, image_size=30):
    
    Student_D_Sample = torch.matmul(dictionary, Student_predicted_sparse_code.T)
    Student_D_Sample = Student_D_Sample.T
    batch_size = Student_D_Sample.size(0)
    Student_D_Sample_2D = torch.zeros((batch_size, image_size, image_size), device=Student_D_Sample.device)

    for batch_idx in range(batch_size):
        array = Student_D_Sample[batch_idx].reshape(image_size, image_size)
        for num in range(1, int(image_size / 2) + 1):
            array[(num * 2) - 1] = torch.flip(array[(num * 2) - 1], dims=[0]) 
        Student_D_Sample_2D[batch_idx] = array 
    Student_D_Sample_2D = Student_D_Sample_2D.unsqueeze(1)  # [B, 1, image_size, image_size]
    
    return Student_D_Sample_2D

class DTEMDNet(nn.Module):
    def __init__(self, model_path, device):
        super(DTEMDNet, self).__init__()
        
        #Load pretrained model
        self.dutemdnet = DTEMDNet().to(device)
        dutemdnet_checkpoint = torch.load(os.path.join(model_path, 'your_pretrained_model.pth'))
        self.dutemdnet.load_state_dict(dutemdnet_checkpoint)
        self.dutemdnet.train()

    def forward(self, inputs_aug1, dictionary, image_size=30):

        output_aug1, sparse_code_aug1 = self.dutemdnet(inputs_aug1, dictionary)

        DRSample2D_aug1 = process_student_sample(dictionary, sparse_code_aug1, image_size)

        return DRSample2D_aug1, sparse_code_aug1, output_aug1
    
#batch test
def test_batch(Original_model, Student_model, inputs_noise, inputs_clean, dictionary):
    batch_size = inputs_noise.size(0)
    snr_original_list, snr_student_list = [], []

    Original_outputs_1d = []
    Student_outputs_1d = []
    Clean_signals_1d = []
    Noise_signals_1d = []

    for i in range(batch_size):
        input_noise = inputs_noise[i].unsqueeze(0) 
        input_clean = inputs_clean[i].unsqueeze(0)  

        #forward flow
        Original_output, _  = Original_model(input_noise, dictionary)
        _1 , _2, Student_output = Student_model(input_noise, dictionary)

        # trans to numpy
        Original_output = Original_output.detach().cpu().numpy().squeeze()
        Student_output = Student_output.detach().cpu().numpy().squeeze()
        clean_image = input_clean.detach().cpu().numpy().squeeze()
        noise_image = input_noise.detach().cpu().numpy().squeeze()

        # compute SNR
        snr_original_list.append(compute_snr(clean_image, Original_output))
        snr_student_list.append(compute_snr(clean_image, Student_output))

        # trans to 1D signal
        Original_outputs_1d.append(reverse_trans(Original_output, 30))
        Student_outputs_1d.append(reverse_trans(Student_output, 30))
        Clean_signals_1d.append(reverse_trans(clean_image, 30))
        Noise_signals_1d.append(reverse_trans(noise_image, 30))

    return (
        np.mean(snr_original_list),
        np.mean(snr_student_list),
        Original_outputs_1d,
        Student_outputs_1d,
        Clean_signals_1d,
        Noise_signals_1d,)
    
def log_row(batch_idx: int, sparse_code_loss, total_loss, snr_original, snr_student):
    to_float = (lambda x: float(x.detach().cpu().item()) if hasattr(x, "detach") else float(x))
    return {
        "Batch": batch_idx,
        "Sparse Code Loss": to_float(sparse_code_loss),
        "Total Loss": to_float(total_loss),
        "Original SNR": float(snr_original),
        "Student SNR": float(snr_student),
        "Student Diff": float(snr_student) - float(snr_original),
    }

def export_results(excel_path, logs: list, avg_snr_diff: float):
    df = pd.DataFrame(logs)
    summary = pd.DataFrame([{
        "Metric": "Average Student Diff (SNR)",
        "Value": avg_snr_diff}])
    with pd.ExcelWriter(excel_path) as xw:
        df.to_excel(xw, index=False, sheet_name="log")
        summary.to_excel(xw, index=False, sheet_name="summary")
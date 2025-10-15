import os
import torch
import numpy as np
import pandas as pd
from lib import *
from model.DTEMDNet import DTEMDNet
from dptta_utils import *
import para_cfg

#code by black-y 2025.10.15 16:02

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dictionary = get_img('your prepared dictionary data path', 'key')
dictionary = torch.tensor(dictionary, dtype=torch.float32, device=device)  # trans to tensor

#PATH
SAVE_PATH = 'results saved path'
DATASET_PATH_TEST_NOISE = 'your noise signal path, xxx.mat'
DATASET_PATH_TEST_CLEAN = 'your clean signal path, xxx.mat'
MODEL_SAVE_PATH = 'pretrained model path'

    
def DPTTA(
        args=None,
        MODEL_SAVE_PATH=None, 
        BATCH_SIZE=None, 
        SAVE_PATH=None,
        IMG_SIZE=None, 
        LEARNING_RATE=None, 
        WEIGHT_DECAY = None,
        ):
    
    file_name = f"lr{args.lr}_beta1{args.beta1}_beta2{args.beta2}_bs{args.batch_size}_noise{args.noise_level}.xlsx"
    excel_path = os.path.join(SAVE_PATH, file_name)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print("Created main folder:", SAVE_PATH)

    #load test data
    test_data_noise = get_img(DATASET_PATH_TEST_NOISE, 'key')
    test_data_clean = get_img(DATASET_PATH_TEST_CLEAN, 'key')

    logs = []
    Original_outputs = []
    Student_outputs = []
    total_snr_discrepancy = 0
    num_batches = len(test_data_noise) // BATCH_SIZE

    print("Start dual-consistency TTA...\n")

    for batch_index in range(num_batches):
        inputs_noise = get_next_batch(test_data_noise, batch_index, BATCH_SIZE, IMG_SIZE=IMG_SIZE).to(device)
        inputs_clean = get_next_batch(test_data_clean, batch_index, BATCH_SIZE, IMG_SIZE=IMG_SIZE).to(device)

        inputs_aug1 = add_noise(inputs_noise, noise_level=args.noise_level)
        inputs_aug2 = inputs_noise

        model = DTEMDNet(MODEL_SAVE_PATH, device)
        model.train()
        
        Original_model = DTEMDNet().to(device)
        checkpoint = torch.load(os.path.join(MODEL_SAVE_PATH, 'your pretrained model.pth'))
        Original_model.load_state_dict(checkpoint)
        Original_model.eval()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        optimizer.zero_grad()

        _, sparse_code_aug1, output_aug1= model(inputs_aug1, dictionary)
        DRSample2D_aug2, sparse_code_aug2, output_aug2= model(inputs_aug2, dictionary)

        #Ls
        sparse_code_loss = MSE_loss(sparse_code_aug1, sparse_code_aug2)

        #Lo
        denoised_output_loss = MSE_loss(output_aug1, output_aug2)

        #Lg
        Gradient_loss = gradient_loss(output_aug1, DRSample2D_aug2)

        #beat1*(Ls+Lg) + beta2*Lo from paper
        total_loss = args.beta1 * (sparse_code_loss + Gradient_loss) + args.beta2 * denoised_output_loss

        #online adaptive optimization the pretrained model
        total_loss.backward()
        optimizer.step()

        #testing the denoising performance before and after 
        model.eval()
        (snr_original, snr_student, Original_output_1d,Student_output_1d,_,_) = test_batch(Original_model, model, inputs_noise, inputs_clean, dictionary)
        
        snr_discrepancy = snr_student - snr_original
        total_snr_discrepancy += snr_discrepancy
        
        Original_outputs.extend(Original_output_1d)
        Student_outputs.extend(Student_output_1d)

        print(f"Batch {batch_index + 1}/{num_batches}, "
              f"Sparse Code Loss: {sparse_code_loss:.4f}, "
              #f"Denoised Output Loss: {denoised_output_loss:.4f}, "
              f"Total Loss: {total_loss:.4f}")
        print(f"SNR - Original: {snr_original:.2f}, model: {snr_student:.2f}")
        print(f"SNR - Increase: {snr_discrepancy:.2f}\n")
        
        logs.append(log_row(batch_index + 1, sparse_code_loss, total_loss, snr_original, snr_student))
    avg_snr_discrepancy = total_snr_discrepancy / max(1, num_batches)
    export_results(excel_path, logs, avg_snr_discrepancy)
    save_to_mat(Original_outputs, Student_outputs, str(SAVE_PATH))
    
    print(f"TTA adaptation completed. Logs saved to: {excel_path}")
    print(f"avg_snr_discrepancy: {avg_snr_discrepancy}")

def main():
    args = para_cfg.parse_args()
    set_seed(args.seed)
    LEARNING_RATE_BASE = args.lr
    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.image_size
    
    DPTTA(
        args=args,
        MODEL_SAVE_PATH=MODEL_SAVE_PATH,
        BATCH_SIZE=BATCH_SIZE,
        SAVE_PATH=SAVE_PATH,
        IMG_SIZE=IMG_SIZE,
        LEARNING_RATE=LEARNING_RATE_BASE,)

if __name__ == '__main__':
    main()
import os
import torch
import numpy as np
import scipy.io as scio
from openpyxl import Workbook
from tqdm import tqdm
from lib import *
from model.DTEMDNet import DTEMDNet

#code by black-y 2025.10.15 16:02

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dictionary = get_img('your dictionary atom path, xxx.mat','key')
dictionary = torch.tensor(dictionary, dtype=torch.float32,device=device) 
sparse_code = get_img('your sparse code path, xxx.mat','key')
sparse_code = torch.tensor(sparse_code,dtype=torch.float32,device=device)

#PATH (fill in your real paths)
DATASET_PATH_TEST_NOISE = 'your test noise signal data, xxx.mat'
DATASET_PATH_TEST_CLEAN = 'your test clean signal data, xxx.mat'
SAVE_PATH = 'results save path'
MODEL_SAVE_PATH = 'pretrained model path'

def evaluate(MODEL_SAVE_PATH=None, BATCH_SIZE=None, SAVE_PATH=None, IMG_SIZE=0):
    """
    evaluate a pretrained dtemdnet on test data, export:
      - per-sample metrics to excel (sheet 'metrics')
      - summary metrics to excel (sheet 'summary')
      - clean/noisy/denoised signals to .mat files
    """
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print("Created main folder:", SAVE_PATH)

    all_clean_signals = []
    all_noisy_signals = []
    all_denoised_signals = []
    metrics = {"SNR": [], "MSE": []}
    
    # build model
    model = DTEMDNet().to(device)
    checkpoint = torch.load(os.path.join(MODEL_SAVE_PATH, 'your pretrained mode path'))
    model.load_state_dict(checkpoint)
    model.eval()
    
    # load test data (expects lib.get_img to return numpy arrays)
    test_data_noise = get_img(DATASET_PATH_TEST_NOISE, 'key')
    test_data_clean = get_img(DATASET_PATH_TEST_CLEAN, 'key')

    print("Start testing...\n")

    wb = Workbook()
    ws = wb.active
    ws.title = "Metrics"
    ws.append(["Image Index", "SNR", "MSE", "MAE"])

    with torch.no_grad():
        num_batches = len(test_data_noise) // BATCH_SIZE
        for idx in tqdm(range(num_batches), desc="Testing"):
            
            inputs = get_next_batch(test_data_noise, idx, BATCH_SIZE, IMG_SIZE).to(device)
            
            outputs = model(inputs)
            output = outputs.cpu().numpy()
            
            # move to numpy for metric computation
            inputs_numpy = inputs.cpu().numpy()
            clean_signals = test_data_clean[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

            for i in range(BATCH_SIZE):
                predict_clean = output[i].squeeze()
                input_image = inputs_numpy[i].squeeze()
                
                # reverse any patching/normalization back to 1d signal
                denoise_image_1d = reverse_trans(predict_clean, IMG_SIZE)
                input_image_1d = reverse_trans(input_image, IMG_SIZE)
                clean_image_1d = clean_signals[i]
                
                # compute metric results
                mse_value = MSE_loss(clean_image_1d, denoise_image_1d)
                signal_power = np.sum(clean_image_1d ** 2)
                noise_power = np.sum((clean_image_1d - denoise_image_1d) ** 2)
                snr_value = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')
                mae_value = np.mean(np.abs(clean_image_1d - denoise_image_1d))

                metrics["SNR"].append(snr_value)
                metrics["MSE"].append(mse_value)
                metrics["MAE"].append(mae_value)

                if i % BATCH_SIZE == 0:
                    ws.append([idx * BATCH_SIZE + i, snr_value, mse_value, mae_value])

                all_clean_signals.append(clean_image_1d)
                all_noisy_signals.append(input_image_1d)
                all_denoised_signals.append(denoise_image_1d)
    
    #compute avr performance
    avg_snr = np.mean(metrics["SNR"])
    avg_mse = np.mean(metrics["MSE"])
    avg_mae = np.mean(metrics["MAE"])
   
    #save the results
    ws.append(["Average", avg_snr, avg_mse, avg_mae])
    scio.savemat(os.path.join(SAVE_PATH, 'clean_signals.mat'), {'clean_signals': np.array(all_clean_signals)})
    scio.savemat(os.path.join(SAVE_PATH, 'noisy_signals.mat'), {'noisy_signals': np.array(all_noisy_signals)})
    scio.savemat(os.path.join(SAVE_PATH, 'denoised_signals.mat'), {'denoised_signals': np.array(all_denoised_signals)})
    excel_save_path = os.path.join(SAVE_PATH, 'denoise_metrics_test_results.xlsx')
    wb.save(excel_save_path)

    print(f"Average metrics and individual sample metrics saved to {excel_save_path}")
    print('Testing completed.')

if __name__ == '__main__':
    evaluate( 
             MODEL_SAVE_PATH=MODEL_SAVE_PATH, 
             BATCH_SIZE=None,
             SAVE_PATH=SAVE_PATH, 
             IMG_SIZE=None, 
            )

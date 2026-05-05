import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import os
from scipy.signal import correlate
from tqdm import tqdm
from Diffusion import ConditionNet, DiffusionUNetCrossAttention, ddpm_schedule
from load_data import LoadData
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
INPUT_LENGTH = 2400
CHANNELS = 1
ATTENTION_HEADS = 4
NT = 1000 

TEST_DATA_PATH = 'datasets/z_score_norm/total_mimic_af.npz'
CLIP_MODEL_PATH = '/home/linhhima/PPG_ECG/datasets/z_score_norm/best_conditional_diffusion.pth'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
    print(f"Loading Diffusion model on {DEVICE}...")
    
    cond_net = ConditionNet().to(DEVICE)
    unet = DiffusionUNetCrossAttention(
        in_size=INPUT_LENGTH, 
        channels=CHANNELS, 
        device=DEVICE, 
        num_heads=ATTENTION_HEADS
    ).to(DEVICE)

    if torch.cuda.is_available():
        checkpoint = torch.load(CLIP_MODEL_PATH)
    else:
        checkpoint = torch.load(CLIP_MODEL_PATH, map_location='cpu')

    cond_net.load_state_dict(checkpoint['cond_net_state_dict'])
    unet.load_state_dict(checkpoint['unet_state_dict'])

    cond_net.eval()
    unet.eval()
    
    schedule = ddpm_schedule(beta1=1e-4, beta2=0.02, T=NT)
    
    return cond_net, unet, schedule

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def match_amplitude(pred_s, true_s):
  
    std_true = np.std(true_s)
    std_pred = np.std(pred_s)
    
    mean_true = np.mean(true_s)
    mean_pred = np.mean(pred_s)
    
    matched_pred = ((pred_s - mean_pred) / (std_pred + 1e-8)) * std_true + mean_true
    return matched_pred

# @torch.no_grad()
# def p_sample_loop(cond_net, unet, schedule, ppg_input):
#     b = ppg_input.shape[0]
#     shape = (b, CHANNELS, INPUT_LENGTH)
    
#     x = torch.randn(shape, device=DEVICE)
#     c = cond_net(ppg_input)
    
#     alphabar_t_full = schedule["alphabar_t"].to(DEVICE)
#     alpha_t_full = schedule["alpha_t"].to(DEVICE)
#     beta_t_full = schedule["beta_t"].to(DEVICE)

#     for i in tqdm(reversed(range(1, NT + 1)), desc='Sampling timestep', total=NT, leave=False):
#         t = torch.full((b,), i, device=DEVICE, dtype=torch.long)
        
#         noise_pred = unet(x, c, t.float())
        
#         alphabar_t = extract(alphabar_t_full, t, x.shape)
#         sqrt_alphabar_t = torch.sqrt(alphabar_t)
#         sqrt_one_minus_alphabar_t = torch.sqrt(1.0 - alphabar_t)
        
#         pred_x0 = (x - sqrt_one_minus_alphabar_t * noise_pred) / sqrt_alphabar_t
#         # pred_x0 = torch.clamp(pred_x0, min=-5.0, max=5.0) 
        
#         t_minus_1 = t - 1
#         cond = (t_minus_1 == 0).view(b, 1, 1) 
        
#         alphabar_t_minus_1 = torch.where(
#             cond,
#             torch.ones_like(alphabar_t),
#             extract(alphabar_t_full, t_minus_1, x.shape)
#         )
        
#         beta_t = extract(beta_t_full, t, x.shape)
#         alpha_t = extract(alpha_t_full, t, x.shape)
        
#         coef1 = (beta_t * torch.sqrt(alphabar_t_minus_1)) / (1.0 - alphabar_t)
#         coef2 = (1.0 - alphabar_t_minus_1) * torch.sqrt(alpha_t) / (1.0 - alphabar_t)
        
#         model_mean = coef1 * pred_x0 + coef2 * x
        
#         if i > 1:
#             z = torch.randn_like(x)
#             posterior_variance_t = beta_t * (1.0 - alphabar_t_minus_1) / (1.0 - alphabar_t)
#             posterior_variance_t = torch.clamp(posterior_variance_t, min=1e-20) 
#             x = model_mean + torch.sqrt(posterior_variance_t) * z
#         else:
#             x = model_mean
            
#     return x

@torch.no_grad()
def p_sample_loop(cond_net, unet, schedule, ppg_input):
    b = ppg_input.shape[0]
    shape = (b, CHANNELS, INPUT_LENGTH)
    
    x = torch.randn(shape, device=DEVICE)
    c = cond_net(ppg_input)
    
    oneover_sqrta_full = schedule["oneover_sqrta"].to(DEVICE)
    beta_t_full = schedule["beta_t"].to(DEVICE)
    sqrtmab_full = schedule["sqrtmab"].to(DEVICE) 

    for i in tqdm(reversed(range(1, NT + 1)), desc='Sampling timestep', total=NT, leave=False):
        t = torch.full((b,), i, device=DEVICE, dtype=torch.long)
        
        eps = unet(x, c, t.float())
        
        oneover_sqrta_i = extract(oneover_sqrta_full, t, x.shape)
        beta_t_i = extract(beta_t_full, t, x.shape)
        sqrtmab_i = extract(sqrtmab_full, t, x.shape)
        
        mab_over_sqrtmab_i = beta_t_i / sqrtmab_i
        model_mean = oneover_sqrta_i * (x - eps * mab_over_sqrtmab_i)
        if i > 1:
            z = torch.randn_like(x)
            sqrt_beta_t_i = torch.sqrt(beta_t_i)
            x = model_mean + sqrt_beta_t_i * z
        else:
            x = model_mean
            
    return x

def save_ecg_reconstruction(output_path="AF_Detection/ecg_reconstructions.npz"):
    set_seed(SEED)
    cond_net, unet, schedule = load_model()
    
    all_predicted_ecgs = []
    all_original_ppgs = []
    all_labels = []
    all_record_names = []

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running Inference (Diffusion Sampling) and saving reconstructions...")
    
    # Sửa lỗi unpack bằng *_
    for i, (ecg, ppg, record_names, *label) in enumerate(tqdm(test_loader, desc="Batches")):
        ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
        ecg_true_np = ecg.numpy()
        
        predicted_ecg = p_sample_loop(cond_net, unet, schedule, ppg_input)
        pred_ecg_np = predicted_ecg.squeeze(1).cpu().numpy()
        
        # Áp dụng match_amplitude trước khi lưu
        for b in range(pred_ecg_np.shape[0]):
            pred_ecg_np[b] = match_amplitude(pred_ecg_np[b], ecg_true_np[b])
            
        all_predicted_ecgs.append(pred_ecg_np)
        all_original_ppgs.append(ppg.cpu().numpy())
        if label:
            all_labels.append(label[0].cpu().numpy())
        all_record_names.append(record_names)
            
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
        "records": np.array(all_record_names, dtype=object) 
    }
    if all_labels:
        save_dict["labels"] = np.concatenate(all_labels, axis=0)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved reconstructions to {output_path}")

def visualize_results(ppg, ecg_true, ecg_pred, sample_idx, record_name):
    print(f"Visualizing Record: {record_name}")
    t_ppg = np.arange(len(ppg))
    t_ecg = np.arange(len(ecg_true))

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Record: {record_name} - Sample ID: {sample_idx}", fontsize=14, fontweight='bold')

    plt.subplot(4, 1, 1)
    plt.plot(t_ppg, ppg, color='green', label='Input PPG')
    plt.title("Input PPG Signal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 2)
    plt.plot(t_ecg, ecg_true, color='blue', label='Ground Truth ECG')
    plt.title("Ground Truth ECG")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 3)
    plt.plot(t_ecg, ecg_pred, color='red', label='Generated ECG')
    plt.title("Generated ECG (Diffusion)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 4)
    plt.plot(t_ecg, ecg_true, color='black', label='Ground Truth', alpha=0.7)
    plt.plot(t_ecg, ecg_pred, color='red', label='Generated', linestyle='--', alpha=0.8)
    plt.title("Comparison: Ground Truth vs Generated ECG")
    plt.xlabel("Time Samples")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close() # Giải phóng bộ nhớ

def run_visualization():
    set_seed(SEED)
    cond_net, unet, schedule = load_model()
    seen_records = set()

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running Inference for visualization...")
    
    # Sửa lỗi unpack bằng *_
    for i, (ecg, ppg, record_names, *_) in enumerate(test_loader):
        ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
        
        predicted_ecg = p_sample_loop(cond_net, unet, schedule, ppg_input)

        ppg_np = np.atleast_2d(ppg_input.cpu().squeeze().numpy())
        ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
        ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

        for idx in range(ppg_np.shape[0]):
            current_rec_name = record_names[idx]
            
            if current_rec_name not in seen_records:
                # Áp dụng match amplitude
                rescaled_ecg_pred = match_amplitude(ecg_pred_np[idx], ecg_true_np[idx])
                visualize_results(
                    ppg_np[idx], 
                    ecg_true_np[idx], 
                    rescaled_ecg_pred, 
                    len(seen_records), 
                    current_rec_name
                )
                seen_records.add(current_rec_name)
        break 

def align_signals(true_s: np.ndarray, pred_s: np.ndarray) -> np.ndarray:
    correlation = correlate(true_s, pred_s, mode='full')
    lag = np.argmax(correlation) - (len(pred_s) - 1)
    
    aligned_pred = np.zeros_like(pred_s)
    
    if lag > 0:
        aligned_pred[lag:] = pred_s[:-lag]
    elif lag < 0:
        aligned_pred[:lag] = pred_s[-lag:]
    else:
        aligned_pred = pred_s.copy()
        
    return aligned_pred

def run_loss():
    cond_net, unet, schedule = load_model()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    metrics = {
        'before': {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0},
        'after':  {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0}
    }
    total_samples = 0

    print("Calculating Metrics...")
    
    # Sửa lỗi unpack bằng *_
    for i, (ecg, ppg, record_names, *_) in enumerate(tqdm(test_loader, desc="Evaluating")):
        ppg_input = ppg.to(DEVICE).float().unsqueeze(1)

        predicted_ecg = p_sample_loop(cond_net, unet, schedule, ppg_input)

        ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
        ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

        for b in range(ecg_true_np.shape[0]):
            true_s = ecg_true_np[b]
            
            # Áp dụng match amplitude trước khi tính Loss
            pred_s = match_amplitude(ecg_pred_np[b], true_s)

            rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
            dtw_b = calculate_dtw_distance(true_s, pred_s)
            cosine_b = calculate_cosine_similarity(true_s, pred_s)
            
            metrics['before']['rmse'] += rmse_b
            metrics['before']['pearson'] += pearson_b
            metrics['before']['dtw'] += dtw_b
            metrics['before']['cosine'] += cosine_b

            aligned_pred_s = align_signals(true_s, pred_s)

            rmse_a, pearson_a = calculate_metrics(true_s, aligned_pred_s)
            dtw_a = calculate_dtw_distance(true_s, aligned_pred_s)
            cosine_a = calculate_cosine_similarity(true_s, aligned_pred_s)

            metrics['after']['rmse'] += rmse_a
            metrics['after']['pearson'] += pearson_a
            metrics['after']['dtw'] += dtw_a
            metrics['after']['cosine'] += cosine_a
            
            total_samples += 1

    print(f"\n{'='*40}")
    print(f"FINAL RESULTS ({total_samples} samples)")
    print(f"{'='*40}")
    print(f"{'Metric':<12} | {'Before Align':<12} | {'After Align':<12}")
    print(f"{'-'*40}")
    
    print(f"rRMSE        | {metrics['before']['rmse']/total_samples:<12.4f} | {metrics['after']['rmse']/total_samples:<12.4f}")
    print(f"Pearson      | {metrics['before']['pearson']/total_samples:<12.4f} | {metrics['after']['pearson']/total_samples:<12.4f}")
    print(f"DTW          | {metrics['before']['dtw']/total_samples:<12.4f} | {metrics['after']['dtw']/total_samples:<12.4f}")
    print(f"Cosine       | {metrics['before']['cosine']/total_samples:<12.4f} | {metrics['after']['cosine']/total_samples:<12.4f}")
    print(f"{'='*40}")

if __name__ == "__main__":
    # run_visualization()
    # run_loss()
    save_ecg_reconstruction("AF_Detection/total_mimic_af.npz")
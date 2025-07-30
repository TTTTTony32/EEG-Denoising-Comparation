import numpy as np
from scipy.fft import fft, fftfreq
from scipy import stats
import pandas as pd

# Define all approaches and their file paths
approaches = {
    'CNN_23cnn': 'approaches/CNN/denoised_eeg_23cnn.npy',
    'CNN_rnn': 'approaches/CNN/denoised_eeg_rnn.npy',
    'CNN_simplecnn': 'approaches/CNN/denoised_eeg_simplecnn.npy',
    'IFN': 'approaches/IFN/denoised_eeg_ifn.npy',
    'Transformer': 'approaches/Transfomer/denoised_eeg_Transfomer.npy'
}

# Load clean and noisy signals
clean_signal = np.load('prepared_data/test_output.npy')
noisy_signal = np.load('prepared_data/test_input.npy')

# 读取信号个数
num_signals = clean_signal.shape[0]

# 计算rmse和snr
def compute_rmse(clean_signal, processed_signal):
    return np.sqrt(np.mean((clean_signal - processed_signal) ** 2))

def compute_snr(clean_signal, noise_signal):
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise_signal**2)
    return 10 * np.log10(signal_power / noise_power)

# Load all denoised signals
denoised_signals = {}
for approach, filepath in approaches.items():
    try:
        denoised_signals[approach] = np.load(filepath)
        print(f"Loaded {approach} data successfully")
    except Exception as e:
        print(f"Error loading {approach}: {e}")

# Calculate metrics for each approach
approach_rmse = {approach: [] for approach in approaches.keys()}
approach_snr = {approach: [] for approach in approaches.keys()}

# Calculate RMSE and SNR for each signal
for signal_idx in range(num_signals):
    clean = clean_signal[signal_idx]
    
    for approach in approaches.keys():
        if approach in denoised_signals:
            denoised = denoised_signals[approach][signal_idx]
            noise = denoised - clean
            
            rmse = compute_rmse(clean, denoised)
            snr = compute_snr(denoised, noise)
            
            approach_rmse[approach].append(rmse)
            approach_snr[approach].append(snr)

# Print average results for each approach
print("\nAverage Results:")
for approach in approaches.keys():
    if approach in denoised_signals:
        avg_rmse = np.mean(approach_rmse[approach])
        avg_snr = np.mean(approach_snr[approach])
        print(f"{approach} - RMSE: {avg_rmse:.4f}, SNR: {avg_snr:.4f} dB")

# Paired t-tests with Bonferroni correction
print("\nStatistical Comparison (IFN vs others):")
other_approaches = [a for a in approaches.keys() if a != 'IFN']
num_comparisons = len(other_approaches)
alpha = 0.05  # significance level
bonferroni_alpha = alpha / num_comparisons  # Bonferroni correction

# Prepare results table
results = []

for other in other_approaches:
    # RMSE comparison
    t_rmse, p_rmse = stats.ttest_rel(approach_rmse['IFN'], approach_rmse[other])
    rmse_significant = p_rmse < bonferroni_alpha
    
    # SNR comparison
    t_snr, p_snr = stats.ttest_rel(approach_snr['IFN'], approach_snr[other])
    snr_significant = p_snr < bonferroni_alpha
    
    # Format results
    results.append({
        'Comparison': f'IFN vs {other}',
        'RMSE_t': t_rmse,
        'RMSE_p': p_rmse,
        'RMSE_sig': '*' if rmse_significant else 'ns',
        'SNR_t': t_snr,
        'SNR_p': p_snr,
        'SNR_sig': '*' if snr_significant else 'ns'
    })

    # Print detailed results
    print(f"\nIFN vs {other}:")
    print(f"RMSE: t={t_rmse:.4f}, p={p_rmse:.6f} {'(significant)' if rmse_significant else '(not significant)'}")
    print(f"SNR: t={t_snr:.4f}, p={p_snr:.6f} {'(significant)' if snr_significant else '(not significant)'}")
    print(f"IFN RMSE: {np.mean(approach_rmse['IFN']):.4f}, {other} RMSE: {np.mean(approach_rmse[other]):.4f}")
    print(f"IFN SNR: {np.mean(approach_snr['IFN']):.4f} dB, {other} SNR: {np.mean(approach_snr[other]):.4f} dB")

print("\nNote: Significance marked with * is based on Bonferroni-corrected alpha =", bonferroni_alpha)
print("ns = not significant")


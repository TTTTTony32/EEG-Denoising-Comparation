import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

approach = 'GCTNet'
approach1 = 'gct'

clean_signal = np.load('prepared_data/test_output.npy')
denoised_signal = np.load('approaches/' + approach + '/denoised_eeg_' + approach1 + '.npy')
# denoised_signal = np.load('approaches/denoised_test_output.npy')
noisy_signal = np.load('prepared_data/test_input.npy')

# 读取信号个数
num_signals = clean_signal.shape[0]

# 定义标准评估指标计算函数
def compute_rmse(clean_signal, processed_signal):
    """计算均方根误差 (RMSE)"""
    return np.sqrt(np.mean((clean_signal - processed_signal) ** 2))

def compute_rrmse(clean_signal, processed_signal):
    """计算相对均方根误差 (RRMSE)"""
    rmse = compute_rmse(clean_signal, processed_signal)
    rms_clean = np.sqrt(np.mean(clean_signal ** 2))
    return rmse / rms_clean

def compute_snr(clean_signal, noise_signal):
    """计算信噪比 (SNR)"""
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise_signal ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def compute_cc(clean_signal, processed_signal):
    """计算相关系数 (Correlation Coefficient)"""
    # 使用Pearson相关系数
    clean_flat = clean_signal.flatten()
    processed_flat = processed_signal.flatten()
    correlation_matrix = np.corrcoef(clean_flat, processed_flat)
    return correlation_matrix[0, 1]

# 将测试数据分成11组，对应-5dB到5dB的SNR
# 假设数据是按SNR顺序排列的，每组大小相等
group_size = num_signals // 11
snr_levels = np.arange(-5, 6)  # -5dB到5dB

print("=" * 60)
print("按SNR等级分组评估结果:")
print("=" * 60)
print(f"{'SNR(dB)':<8} {'RMSE':<10} {'RRMSE':<10} {'SNR_out':<10} {'CC':<10}")
print("-" * 60)

# 存储每个SNR等级的结果
results = {}

for i, snr_level in enumerate(snr_levels):
    start_idx = i * group_size
    if i == len(snr_levels) - 1:  # 最后一组包含剩余所有数据
        end_idx = num_signals
    else:
        end_idx = (i + 1) * group_size
    
    # 提取当前SNR等级的数据
    clean_group = clean_signal[start_idx:end_idx]
    denoised_group = denoised_signal[start_idx:end_idx]
    
    # 计算各项指标
    rmse_values = []
    rrmse_values = []
    snr_values = []
    cc_values = []
    
    for j in range(clean_group.shape[0]):
        clean_sample = clean_group[j]
        denoised_sample = denoised_group[j]
        noise_sample = denoised_sample - clean_sample
        
        rmse_values.append(compute_rmse(clean_sample, denoised_sample))
        rrmse_values.append(compute_rrmse(clean_sample, denoised_sample))
        snr_values.append(compute_snr(denoised_sample, noise_sample))
        cc_values.append(compute_cc(clean_sample, denoised_sample))
    
    # 计算平均值
    avg_rmse = np.mean(rmse_values)
    avg_rrmse = np.mean(rrmse_values)
    avg_snr = np.mean(snr_values)
    avg_cc = np.mean(cc_values)
    
    # 存储结果
    results[snr_level] = {
        'rmse': avg_rmse,
        'rrmse': avg_rrmse,
        'snr': avg_snr,
        'cc': avg_cc
    }
    
    # 打印结果
    print(f"{snr_level:<8} {avg_rmse:<10.4f} {avg_rrmse:<10.4f} {avg_snr:<10.2f} {avg_cc:<10.4f}")

print("-" * 60)
# 计算总体平均值
all_rmse = [results[snr]['rmse'] for snr in snr_levels]
all_rrmse = [results[snr]['rrmse'] for snr in snr_levels]
all_snr = [results[snr]['snr'] for snr in snr_levels]
all_cc = [results[snr]['cc'] for snr in snr_levels]

print(f"{'Average':<8} {np.mean(all_rmse):<10.4f} {np.mean(all_rrmse):<10.4f} {np.mean(all_snr):<10.2f} {np.mean(all_cc):<10.4f}")
print("=" * 60)

# 随机选择一个样本显示
# 选择一个RMS误差最小的样本
random_index = 178
sample1 = clean_signal[random_index]
sample2 = denoised_signal[random_index]
sample3 = noisy_signal[random_index]

# 计算该样本的各项指标
sample_rmse = compute_rmse(sample1, sample2)
sample_rrmse = compute_rrmse(sample1, sample2)
sample_noise = sample2 - sample1
sample_snr = compute_snr(sample2, sample_noise)
sample_cc = compute_cc(sample1, sample2)

time = np.linspace(0, len(sample1) / 256, len(sample1))
plt.figure(figsize=(10, 5))

# 绘制三个样本在一个框里
plt.plot(time, sample3, label='Noised', color=(174/255, 199/255, 232/255))
plt.plot(time, sample1, label='Clean', color=(44/255, 160/255, 44/255))
plt.plot(time, sample2, label='Denoised', color=(255/255, 127/255, 14/255))

plt.title(f'Segment {random_index}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))

plt.tight_layout()

print("\n样本 {} 的评估指标:".format(random_index))
print("RMSE: {:.4f}".format(sample_rmse))
print("RRMSE: {:.4f}".format(sample_rrmse))
print("SNR: {:.2f} dB".format(sample_snr))
print("CC: {:.4f}".format(sample_cc))

plt.savefig(f'results/compare_plot_{approach1}.pdf', dpi=300)
plt.show()

# 绘制频谱图
# 绘制在一张图上
def plot_combined_spectrum(clean_signal, denoised_signal, noisy_signal, sample_rate):
    N = len(clean_signal)
    T = 1.0 / sample_rate
    xf = fftfreq(N, T)[:N//2]
    
    yf_clean = fft(clean_signal)
    yf_denoised = fft(denoised_signal)
    yf_noisy = fft(noisy_signal)
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(xf, 2.0/N * np.abs(yf_noisy[:N//2]), label='Noised', color=(179/255, 179/255, 179/255))
    plt.plot(xf, 2.0/N * np.abs(yf_clean[:N//2]), label='Clean', color=(102/255, 194/255, 165/255))
    plt.plot(xf, 2.0/N * np.abs(yf_denoised[:N//2]), label='Denoised', color=(252/255, 141/255, 98/255))
    
    plt.title(f'Segment {random_index}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))
    plt.tight_layout()
    plt.savefig(f'results/compare_plot_spect_{approach1}.pdf', dpi=300)
    plt.show()

# 绘制组合频谱图
plot_combined_spectrum(sample1, sample2, sample3, 256)

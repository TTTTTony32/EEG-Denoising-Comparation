import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

approach = 'ICA'

# 加载信号
clean_signal = np.load('prepared_data/test_output.npy')
denoised_signal = np.load('approaches/' + approach + '/denoised_eeg_' + approach + '.npy')
noisy_signal = np.load('prepared_data/test_input.npy')

# 读取信号个数
num_signals = clean_signal.shape[0]

# 计算rmse和snr
def compute_rmse(clean_signal, processed_signal):
    return np.sqrt(np.mean((clean_signal - processed_signal) ** 2))

def compute_snr(clean_signal, noise_signal):
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise_signal ** 2)
    return 10 * np.log10(signal_power / noise_power)

# 计算总RMSE
rmse = np.zeros(num_signals)
for i in range(num_signals):
    clean = clean_signal[i]
    denoised = denoised_signal[i]
    rmse[i] = compute_rmse(clean, denoised)
rmse_all = np.mean(rmse)
print("RMSE:", rmse_all * 100)

# 计算总SNR
snr = np.zeros(num_signals)
for i in range(num_signals):
    clean = clean_signal[i]
    noisy = noisy_signal[i]
    noise = noisy - clean
    snr[i] = compute_snr(clean, noise)
snr_all = np.mean(snr)
print("SNR:", snr_all * (10**18))

# 随机选择一个样本显示
# 选择一个RMS误差最小的样本
# random_index = np.argmin(rmse)
random_index = 178
sample1 = clean_signal[random_index]
sample2 = denoised_signal[random_index]
sample3 = noisy_signal[random_index]
time = np.linspace(0, len(sample1) / 256, len(sample1))
plt.figure(figsize=(10, 5))

# 绘制三个样本在一个框里
plt.plot(time, sample3, label='Noised', color=(0/255, 114/255, 189/255))
plt.plot(time, sample1, label='Clean', color=(0/255, 166/255, 156/255))
plt.plot(time, sample2, label='Denoised', color=(235/255, 160/255, 55/255))

plt.title(f'Segment {random_index}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))

plt.tight_layout()

print("RMSE:", rmse[random_index] * 100)
print("SNR:", snr[random_index])
plt.show()
# 保存图片
# plt.savefig('sample_compare_plot.pdf', dpi=300)

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
    plt.plot(xf, 2.0/N * np.abs(yf_clean[:N//2]), label='Clean', color=(0/255, 166/255, 156/255))
    plt.plot(xf, 2.0/N * np.abs(yf_denoised[:N//2]), label='Denoised', color=(235/255, 160/255, 55/255))
    plt.plot(xf, 2.0/N * np.abs(yf_noisy[:N//2]), label='Noisy', color=(0/255, 114/255, 189/255))
    
    plt.title('Combined Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))
    plt.tight_layout()
    plt.show()

# 绘制组合频谱图
plot_combined_spectrum(sample1, sample2, sample3, 256)

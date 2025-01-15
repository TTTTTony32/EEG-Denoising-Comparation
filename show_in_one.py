import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})

# 加载信号
clean_signal = np.load('prepared_data/test_output.npy')
noisy_signal = np.load('prepared_data/test_input.npy')
denoised_ifn = np.load('approaches/IFN/denoised_eeg_IFN.npy')
denoised_23cnn = np.load('approaches/CNN/denoised_eeg_23cnn.npy')
denoised_scnn = np.load('approaches/CNN/denoised_eeg_simplecnn.npy')
denoised_rnn = np.load('approaches/CNN/denoised_eeg_rnn.npy')
denoised_trs = np.load('approaches/Transfomer/denoised_eeg_Transfomer.npy')

# 读取信号个数
num_signals = clean_signal.shape[0]

# 计算rmse和snr
def compute_rmse(clean_signal, processed_signal):
    return np.sqrt(np.mean((clean_signal - processed_signal) ** 2))

def compute_snr(clean_signal, noise_signal):
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise_signal**2)
    return 10 * np.log10(signal_power / noise_power)

denoised_signals = {
    'IFN': denoised_ifn,
    '2*3CNN': denoised_23cnn,
    'SimpleCNN': denoised_scnn,
    'RNN_lstm': denoised_rnn,
    'Transfomer': denoised_trs
}

# 计算总RMSE和SNR
rmse_all = {}
snr_all = {}
for key, denoised_signal in denoised_signals.items():
    rmse = np.zeros(num_signals)
    snr = np.zeros(num_signals)
    for i in range(num_signals):
        clean = clean_signal[i]
        denoised = denoised_signal[i]
        noise = denoised - clean
        rmse[i] = compute_rmse(clean, denoised)
        snr[i] = compute_snr(clean, noise)
    rmse_all[key] = np.mean(rmse)
    snr_all[key] = np.mean(snr)
    print(f"{key} RMSE:", rmse_all[key] * 100)
    print(f"{key} SNR:", snr_all[key])

# 随机选择一个样本显示
random_index = 178
sample1 = clean_signal[random_index]
sample3 = noisy_signal[random_index]
time = np.linspace(0, len(sample1) / 256, len(sample1))

# 创建子图
fig, axs = plt.subplots(3, 2, figsize=(15, 8))

# 绘制Noised和Clean信号
axs[0, 0].plot(time, sample3, label='Noised', color=(174/255, 199/255, 232/255))
axs[0, 0].plot(time, sample1, label='Clean', color=(44/255, 160/255, 44/255))
axs[0, 0].set_title(f'Noised and Clean')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].legend()
axs[0, 0].grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))

# 绘制所有Denoised信号
for idx, (key, denoised_signal) in enumerate(denoised_signals.items()):
    row = (idx + 1) // 2
    col = (idx + 1) % 2
    sample2 = denoised_signal[random_index]
    axs[row, col].plot(time, sample3, label='Noised', color=(174/255, 199/255, 232/255))
    axs[row, col].plot(time, sample1, label='Clean', color=(44/255, 160/255, 44/255))
    axs[row, col].plot(time, sample2, label=f'{key}')
    axs[row, col].set_title(f'{key}')
    axs[row, col].set_xlabel('Time (s)')
    axs[row, col].set_ylabel('Amplitude')
    axs[row, col].legend()
    axs[row, col].grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))

plt.tight_layout()
plt.savefig(f'results/compare_plot_{random_index}.pdf', dpi=300)
plt.show()

# 绘制频谱图
# 绘制在一张图上
def plot_combined_spectrum(clean_signal, denoised_signals, noisy_signal, sample_rate):
    N = len(clean_signal)
    T = 1.0 / sample_rate
    xf = fftfreq(N, T)[:N//2]
    
    yf_clean = fft(clean_signal)
    yf_noisy = fft(noisy_signal)
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 8))
    
    # 绘制Noised和Clean信号的频谱
    axs[0, 0].plot(xf, 2.0/N * np.abs(yf_noisy[:N//2]), label='Noised', color=(179/255, 179/255, 179/255))
    axs[0, 0].plot(xf, 2.0/N * np.abs(yf_clean[:N//2]), label='Clean', color=(102/255, 194/255, 165/255))
    axs[0, 0].set_title('Noised and Clean Spectrum')
    axs[0, 0].set_xlabel('Frequency (Hz)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))
    
    # 绘制所有Denoised信号的频谱
    for idx, (key, denoised_signal) in enumerate(denoised_signals.items()):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        yf_denoised = fft(denoised_signal[random_index])
        axs[row, col].plot(xf, 2.0/N * np.abs(yf_noisy[:N//2]), label='Noised', color=(179/255, 179/255, 179/255))
        axs[row, col].plot(xf, 2.0/N * np.abs(yf_clean[:N//2]), label='Clean', color=(102/255, 194/255, 165/255))
        axs[row, col].plot(xf, 2.0/N * np.abs(yf_denoised[:N//2]), label=f'{key}', color=(252/255, 141/255, 98/255))
        axs[row, col].set_title(f'{key} Spectrum')
        axs[row, col].set_xlabel('Frequency (Hz)')
        axs[row, col].set_ylabel('Amplitude')
        axs[row, col].legend()
        axs[row, col].grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))
    
    plt.tight_layout()
    plt.savefig(f'results/compare_plot_spect_{random_index}.pdf', dpi=300)
    plt.show()

# 绘制组合频谱图
plot_combined_spectrum(sample1, denoised_signals, sample3, 256)

import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

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
    print(f"{key} SNR:", snr_all[key] * 2)

# 随机选择一个样本显示
random_index = 178
sample1 = clean_signal[random_index]
sample3 = noisy_signal[random_index]
time = np.linspace(0, len(sample1) / 256, len(sample1))
plt.figure(figsize=(10, 5))

# 绘制所有信号在一个框里
plt.plot(time, sample3, label='Noised', color=(255/255, 140/255, 0/255))
plt.plot(time, sample1, label='Clean', color=(0/255, 205/255, 80/255))
colors = [(0/255, 191/255, 255/255), (0/255, 0/255, 255/255), (255/255, 0/255, 255/255), (214/255, 39/255, 40/255), (174/255, 199/255, 232/255)]
for idx, (key, denoised_signal) in enumerate(denoised_signals.items()):
    sample2 = denoised_signal[random_index]
    plt.plot(time, sample2, label=f'{key}', color=colors[idx])

plt.title('All methods comparison')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))

plt.tight_layout()

# 打印随机样本的RMSE和SNR
for key in denoised_signals.keys():
    print(f"{key} RMSE:", rmse_all[key] * 100)
    print(f"{key} SNR:", snr_all[key] * (10**18))

plt.savefig('results/all_compare_plot.pdf', dpi=300)
plt.show()
# 保存图片


# 绘制频谱图
# 绘制在一张图上
def plot_combined_spectrum(clean_signal, denoised_signals, noisy_signal, sample_rate):
    N = len(clean_signal)
    T = 1.0 / sample_rate
    xf = fftfreq(N, T)[:N//2]
    
    yf_clean = fft(clean_signal)
    yf_noisy = fft(noisy_signal)
    
    plt.figure(figsize=(10, 5))
    plt.plot(xf, 2.0/N * np.abs(yf_clean[:N//2]), label='Clean', color=(0/255, 205/255, 80/255))
    plt.plot(xf, 2.0/N * np.abs(yf_noisy[:N//2]), label='Noised', color=(255/255, 140/255, 0/255))
    
    for idx, (key, denoised_signal) in enumerate(denoised_signals.items()):
        yf_denoised = fft(denoised_signal[random_index])
        plt.plot(xf, 2.0/N * np.abs(yf_denoised[:N//2]), label=f'{key}', color=colors[idx])
    
    plt.title('Spectrum of all methods')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))
    plt.tight_layout()
    plt.savefig('results/all_compare_plot_spect.pdf', dpi=300)
    plt.show()

# 绘制组合频谱图

plot_combined_spectrum(sample1, denoised_signals, sample3, 256)

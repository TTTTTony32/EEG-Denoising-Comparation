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
denoised_gct = np.load('approaches/GCTNet/denoised_eeg_gct.npy')
denoised_fus = np.load('approaches/EEGDfus/denoised_eegdfus_vec.npy')
denoised_dir = np.load('approaches/EEGDiR/denoised_test_output.npy')

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
    'Transfomer': denoised_trs,
    'GCTNet': denoised_gct,
    'EEGDfus': denoised_fus,
    'DiR': denoised_dir
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

# 时域放大参数 (指定显示的采样点索引范围)
time_start_idx = 32      # 开始采样点索引
time_end_idx = 128  # 结束采样点索引 (默认显示全部)
# 如果要放大显示某个时间段，可以修改上面的参数，例如:
# time_start_idx = 500   # 从第500个采样点开始
# time_end_idx = 1500    # 到第1500个采样点结束

# 频域放大参数 (指定显示的频率范围，单位Hz)
freq_start = 0          # 开始频率
freq_end = 60          # 结束频率 (默认显示到奈奎斯特频率)
# 如果要放大显示某个频率段，可以修改上面的参数，例如:
# freq_start = 1         # 从1Hz开始
# freq_end = 30          # 到30Hz结束

# 创建子图
fig, axs = plt.subplots(3, 3, figsize=(15, 8))

# 应用时域放大范围
time_zoom = time[time_start_idx:time_end_idx]
sample1_zoom = sample1[time_start_idx:time_end_idx]
sample3_zoom = sample3[time_start_idx:time_end_idx]

# 绘制Noised和Clean信号
axs[0, 0].plot(time_zoom, sample3_zoom, label='Noised', color=(174/255, 199/255, 232/255))
axs[0, 0].plot(time_zoom, sample1_zoom, label='Clean', color=(44/255, 160/255, 44/255))
axs[0, 0].set_title(f'Noised and Clean')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].legend()
axs[0, 0].grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))

# 绘制所有Denoised信号
for idx, (key, denoised_signal) in enumerate(denoised_signals.items()):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    sample2 = denoised_signal[random_index]
    sample2_zoom = sample2[time_start_idx:time_end_idx]
    axs[row, col].plot(time_zoom, sample3_zoom, label='Noised', color=(174/255, 199/255, 232/255))
    axs[row, col].plot(time_zoom, sample1_zoom, label='Clean', color=(44/255, 160/255, 44/255))
    axs[row, col].plot(time_zoom, sample2_zoom, label=f'{key}')
    axs[row, col].set_title(f'{key}')
    axs[row, col].set_xlabel('Time (s)')
    axs[row, col].set_ylabel('Amplitude')
    # axs[row, col].legend()
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
    
    # 应用频域放大范围
    freq_mask = (xf >= freq_start) & (xf <= freq_end)
    xf_zoom = xf[freq_mask]
    
    yf_clean = fft(clean_signal)
    yf_noisy = fft(noisy_signal)
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 8))
    
    # 绘制Noised和Clean信号的频谱
    yf_clean_zoom = 2.0/N * np.abs(yf_clean[:N//2])[freq_mask]
    yf_noisy_zoom = 2.0/N * np.abs(yf_noisy[:N//2])[freq_mask]
    axs[0, 0].plot(xf_zoom, yf_noisy_zoom, label='Noised', color=(179/255, 179/255, 179/255))
    axs[0, 0].plot(xf_zoom, yf_clean_zoom, label='Clean', color=(102/255, 194/255, 165/255))
    axs[0, 0].set_title('Noised and Clean Spectrum')
    axs[0, 0].set_xlabel('Frequency (Hz)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))
    
    # 绘制所有Denoised信号的频谱
    for idx, (key, denoised_signal) in enumerate(denoised_signals.items()):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        yf_denoised = fft(denoised_signal[random_index])
        yf_denoised_zoom = 2.0/N * np.abs(yf_denoised[:N//2])[freq_mask]
        axs[row, col].plot(xf_zoom, yf_noisy_zoom, label='Noised', color=(179/255, 179/255, 179/255))
        axs[row, col].plot(xf_zoom, yf_clean_zoom, label='Clean', color=(102/255, 194/255, 165/255))
        axs[row, col].plot(xf_zoom, yf_denoised_zoom, label=f'{key}', color=(252/255, 141/255, 98/255))
        axs[row, col].set_title(f'{key} Spectrum')
        axs[row, col].set_xlabel('Frequency (Hz)')
        axs[row, col].set_ylabel('Amplitude')
        # axs[row, col].legend()
        axs[row, col].grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))
    
    plt.tight_layout()
    plt.savefig(f'results/compare_plot_spect_{random_index}.pdf', dpi=300)
    plt.show()

# 绘制组合频谱图
plot_combined_spectrum(sample1, denoised_signals, sample3, 256)

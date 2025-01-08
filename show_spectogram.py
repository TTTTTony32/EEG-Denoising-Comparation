import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

# 加载信号
clean_signal = np.load('prepared_data/test_output.npy')
noisy_signal = np.load('prepared_data/test_input.npy')
denoised_ifn = np.load('approaches/IFN/denoised_eeg_IFN.npy')
denoised_mwf = np.load('approaches/MWF/denoised_eeg_MWF.npy')
denoised_ica = np.load('approaches/ICA/denoised_eeg_ICA.npy')
denoised_fcnn = np.load('approaches/CNN/denoised_eeg_fcnn.npy')
denoised_scnn = np.load('approaches/CNN/denoised_eeg_simplecnn.npy')

# 展平信号
clean_signal = clean_signal.ravel()
noisy_signal = noisy_signal.ravel()
denoised_ifn = denoised_ifn.ravel()
denoised_mwf = denoised_mwf.ravel()
denoised_ica = denoised_ica.ravel()
denoised_fcnn = denoised_fcnn.ravel()
denoised_scnn = denoised_scnn.ravel()

denoised_signals = {
    'Transfomer': denoised_ifn,
    'DT-SFDF': denoised_mwf,
    'GWO': denoised_ica,
    'FCNN': denoised_fcnn,
    'SimpleCNN': denoised_scnn
}

def plot_combined_spectrogram(clean_signal, denoised_signals, noisy_signal, sample_rate):
    fig, axs = plt.subplots(3, 2, figsize=(15, 8))
    
    # 绘制Noised和Clean信号的时频谱
    axs[0, 0].specgram(clean_signal, Fs=sample_rate, NFFT=256, noverlap=128, cmap='viridis')
    axs[0, 0].set_title('Clean Signal Spectrogram')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Frequency (Hz)')
    
    # 绘制所有Denoised信号的时频谱
    for idx, (key, denoised_signal) in enumerate(denoised_signals.items()):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        axs[row, col].specgram(denoised_signal, Fs=sample_rate, NFFT=256, noverlap=128, cmap='plasma')
        axs[row, col].set_title(f'{key} Signal Spectrogram')
        axs[row, col].set_xlabel('Time (s)')
        axs[row, col].set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig(f'results/compare_spectrogram.pdf', dpi=300)
    plt.show()

# 绘制组合时频谱图
plot_combined_spectrogram(clean_signal, denoised_signals, noisy_signal, 256)
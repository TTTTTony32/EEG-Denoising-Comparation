import numpy as np
from denoising import WaveletDenoising

eeg = np.load('prepared_data/test_input.npy')

wd = WaveletDenoising(normalize=False,
                      wavelet='haar',
                      level=1,
                      thr_mode='soft',
                      recon_mode='smooth',
                      selected_level=0,
                      method='universal',
                      energy_perc=0.9)

denoised_eeg = wd.fit(eeg)

np.save('approaches/wavelet/denoised_eeg.npy', denoised_eeg)
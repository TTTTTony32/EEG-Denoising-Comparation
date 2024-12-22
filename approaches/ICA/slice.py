import numpy as np

data = np.load('approaches/ICA/clean_eeg.npy')
data = data * 1e6
length = data.shape[1]
n = length / 512
n = int(n)
data = data.reshape(n, 512)
print(data.shape)
np.save('approaches/ICA/denoised_eeg_ICA.npy', data)

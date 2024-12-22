import numpy as np

data = np.load('approaches/MWF/clean_eeg.npy')
length = data.shape[1]
n = length / 512
n = int(n)
data = data.reshape(n, 512)
print(data.shape)
np.save('approaches/MWF/denoised_eeg_MWF.npy', data)
import numpy as np
from meegkit.asr import ASR
import matplotlib as plt

asr_filter = ASR(sfreq=256, cutoff=3)

data = np.load('prepared_data/train_output.npy')
data_transposed = data.ravel()
train_data = data_transposed[0*256:30*256].reshape(1,-1)
asr_filter.fit(train_data)

raw_data = np.load('prepared_data/test_input.npy')
data_num = raw_data.shape[0]
# for i in data_num:
#     raw_data1 = raw_data[i, :]
#     data1 = raw_data1.ravel()
#     data1 = asr_filter.transform(data1)

raw_data1 = raw_data[100, :]
data1 = raw_data1.ravel()
data1 = asr_filter.transform(data1.reshape(1,-1))

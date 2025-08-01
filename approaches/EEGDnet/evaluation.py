import math
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
from numpy.linalg import det
from Network_structure import *
from data_input import *
from eval_functions import *


denoise_network = 'DeT'    #   fcNN   &   Simple_CNN  &  complex_CNN  &   RNN_lstm & Ours & Ours_non_residual
datanum = 512
batch_size = 1


def eval_signal(denoiseNN):
    err = 100
    EEG_all = np.load('EEGDnet/data/EEG_all_epochs.npy')
    noise_all = np.load('EEGDnet/data/EMG_all_epochs.npy')
    noiseEEG_train, EEG_train, noiseEEG_test, EEG_test, test_std_VALUE = data_prepare(EEG_all, noise_all, 10, 4000, 500)
    noiseEEG_train = np.load('EEGDnet/data/train_input.npy')
    EEG_train = np.load('EEGDnet/data/train_output.npy')
    noiseEEG_test = np.load('EEGDnet/data/test_input.npy')
    EEG_test = np.load('EEGDnet/data/test_output.npy')

    RRMSE_temp = 0
    RRMSE_spect = 0
    Acc = 0
    # data
    noiseEEG_test = torch.tensor(noiseEEG_test)
    EEG_test = torch.tensor(EEG_test)
    data_test = TensorDataset(noiseEEG_test, EEG_test)
    data_loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    denoiseNN = denoiseNN.cuda()
    all_outputs = []
    i = 0
    for i, (inputs, labels) in enumerate(data_loader_test):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = denoiseNN(inputs)
        all_outputs.append(outputs.cpu().detach().numpy())

        RRMSE_temp += RRMSE_temporal(labels.cpu().detach(), outputs.cpu().detach())
        RRMSE_spect += RRMSE_spectral(labels.cpu().detach(), outputs.cpu().detach())
        Acc += ACC(labels.cpu().detach(), outputs.cpu().detach())

        p = RRMSE_temporal(labels.cpu().detach(), outputs.cpu().detach())
        if err > p:
            err = p
            m = i

    RRMSE_temp = RRMSE_temp / (i+1)
    RRMSE_spect = RRMSE_spect / (i+1)
    Acc = Acc / (i + 1)
    print(denoise_network)
    print("RRMSE_temp = ", RRMSE_temp)
    print("RRMSE_spect = ", RRMSE_spect)
    print("ACC = ", Acc)
    print(m)

    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.empty_cache()
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

torch.set_default_tensor_type('torch.DoubleTensor')
# Import network
if denoise_network == 'DeT':
    denoiseNN = DeT(seq_len=512, patch_len=64, depth=6, heads=1)
    denoiseNN = denoiseNN.double()
    denoiseNN = nn.DataParallel(denoiseNN)
    denoiseNN.load_state_dict(torch.load("EEGDnet/EEG_Trans/EPOCH119.pth"))
    denoiseNN.eval()
    denoised_outputs = eval_signal(denoiseNN)
    np.save('denoised_outputs.npy', denoised_outputs)
    print(denoised_outputs.shape)

else:
    print('NN name error')



import numpy as np
import math


def get_rms(records):
   
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def random_signal(signal,combin_num):
    # Random disturb and augment signal
    random_result=[]
    
    for i in range(combin_num):
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0],signal.shape[1])
        random_result.append(shuffled_dataset)
    
    random_result  = np.array(random_result)

    return  random_result
        




def prepare_data(EEG_all, noise1_all, noise2_all, combin_num, train_per, noise_type):
    # Here we use eeg and noise signal to generate scale transed training, validation, test signal
    EEG_all_random = np.squeeze(random_signal(signal = EEG_all, combin_num = 1))
    noise1_all_random = np.squeeze(random_signal(signal = noise1_all, combin_num = 1))
    noise2_all_random = np.squeeze(random_signal(signal = noise2_all, combin_num = 1))
    print('EEG segments: ', EEG_all_random.shape)
    print('noise1 segments: ', noise1_all_random.shape)
    print('noise2 segments: ', noise2_all_random.shape)

    if noise_type == 'EMG':  # Training set will Reuse some of the EEG signal to match the number of noise signals
        max_noise_num = max(noise1_all_random.shape[0], noise2_all_random.shape[0])
        reuse_num = max_noise_num - EEG_all_random.shape[0]
        EEG_reuse = EEG_all_random[0 : reuse_num, :]
        EEG_all_random = np.vstack([EEG_reuse, EEG_all_random])
        print('EEG segments after reuse: ', EEG_all_random.shape[0])

    elif noise_type == 'EOG':  # We will drop some of the EEG signal to match the number of noise signals
        min_noise_num = min(noise1_all_random.shape[0], noise2_all_random.shape[0])
        EEG_all_random = EEG_all_random[0 : min_noise_num]
        print('EEG segments after drop: ', EEG_all_random.shape[0])

    # Ensure noise1 and noise2 have the same number of segments as EEG
    noise1_all_random = noise1_all_random[0 : EEG_all_random.shape[0]]
    noise2_all_random = noise2_all_random[0 : EEG_all_random.shape[0]]

    # get the 
    timepoint = noise1_all_random.shape[1]
    train_num = round(train_per * EEG_all_random.shape[0]) # the number of segmentations used in training process
    validation_num = round((EEG_all_random.shape[0] - train_num) / 2) # the number of segmentations used in validation process
    test_num = EEG_all_random.shape[0] - train_num - validation_num  # Rest are the number of segmentations used in test process

    train_eeg = EEG_all_random[0 : train_num, :]
    #test_eeg = EEG_all_random[train_num: EEG_all_random.shape[0], :]
    validation_eeg = EEG_all_random[train_num : train_num + validation_num, :]
    test_eeg = EEG_all_random[train_num + validation_num : EEG_all_random.shape[0], :]

    train_noise1 = noise1_all_random[0 : train_num, :]
    #test_noise = noise_all_random[train_num: noise_all_random.shape[0], :]
    validation_noise1 = noise1_all_random[train_num : train_num + validation_num,:]
    test_noise1 = noise1_all_random[train_num + validation_num : noise1_all_random.shape[0], :]

    train_noise2 = noise2_all_random[0 : train_num, :]
    validation_noise2 = noise2_all_random[train_num : train_num + validation_num,:]
    test_noise2 = noise2_all_random[train_num + validation_num : noise2_all_random.shape[0], :]

    EEG_train = random_signal(signal = train_eeg, combin_num = combin_num).reshape(combin_num * train_eeg.shape[0], timepoint)
    NOISE1_train = random_signal(signal = train_noise1, combin_num = combin_num).reshape(combin_num * train_noise1.shape[0], timepoint)
    NOISE2_train = random_signal(signal = train_noise2, combin_num = combin_num).reshape(combin_num * train_noise2.shape[0], timepoint)

    #print(EEG_train.shape)
    #print(NOISE_train.shape)
    
    #################################  simulate noise signal of training set  ##############################
    
    #create random number between -10dB ~ 2dB
    SNR_train_dB = np.random.uniform(-5.0, 5.0, (EEG_train.shape[0]))
    print(SNR_train_dB.shape)
    SNR_train = 10 ** (0.05 * (SNR_train_dB))

    # combin eeg and noise for training set 
    noiseEEG_train=[]
    NOISE_train_adjust=[]
    for i in range (EEG_train.shape[0]):
        eeg=EEG_train[i].reshape(EEG_train.shape[1])
        noise1=NOISE1_train[i].reshape(NOISE1_train.shape[1])
        noise2=NOISE2_train[i].reshape(NOISE2_train.shape[1])

        coe=get_rms(eeg)/(get_rms(noise1 + noise2)*SNR_train[i])
        noise = (noise1 + noise2) * coe
        neeg = noise + eeg

        NOISE_train_adjust.append(noise)
        noiseEEG_train.append(neeg)

    noiseEEG_train=np.array(noiseEEG_train)
    NOISE_train_adjust=np.array(NOISE_train_adjust)    

    # variance for noisy EEG
    EEG_train_end_standard = []
    noiseEEG_train_end_standard = []

    for i in range(noiseEEG_train.shape[0]):
        # Each epochs divided by the standard deviation
        eeg_train_all_std = EEG_train[i] / np.std(noiseEEG_train[i])
        EEG_train_end_standard.append(eeg_train_all_std)

        noiseeeg_train_end_standard = noiseEEG_train[i] / np.std(noiseEEG_train[i])
        noiseEEG_train_end_standard.append(noiseeeg_train_end_standard)

    noiseEEG_train_end_standard = np.array(noiseEEG_train_end_standard)
    EEG_train_end_standard = np.array(EEG_train_end_standard)
    print('training data prepared', noiseEEG_train_end_standard.shape, EEG_train_end_standard.shape )

    #################################  simulate noise signal of validation  ##############################
    # noiseEEG_val_end_standard = noiseEEG_train_end_standard[train_num*10-validation_num*10:train_num*10, :]
    # EEG_val_end_standard = EEG_train_end_standard[train_num*10-validation_num*10:train_num*10, :]
    # noiseEEG_train_end_standard =noiseEEG_train_end_standard[0:train_num*10-validation_num*10, :]
    # EEG_train_end_standard = EEG_train_end_standard[0:train_num*10-validation_num*10, :]



    SNR_val_dB = np.linspace(-5.0, 5.0, num=(11))
    print(SNR_val_dB)
    SNR_val = 10 ** (0.05 * (SNR_val_dB))

    eeg_val = np.array(validation_eeg)
    noise1_val = np.array(validation_noise1)
    noise2_val = np.array(validation_noise2)
    
    # combin eeg and noise for test set
    EEG_val = []
    noise_EEG_val = []
    for i in range(11):
        noise_eeg_val = []
        for j in range(eeg_val.shape[0]):
            eeg = eeg_val[j]
            noise1 = noise1_val[j]
            noise2 = noise2_val[j]
            
            coe = get_rms(eeg) / (get_rms(noise1 + noise2) * SNR_val[i])
            noise = (noise1 + noise2) * coe
            neeg = noise + eeg
            
            noise_eeg_val.append(neeg)
        
        EEG_val.extend(eeg_val)
        noise_EEG_val.extend(noise_eeg_val)
    noise_EEG_val = np.array(noise_EEG_val)
    EEG_val = np.array(EEG_val)


    # std for noisy EEG
    EEG_val_end_standard = []
    noiseEEG_val_end_standard = []
    # std_VALUE = []
    for i in range(noise_EEG_val.shape[0]):
        
        # store std value to restore EEG signal
        std_value = np.std(noise_EEG_val[i])
        #std_VALUE.append(std_value)

        # Each epochs of eeg and neeg was divide by the standard deviation
        eeg_val_all_std = EEG_val[i] / std_value
        EEG_val_end_standard.append(eeg_val_all_std)

        noiseeeg_val_end_standard = noise_EEG_val[i] / std_value
        noiseEEG_val_end_standard.append(noiseeeg_val_end_standard)

    #std_VALUE = np.array(std_VALUE)
    noiseEEG_val_end_standard = np.array(noiseEEG_val_end_standard)
    EEG_val_end_standard = np.array(EEG_val_end_standard)
    print('validation data prepared, validation data shape: ', noiseEEG_val_end_standard.shape, EEG_val_end_standard.shape)

    #################################  simulate noise signal of test  ##############################

    SNR_test_dB = np.linspace(-5.0, 5.0, num=(11))
    SNR_test = 10 ** (0.05 * (SNR_test_dB))

    eeg_test = np.array(test_eeg)
    noise1_test = np.array(test_noise1)
    noise2_test = np.array(test_noise2)
    
    # combin eeg and noise for test set 
    EEG_test = []
    noise_EEG_test = []
    for i in range(11):
        noise_eeg_test = []
        for j in range(eeg_test.shape[0]):
            eeg = eeg_test[j]
            noise1 = noise1_test[j]
            noise2 = noise2_test[j]
            
            coe = get_rms(eeg) / (get_rms(noise1 + noise2) * SNR_test[i])
            noise = (noise1 + noise2) * coe
            neeg = noise + eeg
            
            noise_eeg_test.append(neeg)
        
        EEG_test.extend(eeg_test)
        noise_EEG_test.extend(noise_eeg_test)
    noise_EEG_test = np.array(noise_EEG_test)
    EEG_test = np.array(EEG_test)


    # std for noisy EEG
    EEG_test_end_standard = []
    noiseEEG_test_end_standard = []
    std_VALUE = []
    for i in range(noise_EEG_test.shape[0]):
        
        # store std value to restore EEG signal
        std_value = np.std(noise_EEG_test[i])
        std_VALUE.append(std_value)

        # Each epochs of eeg and neeg was divide by the standard deviation
        eeg_test_all_std = EEG_test[i] / std_value
        EEG_test_end_standard.append(eeg_test_all_std)

        noiseeeg_test_end_standard = noise_EEG_test[i] / std_value
        noiseEEG_test_end_standard.append(noiseeeg_test_end_standard)

    std_VALUE = np.array(std_VALUE)
    noiseEEG_test_end_standard = np.array(noiseEEG_test_end_standard)
    EEG_test_end_standard = np.array(EEG_test_end_standard)
    print('test data prepared, test data shape: ', noiseEEG_test_end_standard.shape, EEG_test_end_standard.shape)

    return noiseEEG_train_end_standard, EEG_train_end_standard, noiseEEG_val_end_standard, EEG_val_end_standard, noiseEEG_test_end_standard, EEG_test_end_standard, std_VALUE, SNR_train

import numpy as np
import scipy.io

def load_mat_file(file_path):
    mat_data = scipy.io.loadmat(file_path)
    return {key: np.array(value) for key, value in mat_data.items() if not key.startswith('__')}

file_path = 'approaches/MWF/matlab.mat'
data = load_mat_file(file_path)

# Save each item in the data dictionary to a separate .npy file
for key, value in data.items():
    np.save("approaches/MWF/clean_EEG.npy", value)
    print(f"Saved {key} to {key}.npy")
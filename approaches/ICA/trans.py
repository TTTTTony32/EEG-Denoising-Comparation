import mne
import numpy as np

# import numpy as np
# import scipy.io

# def load_mat_file(file_path):
#     mat_data = scipy.io.loadmat(file_path)
#     return {key: np.array(value) for key, value in mat_data.items() if not key.startswith('__')}

# file_path = 'approaches/ICA/data.mat'
# data = load_mat_file(file_path)

# # Save each item in the data dictionary to a separate .npy file
# for key, value in data.items():
#     np.save("approaches/ICA/clean_EEG.npy", value)
#     print(f"Saved {key} to {key}.npy")
#     print(f"Shape: {value.shape}")

# Load the .set file
file_path = 'approaches/ICA/clean.set'
raw = mne.io.read_raw_eeglab(file_path, preload=True)

# Get the data as a numpy array
data = raw.get_data()

# Save the numpy array to a .npy file
np.save("approaches/ICA/clean_EEG.npy", data)
print(f"Saved data to clean_EEG.npy")
print(f"Shape: {data.shape}")
# Select a portion of the data to display
start, stop = 1, 150  # For example, select the first 5 channels
selected_data = data[0, start:stop]

# Print the selected data
print("Selected data:")
print(selected_data)
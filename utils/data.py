import torch
import numpy as np

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, window_size, device):
        device = torch.device(device)
        self.window_size = window_size
        # Load the data from the text file
        with open(file_path, "r") as f:
            data = [float(line.strip()) for line in f.readlines()]

        # Calculate the number of windows
        num_windows = len(data) // window_size
        self.num_windows = num_windows
        # Truncate the data to a multiple of the window size
        self.data = data[:num_windows * window_size]

        # Reshape the data into windows
        self.windows = torch.tensor(self.data).view(num_windows, 1, window_size).to(device)
    def get_input(self):
        return self.data
    
    def get_window(self, index):
        return np.array(self.data).reshape(self.num_windows, self.window_size)[index]
        

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]

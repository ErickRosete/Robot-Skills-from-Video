import glob
import logging
import os
import pickle
import hydra
import numpy as np
from torch.utils.data import Dataset
 
logger = logging.getLogger(__name__)

class BaseRelayKitchen(Dataset):
    """ Common Relay Kitchen Class, it will be extended by other datasets """
    def __init__(self, datasets_dir, max_window_size=32):
        self.datasets_dir = datasets_dir
        self.max_window_size = max_window_size
        #abs_datasets_dir = os.path.join(hydra.utils.get_original_cwd(), datasets_dir)
        abs_datasets_dir = os.path.join(os.getcwd(), datasets_dir)
        self.data_files = glob.glob(os.path.join(abs_datasets_dir, "*.pkl"))
        assert self.data_files
        self.file_index = []  # To map from dataloader idx -> file idx
        logger.info("loading dataset.....")
        self.data = self.load_data(self.data_files)
        logger.info("finished loading dataset")

    def getSequences(self, idx, window_size):
        file_idx = self.file_index[idx]
        x = self.data[file_idx]
        observations = x['observations']
        images = x['images']
        actions = x['actions']
        seq_obs, seq_imgs, seq_acts = [], [], []
        # Get the first dataloader index from the file with index file_idx
        first_dlidx_file = next(i for i, fidx in enumerate(self.file_index) if fidx == file_idx)
        seq_idx = idx - first_dlidx_file
 
        if 'val' in self.datasets_dir:
            # For validation data
            # return: [current_img, goal_img] , current_obs, current_action
            seq_obs = observations[seq_idx]
            seq_imgs = [images[seq_idx], images[seq_idx + window_size]]
            seq_acts = actions[seq_idx]
        else:
            seq_obs = observations[seq_idx: seq_idx + window_size]
            seq_imgs = images[seq_idx: seq_idx + window_size]
            seq_acts = actions[seq_idx: seq_idx + window_size]
        seq_imgs = np.transpose(seq_imgs, (0, 3, 1, 2))  # Change to S, C, H, W
        return seq_obs.astype(np.float32), seq_imgs.astype(np.float32), seq_acts.astype(np.float32)

    def load_data(self, file_names):
        paths = []
        for file_idx, data_file in enumerate(file_names):
            path = {'observations': [], 'images': [], 'actions': []}  # Only retrieve this keys
            if os.path.getsize(data_file) > 0:  # Check if the file is not empty
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    for key in path.keys():
                        if key == "observations":
                            path[key] = data[key][:, :9]
                        else:
                            path[key] = data[key]
 
            # Possible initial indices in the recorded path that will be able to create a window of experience
            # Starting in that point
            possible_indices = path["observations"].shape[0] - self.max_window_size
            for _ in range(possible_indices):
                self.file_index.append(file_idx)
            paths.append(path)
        return paths
 
    def __len__(self):
        return len(self.file_index)
 
class ConstantRelayKitchen(BaseRelayKitchen):
    """ This dataloader creates batches of a single window size """
    def __init__(self, datasets_dir, window_size):
        self.window_size = window_size
        super().__init__(datasets_dir, self.window_size)

    def __getitem__(self, idx):
        return self.getSequences(idx, self.window_size)

class PaddedRelayKitchen(BaseRelayKitchen):
    """ This dataloader pads the window size, to the max window size """
    def __init__(self, datasets_dir, ws_range=[16, 32]):
        self.ws_range = ws_range
        super().__init__(datasets_dir, ws_range[1])
 
    def __getitem__(self, idx):
        window_size = np.random.randint(*self.ws_range)
        seq_obs, seq_imgs, seq_acts = self.getSequences(idx, window_size)
        if 'train' in self.datasets_dir:
            pad_size = self.ws_range[1] - window_size
            seq_obs = np.pad(seq_obs, ((pad_size,0),(0,0)), 'constant', constant_values=0)
            seq_acts = np.pad(seq_acts, ((pad_size,0),(0,0)), 'constant', constant_values=0)
            seq_imgs = np.pad(seq_imgs, ((pad_size,0),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
        return seq_obs, seq_imgs, seq_acts

class VaryingRelayKitchen(BaseRelayKitchen):
    """ This dataloader creates batches of a different window size
        every batch. """
    def __init__(self, datasets_dir, ws_range=[16, 32]):
        self.ws_range = ws_range
        super().__init__(datasets_dir, ws_range[1])

    def __getitem__(self, idx):
        return self.getSequences(*idx)


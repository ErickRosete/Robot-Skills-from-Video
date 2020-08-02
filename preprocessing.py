import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import glob

#Load data from specific packages
def load_data(file_names):
    paths = []
    for data_file in file_names:
        path = {'observations': [], 'images': [], 'actions': []} #Only retrieve this keys
        if os.path.getsize(data_file) > 0:   #Check if the file is not empty   
            with open(data_file, 'rb') as f:
                data = pickle.load(f) 
                for key in path.keys():
                    if key == "observations":
                        path[key] = data[key][:, :9]
                    else:    
                        path[key] = data[key]
        paths.append(path)
    return paths

#Load data from ALL packages in a folder
def read_data(datasets_dir="./data/training"):
    name_str = os.path.basename(os.path.normpath(datasets_dir))
    print("Read %s data ..."%name_str)
    file_names = get_filenames(datasets_dir)
    return load_data(file_names)

def get_filenames(datasets_dir="./data/training"):
    file_names = glob.glob(os.path.join(datasets_dir, "*.pkl"))
    np.random.shuffle(file_names)
    return file_names

#For validation
def mult_wind_preprocessing(paths, min_ws, max_ws, batch_size=64):
    #Complete paths -> List of windows
    seq_obs, seq_imgs, seq_acts = [], [], []
    for path in paths:
        observations = path['observations']
        images = path['images']
        actions = path['actions']
        n = observations.shape[0]
        t = 0
        window_size = np.random.randint(min_ws, max_ws + 1)
        while t + window_size <= n - 1: 
            seq_obs.append(observations[t])
            seq_imgs.append(np.array([images[t], images[t+window_size]]))
            seq_acts.append(actions[t])
            window_size = np.random.randint(min_ws, max_ws + 1)
            t += 1
    
    #List -> numpy array
    seq_obs = np.stack(seq_obs, axis=0) # B, O
    seq_imgs = np.stack(seq_imgs, axis=0) # B, S, H, W, C
    seq_imgs = np.transpose(seq_imgs, (0, 1, 4, 2, 3)) # B, S, C, H, W
    seq_acts = np.stack(seq_acts, axis=0) # B, A

    #Shuffle inds
    inds = np.arange(seq_obs.shape[0])
    np.random.shuffle(inds)
    seq_obs = seq_obs[inds]
    seq_imgs = seq_imgs[inds]
    seq_acts = seq_acts[inds]

    #Split to batches
    num_splits = seq_obs.shape[0]//batch_size
    seq_obs = np.array_split(seq_obs, num_splits)
    seq_imgs = np.array_split(seq_imgs, num_splits)
    seq_acts = np.array_split(seq_acts, num_splits)
    return seq_obs, seq_imgs, seq_acts
    
def preprocess_data(paths, window_size=16, batch_size=64, validation=False, reset_seed=False):
    #Complete paths -> List of windows
    seq_obs, seq_imgs, seq_acts = [], [], []
    for path in paths:
        observations = path['observations']
        images = path['images']
        actions = path['actions']
        n = observations.shape[0]
        t = 0
        while t + window_size <= n - 1: 
            if(validation):
                #For validation data
                #return: [current_img, goal_img] , current_obs, current_action
                seq_obs.append(observations[t])
                seq_imgs.append(np.array([images[t], images[t+window_size]]))
                seq_acts.append(actions[t])
            else:
                seq_obs.append(observations[t: t+window_size])
                seq_imgs.append(images[t: t+window_size])
                seq_acts.append(actions[t: t+window_size])
            t += 1
    
    #List -> numpy array
    seq_obs = np.stack(seq_obs, axis=0) #Train = B, S, O | Val = B, O
    seq_imgs = np.stack(seq_imgs, axis=0) #B, S, H, W, C
    seq_imgs = np.transpose(seq_imgs, (0, 1, 4, 2, 3)) #B, S, C, H, W
    seq_acts = np.stack(seq_acts, axis=0) #Train = B, S, A | Val = B, A

    #Shuffle inds
    if(reset_seed):
         np.random.seed(0) #same numbers..
    inds = np.arange(seq_obs.shape[0])
    np.random.shuffle(inds)
    seq_obs = seq_obs[inds]
    seq_imgs = seq_imgs[inds]
    seq_acts = seq_acts[inds]

    #Split to batches
    num_splits = seq_obs.shape[0]//batch_size
    seq_obs = np.array_split(seq_obs, num_splits)
    seq_imgs = np.array_split(seq_imgs, num_splits)
    seq_acts = np.array_split(seq_acts, num_splits)
    return seq_obs, seq_imgs, seq_acts

def path_to_batches(path, window_size=16, batch_size=64, validation=False):
    #Path-> List of batches without changing order
    seq_obs, seq_imgs, seq_acts = [], [], []
    observations = path['observations']
    images = path['images']
    actions = path['actions']
    n = observations.shape[0]
    t = 0
    while t + window_size <= n - 1: 
        if(validation):
            seq_obs.append(observations[t])
            seq_imgs.append(np.array([images[t], images[t+window_size]]))
            seq_acts.append(actions[t])
        else:
            seq_obs.append(observations[t: t+window_size])
            seq_imgs.append(images[t: t+window_size])
            seq_acts.append(actions[t: t+window_size])
        t += 1
    
    #List -> numpy array
    seq_obs = np.stack(seq_obs, axis=0) #Train = B, S, O | Val = B, O
    seq_imgs = np.stack(seq_imgs, axis=0) #B, S, H, W, C
    seq_imgs = np.transpose(seq_imgs, (0, 1, 4, 2, 3)) #B, S, C, H, W
    seq_acts = np.stack(seq_acts, axis=0) #Train = B, S, A | Val = B, A

    #Split to batches
    num_splits = seq_obs.shape[0]//batch_size
    seq_obs = np.array_split(seq_obs, num_splits)
    seq_imgs = np.array_split(seq_imgs, num_splits)
    seq_acts = np.array_split(seq_acts, num_splits)
    return seq_obs, seq_imgs, seq_acts
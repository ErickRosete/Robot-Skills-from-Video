import numpy as np
import glob
import pickle 
import matplotlib.pyplot as plt
from network.play_lmp import PlayLMP
import torch
import cv2
import os 
import gym
import sys
sys.path.append("./relay-policy-learning/adept_envs/")
import adept_envs

def print_img_goals(data_dir="./data/validation/", save_folder = "./data/goals/", i=0, n_packages=1, load_all = False):
    data_files = glob.glob(data_dir + "*.pkl")
    if not load_all:
        data_files = data_files[i:i+n_packages]
    print("Printing images...")
    print(data_files)

    data_img = []
    for i,file in enumerate(data_files):
        #load images of file
        with open(file, 'rb') as f:
            if(i==0):
                data_img = pickle.load(f)['images']
            else:
                data_img = np.concatenate(pickle.load(f)['images'], axis=0)
        for i,img in enumerate(data_img):
            save_path = save_folder + os.path.basename(file)[:-4] +"_img_"+str(i)+".png"
            cv2.imwrite(save_path, img)

def test_model():

    #load goal
    goal_path = "./data/goals/friday_kettle_bottomknob_switch_slide_6_path_img_47.png"
    goal = cv2.imread(goal_path)

    #model init
    model = PlayLMP()
    model.load("./models/model_b4780.pth")
    
    #Env init
    env = gym.make('kitchen_relax-v1')
    s = env.reset()
    for i in range(5000):
        curr_img = env.render(mode='rgb_array')
        curr_img = cv2.resize(curr_img , (300,300))
        
        current_and_goal = np.stack((curr_img, goal) , axis=0) #(2, 300, 300, 3)
        current_and_goal = np.expand_dims(current_and_goal.transpose(0,3,1,2), axis=0) #(1, 2, 3, 300, 300)
        current_obs = np.expand_dims(s[:9], axis=0) #(1,9)
        #prediction
        action = model.predict(current_obs, current_and_goal).squeeze(0) #(9)
        s , r, _, _ = env.step(action.cpu().detach().numpy())
        env.render()

if __name__ == '__main__':
    test_model()
    #print_img_goals()
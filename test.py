import numpy as np
import glob
import pickle 
import matplotlib.pyplot as plt
from networks.play_lmp import PlayLMP
import torch
import cv2
import os 
import gym
import sys
sys.path.append("./relay-policy-learning/adept_envs/")
import adept_envs
import utils.constants as constants
import skvideo.io

#Load actions from .pkl file. Use absolute path including extension name
def load_actions_data(file_name):
    path = {'actions': [], 'init_qpos':[], 'init_qvel':[]} #Only retrieve this keys
    if os.path.getsize(data_file) > 0:   #Check if the file is not empty   
        with open(file_name, 'rb') as f:
            data = pickle.load(f) 
            for key in path.keys():
                if key == "observations":
                    path[key] = data[key][:, :9]
                else:    
                    path[key] = data[key]
    return path

#Print imgs from validation packages. Use i to select the nth package.
#n_packages to select how many packages to load (i.e, val_data[i:i+n_packages])
def print_img_goals(data_dir="./data/validation/", save_folder = "./data/goals/", i=0,\
                     n_packages=1, load_all = False):
    data_files = glob.glob(data_dir + "*.pkl")
    if not load_all:
        data_files = data_files[i:i+n_packages]
    print("Printing images...")
    print(data_files)

    data_img = []
    try:
        for i, file in enumerate(data_files):
            #load images of file
            with open(file, 'rb') as f:
                if(i==0):
                    data_img = pickle.load(f)['images']
                else:
                    data_img = np.concatenate(pickle.load(f)['images'], axis=0)
            for i,img in enumerate(data_img):
                    save_path = save_folder + os.path.basename(file)[:-4] +"_img_"+str(i)+".png"
                    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) #save as blue shelfs
    except Exception as e:
        print(e)      

    print("done!")

#init environment with pos and vel from given file
def init_env(env, file_name):
    if os.path.getsize(data_file) > 0:   #Check if the file is not empty   
        with open(file_name, 'rb') as f:
            data = pickle.load(f) 
            env.sim.data.qpos[:] = data['init_qpos'].copy()
            env.sim.data.qvel[:] = data['init_qpos'].copy()
            env.sim.forward()
    return env

#Proxy function to either save or render a model
def viewer(env, mode='initialize', filename='video'):
    global render_buffer
    if mode == 'initialize':
        render_buffer = []
        mode = 'render'

    if mode == 'render':
        curr_frame = env.render(mode='rgb_array')
        render_buffer.append(curr_frame)

    if mode == 'save':
        skvideo.io.vwrite(filename, np.asarray(render_buffer))
        print("\noffscreen buffer saved", filename)

#Reproduced saved actions from file in the env. Optionally save a video.
#Load file should include file path + filename + extension
#save_filename is the name for the video to be saved. It is stored under './data/videos/'
def reproduce_file_actions(load_file, save_filename = "video.mp4", show_video=True, save=False):
    data = load_actions_data(load_file)[0]#load first package
    #Env init
    gym_env = gym.make('kitchen_relax-v1')
    env = gym_env.env
    s = env.reset()

    # prepare env
    env.sim.data.qpos[:] = data['init_qpos'].copy()
    env.sim.data.qvel[:] = data['init_qpos'].copy()
    env.sim.forward()

    #Viewer
    FPS = 30
    render_skip = max(1, round(1. / \
        (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    viewer(env, mode='initialize')

    for action in data['actions']:
        s , r, _, _ = env.step(action)
        if(show_video):
            if i_frame % render_skip == 0:
                viewer(env, mode='render')
                print(i_frame, end=', ', flush=True)
    
    if(save):
        save_path = "./data/videos/" + save_filename
        viewer(env, mode='save', filename=save_filename)

def render_model(env, model, goal_path,  save_filename = "video.mp4", n_steps = 300 ):
    FPS = 30
    render_skip = max(1, round(1. / \
        (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    t0 = timer.time()

    viewer(env, mode='initialize')
    for i_frame in range(n_steps):
        if i_frame % render_skip == 0:
            viewer(env, mode='render')
            print(i_frame, end=', ', flush=True)
    
    viewer(env, mode='save', filename=filename)
    print("time taken = %f" % (timer.time() - t0))

def test_model(model, goal_path, show_goal=False, n_steps = 200):
    #load goal
    if(show_goal):
        goal = plt.imread(goal_path) #read as RGB, blue shelfs
        plt.axis('off')
        plt.suptitle("Goal")
        plt.imshow(goal)
        plt.show()

    #Env init
    gym_env = gym.make('kitchen_relax-v1')
    env = gym_env.env
    s = env.reset()

    for i in range(5000):
        curr_img = env.render(mode='rgb_array')
        curr_img = cv2.resize(curr_img , (300,300))

        #goal_path = "./data/goals/friday_microwave_kettle_topknob_hinge_0_path_img_%d.png" % (i+16)
        #goal = plt.imread(goal_path) #read as RGB, blue shelfs
        current_and_goal = np.stack((curr_img, goal) , axis=0) #(2, 300, 300, 3)
        current_and_goal = np.expand_dims(current_and_goal.transpose(0,3,1,2), axis=0) #(1, 2, 3, 300, 300)
        current_obs = np.expand_dims(s[:9], axis=0) #(1,9)
        #prediction
        #if(i % 30 == 0):
        #    plan = model.get_pp_plan(current_obs,current_and_goal)
        #action = model.predict_with_plan(current_obs, current_and_goal, plan).squeeze(0) #(9)
        action = model.predict(current_obs, current_and_goal).squeeze(0) #(9)
        s , r, _, _ = env.step(action.cpu().detach().numpy())
        env.render()

if __name__ == '__main__':
    #model init
    #model = PlayLMP(constants.LEARNING_RATE, constants.BETA, \
    #                  num_mixtures=1, use_logistics=False)
    #model.load("./models/model_b62880.pth")
    #test_model(goal_path = "./data/goals/friday_microwave_kettle_topknob_hinge_0_path_img_50.png")

    #print_img_goals(data_dir = "./data/validation/")
    reproduce_file_actions("./data/validation/friday_microwave_kettle_topknob_hinge_8_path.pkl", show_video=True, save=True)
    #reproduce_file_actions("./data/validation/friday_microwave_kettle_topknob_hinge_8_path.pkl")
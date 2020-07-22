#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:01:08 2020

@author: suresh, eric, jessica
"""

import utils
# add path to corresponding packages
utils.add_path('/media/suresh/research/github/robotics/rl_robotics/packages/relay_policy_learning/adept_envs')
utils.add_path('/media/suresh/research/github/robotics/rl_robotics/packages/puppet/vive/source')
utils.add_path('/media/suresh/research/github/robotics/rl_robotics/packages/mjrl')

import os

import pickle
import numpy as np
from parse_mjl import parse_mjl_logs, viz_parsed_mjl_logs
from mjrl.utils.gym_env import GymEnv
import adept_envs
import gym
import cv2

# playback demos and get data(physics respected)
def gather_training_data(env, data, pkl_seq, save_path, width = 300, height = 300, render=None):

    env = env.env

    # initialize
    env.reset()
    init_qpos = data['qpos'][0].copy()
    init_qvel = data['qvel'][0].copy()
    act_mid = env.act_mid
    act_rng = env.act_amp

    # prepare env
    env.sim.data.qpos[:] = init_qpos
    env.sim.data.qvel[:] = init_qvel
    env.sim.forward()

    # step the env and gather data
    img_seq = 0
    path_obs = None
    path_img = None
    path_act = None
    for i_frame in range(data['ctrl'].shape[0] - 1):
        if render == 'OFF':
            curr_frame = env.render(mode='rgb_array')

            # resize image and save
            img_seq += 1
            img_name = '/images/' + str(pkl_seq).zfill(5) + '_' + str(img_seq).zfill(5) + '.png'
            curr_frame = cv2.resize(curr_frame, (width, height))
            cv2.imwrite(save_path+img_name, curr_frame)

            if path_img is None:
                path_img = np.array([img_name])
            else:
                path_img = np.vstack((path_img, np.array([img_name])))
        else: # 'ON'
            env.mj_render()

        obs = env._get_obs()

        # Construct the action
        ctrl = (data['ctrl'][i_frame] - obs[:9])/(env.skip*env.model.opt.timestep)
        act = (ctrl - act_mid) / act_rng
        act = np.clip(act, -0.999, 0.999)
        next_obs, reward, done, env_info = env.step(act)
        if path_obs is None:
            path_obs = obs
            path_act = act
        else:
            path_obs = np.vstack((path_obs, obs))
            path_act = np.vstack((path_act, act))

    # note that <init_qpos, init_qvel> are one step away from <path_obs[0], path_act[0]>
    return path_obs, path_img, path_act, init_qpos, init_qvel

def main(env, demo_dir, skip, view, save_path, width, height, render):
    pkl_dir = save_path+'/pkl/'
    images_dir = save_path+'/images/'
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
        os.makedirs(images_dir)

    pkl_seq = 0
    gym_env = gym.make(env)
    for root, dirs, files in os.walk(demo_dir, topdown=False):

        for f_name in files:
            pkl_seq += 1
            file = os.path.join(root, f_name)
            # process logs
            print("processing => " + f_name, end=': ')
            try:
                data = parse_mjl_logs(file, skip)
            except :
                print(' experienced an error')
                print('-------------')
                continue


            # playback logs and gather data
            if view == 'playback':
                try:
                    obs, img, act, init_qpos, init_qvel = gather_training_data(gym_env, data, pkl_seq, save_path, width, height, render)
                except Exception as e:
                    print(e)
                    continue
                path = {
                    'observations': obs,
                    'images': img,
                    'actions': act,
                    'goals': obs,
                    'init_qpos': init_qpos,
                    'init_qvel': init_qvel
                }

            # save traiing data
            pkl_name = env + '_' + str(pkl_seq).zfill(5) + '.pkl'
            pickle.dump(path, open(pkl_dir + pkl_name, 'wb'))
            print('done')
        # if pkl_seq == 1: # remove this if-block
        #     break
    gym_env.close()

if __name__ == '__main__':
    demo_dir = '/media/suresh/research/github/robotics/rl_robotics/packages/relay_policy_learning/kitchen_demos_multitask/friday_kettle_bottomknob_switch_slide/'
    save_path = '../dataset/train/friday_kettle_bottomknob_switch_slide'

    skip = 40
    view = 'playback'
    render = 'OFF' # or 'ON'
    width = 299
    height = 299

    env = 'kitchen_relax-v1'
    main(env, demo_dir, skip, view, save_path, width, height, render)

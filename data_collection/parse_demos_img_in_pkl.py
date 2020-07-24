#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append("./mjrl")
sys.path.append("./relay-policy-learning/adept_envs/")
sys.path.append("./puppet/vive/source/")

import click
import glob
import pickle
import numpy as np
from parse_mjl import parse_mjl_logs, viz_parsed_mjl_logs
from mjrl.utils.gym_env import GymEnv
import adept_envs
import time as timer
import skvideo.io
import gym
import cv2
from pathlib import Path

# headless renderer
render_buffer = []  # rendering buffer


def viewer(env,
           mode='initialize',
           filename='video',
           frame_size=(640, 480),
           camera_id=0,
           render=None):
    if render == 'onscreen':
        env.mj_render()

    elif render == 'offscreen':

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

    elif render == 'None':
        pass

    else:
        print("unknown render: ", render)


# view demos (physics ignored)
def render_demos(env, data, filename='demo_rendering.mp4', render=None):
    FPS = 30
    render_skip = max(1, round(1. / \
        (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    t0 = timer.time()

    viewer(env, mode='initialize', render=render)
    for i_frame in range(data['ctrl'].shape[0]):
        env.sim.data.qpos[:] = data['qpos'][i_frame].copy()
        env.sim.data.qvel[:] = data['qvel'][i_frame].copy()
        env.sim.forward()
        if i_frame % render_skip == 0:
            viewer(env, mode='render', render=render)
            print(i_frame, end=', ', flush=True)

    viewer(env, mode='save', filename=filename, render=render)
    print("time taken = %f" % (timer.time() - t0))


# playback demos and get data(physics respected)
def gather_training_data(env, data, filename='demo_playback.mp4', render=None):
    env = env.env
    FPS = 30
    render_skip = max(1, round(1. / \
        (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    t0 = timer.time()

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
    viewer(env, mode='initialize', render=render)

    # step the env and gather data
    path_obs = None
    for i_frame in range(data['ctrl'].shape[0] - 1):
        obs = env._get_obs()
        img = env.render(mode='rgb_array') # 1920, 2560, 3
        img = cv2.resize(img, (300, 300))
        if i_frame % 20 == 0:
            cv2.imwrite("./images/img_" + str(i_frame) + ".png", img) 
        img = np.expand_dims(img, 0)
        ctrl = (data['ctrl'][i_frame] - obs[:9])/(env.skip*env.model.opt.timestep)
        act = (ctrl - act_mid) / act_rng
        act = np.clip(act, -0.999, 0.999)
        next_obs, reward, done, env_info = env.step(act)
        if path_obs is None:
            path_obs = obs
            path_act = act
            path_img = img
        else:
            path_obs = np.vstack((path_obs, obs))
            path_act = np.vstack((path_act, act))
            path_img = np.vstack((path_img, img))

        # render when needed to maintain FPS
        if i_frame % render_skip == 0:
            viewer(env, mode='render', render=render)
            print(i_frame, end=', ', flush=True)

    # finalize
    if render:
        viewer(env, mode='save', filename=filename, render=render)

    t1 = timer.time()
    print("time taken = %f" % (t1 - t0))

    # note that <init_qpos, init_qvel> are one step away from <path_obs[0], path_act[0]>
    return path_obs, path_img, path_act, init_qpos, init_qvel


# MAIN =========================================================
@click.command(help="parse tele-op demos")
@click.option('--env', '-e', type=str, help='gym env name', required=True)
@click.option(
    '--demo_dir',
    '-d',
    type=str,
    help='directory with tele-op logs',
    required=True)
@click.option(
    '--skip',
    '-s',
    type=int,
    help='number of frames to skip (1:no skip)',
    default=1)
@click.option('--graph', '-g', type=bool, help='plot logs', default=False)
@click.option('--save_logs', '-l', type=bool, help='save logs', default=False)
@click.option(
    '--view', '-v', type=str, help='render/playback', default='render')
@click.option(
    '--render', '-r', type=str, help='onscreen/offscreen', default='onscreen')
def main(env, demo_dir, skip, graph, save_logs, view, render):

    gym_env = gym.make(env)
    #paths = []
    
    #demo_dirs = ["friday_kettle_bottomknob_hinge_slide", "friday_kettle_bottomknob_switch_slide", 
    #    "friday_kettle_switch_hinge_slide", "friday_kettle_topknob_bottomknob_slide", 
    #    "friday_kettle_topknob_switch_slide", "friday_microwave_bottomknob_hinge_slide",
    #    "friday_microwave_bottomknob_switch_slide", "friday_microwave_kettle_bottomknob_hinge", 
    #    "friday_microwave_kettle_bottomknob_slide","friday_microwave_kettle_hinge_slide",
    #    "friday_microwave_kettle_switch_slide", "friday_microwave_kettle_topknob_hinge"] 
        
    demo_dirs = [ "friday_microwave_kettle_topknob_switch", "friday_microwave_topknob_bottomknob_hinge", 
        "friday_microwave_topknob_bottomknob_slide", "friday_topknob_bottomknob_switch_slide", 
        "postcorl_kettle_bottomknob_switch_hinge", "postcorl_kettle_topknob_bottomknob_hinge", 
        "postcorl_kettle_topknob_bottomknob_switch", "postcorl_microwave_bottomknob_switch_slide", 
        "postcorl_microwave_kettle_switch_hinge", "postcorl_microwave_switch_hinge_slide", 
        "postcorl_microwave_topknob_bottomknob_switch", "postcorl_microwave_topknob_switch_hinge"]
    #create dirs 
    video_dir =  "./videos/"     
    data_dir =  "./data/"          
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for demo_dir in demo_dirs:
        print("Scanning demo_dir: " + demo_dir + "=========")
        comp_demo_dir = "./kitchen_demos_multitask/" + demo_dir + "/"
        for ind, file in enumerate(glob.glob(comp_demo_dir + "*.mjl")):

            # process logs
            print("processing: " + file, end=': ')

            data = parse_mjl_logs(file, skip)

            print("log duration %0.2f" % (data['time'][-1] - data['time'][0]))

            # plot logs
            if (graph):
                print("plotting: " + file)
                viz_parsed_mjl_logs(data)

            # save logs
            if (save_logs):
                pickle.dump(data, open(file[:-4] + ".pkl", 'wb'))

            # render logs to video
            if view == 'render':
                render_demos(
                    gym_env,
                    data,
                    filename=data['logName'][:-4] + '_demo_render.mp4',
                    render=render)

            # playback logs and gather data
            elif view == 'playback':
                try:
                    obs, imgs, act, init_qpos, init_qvel = gather_training_data(gym_env, data,\
                    filename=video_dir + demo_dir + "_" + str(ind) + '.mp4', render=render)
                    #data['logName'][:-4]+'_playback.mp4', render=render)
                except Exception as e:
                    print(e)
                    continue
                path = {
                    'observations': obs,
                    'images': imgs,
                    'actions': act,
                    'goals': obs,
                    'init_qpos': init_qpos,
                    'init_qvel': init_qvel
                }
                #paths.append(path)
                pickle.dump(path, open(data_dir + demo_dir + "_" + str(ind) + "_path.pkl", 'wb'))
                #print(demo_dir + env + file + "_path.pkl")
if __name__ == '__main__':
    main()
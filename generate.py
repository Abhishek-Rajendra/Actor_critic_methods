'''
Generate video for Max policy and sampling policy for all the 
models given in a variant folder path

run pyhton gerenate.py -h for futher instruction to use

'''

import numpy as np 
import gym
import sys
import os
import argparse
from reignforce import Agent

def Generate_episode_max(env,agent,path):
    path = path + "/max/"
    if not os.path.exists(path):
        os.makedirs(path)

    env = gym.wrappers.Monitor(env,path,video_callable=lambda episode_id: episode_id==0,force = True)
    done = False
    observation = env.reset()
    while not done:
            #Get the probability of performing actions
            action_prob = agent.policy.predict(observation[np.newaxis, :])
            #Get the location(action number) by finding the max position
            action = np.argmax(action_prob)
            observation_,reward, done, info = env.step(action)
            observation = observation_

    print("done")


def Generate_episode_sampling(env,agent,path):
    path = path + "/sampling/"
    if not os.path.exists(path):
        os.makedirs(path)
    env = gym.wrappers.Monitor(env,path,video_callable=lambda episode_id: episode_id==0,force = True)
    done = False
    observation = env.reset()
    while not done:
            action = agent.choose_action(observation)
            observation_,reward, done, info = env.step(action)
            observation = observation_
    print("done")
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("path",help = "Folder Path to parents folder of weights of network, it should contain final.h5 and optimal.h5")
    parser.add_argument("environment",help="CartPole or Acrobot or LunarLander or MountainCar")
    args = parser.parse_args()

    envs = {'CartPole': 'CartPole-v0', 
            'Acrobot': 'Acrobot-v1',
            'LunarLander': 'LunarLander-v2',
            'MountainCar':'MountainCar-v0'}
    
    environment = args.environment

    if environment in envs:
        opengym_env = envs.get(environment, None)
    else:
        print("Please provide the right environment")
        exit()
    
    env = gym.make(opengym_env)

    n_actions = env.action_space.n
    n_states = len(env.observation_space.low)
    
    givePath = args.path
    if givePath[-1] != "/":
        givePath+="/"

    lists = ["final","optimal"]
    
    for filename in os.listdir(givePath):
        for types in lists:
            #weights path
            weightPath = givePath +filename + "/" + types +".h5"
            #Video path
            path = givePath + filename + "/" + types + "/video"
            print(weightPath, path)

            if not os.path.exists(path):
                os.makedirs(path)
            #Getting the paramethers for the Policy Network
            temp = filename.rsplit("_")
            print(temp)
            label = temp[0]
            gamma = 0.99
            alpha = float(temp[1])
            layers = ""
            h1_layer = 0
            h2_layer = 0
            if(temp[2]=='1'):
                h1_layer = int(temp[3])
            elif(temp[2]=='2'):
                h1_layer = int(temp[3])
                h2_layer = int(temp[4])
            
            print(h1_layer,h2_layer)
            agent = Agent(ALPHA = alpha, input_dims=n_states, 
             Gamma= gamma, n_actions = n_actions,layer1_size=h1_layer, layer2_size=h2_layer)
            #Load weights
            try:
                agent.policy.load_weights(weightPath)
            except IOError:
                continue

            print(path)
            print("--------Max Policy-------")
            Generate_episode_max(env,agent,path)
            print("--------Sampling Policy-------")
            Generate_episode_sampling(env,agent,path)
import numpy as np 
import sys
from reignforce import Agent
import os
import gym
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("path",help = "Folder Path to weights")
    args = parser.parse_args()

    envs = {'CartPole': 'CartPole-v0', 
            'Acrobot': 'Acrobot-v1',
            'LunarLander': 'LunarLander-v2',
            'MountainCar':'MountainCar-v0'}


    path = args.path
    
    directory = os.path.dirname(path)
    #Model data
    modeldata = os.path.split(directory)[1]
    parentdir = os.path.dirname(directory)
    #Variant
    variant = os.path.split(parentdir)[1]
    #Environment
    environment = os.path.dirname(parentdir)

    print(environment)
    if environment in envs:
        opengym_env = envs.get(environment, None)
    else:
        print("Please provide the right environment")
        exit()

    env = gym.make(opengym_env)
    n_actions = env.action_space.n
    n_states = len(env.observation_space.low)


    temp = modeldata.rsplit("_")
    print(temp)
    h1_layer = 0
    h2_layer = 0
    if(variant == "Reinforce"):
        if(temp[3]=='1'):
            h1_layer = temp[4]
        elif(temp[3]=='2'):
            h1_layer = int(temp[4])
            h2_layer = int(temp[5])
    else:
        if(temp[2]=='1'):
            h1_layer = temp[3]
        elif(temp[2]=='2'):
            h1_layer = int(temp[3])
            h2_layer = int(temp[4])



    agent = Agent(input_dims=n_states, n_actions = n_actions,layer1_size=h1_layer, layer2_size=h1_layer)

    # Load_model = keras.models.load_model(sys.argv[1])
    agent.policy.load_weights(path)

    print(agent.policy.get_weights())
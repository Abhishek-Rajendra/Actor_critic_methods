import numpy as np 
import gym
import argparse
from reignforce import Agent
import sys
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot(episode_rewards, policy, label, alpha, gamma, plot_path):

    plt.figure()
    plt.suptitle(policy)
    plt.title(environment+r", $\alpha $ = "+str(alpha)+", $\gamma$ = "+str(gamma))
    plt.plot(range(len(episode_rewards)),episode_rewards, '.-',label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Total reward')
    plt.legend()
    plt.savefig(plot_path+"/" + policy +"Reward.png")

    plt.figure()
    plt.suptitle(policy)
    z1=pd.Series(episode_rewards).rolling(50).mean()
    plt.title(environment+r", $\alpha $ = "+str(alpha)+", $\gamma$ = "+str(gamma)+ ",  Best average reward: "+ str(np.max(z1)))
    plt.plot(z1,label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Rewards over past 50 episodes')
    plt.legend()

    plt.savefig(plot_path+"/" + policy +"cumulative.png")
    # plt.show()

def policy_sampling(env,agent,label, alpha, gamma, plot_path,ep=1000):

    score_history = []
    n_episodes = ep

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_,reward, done, info = env.step(action)
            observation = observation_
            score += reward
        score_history.append(score)
        print('episode ', i,'score %.1f' % score,
        'average_score %.1f' % np.mean(score_history[-50:]))
    
    plot(score_history, "Sampling_Policy",label, alpha, gamma, plot_path)

    return [np.mean(score_history), np.std(score_history)]

def policy_max(env,agent,label, alpha, gamma, plot_path,ep=1000):

    score_history = []
    n_episodes = ep

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            #Get the probability of performing actions
            action_prob = agent.policy.predict(observation[np.newaxis, :])
            #Get the location(action number) by finding the max position
            action = np.argmax(action_prob)
            observation_,reward, done, info = env.step(action)
            observation = observation_
            score += reward
        score_history.append(score)
        print('episode ', i,'score %.1f' % score,
        'average_score %.1f' % np.mean(score_history[-50:]))

    plot(score_history,"Max_Policy",label, alpha, gamma, plot_path)

    return [np.mean(score_history), np.std(score_history)]

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
            temp = filename.rsplit("_")
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
            
            for t in temp[3:]:
                layers = layers + t

            #Weights path
            weight_path = givePath +filename + "/" + types +".h5"
            #plots path
            plot_path = givePath +filename + "/" + types + "/plot"
            #mean std path
            meanStd_path = givePath +filename + "/" + types + "/meanstd"

            print(weight_path,plot_path,meanStd_path,h1_layer,h2_layer,layers,label,gamma,alpha)

            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

        
            agent = Agent(ALPHA = alpha, input_dims=n_states, 
            Gamma= gamma, n_actions = n_actions,layer1_size=h1_layer, layer2_size=h2_layer)

            agent.policy.load_weights(weight_path)

            print(agent.policy.get_weights())

            episodes = 200
            
            mean_std = []
            print("----Max Policy----")
            mean_std.append(policy_max(env, agent,label, alpha, gamma, plot_path,ep=episodes))
            print("----Sampling Policy----")
            mean_std.append(policy_sampling(env, agent,label, alpha, gamma, plot_path,ep=episodes))

            print(mean_std)
            
            np.save(meanStd_path,mean_std)
    

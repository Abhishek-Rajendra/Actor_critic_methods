'''
This code implements 3 types for Reignforce algorithm total, without and with.
Total - Will take total reward as the advantages function to guide the policy 
network.
Without - Will use total future reward from that state to terminal state as 
advantage function
With - It is same as without but uses a baseline that is the average Total 
reward of the episode till now

Parameter to tweek: Environment, varient, n_episodes, alpha, gamma, thershold for 
MountainCar environment, Save variable to save the plots, weights and rewards(score) 
and layers(max u can use 2 layers) - "1_8" for single layer, "2_16_16" for two layers

You can even Load a model, but make sure the layer parameter is set right
'''
import argparse
import gym
import matplotlib.pyplot as plt 
import numpy as np 
from reignforce import Agent
import pandas as pd
import os



###Caluculation of mean and standard deviation and plot
def plot(episode_rewards, policy, label, alpha, gamma, plot_path):

    plt.figure()
    plt.suptitle(policy)
    plt.title(environment+r"$\alpha $ = "+str(alpha)+r", $\gamma$ = "+str(gamma))
    plt.plot(range(len(episode_rewards)),episode_rewards, '.-',label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig(plot_path+"/" + policy +"Reward.png")

    plt.figure()
    plt.suptitle(policy)
    z1=pd.Series(episode_rewards).rolling(50).mean()
    plt.title(environment+r"$\alpha $ = "+str(alpha)+r", $\gamma$ = "+str(gamma)+ ", Best average reward: "+ str(np.max(z1)))
    plt.plot(z1,label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Rewards over previous 50 episodes')
    plt.legend()

    plt.savefig(plot_path+"/" + policy +"cumulative.png")
    # plt.show()

def policy_sampling(env,actor,label, alpha, gamma, plot_path,ep=1000):

    score_history = []
    n_episodes = ep

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = actor.choose_action(observation)
            observation_,reward, done, info = env.step(action)
            observation = observation_
            score += reward
        score_history.append(score)
        print('episode ', i,'score %.1f' % score,
        'average_score %.1f' % np.mean(score_history[-50:]))
    
    plot(score_history, "Sampling_Policy",label, alpha, gamma, plot_path)

    return [np.mean(score_history), np.std(score_history)]

def policy_max(env,actor,label, alpha, gamma, plot_path,ep=1000):

    score_history = []
    n_episodes = ep

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            #Get the probability of performing actions
            action_prob = actor.policy.predict(observation[np.newaxis, :])
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
#######################################################################
#Generating Video of an episode
def Generate_episode_sampling(env,actor,path):
    path = path + "/sampling/"
    if not os.path.exists(path):
        os.makedirs(path)
    env = gym.wrappers.Monitor(env,path,video_callable=lambda episode_id: episode_id==0,force = True)
    done = False
    observation = env.reset()
    while not done:
            action = actor.choose_action(observation)
            observation_,reward, done, info = env.step(action)
            observation = observation_

def Generate_episode_max(env,actor,path):
    path = path + "/max/"
    if not os.path.exists(path):
        os.makedirs(path)

    env = gym.wrappers.Monitor(env,path,video_callable=lambda episode_id: episode_id==0,force = True)
    done = False
    observation = env.reset()
    while not done:
            #Get the probability of performing actions
            action_prob = actor.policy.predict(observation[np.newaxis, :])
            #Get the location(action number) by finding the max position
            action = np.argmax(action_prob)
            observation_,reward, done, info = env.step(action)
            observation = observation_
##########################################
#Plot training
##########################################
def plotLearning(title,alpha,gamma,loss, episode_rewards, path, label, save):

    plt.figure()
    plt.suptitle(label+" - "+environment)
    plt.title(r"$\alpha $ = "+alpha+r", $\gamma$ = "+gamma)
    plt.plot(range(len(episode_rewards)),episode_rewards, '.-',label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Total Rewards')
    plt.legend()
    if(save):
        plt.savefig(path+"/Reward.png")

    plt.figure()
    plt.suptitle(label+" - "+environment)
    plt.title(r"$\alpha$ = "+alpha+r", $\gamma$ = "+gamma)
    plt.plot(range(len(loss)),loss, '.-',label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('loss in each episode')
    plt.legend()
    if(save):
        plt.savefig(path+"/loss.png")

    plt.figure()
    plt.suptitle(variant+" - "+environment)
    z1=pd.Series(episode_rewards).rolling(50).mean()
    plt.title(r"$\alpha$ = "+alpha +r", $\gamma$ = "+gamma + ",  Best average reward: "+ str(np.max(z1)))
    plt.plot(z1,label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Rewards over past 50 episodes')
    plt.legend()
    if(save):
        plt.savefig(path+"/cumulative.png")
    plt.show()

###########################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--load",default="",type=str,help = "Give a .h5 model of weights for policy network")
    args = parser.parse_args()

    #Whether to save the output
    Save = True
    types = ["total","without","with"]
    #Environment details
    envs = {'CartPole': 'CartPole-v0', 
            'Acrobot': 'Acrobot-v1',
            'LunarLander': 'LunarLander-v2',
            'MountainCar':'MountainCar-v0'}

    environment = "CartPole"
    folder = "Reinforce1"
    variant = "with"

    if variant not in types:
        print("This variant doesn't exist")
        exit()
    if environment in envs:
        opengym_env = envs.get(environment, None)
    else:
        print("Please provide the right environment")
        exit()

    #Environment Details
    env = gym.make(opengym_env)

    if(environment == "MountainCar"):
        env = gym.make(opengym_env).env
    
    n_actions = env.action_space.n
    n_states = len(env.observation_space.low)   
    
    ##############################
    #Policy Netwrok
    ##############################
    alpha = 0.001
    layers = "0"
    gamma = 0.99

    h1_actor = 0
    h2_actor = 0
    if(layers!='0'):
        temp = layers.rsplit("_")
        if(temp[0]=='1'):
            h1_actor = int(temp[1])
        elif(temp[0]=='2'):
            h1_actor = int(temp[1])
            h2_actor = int(temp[2])
        else:
            print("Wrong actor network config, or hidden layers greater than 2")

    actor = Agent(ALPHA = alpha, input_dims=n_states, 
    Gamma= gamma, n_actions = n_actions,layer1_size=h1_actor, layer2_size=h2_actor)

    if(args.load != ""):
        print("---Loading weights---")
        actor.load_weights(args.load)

    #Number of trajectories to perform
    n_episodes = 10
    save_point = n_episodes//2
    ###########################
    #Used for mountain car
    ###########################
    threshold = 500
    max = -threshold
    optimal_episode = 0
#############################################################################

    path = environment + "/" + folder + "/"+variant + "_" + str(gamma) + "_"+ str(alpha)+"_"+layers

    if(Save):
        if not os.path.exists(path):
            os.makedirs(path)
    
    if(Save):
        actor.save_model(path + "/initial.h5")

    #Store Total reward history and loss
    score_history = []
    loss = []

    #Loop for the trajectories
    for i in range(n_episodes):
        done = False
        score = 0
        #Set initial state 
        observation = env.reset()
        count = 0
        while not done:
            count +=1
            #Choose action
            action = actor.choose_action(observation)
            #Perform action
            observation_,reward, done, info = env.step(action)
            #Store all the parameters for later use
            actor.store_transition(observation, action, reward)
            observation = observation_
            score += reward
            if(environment == "MountainCar"):
                if(count==threshold):
                    break

        score_history.append(score)
        #save at mid point
        if(i==save_point):
            if(Save):
                actor.save_model(path + "/middle.h5")
        #Save Optimal score model
        if(max<=score):
            max=score
            optimal_episode = i
            if(Save):
                actor.save_model(path +"/optimal.h5")
                np.save(path+"/score",score_history)
                np.save(path+"/score",score_history)
                
        #Learning the policy
        loss.append(actor.learn(variant))

        print('episode ', i,'loss %.1f' % loss[i],'score %.1f' % score,
        'average_score %.1f' % np.mean(score_history[-100:]))

    if(Save):
        actor.save_model(path + "/final.h5")
        np.save(path+"/score",score_history)

    plotLearning(environment,str(alpha), str(gamma), loss,score_history, path = path, label=variant, save =Save)
    print(max,optimal_episode)

    if Save:
        #Loop through final and optimal
        lists = ["final","optimal"]
        for types in lists:

            #weights path
            weightPath = path + "/" + types +".h5"

            #Generate Video
            video_path = path +"/" + types + "/video"
            
            if not os.path.exists(video_path):
                os.makedirs(video_path)

            #Load Weights
            actor.policy.load_weights(weightPath)

            print("--------Max Policy-------")
            Generate_episode_max(env,actor,video_path)
            print("--------Sampling Policy-------")
            Generate_episode_sampling(env,actor,video_path)

            #Get statistics

            meanStd_path = path+"/" + types + "/meanstd"
            #plot path
            plot_path = path + "/" + types + "/plot"

            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

            episodes = 200 
            mean_std = []
            print("----Max Policy----")
            mean_std.append(policy_max(env, actor,variant, alpha, gamma, plot_path, ep=episodes))
            print("----Sampling Policy----")
            mean_std.append(policy_sampling(env, actor,variant, alpha, gamma, plot_path, ep=episodes))

            print(mean_std)
            
            np.save(meanStd_path,mean_std)
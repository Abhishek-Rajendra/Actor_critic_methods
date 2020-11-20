'''
Here we are implementing Konda-Actor Critic Algorithm, but here we are not
TD lambda rather I am implementing TD(1) by using future reward from each state
as target to update the critic.
Also note that in konda we are using phi from actor as input to critic network

Parameter to tweek: Environment, n_episodes, alpha1, alpha2, gamma, thershold for 
MountainCar environment, Save variable to save the plots, weights, rewards(score)
and video, and actor_info and critic_info which are layer info
(at most you can use 2 layers) - "1_8" for single layer, "2_16_16" for two layers

'''
import argparse
import gym
import matplotlib.pyplot as plt 
import numpy as np 
from konda_TD_1 import Actor, Critic
import pandas as pd
import os
from tqdm import tqdm
import time

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
##################################################################
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

##################################################################
#Plot trinaing

def plotLearning(title,alpha,gamma,loss, episode_rewards, path, label, save):

    plt.figure()
    plt.suptitle(label+" - "+title)
    plt.title(r"$\alpha $ = "+alpha+r", $\gamma$ = "+gamma)
    plt.plot(range(len(episode_rewards)),episode_rewards, '.-',label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Total Rewards')
    plt.legend()
    if(save):
        plt.savefig(path+"/Reward.png")

    # plt.figure()
    # plt.suptitle(label+" - "+title)
    # plt.title(r"$\alpha$ = "+alpha+r", $\gamma$ = "+gamma)
    # plt.plot(range(len(loss)),loss, '.-',label=label)
    # plt.xlabel('Number of Episodes')
    # plt.ylabel('loss in each episode')
    # plt.legend()
    # if(save):
    #     plt.savefig(path+"/loss.png")

    plt.figure()
    plt.suptitle(variant+" - "+title)
    z1=pd.Series(episode_rewards).rolling(50).mean()
    plt.title(r"$\alpha$ = "+alpha +r", $\gamma$ = "+gamma + ",  Best average reward: "+ str(np.max(z1)))
    plt.plot(z1,label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Rewards over past 50 episodes')
    plt.legend()
    if(save):
        plt.savefig(path+"/cumulative.png")
    plt.show()
##################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_actor",default="",type=str,help = "Give a .h5 model of weights for Actor network")
    parser.add_argument("--load_critic",default="",type=str,help = "Give a .h5 model of weights for Critic network")
    args = parser.parse_args()

    #Whether to save the output
    Save = False

    #Environment details
    envs = {'CartPole': 'CartPole-v0', 
            'Acrobot': 'Acrobot-v1',
            'LunarLander': 'LunarLander-v2',
            'MountainCar':'MountainCar-v0'}

    environment = "LunarLander"
    #Variant details
    variant = "kondaTD(1)"

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
    #Actor Network
    alpha1 = 0.0002
    actor_info = "1_64"

    h1_actor = 0
    h2_actor = 0
    #Input to critic
    critic_input = n_states
    if(actor_info!='0'):
        temp = actor_info.rsplit("_")
        if(temp[0]=='1'):
            h1_actor = int(temp[1])
            critic_input = h1_actor
        elif(temp[0]=='2'):
            h1_actor = int(temp[1])
            h2_actor = int(temp[2])
            critic_input = h2_actor
        else:
            print("Wrong actor network config, or hidden layers greater than 2")

    #Critic Network
    alpha2 = 0.005
    gamma = 0.99
    
    critic_info="0" #Dont Change, let it be zero

    actor = Actor(ALPHA = alpha1, input_dims=n_states,
    n_actions = n_actions,layer1_size=h1_actor, layer2_size=h2_actor)

    #Input for the critic linear approximation is the last hidden layer o fthe actor network
    critic = Critic(ALPHA = alpha2, input_dims=critic_input*n_actions, 
    Gamma= gamma, n_actions = n_actions)

    if(args.load_actor != ""):
        print("---Loading actor weights---")
        actor.load_weights(args.load_actor)
    
    if(args.load_critic != ""):
        print("---Loading critic weights---")
        critic.load_weights(args.load_critic)

    #Number of trajectories to perform
    n_episodes = 10
    save_point = n_episodes//2

    #Used for mountain car
    threshold = 500
    max = -threshold
##################################################################

    optimal_episode = 0
    Qpi =0
    #Create a path to save the output
    path = environment +"/" + variant + "/" + variant + "_" \
    + str(alpha1)+"_" +actor_info +"_" +str(alpha2)+"_" +critic_info \
    +"_"+ str(gamma)
    #Check whether the path exists already
    print(path)
    if(Save):
        if not os.path.exists(path):
            os.makedirs(path)

    #Save Initial weights
    if(Save):
        actor.save_model(path+"/initial.h5")
        critic.save_model(path+"/initial_critic.h5")

    #Store Total reward history and loss
    score_history = []
    loss = []

    #Loop for the trajectories
    for i in (range(n_episodes)):
        done = False
        score = 0
        #Set initial state 
        observation = env.reset()
        #choose action 
        action = actor.choose_action(observation)

        count = 0
        #buiding each trajectory
        while not done:
            count += 1
            #Perform action
            observation_,reward, done, info = env.step(action)
            #Get next action for the next state
            action_ = actor.choose_action(observation_)

            #Giving one more dimension to row
            state = observation[np.newaxis,:]
            state_ = observation_[np.newaxis,:]
            #probability of actions
            probs = actor.policy.predict(state)[0]
            #Output of last hidden layer of actor network
            actor_phi = actor.phi.predict(state)[0]

            phi = critic.getPhi(actor_phi,probs,action)

            #Get Q-value for present state-action taken
            Qpi = critic.critic.predict(phi[np.newaxis,:])[0]
            #Store all the parameters for later use
            actor.store_transition(observation, action, Qpi)
            #store all the parameters for critic update
            critic.store_transition(phi, reward)

            action = action_
            observation = observation_
            score += reward
            if(environment == "MountainCar"):
                if(count==threshold):
                    break

        #Store scores for each trajectory
        score_history.append(score)

        #Update Critic network
        critic.learn()
        #Update actor network
        actor.learn()

        if(i==save_point):
            if(Save):
                actor.save_model(path+"/middle.h5")
                critic.save_model(path+"/middle_critic.h5")

        if(max<=score):
            max=score
            optimal_episode = i
            if(Save):

                actor.save_model(path+"/optimal.h5")
                critic.save_model(path+"/optimal_critic.h5")
                np.save(path+"/score",score_history)

        print('episode ', i,'score %.1f' % score,
        'average_score %.1f' % np.mean(score_history[-100:]),'Q',Qpi)

    if(Save):
        actor.save_model(path+"/final.h5")
        critic.save_model(path+"/final_critic.h5")
        np.save(path+"/score",score_history)
    # filename = environment +"_"+variant+str(alpha1)+"_"+layers+'.png'
    plotLearning(environment,str(alpha1)+","+str(alpha2),str(gamma),loss,score_history, path= path, label=variant, save =Save)
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
            actor.actor.load_weights(weightPath)

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
            mean_std.append(policy_max(env, actor,variant, alpha1, gamma, plot_path, ep=episodes))
            print("----Sampling Policy----")
            mean_std.append(policy_sampling(env, actor,variant, alpha1, gamma, plot_path, ep=episodes))

            print(mean_std)
            
            np.save(meanStd_path,mean_std)
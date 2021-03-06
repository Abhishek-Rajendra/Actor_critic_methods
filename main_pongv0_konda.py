'''
This code is used to run pong-v0 using konda actor critic.
Note that you can use linear fully connected network or CNN, please check pongv0_konda.py.
and uncomment. 
Also note that for critic input_dims argument put 320, if you are using CNN arcitecture
If you are using fully connected layers then input critic_input*2( because two actions)

Actions used
UP_ACTION = 2
DOWN_ACTION = 3

Here we preprocess 210x160x3 uint8 frame into (80,80) 2D float vector, this is two remove
unnecessary parts of the frame.

Parameters to look out for num_games, critic_input, alpha, gamma
'''
import gym
import matplotlib.pyplot as plt 
import numpy as np 
from pongv0_konda import Actor, Critic
import pandas as pd
import os
from tqdm import tqdm
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
import argparse


# os.environ["CUDA_VISIBLE_DEVICES"]="0" 

#Plot the learning graph
def plotLearning(title ,alpha, gamma, episode_rewards, path, label, save):

    plt.figure()
    plt.suptitle(label+" - "+environment)
    plt.title(r"$\alpha $ = "+alpha+", $\gamma$ = "+gamma)
    plt.plot(range(len(episode_rewards)),episode_rewards, '.-',label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Total Rewards')
    plt.legend()
    if(save):
        plt.savefig(path+"/Reward.png")


    plt.figure()
    plt.suptitle(variant+" - "+environment)
    z1=pd.Series(episode_rewards).rolling(50).mean()
    plt.title(r"$\alpha$ = "+alpha +", $\gamma$ = "+gamma + ",  Best average reward: "+ str(np.max(z1)))
    plt.plot(z1,label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Rewards over past 50 episodes')
    plt.legend()
    if(save):
        plt.savefig(path+"/cumulative.png")
    plt.show()

# preprocessing used by Karpathy with small change, output shape is (80,80) not 80*80 (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def prepro(I):
  """ prepro 210x160x3 uint8 frame into (80,80) 2D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  #Gives output as (80,80)
  return I.astype(np.float)

if __name__ == '__main__':

    #Whether to save the output
    Save = True
    
    #Variant details
    variant = "konda"
    #Environment Details
    environment = "PingPong" #Used for nameing of plot and creating a folder

    env = gym.make("Pong-v0")

    num_games = 1000
    save_point = num_games//2
    render = False
    
    #Actor Network
    alpha1 = 0.0001
    #Can use cnn arcitecture, check pongv0_konda.py file and uncomment
    actor_info = "1_500"
    h1_actor = 0
    h2_actor = 0
    #Input to critic
    critic_input = 80*80 #If no hidden layers are used
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
    alpha2 = 0.00025
    gamma = 0.99
    

    actor = Actor(ALPHA = alpha1, n_actions = 2,layer1_size=h1_actor,layer2_size=h2_actor)


    # if resume:
	#     actor.actor.load_weights("weights.h5")

    #Input for the critic linear approximation is the last hidden layer of the actor network
    #Use 320 if you are using CNN for input_dims, otherwise use critic_input * 2
    critic = Critic(ALPHA = alpha2, input_dims=320, 
    Gamma= gamma, n_actions = 2)

    #Create a path to save the output
    path = environment +"/" + variant + "/" + variant + "_" \
    + str(alpha1)+"_" + str(actor_info)    +"_" +str(alpha2)+"_" +str(critic_input) \
    +"_"+ str(gamma)
    #Check whether the path exists already
    print(path)
    if(Save):
        if not os.path.exists(path):
            os.makedirs(path)


    score_history = []
    n_steps = 0
    best_score = -21

    optimal_episode = 0
    Qpi =0

    UP_ACTION = 2
    DOWN_ACTION = 3
    # Mapping from action values to outputs from the policy network
    action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

    ep_num = 0 
    #Loop for the trajectories
    while True:
        ep_num +=1
        done = False
        score = 0

        last_observation = env.reset()
        last_observation = prepro(last_observation)
        #Perform a random action at the beginning
        action = env.action_space.sample()
        observation, _, _, _ = env.step(action)
        observation = prepro(observation)
        n_steps = 1
        #buiding each trajectory
        while not done:
            if render:
                env.render()
            
            observation_delta = observation - last_observation
            #To get the notion of direction we take the difference of consecutive frames
            last_observation = observation
            #choose action 
            action = actor.choose_action(observation)
            #Perform action
            observation,reward, done, info = env.step(action)
            observation = prepro(observation)
            score += reward
            n_steps += 1

            #Giving one more dimension to row
            state = observation_delta[np.newaxis,:]
            # state_ = observation_[np.newaxis,:]
            #probability of actions
            probs = actor.policy.predict(state)[0]
            probs = np.append(probs[0],1-probs[0])
            #Output of last hidden layer of actor network
            actor_phi = actor.phi.predict(state)[0]
            #Get Phi's for critic
            action = action_dict[action]
            phi = critic.getPhi(actor_phi,probs,action)

            #Get Q-value for present state-action taken
            Qpi = critic.critic.predict(phi[np.newaxis,:])[0]
            #Store all the parameters for later use
            
            actor.store_transition(observation, action, Qpi)
            #store all the parameters for critic update
            critic.store_transition(phi, reward)

            
        #Store scores for each trajectory
        score_history.append(score)

        #Update Critic network
        critic.learn()
        #Update actor network
        actor.learn()

        # if(i==save_point):
        #     if(Save):
        #         actor.save_model(path+"/middle.h5")
        #         critic.save_model(path+"/middle_critic.h5")

        avg_score = np.mean(score_history[-50:])

        if avg_score > best_score:
            # optimal_episode = i
            print("Average score %.2f is better then best score %.2f" % 
                    (avg_score,best_score))
            if(Save):
                actor.save_model(path+"/optimal.h5")
                critic.save_model(path+"/optimal_critic.h5")
                np.save(path+"/score",score_history)
            best_score = avg_score
        
        if(Save):
            np.save(path+"/score",score_history)

        print('episode ', ep_num,'score %.1f' % score,
        'average_score %.1f' % np.mean(score_history[-50:]),'steps', n_steps,'Q',Qpi)

    if(Save):
        actor.save_model(path+"/final.h5")
        critic.save_model(path+"/final_critic.h5")
        np.save(path+"/score",score_history)

    print(best_score,optimal_episode)
    plotLearning(environment,str(alpha1)+","+str(alpha2),str(gamma),score_history, path= path, label=variant, save =Save)   


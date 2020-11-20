import gym
import matplotlib.pyplot as plt 
import numpy as np 
from TDac import Actor, Critic
import pandas as pd
import os

###Caluculation of mean and standard deviation and plot
def plot(episode_rewards, policy, label, alpha, gamma, plot_path):

    plt.figure()
    plt.suptitle(policy)
    plt.title(environment+r", $\alpha $ = "+str(alpha)+r", $\gamma$ = "+str(gamma))
    plt.plot(range(len(episode_rewards)),episode_rewards, '.-',label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig(plot_path+"/" + policy +"Reward.png")

    plt.figure()
    plt.suptitle(policy)
    z1=pd.Series(episode_rewards).rolling(50).mean()
    plt.title(environment+r", $\alpha $ = "+str(alpha)+r", $\gamma$ = "+str(gamma)+ ", Best average reward: "+ str(np.max(z1)))
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
            action_prob = agent.actor.predict(observation[np.newaxis, :])
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
            actor.store_transition(observation, action, reward)
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
            action_prob = actor.actor.predict(observation[np.newaxis, :])
            #Get the location(action number) by finding the max position
            action = np.argmax(action_prob)
            observation_,reward, done, info = env.step(action)
            actor.store_transition(observation, action, reward)
            observation = observation_

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
    plt.title(r", $\alpha$ = "+alpha+r", $\gamma$ = "+gamma)
    plt.plot(range(len(loss)),loss, '.-',label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('loss in each episode')
    plt.legend()
    if(save):
        plt.savefig(path+"/loss.png")

    plt.figure()
    plt.suptitle(variant+" - "+environment)
    z1=pd.Series(episode_rewards).rolling(50).mean()
    plt.title(r", $\alpha$ = "+alpha +r", $\gamma$ = "+gamma + ",  Best average reward: "+ str(np.max(z1)))
    plt.plot(z1,label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Rewards over past 50 episodes')
    plt.legend()
    if(save):
        plt.savefig(path+"/cumulative.png")
    plt.show()


if __name__ == '__main__':

    #Whether to save the output
    Save = True

    #Environment details
    envs = {'CartPole': 'CartPole-v0', 
            'Acrobot': 'Acrobot-v1',
            'LunarLander': 'LunarLander-v2',
            'MountainCar':'MountainCar-v0'}

    environment = "Acrobot"
    #Variant details
    variant = "TD"

    if environment in envs:
        opengym_env = envs.get(environment, None)
    else:
        print("Please provide the right environment")
        exit()
    
    #Environment Details
    env = gym.make(opengym_env)
    n_actions = env.action_space.n
    n_states = len(env.observation_space.low)
    #Actor Network
    alpha1 = 0.005
    actor_batch_size = 20
    actor_info = "0"
    h1_actor = 0
    h2_actor = 0
    if(actor_info!='0'):
        temp = actor_info.rsplit("_")
        if(temp[0]=='1'):
            h1_actor = int(temp[1])
        elif(temp[0]=='2'):
            h1_actor = int(temp[1])
            h2_actor = int(temp[2])
        else:
            print("Wrong actor network config, or hidden layers greater than 2")

    #Critic Network
    Lambda = 0.9 #Between 0 and 1 inclusive
    alpha2 = 0.005
    gamma = 1
    critic_info = "1_16"
    h1_critic = 0
    h2_critic = 0
    if(critic_info!='0'):
        temp = critic_info.rsplit("_")
        if(temp[0]=='1'):
            h1_critic = int(temp[1])
        elif(temp[0]=='2'):
            h1_critic = int(temp[1])
            h2_critic = int(temp[2])
        else:
            print("Wrong critic network config, or hidden layers greater than 2")


    actor = Actor(ALPHA = alpha1, input_dims=n_states, 
     n_actions = n_actions,layer1_size=h1_actor, layer2_size=h2_actor)

    critic = Critic(ALPHA = alpha2,lambda_=Lambda, input_dims=n_states, 
    Gamma= gamma, n_actions = n_actions,layer1_size=h1_critic, layer2_size=h2_critic)

    #Number of trajectories to perform
    n_episodes = 5000
    save_point = n_episodes//2

    #Used for mountain car
    threshold = 500
    max = -threshold

    optimal_episode = 0
    Q =[]
    #Create a path to save the output
    path = environment + "/"+variant+"/" + "/" + variant + "_" + str(alpha1)\
        +"_" +actor_info + "_" +str(alpha2)+"_" +critic_info \
        +"_"+ str(Lambda) + "_"+str(gamma)+"_"+str(actor_batch_size)
    #Check whether the path exists already
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
    for i in range(n_episodes):
        done = False
        score = 0
        #Set initial state 
        observation = env.reset()
        #choose action 
        action = actor.choose_action(observation)
        #Initialize eligibility
        critic.initialize_eligibility(observation,action)
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
            
            Q = critic.critic.predict(state)
            #Get Q-value for present state-action taken
            Qpi = np.array([Q[0,action]])
            #Store all the parameters for later use
            actor.store_transition(observation, action, Qpi)
            #Update weights
            critic.learn(reward, state_, action_, Qpi, done)

            action = action_
            observation = observation_
            score += reward
            if(environment == "MountainCar"):
                if(count==threshold):
                    break

        #Store scores for each trajectory
        score_history.append(score)
        loss.append(actor.accumulate_gradient())

        if((i+1)%actor_batch_size==0):
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
        'average_score %.1f' % np.mean(score_history[-100:]),'Q',Q)

    if(Save):
        actor.save_model(path+"/final.h5")
        critic.save_model(path+"/final_critic.h5")
        np.save(path+"/score",score_history)
    # filename = environment +"_"+variant+str(alpha1)+"_"+layers+'.png'
    plotLearning(environment,str(alpha1)+","+str(alpha2),str(gamma),loss,score_history, path= path, label=variant, save =Save)
    print(max,optimal_episode)

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
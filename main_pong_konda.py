'''
This can be used to other ataari games other than pong with changing teh environment
with minor changes in PreProcessFrame.
This here will change the how the gym wrapper works, it fixes the action performed 
in 4 frames rather than performing each action  for a duration of k frames, 
where k is uniformly sampled from {2,3,4}.
And sending it to neural network, because from one frame you cant make out
the direction of ball and bat so multiple frames are required.
We can you only two frames tooo, which is given in main_pongv0_konda.py 


'''
import gym
import matplotlib.pyplot as plt 
import numpy as np 
from pingpong_konda import Actor, Critic
import pandas as pd
import os
from tqdm import tqdm
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

# os.environ["CUDA_VISIBLE_DEVICES"]="0" 
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

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


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        self._obs_buffer = []
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(80,80,1), dtype=np.uint8)
    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):

        new_frame = np.reshape(frame, frame.shape).astype(np.float32)

        new_frame = 0.299*new_frame[:,:,0] + 0.587*new_frame[:,:,1] + \
                    0.114*new_frame[:,:,2]

        # need to change this according to the environment...
        new_frame = new_frame[35:195:2, ::2].reshape(80,80,1)

        return new_frame.astype(np.uint8)

class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                            shape=(self.observation_space.shape[-1],
                                   self.observation_space.shape[0],
                                   self.observation_space.shape[1]),
                            dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(n_steps, axis=0),
                             env.observation_space.high.repeat(n_steps, axis=0),
                             dtype=np.float32)

    def reset(self):
        self.buffer = np.array(np.zeros_like(self.observation_space.low, dtype=np.float32))
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer  

def make_env(env_name):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)

if __name__ == '__main__':

    #Whether to save the output
    Save = True
    environment = "PingPong"
    #Variant details
    variant = "konda"
    #Environment Details
    env = make_env("PongNoFrameskip-v4")
    n_actions = 2

    num_games = 10000
    save_point = num_games//2
    load_checkpoint = False
    
   
    #Actor Network
    alpha1 = 0.0001
    last_layer = 300
    #Critic Network
    alpha2 = 0.0005
    gamma = 0.99
    
    critic_input = last_layer * 2

    actor = Actor(ALPHA = alpha1, input_dims=(4,80,80),last_size=last_layer,
    n_actions = n_actions)

    #Input for the critic linear approximation is the last hidden layer o fthe actor network
    critic = Critic(ALPHA = alpha2, input_dims=critic_input, 
    Gamma= gamma, n_actions = n_actions)

    #Create a path to save the output
    path = environment +"/" + variant + "/" + variant + "_" \
    + str(alpha1)+"_" + str(last_layer)    +"_" +str(alpha2)+"_" +str(critic_input) \
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


    score_history = []
    n_steps = 0
    best_score = -21

    optimal_episode = 0
    Qpi =0

    UP_ACTION = 2
    DOWN_ACTION = 3
    # Mapping from action values to outputs from the policy network
    action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

    #Loop for the trajectories
    for i in range(num_games):
        done = False
        score = 0
        #Set initial state 
        observation = env.reset()
        #buiding each trajectory
        while not done:
            #choose action 
            action = actor.choose_action(observation)
            #Perform action
            observation_,reward, done, info = env.step(action)
            n_steps += 1


            #Giving one more dimension to row
            state = observation[np.newaxis,:]
            state_ = observation_[np.newaxis,:]
            #probability of actions
            probs = actor.policy.predict(state)[0]
            probs = np.append(probs[0],1-probs[0])
            #Output of last hidden layer of actor network
            actor_phi = actor.phi.predict(state)[0]

            action = action_dict[action]
            phi = critic.getPhi(actor_phi,probs,action)

            #Get Q-value for present state-action taken
            Qpi = critic.critic.predict(phi[np.newaxis,:])[0]
            #Store all the parameters for later use
            actor.store_transition(observation, action, Qpi)
            #store all the parameters for critic update
            critic.store_transition(phi, reward)

            observation = observation_
            score += reward


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

        avg_score = np.mean(score_history[-50:])

        if avg_score > best_score:
            optimal_episode = i
            print("Average score %.2f is better then best score %.2f" % 
                    (avg_score,best_score))
            if(Save):
                actor.save_model(path+"/optimal.h5")
                critic.save_model(path+"/optimal_critic.h5")
                np.save(path+"/score",score_history)
            best_score = avg_score

        print('episode ', i,'score %.1f' % score,
        'average_score %.1f' % np.mean(score_history[-50:]),'steps', n_steps,'Q',Qpi)

    if(Save):
        actor.save_model(path+"/final.h5")
        critic.save_model(path+"/final_critic.h5")
        np.save(path+"/score",score_history)

    print(best_score,optimal_episode)
    plotLearning(environment,str(alpha1)+","+str(alpha2),str(gamma),score_history, path= path, label=variant, save =Save)

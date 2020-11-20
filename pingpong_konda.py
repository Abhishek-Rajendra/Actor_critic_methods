import numpy as np 
import tensorflow as tf
import keras.backend as K 
from keras.layers import  Dense, Activation, Input, Conv2D, Flatten
from keras.models import Model, load_model 
from keras.optimizers import Adam,RMSprop
from keras.regularizers import l2
import time


class Actor():
    def __init__(self, ALPHA, n_actions =4,
        last_size=16, input_dims = 8):

        self.lr = ALPHA
        self.input_dims = input_dims
        self.h1_dims = last_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.Q_memory = []

        # Action values to send to gym environment to move paddle up/down
        self.UP_ACTION = 2
        self.DOWN_ACTION = 3

        self.actor, self.phi, self.policy = self.build_polic_network()

        self.actions_space = [i for i in range(n_actions)]

    def build_polic_network(self):

        input = Input(shape=(*self.input_dims,))
        advantages = Input(shape=[1])
        conv1 = Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
         data_format='channels_first')(input)
        conv2 = Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                        data_format='channels_first')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                        data_format='channels_first')(conv2)
        flat = Flatten()(conv3)
        dense1 = Dense(self.h1_dims, activation='relu')(flat)
        probs = Dense(1, activation='sigmoid')(dense1)

        #Loss funciton implimenting Cross Entropy
        def custum_loss(y_true,y_pred):
            #Clipping to ignore getting 0 and 1 has input from softmax layer
            out = K.clip(y_pred, 1e-8,1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*advantages)

        #Generate the model with proper inputs and outputs
        actor = Model(inputs=[input,advantages], outputs=[probs])
        actor.compile(optimizer=RMSprop(lr=self.lr), loss=custum_loss)
        actor.summary()

        phi = Model(inputs=[input], outputs=[dense1])

        predict = Model(input=[input], output=[probs])
        # phi.compile(optimizer=Adam(lr=self.lr), loss="custum_loss(Q,y_true,y_pred)")

        return actor, phi, predict
    
        
    def store_transition(self, observation, action, Q):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.Q_memory.append(Q)
    
    def learn(self):

        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        Q_memory = np.array(self.Q_memory)

        Q_memory = Q_memory - np.mean(Q_memory)
        Q_memory/=np.std(Q_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)),action_memory] = 1

        cost = self.actor.train_on_batch([state_memory, Q_memory], actions)

        self.state_memory = []
        self.action_memory = []
        self.Q_memory = []

        # return cost

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        #Get the probability for each action
        probability = self.policy.predict(state)[0]
        #Get the action by sampling from the given probability
        if np.random.uniform() < probability:
            action = self.UP_ACTION
        else:
            action = self.DOWN_ACTION

        return action

        return action

    def save_model(self,name):
        self.actor.save(name)

    def load_model(self,name):
        self.actor = load_model(name)

class Critic():
    def __init__(self, ALPHA, Gamma = 0.99, n_actions =4,
        input_dims = 8,decay=0.01):

        self.gamma = Gamma
        self.lr = ALPHA
        self.decay = decay
        self.input_dims = input_dims
        self.n_actions = n_actions        
        self.actions_space = [i for i in range(n_actions)]
        self.phi_memory = []
        self.reward_memory = []

        self.critic = self.build_polic_network()

    def build_polic_network(self):
        input = Input(shape=(self.input_dims,))
        Qvalue = Dense(1, activation='linear',use_bias=False)(input)

        critic = Model(input = [input], output = [Qvalue])
        critic.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error')
        critic.summary()

        return critic

    def store_transition(self, phi, reward):
        self.phi_memory.append(phi)
        self.reward_memory.append(reward)

    def getPhi(self, actor_hidden_out, probs, action):
        
        Q_input = np.zeros(self.input_dims)

        probs = [-prob for prob in probs]
        probs[action] = 1 + probs[action]
        output_size = len(actor_hidden_out)
        #Geting input for critic linear approximation as given 
        # in Policy Gradient Methods for
        #Reinforcement Learning with FunctionApproximation
        #by Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour
        for i in range(self.n_actions):
            Q_input = Q_input + np.pad(actor_hidden_out,\
                (i*output_size,(self.n_actions-1-i)*output_size)\
                    , 'constant') * probs[i]        

        return Q_input


    def learn(self):
        phi_memory = np.array(self.phi_memory)
        reward_memory = np.array(self.reward_memory)

        #Get future reward for each state in an episode
        G = np.array(np.zeros_like(reward_memory))[:,np.newaxis]
        for r in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(r, len(reward_memory)):
                G_sum +=reward_memory[k]*discount
                discount *= self.gamma
                if reward_memory[k] != 0:
                # Don't count rewards from subsequent rounds
                    break
            G[r][0] = G_sum

        #Here we are considering G has target funtion
        #But need to replace G for the current action performed
        #Rest should be equal to their Q value so the loss 
        #for other Q for other actions must be zero
        # target = self.critic.predict(phi_memory)
        # target[np.arange(len(action_memory)),action_memory] = G

        cost = self.critic.train_on_batch(phi_memory, G)

        # print(self.critic.get_weights())
        self.phi_memory = []
        self.reward_memory = []

        # return cost


    def save_model(self,name):
        self.critic.save(name)

    def load_model(self,name):
        self.critic = load_model(name)
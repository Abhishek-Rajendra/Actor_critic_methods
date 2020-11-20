import numpy as np 
import tensorflow as tf
import keras.backend as K 
from keras.layers import  Dense, Activation, Input
from keras.models import Model, load_model 
from keras.optimizers import Adam
from keras.regularizers import l2
import time


class Actor():
    def __init__(self, ALPHA, n_actions =4,
        layer1_size=16,layer2_size=16, input_dims = 8):

        self.lr = ALPHA
        self.input_dims = input_dims
        self.h1_dims = layer1_size
        self.h2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.Q_memory = []

        self.actor, self.phi, self.policy = self.build_polic_network()

        self.actions_space = [i for i in range(n_actions)]


    def build_polic_network(self):
        #Bulding a network
        input = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
         # no hidden layer
        if(self.h1_dims == 0 and self.h2_dims==0):
            probs = Dense(self.n_actions, activation='softmax')(input)
            #Output of this model will be input for the Critic Network
            phi = Model(inputs=[input], outputs=[input])
        #One hidden layer
        elif(self.h1_dims != 0 and self.h2_dims == 0):
            dense1 = Dense(self.h1_dims,activation='relu')(input)
            probs = Dense(self.n_actions, activation='softmax')(dense1)
            #Output of this model will be input for the Critic Network
            phi = Model(inputs=[input], outputs=[dense1])
        #Two hidden layers
        else:
            dense1 = Dense(self.h1_dims,activation='relu')(input)
            dense2 = Dense(self.h2_dims, activation='relu')(dense1)
            probs = Dense(self.n_actions, activation='softmax')(dense2)

            #Output of this model will be input for the Critic Network
            phi = Model(inputs=[input], outputs=[dense2])

        #Loss funciton implimenting Cross Entropy
        def custum_loss(y_true,y_pred):
            #Clipping to ignore getting 0 and 1 has input from softmax layer
            out = K.clip(y_pred, 1e-8,1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*advantages)

        #Generate the model with proper inputs and outputs
        actor = Model(inputs=[input,advantages], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.lr), loss=custum_loss)
        actor.summary()

        predict = Model(inputs=[input], outputs=[probs])
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
        probabilities = self.policy.predict(state)[0]
        #Get the action by sampling from the given probability
        action = np.random.choice(self.actions_space, p=probabilities)

        return action

    def save_model(self,name):
        self.policy.save(name)

    def load_weights(self,name):
        self.policy.load_weights(name)

class Critic():
    def __init__(self, ALPHA, Gamma = 0.99, n_actions =4,
        input_dims = 8,decay=0.01):

        self.gamma = Gamma
        self.lr = ALPHA
        self.decay = decay
        self.input_dims = input_dims
        self.n_actions = n_actions        
        self.phi_memory = []
        self.reward_memory = []

        self.critic = self.build_polic_network()

    def build_polic_network(self):
        input = Input(shape=(self.input_dims,))
        Qvalue = Dense(1, activation='linear',use_bias=False)(input)

        critic = Model(inputs = [input], outputs = [Qvalue])
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
            G[r][0] = G_sum

        cost = self.critic.train_on_batch(phi_memory, G)

        self.phi_memory = []
        self.reward_memory = []

        # return cost


    def save_model(self,name):
        self.critic.save(name)

    def load_weights(self,name):
        self.critic.load_weights(name)
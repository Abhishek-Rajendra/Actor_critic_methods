import numpy as np 
import tensorflow as tf
import keras.backend as K 
from keras.layers import  Dense, Activation, Input, Conv2D, Flatten, Reshape
from keras.models import Model, load_model 
from keras.optimizers import Adam,RMSprop
from keras.regularizers import l2
import time


class Actor():
    def __init__(self, ALPHA,n_actions =2,
        layer1_size=16,layer2_size=16, input_dims = 8):

        self.lr = ALPHA
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.h1_dims = layer1_size
        self.h2_dims = layer2_size
        self.state_memory = []
        self.action_memory = []
        self.Q_memory = []

        # Action values to send to gym environment to move paddle up/down
        self.UP_ACTION = 2
        self.DOWN_ACTION = 3
        # Mapping from action values to outputs from the policy network
        # self.action_dict = {self.DOWN_ACTION: 0, self.UP_ACTION: 1}


        self.actor, self.phi, self.policy = self.build_polic_network()


    def build_polic_network(self):
        
        #CNN layers
        inputs = Input(shape=(80,80))
        advantages = Input(shape=[1])
        channeled_input = Reshape((80,80,1))(inputs) # Conv2D requries (batch, height, width, channels)  so we need to create a dummy channel 
        conv_1 = Conv2D(filters=10,kernel_size=20,padding='valid',activation='relu',strides=(4,4),use_bias=False)(channeled_input)
        conv_2 = Conv2D(filters=20,kernel_size=10,padding='valid',activation='relu',strides=(2,2),use_bias=False)(conv_1)
        conv_3 = Conv2D(filters=40,kernel_size=3,padding='valid',activation='relu',use_bias=False)(conv_2)
        flattened_layer = Flatten()(conv_3)
        probs = Dense(1,activation='sigmoid',use_bias=False)(flattened_layer)

        phi = Model(inputs=[inputs], outputs=[flattened_layer])

        #Simple Fully conencted layers
        # #Bulding a network
        # input = Input(shape=(80,80))
        # flattened_layer = keras.layers.Flatten()(inputs)
        # advantages = Input(shape=[1])
        #  # no hidden layer
        # if(self.h1_dims == 0 and self.h2_dims==0):
        #     probs = Dense(1, activation='sigmoid',use_bias=False)(flattened_layer)
        #     #Output of this model will be input for the Critic Network
        #     phi = Model(inputs=[input], outputs=[input])
        # #One hidden layer
        # elif(self.h1_dims != 0 and self.h2_dims == 0):
        #     dense1 = Dense(self.h1_dims,activation='relu',use_bias=False)(flattened_layer)
        #     probs = Dense(1, activation='sigmoid',use_bias=False)(dense1)
        #     #Output of this model will be input for the Critic Network
        #     phi = Model(inputs=[input], outputs=[dense1])
        # #Two hidden layers
        # else:
        #     dense1 = Dense(self.h1_dims,activation='relu',use_bias=False)(flattened_layer)
        #     dense2 = Dense(self.h2_dims, activation='relu',use_bias=False)(dense1)
        #     probs = Dense(1, activation='sigmoid',use_bias=False)(dense2)
        #     #Output of this model will be input for the Critic Network
        #     phi = Model(inputs=[input], outputs=[dense2])

        #Loss funciton implimenting Cross Entropy
        def custum_loss(y_true,y_pred):
            #Clipping to ignore getting 0 and 1 has input from softmax layer
            out = K.clip(y_pred, 1e-8,1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*advantages)

        #Generate the model with proper inputs and outputs
        actor = Model(inputs=[inputs,advantages], outputs=[probs])
        actor.compile(optimizer=RMSprop(lr=self.lr), loss=custum_loss)
        actor.summary()

        predict = Model(input=[inputs], output=[probs])
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


        cost = self.actor.train_on_batch([state_memory, Q_memory], action_memory)

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
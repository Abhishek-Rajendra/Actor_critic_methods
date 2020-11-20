from keras.layers import  Dense, Activation, Input
from keras.models import Model, load_model 
from keras.optimizers import Adam,RMSprop
import keras.backend as K 
import numpy as np 


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
        self.reward_memory = []

        self.actor, self.policy = self.build_polic_network()

        self.actions_space = [i for i in range(n_actions)]

    def build_polic_network(self):
        input = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
        # no hidden layer
        if(self.h1_dims == 0 and self.h2_dims==0):
            probs = Dense(self.n_actions, activation='softmax')(input)
        #One hidden layer
        elif(self.h1_dims != 0 and self.h2_dims == 0):
            dense1 = Dense(self.h1_dims,activation='relu')(input)
            probs = Dense(self.n_actions, activation='softmax')(dense1)
        #Two hidden layers
        else:
            dense1 = Dense(self.h1_dims,activation='relu')(input)
            dense2 = Dense(self.h2_dims, activation='relu')(dense1)
            probs = Dense(self.n_actions, activation='softmax')(dense2)


        #Loss funciton implimenting Cross Entropy
        def custum_loss(y_true,y_pred):
            #Clipping to ignore getting 0 and 1 has input from softmax layer
            out = K.clip(y_pred, 1e-8,1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*advantages)
        
        actor = Model(inputs = [input, advantages], outputs = [probs])
        actor.compile(optimizer=Adam(lr=self.lr), loss=custum_loss)
        actor.summary()
        
        predict = Model(inputs=[input], outputs=[probs])
        predict.compile(optimizer=Adam(lr=self.lr), loss=custum_loss)

        return actor, predict

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.actions_space, p=probabilities)

        return action

    def save_model(self,name):
        self.policy.save(name)

    def load_weights(self,name):
        self.policy.load_weights(name)

class Critic():
    def __init__(self, ALPHA, Gamma = 0.99, n_actions =4,
        layer1_size=16,layer2_size=16, input_dims = 8):

        self.gamma = Gamma
        self.lr = ALPHA
        #Estimated reward
        self.G = 0
        #Total Reward of each episode is scored
        self.Total_Reward_for_all_episodes = []

        self.input_dims = input_dims
        self.h1_dims = layer1_size
        self.h2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.critic = self.build_polic_network()


    def build_polic_network(self):
        input = Input(shape=(self.input_dims,))

        # no hidden layer
        if(self.h1_dims == 0 and self.h2_dims==0):
            value = Dense(1, activation='linear')(input)
        #One hidden layer
        elif(self.h1_dims != 0 and self.h2_dims == 0):
            dense1 = Dense(self.h1_dims,activation='relu')(input)
            value = Dense(1, activation='linear')(dense1)
        #Two hidden layers
        else:
            dense1 = Dense(self.h1_dims,activation='relu')(input)
            dense2 = Dense(self.h2_dims, activation='relu')(dense1)
            value = Dense(1, activation='linear')(dense2)

        
        critic = Model(inputs = [input], outputs = [value])
        critic.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error')
        critic.summary()

        return critic 

    def save_model(self,name):
        self.critic.save(name)

    def load_weights(self,name):
        self.critic.load_weights(name)
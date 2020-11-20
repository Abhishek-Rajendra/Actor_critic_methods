from keras.layers import  Dense, Activation, Input
from keras.models import Model, load_model 
from keras.optimizers import Adam,RMSprop
import keras.backend as K 
import numpy as np 
from keras.regularizers import l2


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

        self.actor, self.policy = self.build_polic_network()

        self.actions_space = [i for i in range(n_actions)]

    def build_polic_network(self):
        inputs = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
        # no hidden layer
        if(self.h1_dims == 0 and self.h2_dims==0):
            probs = Dense(self.n_actions, activation='softmax')(inputs)
        #One hidden layer
        elif(self.h1_dims != 0 and self.h2_dims == 0):
            dense1 = Dense(self.h1_dims,activation='relu')(inputs)
            probs = Dense(self.n_actions, activation='softmax')(dense1)
        #Two hidden layers
        else:
            dense1 = Dense(self.h1_dims,activation='relu')(inputs)
            dense2 = Dense(self.h2_dims, activation='relu')(dense1)
            probs = Dense(self.n_actions, activation='softmax')(dense2)

        #Loss funciton implimenting Cross Entropy
        def custum_loss(y_true,y_pred):
            #Clipping to ignore getting 0 and 1 has input from softmax layer
            out = K.clip(y_pred, 1e-8,1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*advantages)
        
        actor = Model(inputs = [inputs, advantages], outputs = [probs])
        actor.compile(optimizer=Adam(lr=self.lr), loss=custum_loss)
        actor.summary()
        
        predict = Model(inputs=[inputs], outputs=[probs])
        predict.compile(optimizer=Adam(lr=self.lr), loss=custum_loss)

        return actor, predict

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.actions_space, p=probabilities)

        return action

    def store_transition(self, observation, action, Q):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.Q_memory.append(Q)
    
    def learn(self):

        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        Q_memory = np.array(self.Q_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)),action_memory] = 1

        cost = self.actor.train_on_batch([state_memory, Q_memory], actions)

        self.state_memory = []
        self.action_memory = []
        self.Q_memory = []

        # return cost

    def save_model(self,name):
        self.policy.save(name)

    def load_weights(self,name):
        self.policy.load_weights(name)

class Critic():
    def __init__(self, ALPHA, Gamma = 0.99, n_actions =4,
        layer1_size=16,layer2_size=16, input_dims = 8):

        self.gamma = Gamma
        self.lr = ALPHA
        self.input_dims = input_dims
        self.h1_dims = layer1_size
        self.h2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.critic = self.build_polic_network()


    def build_polic_network(self):
        inputs = Input(shape=(self.input_dims,))
        # no hidden layer
        if(self.h1_dims == 0 and self.h2_dims==0):
            Qvalue = Dense(self.n_actions, activation='linear')(inputs)
        #One hidden layer
        elif(self.h1_dims != 0 and self.h2_dims == 0):
            dense1 = Dense(self.h1_dims,activation='relu',kernel_regularizer=l2(0.01))(inputs)
            Qvalue = Dense(self.n_actions, activation='linear')(dense1)
        #Two hidden layers
        else:
            dense1 = Dense(self.h1_dims,activation='relu',kernel_regularizer=l2(0.01))(inputs)
            dense2 = Dense(self.h2_dims, activation='relu',kernel_regularizer=l2(0.01))(dense1)
            Qvalue = Dense(self.n_actions, activation='linear')(dense2)

        
        critic = Model(inputs = [inputs], outputs = [Qvalue])
        critic.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error')
        critic.summary()

        return critic
    
    def store_transition(self, observation, action, reward):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        #Get future reward for each state in an episode
        G = np.array(np.zeros_like(reward_memory))
        for r in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(r, len(reward_memory)):
                G_sum +=reward_memory[k]*discount
                discount *= self.gamma
            G[r] = G_sum

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)),action_memory] = 1

        #Here we are considering G has target funtion
        #But need to replace G for the current action performed
        #Rest should be equal to their Q value so the loss 
        #for other Q for other actions must be zero
        target = self.critic.predict(state_memory)
        target[np.arange(len(action_memory)),action_memory] = G

        cost = self.critic.train_on_batch(state_memory, target)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        # return cost

    def save_model(self,name):
        self.critic.save(name)

    def load_weights(self,name):
        self.critic.load_weights(name)
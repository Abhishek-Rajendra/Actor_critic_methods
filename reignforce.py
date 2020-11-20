from keras.layers import  Dense, Activation, Input
from keras.models import Model, load_model 
from keras.optimizers import Adam
import keras.backend as K 
import numpy as np 


class Agent(object):
    def __init__(self, ALPHA = 0.01, Gamma = 0.99, n_actions =4,
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
        
        policy = Model(inputs=[input], outputs=[probs])
        policy.compile(optimizer=Adam(lr=self.lr), loss=custum_loss)

        return actor, policy

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.actions_space, p=probabilities)

        return action

    def store_transition(self, observation, action, reward):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self,variant):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)),action_memory] = 1

        #Total Reward
        if(variant=="total"):
            G = np.array(np.ones_like(reward_memory))
            G_sum = 0
            discount = 1
            for k in range(len(reward_memory)):
                G_sum +=reward_memory[k]*discount
                discount *= self.gamma

            self.G = G * G_sum
            #Try this for better results
            # self.G = self.G - np.mean(self.G)
            # self.G /= np.std(self.G)
        else:
        # #Future Reward 
            G = np.array(np.zeros_like(reward_memory))
            for r in range(len(reward_memory)):
                G_sum = 0
                discount = 1
                for k in range(r, len(reward_memory)):
                    G_sum +=reward_memory[k]*discount
                    discount *= self.gamma
                G[r] = G_sum
        
        if(variant=="without"):
            self.G = G
            #Try this for better results
            # self.G = self.G - np.mean(self.G)
            # self.G /= np.std(self.G)
                
        #Taking mean Total reward across all episodes till now
        b = np.mean(self.Total_Reward_for_all_episodes) if len(self.Total_Reward_for_all_episodes) !=0 else 0

        #Final Advantage value
        if(variant=="with"):
            self.G = (G-b)

        #Append the total reward of present episode
        self.Total_Reward_for_all_episodes.append(G[0])

        cost = self.actor.train_on_batch([state_memory, self.G], actions)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        return cost

    def save_model(self, name):
        self.actor.save(name)

    def load_weights(self,name):
        self.actor.load_weights(name)



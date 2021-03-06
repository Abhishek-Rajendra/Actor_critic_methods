import numpy as np 
import tensorflow as tf
import keras.backend as K 
from keras.layers import  Dense, Activation, Input
from keras.models import Model, load_model 
from keras.optimizers import Adam
from keras.regularizers import l2
import time

#Optimizers for Critic Network
class RMSprop:
    def __init__(self, weights, alpha=0.01, decay_rate=0.99, beta1=0.9, beta2=0.999, epsilon=1e-5):
        self.alpha = alpha
        self.rmsprop_cache = np.zeros_like(weights)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.theta = weights
    def backward_pass(self, gradient):
        self.rmsprop_cache = self.decay_rate * self.rmsprop_cache + (1 - self.decay_rate) * gradient**2
        self.theta = self.theta + self.alpha * gradient / (np.sqrt(self.rmsprop_cache) + self.epsilon)
        return self.theta
        
class AdamOptimizer:
    def __init__(self, weights, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.theta = weights

    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        self.theta = self.theta + self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))
        return self.theta


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
        self.V_memory = []


        self.optimizer = Adam(learning_rate=self.lr)
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

        
        # phi.compile(optimizer=Adam(lr=self.lr), loss="custum_loss(Q,y_true,y_pred)")

        predict = Model(inputs=[input], outputs=[probs])

        return actor, phi, predict
    
        
    def store_transition(self, observation, action, V):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.V_memory.append(V)


    def learn(self):

        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        V_memory = np.array(self.V_memory)

        V_memory = V_memory - np.mean(V_memory)
        V_memory/=np.std(V_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)),action_memory] = 1

        cost = self.actor.train_on_batch([state_memory, V_memory], actions)

        self.state_memory = []
        self.action_memory = []
        self.V_memory = []

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
    def __init__(self, ALPHA, lambda_=0, Gamma = 0.99, n_actions =4,
        input_dims = 8,decay=0.0005):

        self.gamma = Gamma
        self.lr = ALPHA
        self.constant_lr = ALPHA
        self.decay = decay
        self.lambda_ = lambda_
        self.input_dims = input_dims
        self.n_actions = n_actions        
        self.actions_space = [i for i in range(n_actions)]
        self.linear_weights = np.random.randn(input_dims, 1).ravel()
        
        self.optimizer = AdamOptimizer(self.linear_weights,alpha=ALPHA)
        # self.eligibilty = np.zeros(input_dims)
    
    def predict(self, phi):    

        return np.dot(self.linear_weights, phi)
   
    def initialize_eligibility(self, actor_output, action):

        self.eligibilty = actor_output

    def learn(self, phi_next_state, TD_error):

        #Normalizing
        td_el = TD_error * self.eligibilty
        norm = np.linalg.norm(td_el)
        if norm != 0.0:
            td_el = td_el/norm 
        

        #Update weights
        self.linear_weights = self.optimizer.backward_pass(td_el)

    
        #Update Eligibility Traces
        self.eligibilty = self.gamma * self.lambda_ * self.eligibilty + phi_next_state


        # print(TD_error)

    def save_model(self,name):
        np.save(name,self.linear_weights)

    def load_weights(self,name):
        self.linear_weights = np.load(name)
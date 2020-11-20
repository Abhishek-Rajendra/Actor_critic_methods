import numpy as np 
import tensorflow as tf
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import  Dense, Activation, Input
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.regularizers import l2

tf.config.experimental_run_functions_eagerly(True)

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
        #Bulding a network
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

        #Generate the model with proper inputs and outputs
        actor = Model(inputs=[input,advantages], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.lr), loss=custum_loss)
        actor.summary()

        predict = Model(inputs=[input], outputs=[probs])

    
        return actor, predict
    
        
    def store_transition(self, observation, action, Q):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.Q_memory.append(Q)
    
    def learn(self):

        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        Q_memory = np.array(self.Q_memory)

        # Q_memory = Q_memory - np.mean(Q_memory)
        # Q_memory/=np.std(Q_memory)

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
    def __init__(self, ALPHA, lambda_=0, Gamma = 0.99, n_actions =4,
        layer1_size=16,layer2_size=16, input_dims = 8):

        self.gamma = Gamma
        self.lr = ALPHA
        self.lambda_ = lambda_
        self.input_dims = input_dims
        self.h1_dims = layer1_size
        self.h2_dims = layer2_size
        self.n_actions = n_actions

        self.critic = self.build_polic_network()
        
        self.optimizer = Adam(learning_rate=ALPHA)


    def build_polic_network(self):
        #Build the Network
        input = Input(shape=(self.input_dims,))
  
        # no hidden layer
        if(self.h1_dims == 0 and self.h2_dims==0):
            Q_values = Dense(self.n_actions, activation='linear')(input)
        #One hidden layer
        elif(self.h1_dims != 0 and self.h2_dims == 0):
            dense1 = Dense(self.h1_dims,activation='relu',kernel_regularizer=l2(0.01))(input)
            Q_values = Dense(self.n_actions, activation='linear')(dense1)
        #Two hidden layers
        else:
            dense1 = Dense(self.h1_dims,activation='relu',kernel_regularizer=l2(0.01))(input)
            dense2 = Dense(self.h2_dims, activation='relu',kernel_regularizer=l2(0.01))(dense1)
            Q_values = Dense(self.n_actions, activation='linear')(dense2)


        critic = Model(inputs = [input], outputs = [Q_values])
        critic.summary()
 
        return critic

    def initialize_eligibility(self, observation, action):

        state = observation[np.newaxis,:]
        #Get gradient of Q function
        with tf.GradientTape() as tape:
            Qvalues = self.critic(state)
            tvs = self.critic.trainable_variables
            Q = Qvalues[0,action]
        #Calculating Gradient on Q of current state and action with respect to weights(bias included) of the network
        grads = tape.gradient(Q, tvs)

        self.eligibilty = grads

    def learn(self, reward, next_state,next_action, Q, done):
        # weights = self.critic.get_weights()

        #Get gradient of Q function
        with tf.GradientTape() as tape:
            Qvalues = self.critic(next_state)
            tvs = self.critic.trainable_variables
            next_Q = Qvalues[0,next_action]
        #Calculating Gradient on Q of current state and action with respect to weights(bias included) of the network
        grads = tape.gradient(next_Q, tvs)

        Q_ = np.array(next_Q)
        #When done is true no need to take value of next state and change only the target value of present action
        TD_error = reward + self.gamma * Q_ * (1 - int(done)) - Q

        #Update weights
        td_el = TD_error * self.eligibilty

        for grad_el in td_el:
            norm = np.linalg.norm(grad_el)
            if norm != 0.0:
                grad_el = grad_el/norm
        
        self.optimizer.apply_gradients(zip(td_el, self.critic.trainable_variables))

        #Update Eligibility Traces
        self.eligibilty = [(self.gamma * self.lambda_*self.eligibilty[i])+grad for i,grad in enumerate(grads)]



    def save_model(self,name):
        self.critic.save(name)

    def load_weights(self,name):
        self.critic.load_weights(name)
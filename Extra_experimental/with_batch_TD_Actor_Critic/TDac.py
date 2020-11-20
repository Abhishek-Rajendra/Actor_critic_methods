import numpy as np 
import tensorflow as tf
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import  Dense, Activation, Input
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.optimizers import Adam,RMSprop
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

        self.accumulate_grad = []
        self.ep_accumulated = 0
        self.optimizer = Adam(learning_rate=self.lr)
        self.actor = self.build_polic_network()

        self.actions_space = [i for i in range(n_actions)]

    #calculating loss for the actor network for one whole trajectory
    def custum_loss(self,Q,y_true,y_pred):

        #Just clipping of 0s and 1s from the predicted probabilities
        out = K.clip(y_pred, 1e-8,1-1e-8)
        log_lik = y_true*K.log(out)

        return K.sum(-log_lik*Q)


    def build_polic_network(self):
        #Bulding a network
        input = Input(shape=(self.input_dims,))
        
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


        #Generate the model with proper inputs and outputs
        actor = Model(inputs=[input], outputs=[probs])
        actor.summary()

        #Getting the size of weights including bias terms
        tvs = actor.trainable_weights
        #Intitialize the Accumulate gradient to zeros which has same size as trainable variables from the model
        self.accumulate_grad = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]

        return actor
    
        
    def store_transition(self, observation, action, Q):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.Q_memory.append(Q)
    
    def accumulate_gradient(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        Q_memory = np.array(self.Q_memory)

        # Q_memory = (Q_memory - np.mean(Q_memory))/np.std(Q_memory)

        #One hot encoding of actions
        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)),action_memory] = 1

        # loss_history = []

        #GradientTape will record operations as they are executed for calculating gradient
        with tf.GradientTape() as tape:
            logits = self.actor(state_memory)
            #Get trainable variables
            tvs = self.actor.trainable_variables
            #Calculate loss
            loss_value = self.custum_loss(Q_memory,actions, logits)

        # loss_history.append(loss_value.numpy())
        #Calculating Gradient on loss with respect to weights(bias included) of the network
        grads = tape.gradient(loss_value, tvs)
        # print("grads",grads[0])
        #Add to the accumulated grad
        self.ep_accumulated = self.ep_accumulated + 1
        #Storing average of grad till now
        self.accumulate_grad = [self.accumulate_grad[i]*((self.ep_accumulated-1)/self.ep_accumulated) + (grad)/self.ep_accumulated for i, grad in enumerate(grads)]
        # print("acc=",self.accumulate_grad[0])

        #Clear the memories
        self.state_memory = []
        self.action_memory = []
        self.Q_memory = []

        return loss_value.numpy()


    
    def learn(self):
        self.optimizer.apply_gradients(zip(self.accumulate_grad, self.actor.trainable_variables))
        self.ep_accumulated = 0 
        #Assigning the accumulation of gradient ot zero after the batch is reached
        tvs = self.actor.trainable_variables
        self.accumulate_grad = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        #Get the probability for each action
        probabilities = self.actor.predict(state)[0]
        #Get the action by sampling from the given probability
        action = np.random.choice(self.actions_space, p=probabilities)

        return action

    def save_model(self,name):
        self.actor.save(name)

    def load_model(self,name):
        self.actor = load_model(name)

class Critic():
    def __init__(self, ALPHA, lambda_=0, Gamma = 0.99, n_actions =4,
        layer1_size=16,layer2_size=16, input_dims = 8):

        self.gamma = Gamma
        # self.lr = ALPHA
        self.lambda_ = lambda_
        self.input_dims = input_dims
        self.h1_dims = layer1_size
        self.h2_dims = layer2_size
        self.n_actions = n_actions

        self.critic = self.build_polic_network()
        
        self.optimizer = RMSprop(learning_rate=ALPHA)

        self.actions_space = [i for i in range(n_actions)]

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

        #Eligibilty traces are intialized to zero
        # tvs = critic.trainable_variables
        # self.eligibilty = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        
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
        # print(Q,Q_)
        #When done is true no need to take value of next state and change only the target value of present action
        TD_error = reward + self.gamma * Q_ * (1 - int(done)) - Q

        #Update weights
        # weights = weights + self.lr * TD_error * self.eligibilty
        td_el = TD_error * self.eligibilty

        for grad_el in td_el:
            norm = np.linalg.norm(grad_el)
            if norm != 0.0:
                grad_el = grad_el/norm
            # print("he")
        
        self.optimizer.apply_gradients(zip(td_el, self.critic.trainable_variables))

        #Update Eligibility Traces
        # self.eligibilty = [self.gamma * self.lambda_ * elg for elg in self.eligibilty]
        self.eligibilty = [(self.gamma * self.lambda_*self.eligibilty[i])+grad for i,grad in enumerate(grads)]

        # print(TD_error)

    def save_model(self,name):
        self.critic.save(name)

    def load_model(self,name):
        self.critic = load_model(name)
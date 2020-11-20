import numpy as np 
import tensorflow as tf
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import  Dense, Activation, Input
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import time



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
        self.Q_memory = []

        self.accumulate_grad = []
        self.ep_accumulated = 0
        self.optimizer = Adam(learning_rate=self.lr)
        self.actor, self.phi = self.build_polic_network()

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

        #Generate the model with proper inputs and outputs
        actor = Model(inputs=[input], outputs=[probs])
        actor.summary()

        
        # phi.compile(optimizer=Adam(lr=self.lr), loss="custum_loss(Q,y_true,y_pred)")

        #Getting the size of weights including bias terms
        tvs = actor.trainable_weights
        #Intitialize the Accumulate gradient to zeros which has same size as trainable variables from the model
        self.accumulate_grad = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]

        return actor, phi
    
        
    def store_transition(self, observation, action, Q):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.Q_memory.append(Q)
    
    def accumulate_gradient(self):

        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        Q_memory = np.array(self.Q_memory)
        
        #Normalizing Q values
        # Q_memory = Q_memory - np.mean(Q_memory)
        # Q_memory/=np.std(Q_memory)
        
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
        input_dims = 8,decay=0.01):

        self.gamma = Gamma
        # self.lr = ALPHA
        # self.constantlr = ALPHA
        self.decay = decay
        self.lambda_ = lambda_
        self.input_dims = input_dims
        self.n_actions = n_actions        
        self.actions_space = [i for i in range(n_actions)]

        self.linear_weights = np.random.randn(input_dims, 1).ravel() / np.sqrt(input_dims)

        self.optimizer = RMSprop(self.linear_weights,alpha=ALPHA)

        # self.eligibilty = np.zeros(input_dims)
    
    def predict(self, actor_hidden_out, probs, action):
        
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

        return np.dot(self.linear_weights, Q_input)
    
    def initialize_eligibility(self, actor_output, action):
        
        output_size = len(actor_output)
        phi = np.pad(actor_output,(action*output_size,\
            (self.n_actions-1-action)*output_size),'constant')

        self.eligibilty = phi

    def learn(self, reward, next_state,next_action, Q, done, actor,iterations,batch_size):

        #Output of last hidden layer of actor network
        actor_output = actor.phi.predict(next_state)[0]
        output_size = len(actor_output)
        #probability of actions
        probs = actor.actor.predict(next_state)[0]

        #Get Q for next state-action pair
        Q_ = self.predict(actor_output, probs, next_action)
        # print(Q,Q_)
        #When done is true no need to take value of next state and change only the target value of present action
        TD_error = reward + self.gamma * Q_ * (1 - int(done)) - Q

        #Normalizing

        td_el = TD_error * self.eligibilty
        norm = np.linalg.norm(td_el)
        if norm != 0.0:
            td_el = td_el/norm 
        
        # self.lr *= (1. / (1. + self.decay * iterations))

        #Update weights
        # self.linear_weights = self.linear_weights + self.lr * td_el
        self.linear_weights = self.optimizer.backward_pass(td_el)
        # temp = max(self.linear_weights)
        # if(temp>5.0):
        #     print(temp)
        phi_ = np.pad(actor_output,(next_action*output_size,\
            (self.n_actions-1-next_action)*output_size),'constant')

        #Update Eligibility Traces
        self.eligibilty = self.gamma * self.lambda_ * self.eligibilty + phi_
        # norm = np.linalg.norm(self.eligibilty)
        # if(norm != 0.0):
        #     self.eligibilty = self.eligibilty/norm
        
        # if((iterations+1)%batch_size==0):
        #     self.lr = self.constantlr

        # print(TD_error)

    def save_model(self,name):
        np.save(name,self.linear_weights)

    def load_model(self,name):
        self.linear_weights = np.load(name)
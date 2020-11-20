from keras.layers import  Dense, Activation, Input
from keras.models import Model, load_model 
import keras.backend as K 
import numpy as np 
import tensorflow as tf
from tensorflow.keras.optimizers import Adam



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
        self.optimizer = Adam(lr=ALPHA)
        self.actor, self.policy = self.build_polic_network()

        # self.iterate = self.gradCalculator()

        self.actions_space = [i for i in range(n_actions)]


    def custum_loss(self,Q,y_true,y_pred):
        out = K.clip(y_pred, 1e-8,1-1e-8)

        log_lik = y_true*K.log(out)

        return K.sum(-log_lik*Q)


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
        
        #Actor Network
        actor = Model(inputs = [input, advantages], outputs = [probs])

        actor.summary()
        
        predict = Model(inputs=[input], outputs=[probs])
        predict.compile(optimizer=Adam(lr=self.lr), loss=custum_loss)

        tvs = predict.trainable_weights
        self.accumulate_grad = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        # print("1st",self.accumulate_grad[0])

        return actor, predict
    
    def gradCalculator(self):
        Q_memory =[]
        actions = []
        outputTensor = self.policy.output

        listOfVariableTensors = self.policy.trainable_weights

        loss = self.custum_loss(Q_memory,actions, outputTensor)
        grads = K.gradients(loss, listOfVariableTensors)
        iterate = K.function([self.policy.input,Q_memory,actions], [loss,grads])

        return iterate
        
    def store_transition(self, observation, action, Q):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.Q_memory.append(Q)
    
    def accumulate_gradient(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        Q_memory = np.array(self.Q_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)),action_memory] = 1

        # loss_history = []

        #To find the gradient for the loss per episode using tf.GradientTape() when using 
        #Tf.keras model

        # with tf.GradientTape() as tape:
        #     logits = self.policy(state_memory)
        #     tvs = self.policy.trainable_variables
        #     loss_value = self.custum_loss(Q_memory,actions, logits)

        # print("loss=",loss_value.numpy())
        # print("Weights=",tvs)
        # loss_history.append(loss_value.numpy().mean())

        # grads = tape.gradient(loss_value, tvs)

        # print("grads=",grads)
        
        #To find the gradient for the loss per episode in keras
        

        outputTensor = self.policy.output

        listOfVariableTensors = self.policy.trainable_weights

        loss = self.custum_loss(Q_memory,actions, outputTensor)
        grads = K.gradients(loss, listOfVariableTensors)
        iterate = K.function([self.policy.input], [loss,grads])

        loss_value, grads_value = iterate([state_memory])


         #Add to the accumulated grad
        self.ep_accumulated = self.ep_accumulated + 1
        #Storing average of grad till now
        # self.accumulate_grad = [self.accumulate_grad[i]*((self.ep_accumulated-1)/self.ep_accumulated) + (grad)/self.ep_accumulated for i, grad in enumerate(grads)]
        self.accumulate_grad = [self.accumulate_grad[i].assign_add(grad) for i, grad in enumerate(grads_value)]
        # print("acc=",self.accumulate_grad[0])
        self.state_memory = []
        self.action_memory = []
        self.Q_memory = []


    
    def learn(self):
        self.optimizer.apply_gradients(zip(self.accumulate_grad, self.policy.trainable_weights))
        tvs = self.policy.trainable_weights
        self.accumulate_grad = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state,batch_size=len(state))[0]
        action = np.random.choice(self.actions_space, p=probabilities)

        return action

    def save_model(self,name):
        self.policy.save(name)

    def load_model(self,name):
        self.policy = load_model(name, compile=False)

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
        self.optimizer = tf.keras.optimizers.Adam()
        self.actions_space = [i for i in range(n_actions)]

    def build_polic_network(self):
        input = Input(shape=(self.input_dims,))
        # no hidden layer
        if(self.h1_dims == 0 and self.h2_dims==0):
            Q_values = Dense(self.n_actions, activation='linear')(input)
        #One hidden layer
        elif(self.h1_dims != 0 and self.h2_dims == 0):
            dense1 = Dense(self.h1_dims,activation='relu')(input)
            Q_values = Dense(self.n_actions, activation='linear')(dense1)
        #Two hidden layers
        else:
            dense1 = Dense(self.h1_dims,activation='relu')(input)
            dense2 = Dense(self.h2_dims, activation='relu')(dense1)
            Q_values = Dense(self.n_actions, activation='linear')(dense2)

        critic = Model(inputs = [input], outputs = [Q_values])
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

    def load_model(self,name):
        self.critic = load_model(name)
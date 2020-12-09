import numpy as np
import gym
import os

# import necessary modules from keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
gamma = 0.99 # discount factor for reward
resume = False # resume from previous checkpoint?
D = 80 * 80 # input dimensionality: 80x80 grid

# model initialization
# creates a generic neural network architecture
model = Sequential()
model.add(Dense(units=200, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

# compile the model using traditional Machine Learning losses and optimizers
model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

# Methods from Karpathy
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r) #normalizing the result
    discounted_r /= np.std(discounted_r) #idem
    return discounted_r

# gym initialization
env = gym.make("Pong-v0")
observation = env.reset()
prev_input = None

# initialization of variables used in the main loop
xs, ys, rewards = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 1

# load pre-trained model if exist
if (resume and os.path.isfile('Keras\\SmartPong_weights.h5')):
    print("loading previous weights")
    model.load_weights('SmartPong_weights.h5')

# main loop
while (True):

    # preprocess the observation, set input as difference between images
    cur_input = prepro(observation)
    x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
    prev_input = cur_input

    # forward the policy network and sample action according to the proba distribution
    proba = model.predict(np.expand_dims(x, axis=1).T)
    action = 2 if np.random.uniform() < proba else 3
    y = 1 if action == 2 else 0 # 0 and 1 are our labels

    # log the input and label to train later
    xs.append(x)
    ys.append(y)

    # do one step in our environment
    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    reward_sum += reward

    # end of an episode
    if done:
        print('At the end of episode', episode_number, 'the total reward was :', reward_sum)

        # increment episode number
        episode_number += 1

        # training
        model.fit(x=np.vstack(xs), y=np.vstack(ys), verbose=0, sample_weight=discount_rewards(rewards, gamma))

        # Saving the weights used by our model
        if episode_number % batch_size == 0:
            model.save_weights('SmartPong_weights.h5')

        # Log the reward
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

        # Reinitialization
        xs, ys, rewards = [],[],[]
        observation = env.reset()
        reward_sum = 0
        prev_input = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print('ep %d: game finished, reward: %f' % (episode_number, reward) + '' if reward == -1 else '!!!!!!!!')

import gym
import time
import numpy as np
import ppo
import tensorflow as tf
import tensorflow_probability as tfp
tf.enable_eager_execution()
import os
import cv2
import matplotlib.pylab as plt
import env_wrapper
import rollout

#Hyper params:
TRAIN_MODE = False
NUM_ACTION = 4
ENV_GAME_NAME = 'Breakout-v0'
VALUE_FACTOR = 1.0
ENTROPY_FACTOR = 0.01
EPSILON = 0.1
LR = 2.5e-4
LR_DECAY = 'linear'
GRAD_CLIP = 0.5
TIME_HORIZON = 128
BATCH_SIZE = 32
GAMMA = 0.99
LAM = 0.95
EPOCH = 4
ACTORS = 1
FINAL_STEP = 10e6
STATE_SHAPE = [84, 84, 4]

try:  
    os.mkdir('./saved')
except OSError:  
    print ("Creation of the directory failed")
else:  
    print ("Successfully created the directory")

def _process_obs(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs[None, :, :, None] / 256 # Shape (84, 84, 1)
    #return np.reshape(obs, (1, 6, 1))

envs = env_wrapper.EnvWrapper(ENV_GAME_NAME, ACTORS, update_obs=_process_obs)#gym.make('Breakout-v0')
rollouts = [rollout.Rollout() for _ in range(ACTORS)]

global ep_ave_max_q_value
ep_ave_max_q_value = 0
global total_reward
total_reward = 0


def create_tensorboard():
    global_step = tf.train.get_or_create_global_step()

    logdir = "./logs/"
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()
    return global_step, writer

def to_grayscale(im, weights = np.c_[0.2989, 0.5870, 0.1140]):
    """
    Transforms a colour image to a greyscale image by
    taking the mean of the RGB values, weighted
    by the matrix weights
    """
    tile = np.tile(weights, reps=(im.shape[0],im.shape[1],1))
    return np.sum(tile * im, axis=2) / 256.0


def make_epsilon_greedy_policy(model, nA):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        dist, value = model.predict(observation)
        best_action = np.argmax(dist)
        A[best_action] += (1.0 - epsilon)
        return A, value
    return policy_fn

def plti(im, h=8, **kwargs):
    """
    Helper function to plot an image.
    """
    y = im.shape[0]
    x = im.shape[1]
    w = (y/x) * h
    plt.figure(figsize=(w,h))
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off')
    plt.show()

global global_step
global_step, writer = create_tensorboard()
actorCritic = ppo.ActorCritic(NUM_ACTION, LR, EPSILON, FINAL_STEP)
actorCriticOld = ppo.ActorCritic(NUM_ACTION, LR, EPSILON, FINAL_STEP)

try:
    actorCriticOld.load()
except Exception as e:
    print('failed to load')

epsilon_decay_steps = 50000
episode = 0
opti_step = -1
log_probs = []
values    = []
states    = []
actions   = []
rewards   = []
masks     = []

def train(next_value):
    values = []
    rewards = []
    masks = []
    actions = []
    log_probs = []
    states = []

    for rollout in rollouts:
        obs_d, actions_d, rewards_d, values_d, log_probs_d, terminals_d = rollout.get_storage()
        actions.append(actions_d)
        states.append(obs_d)
        rewards.append(rewards_d)
        values.append(values_d)
        log_probs.append(log_probs_d)
        masks.append(terminals_d)

    values = np.array(values)
    rewards = np.array(rewards)
    masks = np.array(masks)
    actions = np.array(actions)
    log_probs = np.array(log_probs)
    states = np.array(states)

    print('values', values.shape)
    print('rewards', rewards.shape)
    print('masks', masks.shape)
    print('actions', actions.shape)
    print('log_probs', log_probs.shape)
    print('states', states.shape)

    #_, next_value = actorCritic.model.predict(stack)
    returns = ppo.compute_returns(rewards, next_value, masks, GAMMA)

    advantage = ppo.compute_gae(rewards, values, next_value, masks, GAMMA, LAM)

    advantage = (advantage - np.mean(advantage)) / np.std(advantage)

    print('advantage', advantage.shape)
    print('returns', returns.shape)

    indices = np.random.permutation(range(TIME_HORIZON))
    states = states[:, indices]
    actions = actions[:, indices]
    log_probs = log_probs[:, indices]
    returns = returns[:, indices]
    advantage = advantage[:, indices]
    masks = masks[:, indices]

    print('states.shape: BEFORE TRAIN', states.shape)
    loss = ppo.update(actorCritic, TIME_HORIZON, EPOCH, BATCH_SIZE, states, actions, log_probs, returns, advantage, GRAD_CLIP, VALUE_FACTOR, ENTROPY_FACTOR, STATE_SHAPE)
    actorCriticOld.hard_copy(actorCritic.model.trainable_variables)

    print('loss:', loss)

    actorCritic.save()

    for rollout in rollouts:
        rollout.flush()

    pass

t = 0
for episode in range(10000):
    global_step.assign_add(1)

    batch_obs = envs.reset()

    j = 0
    ep_ave_max_q_value = 0
    total_reward = 0

    entropy = 0

    batch_stack =[]
    for obs in batch_obs:
        stack = np.concatenate([obs, obs], axis=-1)
        stack = np.concatenate([stack, obs], axis=-1)
        stack = np.concatenate([stack, obs], axis=-1)
        batch_stack.append(stack)

    while not envs.done():
        #batch_stack = batch_obs
        if not TRAIN_MODE:
            envs.render(0)

        actions_t = []
        dists_t = []
        values_t = []
        dist_cat_t = []
        entropy_t = []

        for stack in batch_stack:
            dist, value = actorCriticOld.model.predict(stack)
            distCat = tf.distributions.Categorical(probs=tf.nn.softmax(dist))
            action = distCat.sample(1)[0]

            entropy_t.append(distCat.entropy())
            actions_t.append(action)
            dists_t.append(dist)
            dist_cat_t.append(distCat)
            values_t.append(value)

        obs2s_t, rewards_t, dones_t = envs.step(actions_t)
        total_reward += np.mean(rewards_t)
        entropy += np.mean(entropy_t)

        if t > 0 and (t / ACTORS) % TIME_HORIZON == 0 and TRAIN_MODE:
            next_values = np.reshape(values_t, [-1])
            train(next_values)

        if TRAIN_MODE:
            for i, rollout in enumerate(rollouts):
                log_prob = dist_cat_t[i].log_prob(actions_t[i])
                rollout.add(batch_stack[i][0,:], actions_t[i][0], rewards_t[i], values_t[i][0][0], log_prob[0], 1 - dones_t[i])
            #rollout.add(batch_stack[i][0,:], actions_t[i][0], rewards_t[i] / 10, values_t[i][0][0], log_prob[0], 1 - dones_t[i])

        t += ACTORS
        j += ACTORS


        #batch_obs = obs2s_t
        for i, stack in enumerate(batch_stack):
            stack = stack[:,:,:,1:]
            batch_stack[i] = np.concatenate([stack, obs2s_t[i]], axis=-1)

        if LR_DECAY == 'linear':
            actorCriticOld.decay_clip_param(opti_step)
            actorCriticOld.decay_learning_rate(opti_step)

    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("entropy", entropy / float(j))
        tf.contrib.summary.scalar("reward", total_reward)
    
    print('TOTAL REWARD: ', total_reward, ' ENTROPY: ', entropy / float(j))

env.close()
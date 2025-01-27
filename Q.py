import numpy as np
import random as random
from tqdm.notebook import trange
import gym

def init_q_table(states: tuple, actions: int):
    """Initialize a Q-table as a numpy array of zeros."""
    Q_table = np.zeros(states + (actions, ))
    return Q_table

def greedy_policy(Q_table, state):
    """Returns an action according to the greedy policy."""
    action = np.argmax(Q_table[state])
    return action

def epsilon_greedy_policy(Q_table, state, env, epsilon):
    """Returns an action according to the epsilon greedy policy."""
    random_int = random.uniform(0,1)
    if random_int > epsilon:
        action = greedy_policy(Q_table, state)
    else:
        action = env.action_space.sample()
    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, alpha, gamma, env, max_steps, Q_table, render=False, disc_func=lambda x: x, discretize=True):
    """
    Train with Q-learning.

    Args:
    n_training_episodes (int): Number of training episodes.
    min_epsilon (float): Minimum exploration rate.
    max_epsilon (float): Maximum exploration rate.
    decay_rate (float): Rate of decay for exploration rate.
    alpha (float): Learning rate.
    gamma (float): Discount factor.
    env (gym env): Gym learning environment.
    max_steps (int): Maximum amount of steps in each episode.
    Q_table (numpy array): Q-learning table.
    render (bool, optional): True if information about steps should be printed.
    disc_func (function, optional): A function for discretizing states if necessary.
    
    Returns:
    Q_table (numpy array): Updated Q-learning table.
    all_stats (list): List of episode stats / information.
    """
    all_stats = []

    for episode in trange(n_training_episodes):

        episode_stats = []

        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # Reset environment
        observation = env.reset(discretize=discretize)
        # If observation implemented as dictionary transform to tuple
        if isinstance(observation, dict):
            observation = tuple(observation.values())
        state = disc_func(observation)
        step = 0
        done = False

        # Repeat as long as max_steps or done
        for step in range(max_steps):

            action = epsilon_greedy_policy(Q_table, state, env, epsilon)

            new_observation, reward, done, truncated, info = env.step(action, discretize=discretize)
            if isinstance(new_observation, dict):
                new_observation = tuple(new_observation.values())
            new_state = disc_func(new_observation)

            # Set new value in Q-table
            Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state][action])

            if render:
                env.render(action, reward, info)

            # Set new state
            state = new_state

            step_stats = {
                'state': state,
                'action': action,
                'reward': reward,
                'done': done,
                'truncated': truncated,
                'info': info
            }
            episode_stats.append(step_stats)

            # Finish Episode if done
            if done:
                break

        all_stats.append(episode_stats)

    return Q_table, all_stats

def eval_agent(env, max_steps, n_eval_episodes, Q_table, seed, disc_func=lambda x: x, render=False, discretize=True):
    """
    Evaluate agent trained with Q-learning.

    Args:
        env (gym env): Gym learning environment.
        max_steps (int): Maximum amount of steps in each episode.
        Q_table (numpy array): Q-learning table.
        seed (): Gymnasium environment seed.
        disc_func (function, optional): A function for discretizing states if necessary.
        render (bool, optional): Whether to use the render function of the environment.
        

    Returns:
        mean_reward (float): Mean reward of epsiodes.
        std_reward (float): Standard deviation of rewards of epsiodes.
    """

    episode_rewards = []
    all_stats = []
    for episode in range(n_eval_episodes):
        if seed:
            observation = env.reset(seed=seed[episode], discretize=discretize)
        else:
            observation = env.reset(discretize=discretize)
        if isinstance(observation, dict):
            observation = tuple(observation.values())
        state = disc_func(observation)
        step = 0
        done = False
        total_rewards_ep = 0
        episode_stats = []

        for step in range(max_steps):
            # Take the action (index) that have the maximum reward
            action = greedy_policy(Q_table, state)

            next_observation, reward, done, truncated, info = env.step(action, discretize=discretize)
            if isinstance(next_observation, dict):
                next_observation = tuple(next_observation.values())
            next_state = disc_func(next_observation)
            total_rewards_ep += reward

            step_stats = {
                'state': state,
                'action': action, 
                'reward': reward,
                'done': done,
                'truncated': truncated,
                'info': info
                }
            
            episode_stats.append(step_stats)
            
            if render:
                env.render(action, reward, info)

            if done:
                break

            state = next_state

        episode_rewards.append(total_rewards_ep)
        all_stats.append(episode_stats)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

    return all_stats